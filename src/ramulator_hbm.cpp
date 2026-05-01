#include "ramulator_hbm.hpp"

#include "base/config.h"
#include "base/factory.h"
#include "base/request.h"
#include "frontend/frontend.h"
#include "memory_system/memory_system.h"

#include <limits>
#include <utility>

namespace pim {

namespace {

/// Max full RR sweeps per memory cycle (each sweep tries each PC at most once).
inline constexpr int kMaxWavesPerTick = 256;

}  // namespace

double SimulationResult::effective_gbps() const {
  if (memory_cycles == 0 || tck_ns <= 0.0) {
    return 0.0;
  }
  const double seconds = static_cast<double>(memory_cycles) * (tck_ns * 1.0e-9);
  if (seconds <= 0.0) {
    return 0.0;
  }
  return static_cast<double>(bytes_moved) / seconds / 1.0e9;
}

RamulatorHbmSimulator::RamulatorHbmSimulator(std::string config_path) : config_path_(std::move(config_path)) {
  YAML::Node config = Ramulator::Config::parse_config_file(config_path_, {});
  frontend_ = Ramulator::Factory::create_frontend(config);
  memory_system_ = Ramulator::Factory::create_memory_system(config);
  frontend_->connect_memory_system(memory_system_); 
  memory_system_->connect_frontend(frontend_);
}

void RamulatorHbmSimulator::finalize() {
  if (frontend_) {
    frontend_->finalize();
  }
  if (memory_system_) {
    memory_system_->finalize();
  }
}

SimulationResult RamulatorHbmSimulator::run(std::uint64_t max_memory_cycles) {
  SimulationResult out;
  out.tck_ns = static_cast<double>(memory_system_->get_tCK());
  out.tsv_peak_gbps = tsv_peak_gbps_;

  const double tsv_limit_per_tick = (tsv_peak_gbps_ > 0.0)
                                        ? tsv_peak_gbps_ * 1.0e9 * (out.tck_ns * 1.0e-9)
                                        : std::numeric_limits<double>::infinity();

  std::uint64_t completed = 0;
  std::uint64_t bytes_done = 0;
  std::uint64_t outstanding_requests = 0;
  std::uint64_t standard_sparse_inflight = 0;

  std::uint64_t cycles = 0;

  auto has_work = [&]() {
    //Check if ANY VPU is still executing instructions
    bool any_vpu_active = false;
    for (int i = 0; i < 16; ++i) {
        if (vpus_[i] != nullptr && !vpus_[i]->is_done()) {
            any_vpu_active = true;
            break; 
        }
    }

    //Check if the Piccolo controller is still gathering
    const bool piccolo_active = (piccolo_ != nullptr && !piccolo_->done());

    //Check the memory system
    const bool memory_active = (mux_.total_pending_bursts() > 0 || outstanding_requests > 0);

    return any_vpu_active || piccolo_active || memory_active;
};

  while (cycles < max_memory_cycles && has_work()) {
    if (piccolo_ != nullptr && !piccolo_->done()) {
      piccolo_->tick_issue();
    }
    if (router_ != nullptr) {
      router_->tick();
    }
    std::array<PartialSumAccumulationUnit::VectorType, 16> tree_inputs{};
    bool has_valid_tree_inputs = false;

    for (int i = 0; i < 16; ++i) {
      if (vpus_[i] != nullptr) {
        //Tick the VPU instruction stream
        vpus_[i]->tick(); 

        //Tick the Feeder and Systolic Array for this vault 
        if (feeders_[i] != nullptr && systolic_arrays_[i] != nullptr) {
          auto staggered_acts = feeders_[i]->get_next_activations();
          std::array<float, 16> zero_psums = {0.0f}; // Base aggregation phase
          
          // Capture partial vectors dropping out of the bottom row 
          tree_inputs[i] = systolic_arrays_[i]->tick(staggered_acts, zero_psums);

          // Check if this vault is contributing real data to the reduction tree
          for(float val : tree_inputs[i]) {
            if (std::abs(val) > 1e-6f) { has_valid_tree_inputs = true; break; }
          }
        }
      }
    }

    if (psau_ != nullptr) {
      PartialSumAccumulationUnit::VectorType final_reduced_vector;
      // Pipeline reduction takes 4 cycles 
      bool tree_output_valid = psau_->tick(tree_inputs, has_valid_tree_inputs, final_reduced_vector);

      // If a fully reduced vector emerges, write it back to HBM 
      if (tree_output_valid && router_ != nullptr) {
          MemoryBurst writeback;
          writeback.addr = 0x80000000ull; // Target global address for vertex update
          writeback.size_bytes = 64;      // 16 floats = 64B burst 
          writeback.pc_id = 0;            // Logic-layer directed write
          router_->enqueue_request(writeback);
      }
    }

    double tsv_used_this_tick = 0.0;

    int wave_count = 0;
    bool wave_progress = true;
    while (wave_progress && wave_count < kMaxWavesPerTick) {
      wave_count++;
      wave_progress = false;
      const int sweep_start = mux_.round_robin_cursor();

      for (int k = 0; k < kNumPseudoChannels; k++) {
        const int pc = (sweep_start + k) % kNumPseudoChannels;
        auto& port = mux_.port(pc);
        if (port.empty()) {
          continue;
        }

        MemoryBurst burst = port.front();
        const std::uint32_t sz = burst.size_bytes;
        const double sz_d = static_cast<double>(sz);
        const bool is_standard_sparse_read = (sz >= 64 && !burst.on_complete);

        if (serialize_standard_sparse_reads_ && is_standard_sparse_read &&
            standard_sparse_inflight > 0) {
          continue;
        }

        if (tsv_peak_gbps_ > 0.0 && tsv_used_this_tick + sz_d > tsv_limit_per_tick + 1e-6) {
          continue;
        }

        auto on_complete = burst.on_complete;
        const bool track_as_standard_sparse = is_standard_sparse_read;
        Ramulator::Request req(burst.addr, Ramulator::Request::Type::Read, -1,
                               [&, sz, on_complete, track_as_standard_sparse](Ramulator::Request& /*r*/) {
                                 completed++;
                                 bytes_done += sz;
                                 if (outstanding_requests > 0) {
                                   outstanding_requests--;
                                 }
                                 if (track_as_standard_sparse && standard_sparse_inflight > 0) {
                                   standard_sparse_inflight--;
                                 }
                                 if (on_complete) {
                                   on_complete();
                                 }
                               });

        if (memory_system_->send(std::move(req))) {
          port.pop();
          mux_.advance_after_issue(pc);
          tsv_used_this_tick += sz_d;
          out.tsv_bytes_admitted += sz;
          outstanding_requests++;
          if (track_as_standard_sparse) {
            standard_sparse_inflight++;
          }
          wave_progress = true;
        }
      }
    }

    memory_system_->tick();
    frontend_->tick();
    cycles++;
  }

  out.memory_cycles = cycles;
  out.bytes_moved = bytes_done;
  out.bursts_completed = completed;
  return out;
}

}  // namespace pim
