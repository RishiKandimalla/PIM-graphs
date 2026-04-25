#include "ramulator_hbm.hpp"

#include "base/config.h"
#include "base/factory.h"
#include "base/request.h"
#include "frontend/frontend.h"
#include "memory_system/memory_system.h"

#include <cmath>
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

  const std::uint64_t target_bursts = mux_.total_pending_bursts();
  std::uint64_t completed = 0;
  std::uint64_t bytes_done = 0;

  std::uint64_t cycles = 0;

  while (cycles < max_memory_cycles && completed < target_bursts) {
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

        if (tsv_peak_gbps_ > 0.0 && tsv_used_this_tick + sz_d > tsv_limit_per_tick + 1e-6) {
          continue;
        }

        Ramulator::Request req(
            burst.addr, Ramulator::Request::Type::Read, -1,
            [&, sz](Ramulator::Request& /*r*/) {
              completed++;
              bytes_done += sz;
            });

        if (memory_system_->send(std::move(req))) {
          port.pop();
          mux_.advance_after_issue(pc);
          tsv_used_this_tick += sz_d;
          out.tsv_bytes_admitted += sz;
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
