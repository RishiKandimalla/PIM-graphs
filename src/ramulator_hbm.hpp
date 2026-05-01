#pragma once

#include "piccolo_controller.hpp"
#include "pseudo_channel.hpp"
#include "simd_core.hpp"
#include "hierarchical_router.hpp"
#include "psau.hpp"
#include "tgn_twophase.hpp"

#include <cstdint>
#include <string>
#include <array>

namespace Ramulator {
class IFrontEnd;
class IMemorySystem;
}  // namespace Ramulator

namespace pim {

struct SimulationResult {
  std::uint64_t memory_cycles = 0;
  std::uint64_t bytes_moved = 0;
  std::uint64_t bursts_completed = 0;
  double tck_ns = 0.0;
  /// Bytes charged to the TSV (logic ↔ stack) budget during admission this run.
  std::uint64_t tsv_bytes_admitted = 0;
  /// Peak TSV cap used (GB/s, decimal); 0 means no TSV limit was applied.
  double tsv_peak_gbps = 0.0;
  /// Effective GB/s (decimal) using bytes_moved / (memory_cycles * tck_ns * 1e-9).
  [[nodiscard]] double effective_gbps() const;

  std::uint32_t vertices_processed = 0;
  VertexProgramStats vertex_program_stats{};
};

/// Drives Ramulator 2 (GEM5 frontend + GenericDRAM) with a 16-way PC multiplexer.
class RamulatorHbmSimulator {
 public:
  explicit RamulatorHbmSimulator(std::string config_path);

  RamulatorHbmSimulator(const RamulatorHbmSimulator&) = delete;
  RamulatorHbmSimulator& operator=(const RamulatorHbmSimulator&) = delete;

  [[nodiscard]] PseudoChannelMultiplexer& mux() { return mux_; }
  [[nodiscard]] const PseudoChannelMultiplexer& mux() const { return mux_; }

  /// Cap bytes admitted per memory clock over the TSV (0 = unlimited). Default 512 GB/s per proposal.
  void set_tsv_peak_gbps(double gbps) { tsv_peak_gbps_ = gbps; }
  [[nodiscard]] double tsv_peak_gbps() const { return tsv_peak_gbps_; }
  void attach_piccolo(PiccoloGatherController* piccolo) { piccolo_ = piccolo; }
  void attach_vpu(int index, class SIMDCore* vpu) { vpus_[index] = vpu; }
  void attach_feeder(int index, class ArrayFeeder* feeder) { feeders_[index] = feeder; }
  void attach_systolic_array(int index, class SystolicArray* arr) { systolic_arrays_[index] = arr; }  
  void attach_psau(PartialSumAccumulationUnit* psau) { psau_ = psau; }
  void attach_router(class HierarchicalRouter* router) { router_ = router; }
  void attach_vertex_program(TGNVertexProgram* vp) { vertex_program_ = vp; }

  /// Optional modeling knob for Week-3 comparisons:
  /// serialize non-Piccolo 64B sparse reads to emulate cache-line gather latency.
  void set_serialize_standard_sparse_reads(bool enabled) {
    serialize_standard_sparse_reads_ = enabled;
  }

  /// Run until all bursts are completed or `max_memory_cycles` reached.
  [[nodiscard]] SimulationResult run(std::uint64_t max_memory_cycles = 500000000ull);

  void finalize();

 private:
  std::string config_path_;
  std::array<class SIMDCore*, 16> vpus_ = {nullptr};
  std::array<class ArrayFeeder*, 16> feeders_ = {nullptr};
  std::array<class SystolicArray*, 16> systolic_arrays_ = {nullptr};
  PartialSumAccumulationUnit* psau_ = nullptr;

  class HierarchicalRouter* router_ = nullptr;
  PseudoChannelMultiplexer mux_;
  PiccoloGatherController* piccolo_ = nullptr;

  TGNVertexProgram* vertex_program_ = nullptr;

  Ramulator::IFrontEnd* frontend_ = nullptr;
  Ramulator::IMemorySystem* memory_system_ = nullptr;
  /// 0 = no TSV byte limit; otherwise max bytes per memory cycle ≈ gbps×1e9×tCK.
  double tsv_peak_gbps_ = 512.0;
  bool serialize_standard_sparse_reads_ = false;
};

}  // namespace pim
