#pragma once

#include "pseudo_channel.hpp"

#include <cstdint>
#include <memory>
#include <string>

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

  /// Run until all bursts are completed or `max_memory_cycles` reached.
  [[nodiscard]] SimulationResult run(std::uint64_t max_memory_cycles = 500000000ull);

  void finalize();

 private:
  std::string config_path_;
  PseudoChannelMultiplexer mux_;
  Ramulator::IFrontEnd* frontend_ = nullptr;
  Ramulator::IMemorySystem* memory_system_ = nullptr;
  /// 0 = no TSV byte limit; otherwise max bytes per memory cycle ≈ gbps×1e9×tCK.
  double tsv_peak_gbps_ = 512.0;
};

}  // namespace pim
