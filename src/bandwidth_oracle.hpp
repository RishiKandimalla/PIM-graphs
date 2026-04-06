#pragma once

#include <cmath>
#include <cstdint>
#include <limits>

namespace pim {

/// Byte-smoothed bandwidth model (no DRAM timing): cycles needed to move `total_bytes`
/// at a fixed effective `bytes_per_cycle` (e.g. derived from a 512 GB/s cap and core clock).
[[nodiscard]] inline double oracle_cycles_for_bytes(std::uint64_t total_bytes, double bytes_per_cycle) {
  if (bytes_per_cycle <= 0.0) {
    return std::numeric_limits<double>::infinity();
  }
  return std::ceil(static_cast<double>(total_bytes) / bytes_per_cycle);
}

/// Convert tCK in nanoseconds to clock frequency in Hz.
[[nodiscard]] inline double tck_ns_to_hz(double tck_ns) {
  if (tck_ns <= 0.0) {
    return 0.0;
  }
  return 1.0e9 / tck_ns;
}

/// Theoretical bytes/cycle if the interface sustained `target_gbps` (decimal GB/s) at `clock_hz`.
[[nodiscard]] inline double bytes_per_cycle_from_target_gbps(double target_gbps, double clock_hz) {
  const double bytes_per_s = target_gbps * 1.0e9;
  return bytes_per_s / clock_hz;
}

}  // namespace pim
