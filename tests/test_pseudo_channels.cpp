#include "bandwidth_oracle.hpp"
#include "pseudo_channel.hpp"
#include "ramulator_hbm.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdlib>
#include <string>

#ifndef PIM_DEFAULT_CONFIG
#define PIM_DEFAULT_CONFIG "configs/hbm_pnm.yaml"
#endif

namespace {

std::string config_path() {
  if (const char* e = std::getenv("PIM_CONFIG")) {
    return std::string(e);
  }
  return PIM_DEFAULT_CONFIG;
}

}  // namespace

TEST(BandwidthOracle, BytesPerCycleFromTarget) {
  const double hz = 1.0e9;  // 1 GHz
  const double bpc = pim::bytes_per_cycle_from_target_gbps(512.0, hz);
  EXPECT_DOUBLE_EQ(bpc, 512.0);
}

TEST(PseudoChannelMux, RegionsAreDisjoint) {
  pim::PseudoChannelMultiplexer mux;
  mux.load_sequential_workload(2, 64);
  EXPECT_EQ(mux.total_pending_bursts(), pim::kNumPseudoChannels * 2u);
  std::uint64_t addr0 = mux.port(0).front().addr;
  std::uint64_t addr1 = mux.port(1).front().addr;
  EXPECT_LT(addr0 + 128, addr1);
}

TEST(RamulatorHbm, SinglePcSmoke) {
  pim::RamulatorHbmSimulator sim(config_path());
  constexpr std::uint32_t kBursts = 4;
  for (std::uint32_t i = 0; i < kBursts; i++) {
    pim::MemoryBurst b;
    b.addr = static_cast<std::uint64_t>(i) * 64;
    b.size_bytes = 64;
    b.pc_id = 0;
    sim.mux().port(0).enqueue(b);
  }

  pim::SimulationResult r = sim.run();
  sim.finalize();

  EXPECT_EQ(r.bursts_completed, kBursts);
  EXPECT_EQ(r.bytes_moved, kBursts * 64);
  EXPECT_EQ(r.tsv_bytes_admitted, r.bytes_moved);
  EXPECT_DOUBLE_EQ(r.tsv_peak_gbps, 512.0);
  EXPECT_GT(r.memory_cycles, 0u);
  EXPECT_GT(r.effective_gbps(), 0.0);
}

TEST(RamulatorHbm, SixteenStreamsSequential) {
  pim::RamulatorHbmSimulator sim(config_path());
  constexpr std::uint32_t kBurstsPerPc = 8;
  sim.mux().load_sequential_workload(kBurstsPerPc, 64);

  const std::uint64_t expected_bursts = pim::kNumPseudoChannels * kBurstsPerPc;
  const std::uint64_t expected_bytes = expected_bursts * 64;

  pim::SimulationResult r = sim.run();
  sim.finalize();

  EXPECT_EQ(r.bursts_completed, expected_bursts);
  EXPECT_EQ(r.bytes_moved, expected_bytes);
  EXPECT_EQ(r.tsv_bytes_admitted, r.bytes_moved);
  EXPECT_DOUBLE_EQ(r.tsv_peak_gbps, 512.0);
  EXPECT_GT(r.memory_cycles, 0u);
  EXPECT_GT(r.effective_gbps(), 0.0);

  const double clock_hz = pim::tck_ns_to_hz(r.tck_ns);
  ASSERT_GT(clock_hz, 0.0);
  const double oracle_bpc = pim::bytes_per_cycle_from_target_gbps(512.0, clock_hz);
  const double oracle_cycles =
      pim::oracle_cycles_for_bytes(r.bytes_moved, oracle_bpc);
  EXPECT_GT(oracle_cycles, 0.0);
  (void)oracle_bpc;
  // Sanity: effective throughput should not exceed an absurd multiple of the 512 GB/s design target.
  EXPECT_LT(r.effective_gbps(), 4096.0);
}

TEST(RamulatorHbm, UnlimitedTsvStillRuns) {
  pim::RamulatorHbmSimulator sim(config_path());
  sim.set_tsv_peak_gbps(0.0);
  sim.mux().load_sequential_workload(2, 64);
  pim::SimulationResult r = sim.run();
  sim.finalize();
  EXPECT_EQ(r.bursts_completed, pim::kNumPseudoChannels * 2u);
  EXPECT_DOUBLE_EQ(r.tsv_peak_gbps, 0.0);
  EXPECT_EQ(r.tsv_bytes_admitted, r.bytes_moved);
}
