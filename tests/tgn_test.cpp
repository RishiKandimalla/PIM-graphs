#include <gtest/gtest.h>

#include "ramulator_hbm.hpp"
#include "tgn_twophase.hpp"
#include "gru.hpp"
#include "psau.hpp"
#include "systolic_array.hpp"
#include "array_feeder.hpp"
#include "piccolo_controller.hpp"
#include "pseudo_channel.hpp"
#include "hierarchical_router.hpp"
#include "simd_core.hpp"

#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

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

void log(const std::string& msg) {
  std::cout << "  [log]  " << msg << "\n";
}

void print_stats(const pim::VertexProgramStats& s) {
  std::cout << "\n  ┌─ VertexProgramStats ──────────────────────────┐\n";
  std::cout << "  │  gather_cycles   : " << std::setw(10) << s.gather_cycles  << "              │\n";
  std::cout << "  │  compute_cycles  : " << std::setw(10) << s.compute_cycles << "              │\n";
  std::cout << "  │  psau_cycles     : " << std::setw(10) << s.psau_cycles    << "              │\n";
  std::cout << "  │  load_h_cycles   : " << std::setw(10) << s.load_h_cycles  << "              │\n";
  std::cout << "  │  gru_cycles      : " << std::setw(10) << s.gru_cycles     << "              │\n";
  std::cout << "  │  total_cycles    : " << std::setw(10) << s.total_cycles   << "              │\n";
  std::cout << "  │  vertices_done   : " << std::setw(10) << s.vertices_done  << "              │\n";
  std::cout << "  └───────────────────────────────────────────────┘\n";
}

void print_sim_result(const pim::SimulationResult& r) {
  std::cout << "\n  ┌─ SimulationResult ────────────────────────────┐\n";
  std::cout << "  │  memory_cycles      : " << std::setw(10) << r.memory_cycles    << "          │\n";
  std::cout << "  │  bytes_moved        : " << std::setw(10) << r.bytes_moved      << "          │\n";
  std::cout << "  │  bursts_completed   : " << std::setw(10) << r.bursts_completed << "          │\n";
  std::cout << "  │  effective_gbps     : " << std::setw(10) << std::fixed
            << std::setprecision(2) << r.effective_gbps()                          << "          │\n";
  std::cout << "  │  tsv_bytes_admitted : " << std::setw(10) << r.tsv_bytes_admitted<<"          │\n";
  std::cout << "  │  vertices_processed : " << std::setw(10) << r.vertices_processed<<"          │\n";
  std::cout << "  └───────────────────────────────────────────────┘\n";
}

std::vector<float> identity_weights() {
  std::vector<float> W(256, 0.0f);
  for (int i = 0; i < 16; ++i) W[i * 16 + i] = 1.0f;
  return W;
}

} // namespace

struct HardwareFixture {
  std::array<std::unique_ptr<pim::SystolicArray>,          16> systolic_arrays;
  std::array<std::unique_ptr<pim::ArrayFeeder>,            16> feeders;
  std::array<std::unique_ptr<pim::PiccoloGatherController>,16> piccolos;
  std::unique_ptr<pim::PartialSumAccumulationUnit>             psau;
  std::unique_ptr<pim::RamulatorHbmSimulator>                  sim;
  std::unique_ptr<pim::HierarchicalRouter>                     router;
  std::array<std::unique_ptr<pim::SIMDCore>, 16>              vpus;
  std::unique_ptr<pim::TGNVertexProgram>                       vertex_program;

  std::array<pim::SystolicArray*,           16> sa_ptrs   = {};
  std::array<pim::ArrayFeeder*,             16> feed_ptrs = {};
  std::array<pim::PiccoloGatherController*, 16> pic_ptrs  = {};

  explicit HardwareFixture(const std::string& config_path) {
    sim = std::make_unique<pim::RamulatorHbmSimulator>(config_path);
    router = std::make_unique<pim::HierarchicalRouter>(sim->mux(), 2.0e9);
    sim->attach_router(router.get());
    psau = std::make_unique<pim::PartialSumAccumulationUnit>();
    sim->attach_psau(psau.get());

    for (int v = 0; v < 16; ++v) {
      systolic_arrays[v] = std::make_unique<pim::SystolicArray>();
      feeders[v]         = std::make_unique<pim::ArrayFeeder>();
      piccolos[v]        = std::make_unique<pim::PiccoloGatherController>(v, sim->mux(), 0, 4, 0);
      sa_ptrs[v] = systolic_arrays[v].get();
      feed_ptrs[v] = feeders[v].get();
      pic_ptrs[v] = piccolos[v].get();
      sim->attach_systolic_array(v, sa_ptrs[v]);
      sim->attach_feeder(v, feed_ptrs[v]);
    }

    for (int v = 0; v < 16; ++v) {
      vpus[v] = std::make_unique<pim::SIMDCore>(v, sim->mux(), *router, *feeders[v], *piccolos[v]);
      vpus[v]->attach_sa_and_feeder_for_gru(sa_ptrs[v], feed_ptrs[v]);
      auto W = identity_weights();
      vpus[v]->load_gru_weights(W, W, W, W);
      sim->attach_vpu(v, vpus[v].get());
    }

    vertex_program = std::make_unique<pim::TGNVertexProgram>(pic_ptrs, *psau, sa_ptrs, feed_ptrs);
    auto W = identity_weights();
    vertex_program->load_gru_weights(W, W, W, W);
    sim->attach_vertex_program(vertex_program.get());
  }

  ~HardwareFixture() { sim->finalize(); }
};

// =============================================================================
// TESTS
// =============================================================================

TEST(TGNTest, PhaseOrdering) {
  HardwareFixture hw(config_path());
  pim::VertexProgramConfig cfg;
  cfg.vertex_addr = 0x1000; cfg.hidden_state_addr = 0x2000; cfg.output_addr = 0x3000;
  cfg.num_neighbors = 4; cfg.neighbor_offsets = {0, 1, 2, 3};

  hw.vertex_program->start_vertex(cfg);
  pim::SimulationResult result = hw.sim->run(200'000);

  EXPECT_EQ(hw.vertex_program->get_phase(), pim::VertexProgramPhase::DONE);
  EXPECT_EQ(result.vertices_processed, 1u);
}

// True isolated unit test for the GRU cell mathematics.
TEST(TGNTest, GruOutputValues) {
  pim::GRUCell gru;
  pim::SystolicArray sa;
  pim::ArrayFeeder feeder;

  auto W = identity_weights();
  gru.load_weights(W, W, W, W);

  std::array<float, 16> h_prev; h_prev.fill(1.0f);
  std::array<float, 16> msg;    msg.fill(0.5f);

  gru.start(h_prev, msg);

  int cycles = 0;
  // Tick the hardware locally to guarantee clean wavefront delivery
  while (!gru.is_done() && cycles++ < 5000) {
    gru.tick(sa, feeder);
  }

  // The true mathematical output of the GRU given identity weights.
const float kExpected = 0.89059f; 
const float kTol      = 0.01f;

  const auto& h_new = gru.get_h_new();
  for (int d = 0; d < 16; ++d) {
    EXPECT_NEAR(h_new[d], kExpected, kTol) << "Mismatch at dimension " << d;
  }
}

TEST(TGNTest, MultipleVertices) {
  HardwareFixture hw(config_path());
  for (int vid = 0; vid < 3; ++vid) {
    pim::VertexProgramConfig cfg;
    cfg.vertex_addr = 0x10000 + vid*0x1000; cfg.num_neighbors = 2; cfg.neighbor_offsets = {0, 1};
    hw.vertex_program->start_vertex(cfg);
    pim::SimulationResult r = hw.sim->run(200'000);
    EXPECT_TRUE(hw.vertex_program->is_done());
  }
}

TEST(TGNTest, CycleBudget) {
  HardwareFixture hw(config_path());
  pim::VertexProgramConfig cfg;
  cfg.num_neighbors = 4; cfg.neighbor_offsets = {0, 1, 2, 3};
  hw.vertex_program->start_vertex(cfg);
  pim::SimulationResult r = hw.sim->run(200'000);
  EXPECT_GT(r.vertex_program_stats.gru_cycles, 0u);
  EXPECT_LE(r.vertex_program_stats.total_cycles, 5000u);
}

TEST(TGNTest, NoDoubleWriteback) {
  HardwareFixture hw(config_path());
  pim::VertexProgramConfig cfg;
  cfg.num_neighbors = 2; cfg.neighbor_offsets = {0, 1};
  hw.vertex_program->start_vertex(cfg);
  pim::SimulationResult r = hw.sim->run(200'000);
  EXPECT_EQ(r.vertices_processed, 1u);
}