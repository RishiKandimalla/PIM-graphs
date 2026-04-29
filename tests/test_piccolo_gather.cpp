#include <gtest/gtest.h>
#include "ramulator_hbm.hpp"
#include "simd_core.hpp"
#include "hierarchical_router.hpp"
#include "array_feeder.hpp"
#include "piccolo_controller.hpp"
#include <vector>
#include <iostream>

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

TEST(PiccoloFIM, GatherVersusBaselineLatency) {
    using namespace pim;

    // =================================================================
    // 1. BASELINE: Scattered Cache Line Fetches (Forcing Row Misses)
    // =================================================================
    RamulatorHbmSimulator baseline_sim(config_path());
    HierarchicalRouter baseline_router(baseline_sim.mux(), 1.0e9);
    baseline_sim.set_serialize_standard_sparse_reads(true);
    ArrayFeeder baseline_feeder;
    
    // Instantiate dummy Piccolo to satisfy SIMD constructor, but do NOT attach it to the sim.
    PiccoloGatherController dummy_piccolo(0, baseline_sim.mux(), 0, 4, 8); 
    SIMDCore baseline_vpu(0, baseline_sim.mux(), baseline_router, baseline_feeder, dummy_piccolo);

    baseline_sim.attach_router(&baseline_router);
    baseline_sim.attach_vpu(0, &baseline_vpu);

    std::vector<SIMDInstruction> baseline_program;
    for (int i = 0; i < 8; ++i) {
        // FIX: Multiply by 4096 to ensure every fetch maps to a different DRAM row.
        // This simulates a truly sparse gather, forcing ~40ns latency per read.
        baseline_program.push_back({SIMDOpcode::LOAD, static_cast<uint32_t>(i), 0, 0, 0x90000000ull + (i * 4096), 64});
    }
    baseline_vpu.load_program(baseline_program);
    
    SimulationResult base_res = baseline_sim.run();
    baseline_sim.finalize();
    uint64_t baseline_cycles = base_res.memory_cycles;

    // =================================================================
    // 2. PICCOLO: Internal Bank Gather (Exploiting Row Hits)
    // =================================================================
    RamulatorHbmSimulator piccolo_sim(config_path());
    HierarchicalRouter piccolo_router(piccolo_sim.mux(), 1.0e9);
    ArrayFeeder piccolo_feeder;
    PiccoloGatherController piccolo(0, piccolo_sim.mux(), 0x80000000ull, 4, 8);
    SIMDCore piccolo_vpu(0, piccolo_sim.mux(), piccolo_router, piccolo_feeder, piccolo);

    piccolo_sim.attach_router(&piccolo_router);
    piccolo_sim.attach_vpu(0, &piccolo_vpu);
    piccolo_sim.attach_piccolo(&piccolo); 

    std::vector<float> mock_offsets = {0.0f, 4.0f, 8.0f, 12.0f, 16.0f, 20.0f, 24.0f, 28.0f};
    std::vector<SIMDInstruction> piccolo_program = {
        {SIMDOpcode::GATHER, 10, 0, 0, 0x80000000ull, 32} 
    };
    piccolo_vpu.load_program(piccolo_program);
    
    // Seed the registers using the backdoor so the GATHER instruction works
    for (size_t i = 0; i < mock_offsets.size(); ++i) {
        piccolo_vpu.write_register(i, mock_offsets[i]);
    }
    
    SimulationResult pic_res = piccolo_sim.run();
    piccolo_sim.finalize();
    uint64_t piccolo_cycles = pic_res.memory_cycles;
    double baseline_latency_ns = baseline_cycles * base_res.tck_ns;
    double piccolo_latency_ns = piccolo_cycles * pic_res.tck_ns;
    // =================================================================
    // 3. DELIVERABLE ASSERTIONS
    // =================================================================
    std::cout << "[ DELIVERABLE ] Baseline Latency: " << baseline_latency_ns << " ns\n";
    std::cout << "[ DELIVERABLE ] Piccolo FIM Latency: " << piccolo_latency_ns << " ns\n";
    std::cout << "Baseline Effective GB/s: " << base_res.effective_gbps() << " GB/s\n";
    std::cout << "Piccolo Effective GB/s: " << pic_res.effective_gbps() << " GB/s\n";
    
   
    EXPECT_NEAR(baseline_latency_ns, 320, 30);
    EXPECT_NEAR(piccolo_latency_ns, 40, 15);
    EXPECT_LT(piccolo_latency_ns, baseline_latency_ns / 4); 
}