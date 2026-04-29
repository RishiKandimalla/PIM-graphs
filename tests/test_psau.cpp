#include <gtest/gtest.h>
#include "ramulator_hbm.hpp"
#include "simd_core.hpp"
#include "hierarchical_router.hpp"
#include "psau.hpp"
#include "systolic_array.hpp"
#include "array_feeder.hpp"

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

TEST(HbmIntegration, EndToEndVpuToPsauWriteback) {
    using namespace pim;

    
    RamulatorHbmSimulator sim(config_path());
    HierarchicalRouter router(sim.mux(), 1.0e9); // 1 GHz clock
    PartialSumAccumulationUnit psau;
    
    sim.attach_router(&router);
    sim.attach_psau(&psau);

    std::vector<std::unique_ptr<SIMDCore>> vpus;
    std::vector<std::unique_ptr<ArrayFeeder>> feeders;
    std::vector<std::unique_ptr<SystolicArray>> arrays;
    std::vector<std::unique_ptr<PiccoloGatherController>> piccolos;

    for (int i = 0; i < 16; ++i) {
        feeders.push_back(std::make_unique<ArrayFeeder>());
        arrays.push_back(std::make_unique<SystolicArray>());
        piccolos.push_back(std::make_unique<PiccoloGatherController>(i, sim.mux(), 0, 4, 8));
        
        std::vector<float> identity(256, 0.0f);
        for(int j = 0; j < 16; ++j) identity[j * 16 + j] = 1.0f;
        arrays.back()->load_weights(identity);

        vpus.push_back(std::make_unique<SIMDCore>(i, sim.mux(), router, *feeders.back(), *piccolos.back()));
        
        sim.attach_vpu(i, vpus.back().get());
        sim.attach_feeder(i, feeders.back().get());
        sim.attach_systolic_array(i, arrays.back().get());

        vpus.back()->load_program({
            {SIMDOpcode::LOAD, 0, 0, 0, 0x10000000ull + (i * 0x1000), 64},
            {SIMDOpcode::PUSH_ARR, 0, 0, 0, 0, 0}
        });
    }

    
    SimulationResult result = sim.run();
    sim.finalize();

    std::cout << "[ DELIVERABLE ] Total Memory Cycles: " << result.memory_cycles << "\n";
    std::cout << "[ DELIVERABLE ] Bytes Moved: " << result.bytes_moved << "\n";

    // Expected: 16 LOADs (64B each) + 1 PSAU Writeback (64B) = 1088 Bytes
    EXPECT_GE(result.bytes_moved, 1088u);
    EXPECT_GT(result.memory_cycles, 0u);
    
  
    EXPECT_GT(result.memory_cycles, 320u); 
}

TEST(HbmIntegration, EndToEndVpuToPsauTrace) {
    using namespace pim;

    // 1. Initialize HBM Simulator and Global Hardware [cite: 32, 33]
    RamulatorHbmSimulator sim(config_path());
    HierarchicalRouter router(sim.mux(), 1.0e9); 
    PartialSumAccumulationUnit psau;
    
    sim.attach_router(&router);
    sim.attach_psau(&psau);

    // 2. Initialize 16 Vaults (VPU + Feeder + Array) [cite: 28, 78, 85]
    std::vector<std::unique_ptr<SIMDCore>> vpus;
    std::vector<std::unique_ptr<ArrayFeeder>> feeders;
    std::vector<std::unique_ptr<SystolicArray>> arrays;
    std::vector<std::unique_ptr<PiccoloGatherController>> piccolos;

    for (int i = 0; i < 16; ++i) {
        feeders.push_back(std::make_unique<ArrayFeeder>());
        arrays.push_back(std::make_unique<SystolicArray>());
        piccolos.push_back(std::make_unique<PiccoloGatherController>(i, sim.mux(), 0, 4, 8));
        
        // Identity Matrix for pass-through verification [cite: 85, 88]
        std::vector<float> identity(256, 0.0f);
        for(int j = 0; j < 16; ++j) identity[j * 16 + j] = 1.0f;
        arrays.back()->load_weights(identity);

        vpus.push_back(std::make_unique<SIMDCore>(i, sim.mux(), router, *feeders.back(), *piccolos.back()));
        
        sim.attach_vpu(i, vpus.back().get());
        sim.attach_feeder(i, feeders.back().get());
        sim.attach_systolic_array(i, arrays.back().get());

        // LOAD from memory -> PUSH to Systolic Array [cite: 36, 107]
        vpus.back()->load_program({
            {SIMDOpcode::LOAD, 0, 0, 0, 0x10000000ull + (i * 0x1000), 64},
            {SIMDOpcode::PUSH_ARR, 0, 0, 0, 0, 0}
        });
    }

    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << " GLOBAL ADDER TREE END-TO-END TRACE\n";
    std::cout << std::string(70, '=') << "\n";

    // 3. Execution Loop with Manual Tracing [cite: 100, 109]
    uint64_t cycles = 0;
    uint64_t total_bytes = 0;
    bool simulation_active = true;

    while (cycles < 5000 && simulation_active) {
        // Run exactly 1 cycle
        SimulationResult step_res = sim.run(1);
        total_bytes += step_res.bytes_moved;

        // --- TRACE LOGIC ---
        // Peek into the pipeline stages [cite: 33, 100]
        auto stages = psau.get_valid_bits();
        bool psau_busy = stages[0] || stages[1] || stages[2] || stages[3];

        // Only print when the pipeline is actually doing work
        if (psau_busy) {
            std::cout << "Cycle " << std::setw(4) << cycles << " | "
                      << "Pipe: [" << (stages[0]?"S1":"..") << "->" 
                                   << (stages[1]?"S2":"..") << "->" 
                                   << (stages[2]?"S3":"..") << "->" 
                                   << (stages[3]?"S4":"..") << "] | ";
            
            // Check for the pipelined writeback event [cite: 28, 100]
            if (stages[3] && !psau.get_valid_bits()[2]) { 
                // This is a simplified check; in run() it happens on the true tick return
                std::cout << "<< WRITEBACK TRIGGERED >>";
            }
            std::cout << "\n";
        }

        // Check if all hardware is done
        bool any_vpu_busy = false;
        for(auto& v : vpus) if(!v->is_done()) any_vpu_busy = true;
        
        simulation_active = any_vpu_busy || psau_busy || (sim.mux().total_pending_bursts() > 0);
        cycles++;
    }

    sim.finalize();

    std::cout << std::string(70, '=') << "\n";
    std::cout << "[RESULT] Bytes Moved: " << total_bytes << " | Cycles: " << cycles << "\n";
    
    // Assertions to ensure the hardware actually fired [cite: 100, 113]
    EXPECT_GE(total_bytes, 1088u); // 16 LOADs + 1 reduction writeback
    EXPECT_GT(cycles, 320u); 
}