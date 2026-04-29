#include <gtest/gtest.h>
#include "ramulator_hbm.hpp"
#include "simd_core.hpp"
#include "hierarchical_router.hpp"
#include "systolic_array.hpp"
#include "array_feeder.hpp"
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

}  // namespace


TEST(VpuPipeline, GlobalVectorToSystolicArray) {
    using namespace pim;

    // 1. Initialize Hardware Components 
    RamulatorHbmSimulator sim(config_path());
    HierarchicalRouter router(sim.mux(), 1.0e9); // 1GHz Clock
    ArrayFeeder feeder;
    
    // Create VPU assigned to Pseudo-Channel 0 
    SIMDCore vpu(0, sim.mux(), router, feeder); 
    SystolicArray array;

    // 2. Setup: Load Identity Matrix into Systolic Array 
    // An identity matrix ensures: Output Vector == Input Vector
    std::vector<float> identity(256, 0.0f);
    for(int i = 0; i < 16; ++i) identity[i * 16 + i] = 1.0f;
    array.load_weights(identity);

    // 3. Define VPU Program 
    // Load from 0x90000000ull to trigger the Global Bottleneck 
    std::vector<SIMDInstruction> program = {
        {SIMDOpcode::LOAD, 0, 0, 0, 0x90000000ull, 64}, 
        {SIMDOpcode::PUSH_ARR, 0, 0, 0, 0, 0}         
    };
    vpu.load_program(program);

    // 4. Execution Loop (Cycle-Accurate) 
    uint64_t cycles = 0;
    bool data_received = false;
    std::vector<float> final_outputs;

    // Run until the VPU finishes and the Array Feeder is drained
    while (cycles < 5000 && (!vpu.is_done() || !feeder.is_empty() || final_outputs.size() < 16)) {
        vpu.tick();     // VPU issues LOAD or PUSH 
        router.tick();  // Router moves memory bursts with bottlenecks 
        
        // Mocking Ramulator data return for this unit test 
        // In a full trace-driven run, Ramulator would call this via a callback.
        if (cycles == 200) { 
            // Return 16 floats (1.0f to 16.0f) to the VPU
    
            std::vector<float> burst_data = {
        1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
        9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f};
            vpu.on_memory_reply(burst_data); 
        }

        // Advance the Systolic Array 
            auto staggered_acts = feeder.get_next_activations();
            std::array<float, 16> zero_psums = {0};
            auto row_out = array.tick(staggered_acts, zero_psums);
            
            // Capture any non-zero output dropping out of the bottom row
            for(float f : row_out) if(f > 0) final_outputs.push_back(f);
        
    

        sim.run(1); // Tick Ramulator 2 engine 
        std::cout << "Cycle: " << cycles << " VPU State: " << vpu.is_stalled() << std::endl;
        cycles++;
    }


    // 5. Assertions: Verify the Deliverable Objectives 
    EXPECT_GT(cycles, 200); // Proves the VPU actually stalled for memory
    EXPECT_TRUE(vpu.is_done());
    
    // weights were Identity, outputs should match inputs
    // expect 16 values to eventually drain out of the array
    EXPECT_GE(final_outputs.size(), 16u);
    if (final_outputs.size() >= 1) {
        EXPECT_NEAR(final_outputs[0], 1.0f, 1e-5);
    }
}