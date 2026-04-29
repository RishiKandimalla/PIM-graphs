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

    // 1. Hardware Initialization
    RamulatorHbmSimulator sim(config_path());
    HierarchicalRouter router(sim.mux(), 1.0e9);
    ArrayFeeder feeder;
    
    PiccoloGatherController piccolo(0, sim.mux(), 0x80000000ull, 4, 8);
    SIMDCore vpu(0, sim.mux(), router, feeder, piccolo); 
    SystolicArray array;

    // 2. Setup: Identity Matrix (Output should match Input)
    std::vector<float> identity(256, 0.0f);
    for(int i = 0; i < 16; ++i) identity[i * 16 + i] = 1.0f;
    array.load_weights(identity);

    // 3. Define VPU Program
    vpu.load_program({
        {SIMDOpcode::LOAD, 0, 0, 0, 0x90000000ull, 64}, 
        {SIMDOpcode::PUSH_ARR, 0, 0, 0, 0, 0}         
    });

    uint64_t cycles = 0;
    std::vector<float> final_outputs;

    // 4. The Master Clock Loop
    while (cycles < 5000 && (!vpu.is_done() || !feeder.is_empty() || final_outputs.size() < 16)) {
        vpu.tick();     // VPU manages instructions and stalls
        router.tick();  // Router applies bandwidth bottlenecks
        
        // --- DATA INJECTION (In the Test) ---
        // We manually inject the 1-16 sequence here to simulate the HBM reply 
        // after the 320ns baseline latency.
        if (cycles == 320) { 
            std::vector<float> burst_data(16);
            for(int i = 0; i < 16; ++i) burst_data[i] = static_cast<float>(i + 1);
            
            vpu.on_memory_reply(burst_data); // Wakes up VPU and loads Regs
        }

        // --- ARRAY FEEDER & SYSTOLIC ARRAY ---
        // Feeder provides staggered activations to the array.
        auto staggered_acts = feeder.get_next_activations();
        std::array<float, 16> zero_psums = {0};
        
        // Tick the array and collect any data dropping out of the bottom row.
        auto row_out = array.tick(staggered_acts, zero_psums);
        for(float f : row_out) {
            if(f > 0.001f) final_outputs.push_back(f);
        }

        sim.run(1); // Keep HBM engine in sync
        cycles++;
    }

    // --- OUTPUT THE ARRAY ---
    std::cout << "\n[ TEST RESULT ] Collected " << final_outputs.size() << " elements: \n[ ";
    for (size_t i = 0; i < final_outputs.size(); ++i) {
        std::cout << final_outputs[i] << (i == final_outputs.size() - 1 ? "" : ", ");
    }
    std::cout << " ]\n" << std::endl;

    // --- VERIFICATION ---
    ASSERT_EQ(final_outputs.size(), 16u) << "Array did not fully drain!";
    for (int i = 0; i < 16; ++i) {
        float expected = static_cast<float>(i + 1);
        EXPECT_NEAR(final_outputs[i], expected, 1e-5) 
            << "Data scrambled at index " << i << "! Feeder or Array timing is off.";
    }

    EXPECT_GT(cycles, 320); 
    EXPECT_TRUE(vpu.is_done());
}