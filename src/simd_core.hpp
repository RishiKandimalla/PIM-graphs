#pragma once

#include "pseudo_channel.hpp"
#include "array_feeder.hpp"
#include "hierarchal_router.hpp"
#include "ramulator_hbm.hpp"
#include "systolic_array.hpp"
#include <cstdint>
#include <vector>
#include <queue>
#include <stdexcept>
#include <iostream>
namespace pim {
/// SimdCore models a single SIMD execution unit with a single frontend port (multiplexed across PCs).

enum class SIMDOpcode{
  NOP,
  LOAD,      // Load from HBM to local register
  STORE,     // Store from local register to HBM
  ADD_VEC,   // Element-wise vector addition
  MUL_VEC,    // Element-wise vector multiplication
  PUSH_ARR,   // Push a vector to the ArrayFeeder 
};

struct SIMDInstruction {
  SIMDOpcode op = SIMDOpcode::NOP;
  std::uint32_t dest_reg = 0;
  std::uint32_t src1_reg = 0;
  std::uint32_t src2_reg = 0;
  std::uint64_t mem_addr = 0;  
  std::uint32_t size_bytes = 64;

};

class SIMDCore {
    public:
        SIMDCore(int pc_id, PseudoChannelMultiplexer& mux, std::uint32_t num_registers = 256)
        : pc_id_(pc_id), mux_(mux) {
        // Initialize a dummy local SRAM / Reg File (storing floats for GNN features/weights)
        registers_.resize(num_registers, 0.0f);
    }    
    void load_program(const std::vector<SIMDInstruction>& program) {
    instruction_queue_ = std::queue<SIMDInstruction>();
    for (const auto& inst : program) {
      instruction_queue_.push(inst);
    }
    is_stalled_ = false;
    pending_dest_reg_ = -1;
  }

    void tick() {
    // 1. If we are waiting for memory, freeze the pipeline.
    if (is_stalled_ || instruction_queue_.empty()) {
      return; 
    }

    const auto& inst = instruction_queue_.front();

    switch (inst.op) {
      case SIMDOpcode::LOAD: {
        MemoryBurst burst;
        burst.addr = inst.mem_addr;
        burst.size_bytes = inst.size_bytes;
        burst.pc_id = pc_id_;
        
        mux_.port(pc_id_).enqueue(burst);
        
        // Lock the core! Remember which register gets the data.
        pending_dest_reg_ = inst.dest_reg;
        is_stalled_ = true;
        
        instruction_queue_.pop(); 
        break;
      }
      case SIMDOpcode::PUSH_ARR {
       std::vector<float> vec;
        for (int i = 0; i < SystolicArray::kDimension; ++i) {
          vec.push_back(registers_[inst.src1_reg + i]);
        }
        //feed the vector into the array feeder, which will handle the staggering for us
        array_feeder_.push_vector(vec);

        instruction_queue_.pop();
        break;
      }
      case SIMDOpcode::ADD_VEC: {
        registers_[inst.dest_reg] = registers_[inst.src1_reg] + registers_[inst.src2_reg];
        instruction_queue_.pop();
        break;
      }
      case SIMDOpcode::MUL_VEC: {
        registers_[inst.dest_reg] = registers_[inst.src1_reg] * registers_[inst.src2_reg];
        instruction_queue_.pop();
        break;
      }
      case SIMDOpcode::STORE:
      case SIMDOpcode::NOP:
      default:
        instruction_queue_.pop();
        break;
    }
  }
    //called by ramulator when a memory response arrives for this core's pending request 
    void on_memory_reply(float simulated_data) {
        if (!is_stalled_ || pending_dest_reg_ == -1) {
        return; // sanity check 
        }

        // Write the data into the SRAM and unlock the core
        registers_[pending_dest_reg_] = simulated_data;
        is_stalled_ = false;
        pending_dest_reg_ = -1;
    }

    [[nodiscard]] bool is_done() const {
        return instruction_queue_.empty() && !is_stalled_;
    }

    private:
        int pc_id_;
        PseudoChannelMultiplexer& mux_;
        std::vector<float> registers_;  // Local SRAM regs
        std::queue<SIMDInstruction> instruction_queue_;  

        //Pipeline state
        bool is_stalled_ = false;  // Waiting for memory response
        int pending_dest_reg_ = -1; // Which reg we're waiting to write back to

        HierarchicalRouter& router_; // Hierarchical router for enqueuing requests
        ArrayFeeder& array_feeder_; // ArrayFeeder for pushing vectors to the systolic array

};
}
