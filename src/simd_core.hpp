#pragma once

#include "pseudo_channel.hpp"
#include "array_feeder.hpp"
#include "hierarchical_router.hpp"
#include "ramulator_hbm.hpp"
#include "systolic_array.hpp"
#include "piccolo_controller.hpp"
#include "gru.hpp"

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
  GATHER,   // Piccolo FIM gather sequence

  //GRU + element wise activation extensions
  SIGMOID,
  TANH,
  HADAMARD,
  TIME_ENCODE,
  GRU_UPDATE,

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
        SIMDCore(
          int pc_id, 
          PseudoChannelMultiplexer& mux, 
          HierarchicalRouter& router, 
          ArrayFeeder& feeder, 
          PiccoloGatherController& piccolo, 
          std::uint32_t num_registers = 256)
    
        : pc_id_(pc_id), mux_(mux), router_(router), array_feeder_(feeder), piccolo_(piccolo) {
        // Initialize a dummy local SRAM / Reg File (storing floats for GNN features/weights)
        registers_.resize(num_registers, 0.0f);

        gru_cell_ = std::make_unique<GRUCell>();
    }    
    void load_program(const std::vector<SIMDInstruction>& program) {
    instruction_queue_ = std::queue<SIMDInstruction>();
    for (const auto& inst : program) {
      instruction_queue_.push(inst);
    }
    is_stalled_ = false;
    pending_dest_reg_ = -1;
    is_waiting_for_piccolo_ = false;
    is_waiting_for_gru_ = false;
  }

  void load_gru_weights(const std::vector<float>& W_r, const std::vector<float>& W_z, const std::vector<float>& W_in, const std::vector<float>& W_hn){
    gru_cell_->load_weights(W_r, W_z, W_in, W_hn);
  }

    void tick() {
    //Piccolo Sync 
    if (is_stalled_ && is_waiting_for_piccolo_) {
        // If Piccolo has finished gathering all requested elements
        if (piccolo_.done()) {
            int i = 0;
            // Drain the Data Buffer into the target registers
            uint32_t batch_size = piccolo_.get_num_elems();
            while (!piccolo_.db.empty() && i < static_cast<int>(batch_size) && pending_dest_reg_ + i < static_cast<int>(registers_.size())) { 
                // bit-casting uint32_t from Piccolo to float for SIMD regs
                uint32_t raw_val = piccolo_.db.front();
                piccolo_.db.pop();
                float fval;
                __builtin_memcpy(&fval, &raw_val, sizeof(float));
                registers_[pending_dest_reg_ + i] = fval;
                i++;
            }
            is_stalled_ = false;
            is_waiting_for_piccolo_ = false;
            pending_dest_reg_ = -1;
        }
        return; // Still waiting for Piccolo hardware to finish
      }
       if(is_stalled && is_waiting_for_gru_){
       if(gru_sa_ != nullptr && gru_feeder_ != nullptr){
          bool gru_done = gru_cell_->tick(*gru_sa_, *gru_feeder_);
          if (gru_done){
            const auto& h_new = gru_cell_->get_h_new();
            for(int i = 0; i < 16 && pending_dest_reg_ + i < static_cast<int>(registers_.size()); ++i)
              registers_[pending_dest_reg_ + i] = h_new[i];
            
          is_stalled_ = false;
          is_waiting_for_gru_ = false;
          pending_dest_reg_ = -1;
          gru_sa_ = nullptr;
          gru_feeder_ = nullptr;
        }
      }
        return;
      }

             
     //If we are waiting for memory, freeze the pipeline.
    if (is_stalled_ || instruction_queue_.empty()) {
      return; 
    }

    const auto& inst = instruction_queue_.front();

    switch (inst.op) {
      case SIMDOpcode::GATHER: {
        piccolo_.set_base_addr(inst.mem_addr);
        uint32_t elements_to_gather = inst.size_bytes / 4;
        piccolo_.reset_for_new_batch(elements_to_gather); // Prepare controller for 8 elements
        for (int i = 0; i < elements_to_gather; ++i) {
            uint32_t offset = static_cast<uint32_t>(registers_[inst.src1_reg + i]);
            piccolo_.ob.push(offset);
        }
        
        pending_dest_reg_ = inst.dest_reg;
        is_stalled_ = true;
        is_waiting_for_piccolo_ = true;
        
        instruction_queue_.pop();
        break;
      }
      case SIMDOpcode::LOAD: {
        MemoryBurst burst;
        burst.addr = inst.mem_addr;
        burst.size_bytes = inst.size_bytes;
        burst.pc_id = pc_id_;
        
        burst.on_complete = [this]() {
          // This executes when DRAM finishes the read

          std::vector<float> mock_payload(16, 1.0f);

          this->on_memory_reply(mock_payload);

          }; 

        router_.enqueue_request(burst);
        
        // Lock the core! Remember which register gets the data.
        pending_dest_reg_ = inst.dest_reg;
        is_stalled_ = true;
        
        instruction_queue_.pop(); 
        break;
      }
      case SIMDOpcode::PUSH_ARR: {
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
      case SIMDOpcode::SIGMOID: {
         for (int i = 0; i < 16 && inst.src1_reg + i < registers_.size() &&inst.dest_reg + i < registers_.size(); ++i) {
          registers_[inst.dest_reg + i] =
              1.0f / (1.0f + std::exp(-registers_[inst.src1_reg + i]));
        }
        instruction_queue_.pop();
        break;
      }
      case SIMDOpcode::TANH: {
        for (int i = 0; i < 16 && inst.src1_reg + i < registers_.size() &&inst.dest_reg + i < registers_.size(); ++i) {
          registers_[inst.dest_reg + i] =
              std::tanh(registers_[inst.src1_reg + i]);
        }
        instruction_queue_.pop();
        break;
      }
      // r[dest+d] = r[src1+d] * r[src2+d]  for d in [0,16)
      case SIMDOpcode::HADAMARD: {
        for (int i = 0; i < 16 && inst.src1_reg + i < registers_.size() && inst.src2_reg + i < registers_.size() &&inst.dest_reg + i < registers_.size(); ++i) {
          registers_[inst.dest_reg + i] =
              registers_[inst.src1_reg + i] * registers_[inst.src2_reg + i];
        }
        instruction_queue_.pop();
        break;
      } 
      // φ(Δt): sinusoidal time-delta embedding as in TGN §2.
      // mem_addr carries the raw int64 timestamp; Δt = timestamp - t_ref.
      // We model a simplified cosine embedding across 16 frequencies.
      // time encoding is computed on-chip to avoid sending
      // raw timestamps over the TSV."
      case SIMDOpcode::TIME_ENCODE: {
        float delta_t = static_cast<float>(inst.mem_addr);
        for (int i = 0; i < 16 && inst.dest_reg + i < registers_.size(); ++i) {
          float freq = 1.0f / std::pow(10000.0f, 2.0f * i / 16.0f);
          registers_[inst.dest_reg + i] =
              (i % 2 == 0) ? std::cos(freq * delta_t)
                           : std::sin(freq * delta_t);
        }
        instruction_queue_.pop();
        break;
      }
      // Triggers the full GRU cell evaluation (gru.hpp).
      // The core stalls until the GRUCell state machine completes.
      // src1_reg: base of h_v(t-1) in register file   (16 floats)
      // src2_reg: base of m_v(t)   in register file   (16 floats)
      // dest_reg: base for h_v(t)  output             (16 floats)
      //
      // The systolic array and feeder needed by the GRU are injected via
      // attach_sa_and_feeder_for_gru() — called by the vertex program before
      // issuing this instruction.
      case SIMDOpcode::GRU_UPDATE: {
        if (gru_sa_ == nullptr || gru_feeder_ == nullptr) {
          // Hardware not yet connected; treat as NOP and retry next cycle.
          break;
        }
        // Read operands from register file
        std::array<float, 16> h_prev{}, msg{};
        for (int i = 0; i < 16; ++i) {
          h_prev[i] = (inst.src1_reg + i < registers_.size())
                          ? registers_[inst.src1_reg + i] : 0.0f;
          msg[i]    = (inst.src2_reg + i < registers_.size())
                          ? registers_[inst.src2_reg + i] : 0.0f;
        }
        gru_cell_->start(h_prev, msg);
        pending_dest_reg_   = inst.dest_reg;
        is_stalled_         = true;
        is_waiting_for_gru_ = true;
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
    void on_memory_reply(const std::vector<float>& burst_data) {
        if (!is_stalled_ || pending_dest_reg_ == -1) {
        return; // sanity check 
        }

        // must not write beyond the end of our register file
        size_t count = std::min<size_t>(16, burst_data.size());
        if (pending_dest_reg_ + count > registers_.size()) {
            count = registers_.size() - pending_dest_reg_; 
        }

        // Write the data into the SRAM and unlock the core
        if (count > 0) {
            std::copy_n(burst_data.begin(), count, registers_.begin() + pending_dest_reg_);
        }
        is_stalled_ = false;
        pending_dest_reg_ = -1;
    }

    void attach_sa_and_feeder_for_gru(SystolicArray* sa, ArrayFeeder* feeder) {
      gru_sa_ = sa;
      gru_feeder_ = feeder;
    }

    [[nodiscard]] bool is_done() const {
        return instruction_queue_.empty() && !is_stalled_;
    }

    //helper function for vpu test 
    [[nodiscard]] bool is_stalled() const { return is_stalled_; }

    [[nodiscard]] bool is_waiting_for_gru() const { return is_waiting_for_gru_; }

    void write_register(std::uint32_t reg, float value) {
        if (reg < registers_.size()) {
            registers_[reg] = value;
        }
    }

    [[nodiscard]] float read_register(std::uint32_t reg) const {
        return (reg < registers_.size()) ? registers_[reg] : 0.0f;
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
        PiccoloGatherController& piccolo_; // Controller for issuing gather requests
        bool is_waiting_for_piccolo_ = false; // Special stall state for waiting on Piccolo gather completion

        bool is_waiting_for_gru_ = false; // Special stall state for waiting on GRU computation completion
        SystolicArray* gru_sa_ = nullptr; 
        ArrayFeeder* gru_feeder_ = nullptr;
        std::unique_ptr<GRUCell> gru_cell_;
};

} // namespace pim