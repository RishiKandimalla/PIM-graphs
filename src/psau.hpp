#pragma once

#include <array>
#include <cstdint>

namespace pim {

class PartialSumAccumulationUnit {
 public:
  static constexpr int kNumVPUs = 16;
  static constexpr int kVectorDim = 16; // Matches SystolicArray::kDimension

  using VectorType = std::array<float, kVectorDim>;

  PartialSumAccumulationUnit() {
    // Initialize the 4 pipeline stages with zeros
    stage1_regs_.fill({0.0f});
    stage2_regs_.fill({0.0f});
    stage3_regs_.fill({0.0f});
    stage4_out_ = {0.0f};
    valid_bits_ = {false, false, false, false};
  }

  /// Ticks the pipelined adder tree.
  /// Takes an array of 16 vectors (from the 16 VPUs).
  /// Returns true if a valid, fully-reduced vector drops out this cycle.
  bool tick(const std::array<VectorType, kNumVPUs>& vpu_outputs, bool input_valid, VectorType& final_result) {
    bool output_valid = valid_bits_[3];
    final_result = stage4_out_;

    // Stage 4: Add 2 vectors into 1 (Final Result)
    for (int i = 0; i < kVectorDim; ++i) {
      stage4_out_[i] = stage3_regs_[0][i] + stage3_regs_[1][i];
    }
    valid_bits_[3] = valid_bits_[2];

    // Stage 3: Add 4 vectors into 2
    for (int j = 0; j < 2; ++j) {
      for (int i = 0; i < kVectorDim; ++i) {
        stage3_regs_[j][i] = stage2_regs_[2*j][i] + stage2_regs_[2*j + 1][i];
      }
    }
    valid_bits_[2] = valid_bits_[1];

    // Stage 2: Add 8 vectors into 4
    for (int j = 0; j < 4; ++j) {
      for (int i = 0; i < kVectorDim; ++i) {
        stage2_regs_[j][i] = stage1_regs_[2*j][i] + stage1_regs_[2*j + 1][i];
      }
    }
    valid_bits_[1] = valid_bits_[0];

    // Stage 1: Add 16 vectors into 8
    for (int j = 0; j < 8; ++j) {
      for (int i = 0; i < kVectorDim; ++i) {
        stage1_regs_[j][i] = vpu_outputs[2*j][i] + vpu_outputs[2*j + 1][i];
      }
    }
    valid_bits_[0] = input_valid;

    return output_valid;
  }

  [[nodiscard]] std::array<bool, 4> get_valid_bits() const {return valid_bits_;}
 private:
  // Pipeline registers for the reduction tree
  std::array<VectorType, 8> stage1_regs_;
  std::array<VectorType, 4> stage2_regs_;
  std::array<VectorType, 2> stage3_regs_;
  VectorType stage4_out_;
  
  // Tracks whether the data in the pipeline represents real workloads or bubbles
  std::array<bool, 4> valid_bits_; 
};

}  // namespace pim