#pragma once

#include <cstdint>
#include <vector>
#include <array>
#include <stdexcept>

namespace pim {

//Mult Acc Unit
struct ProcessingElement {
  float weight = 0.0f;          // Stationary weight
  
  // Latched inputs 
  float act_in = 0.0f;          // Activation coming from the left
  float psum_in = 0.0f;         // Partial sum coming from the top
  
  // Computed outputs 
  float act_out = 0.0f;
  float psum_out = 0.0f;
};

class SystolicArray {
 public:
  static constexpr int kDimension = 16;
  static constexpr int kNumPEs = kDimension * kDimension;

  SystolicArray() {
    pes_.resize(kNumPEs);
  }

  // Pre-load the stationary weights into the PEs.
  // Expects a flat 256-element vector representing a 16x16 matrix.
  void load_weights(const std::vector<float>& weight_matrix) {
    if (weight_matrix.size() != kNumPEs) {
      throw std::invalid_argument("Weight matrix must be exactly 256 elements.");
    }
    for (int i = 0; i < kNumPEs; ++i) {
      pes_[i].weight = weight_matrix[i];
      pes_[i].act_in = 0.0f;
      pes_[i].psum_in = 0.0f;
    }
  }

  // Advances the array by one clock cycle.
  //Takes the 16 new activations entering the left edge, and 
  // 16 new partial sums entering the top edge.
  // Returns the 16 completed partial sums dropping out of the bottom edge.
  [[nodiscard]] std::array<float, kDimension> tick(
      const std::array<float, kDimension>& left_activations,
      const std::array<float, kDimension>& top_psums) {
      
    std::array<float, kDimension> bottom_outputs = {0};

    // --- STEP 1: COMPUTE (All PEs execute MAC simultaneously using latched inputs) ---
    for (int i = 0; i < kNumPEs; ++i) {
      pes_[i].act_out = pes_[i].act_in; // Activations pass through horizontally
      pes_[i].psum_out = pes_[i].psum_in + (pes_[i].act_in * pes_[i].weight); // MAC
    }

    // --- STEP 2: CAPTURE OUTPUTS (Read the bottom row before we overwrite data) ---
    for (int col = 0; col < kDimension; ++col) {
      int bottom_row_idx = (kDimension - 1) * kDimension + col;
      bottom_outputs[col] = pes_[bottom_row_idx].psum_out;
    }

    // --- STEP 3: SHIFT / LATCH (Move outputs to neighbors' inputs for the NEXT cycle) ---
    // We iterate backwards through the array to avoid overwriting data we haven't shifted yet.
    
    // Shift Partial Sums downwards 
    for (int row = kDimension - 1; row > 0; --row) {
      for (int col = 0; col < kDimension; ++col) {
        pes_[row * kDimension + col].psum_in = pes_[(row - 1) * kDimension + col].psum_out;
      }
    }
    // Feed new top inputs into Row 0
    for (int col = 0; col < kDimension; ++col) {
      pes_[col].psum_in = top_psums[col];
    }

    // Shift Activations to the right 
    for (int col = kDimension - 1; col > 0; --col) {
      for (int row = 0; row < kDimension; ++row) {
        pes_[row * kDimension + col].act_in = pes_[row * kDimension + (col - 1)].act_out;
      }
    }
    // Feed new left inputs into Column 0
    for (int row = 0; row < kDimension; ++row) {
      pes_[row * kDimension].act_in = left_activations[row];
    }

    return bottom_outputs;
  }

 private:
  // index = (row * 16) + col
  std::vector<ProcessingElement> pes_;
};

}  // namespace pim