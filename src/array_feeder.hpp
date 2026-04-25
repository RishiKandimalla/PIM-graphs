#pragma once
#include "systolic_array.hpp"
#include <queue>
#include <vector>

namespace pim {

class ArrayFeeder {
 public:
  ArrayFeeder() {}

  /// Takes a flat 16-element vector from the SIMD core and queues it with staggering
  void push_vector(const std::vector<float>& vec) {
    for (int i = 0; i < SystolicArray::kDimension; ++i) {
      // Pad with i zeros to create the spatial skew (wavefront alignment)
      for (int delay = 0; delay < i; ++delay) {
        delay_lines_[i].push(0.0f);
      }
      delay_lines_[i].push(vec[i]);
    }
  }

  /// Pops the next staggered cycle of data to feed into systolic_array.tick()
  std::array<float, SystolicArray::kDimension> get_next_activations() {
    std::array<float, SystolicArray::kDimension> acts = {0.0f};
    for (int i = 0; i < SystolicArray::kDimension; ++i) {
      if (!delay_lines_[i].empty()) {
        acts[i] = delay_lines_[i].front();
        delay_lines_[i].pop();
      }
    }
    return acts;
  }

  [[nodiscard]] bool is_empty() const {
    for (const auto& q : delay_lines_) {
      if (!q.empty()) return false;
    }
    return true;
  }

 private:
  // 16 independent FIFOs to delay inputs to the correct row
  std::array<std::queue<float>, SystolicArray::kDimension> delay_lines_;
};

} // namespace pim