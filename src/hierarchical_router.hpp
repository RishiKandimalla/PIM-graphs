#pragma once
#include "memory_request.hpp"
#include "bandwidth_oracle.hpp"
#include "pseudo_channel.hpp"
#include <queue>

namespace pim {

class HierarchicalRouter {
 public:
  // Define the boundary between local memory and global memory
  static constexpr std::uint64_t kGlobalAddressBoundary = 0x80000000ull; 

  HierarchicalRouter(PseudoChannelMultiplexer& mux, double clock_hz) 
      : mux_(mux), clock_hz_(clock_hz) {
      
    // Set up bottlenecks: Local is fast (512 GB/s), Global is slow (64 GB/s)
    local_bytes_per_tick_ = bytes_per_cycle_from_target_gbps(512.0, clock_hz_);
    global_bytes_per_tick_ = bytes_per_cycle_from_target_gbps(64.0, clock_hz_);
  }

  void enqueue_request(MemoryBurst burst) {
    if (burst.addr >= kGlobalAddressBoundary) {
      global_queue_.push(burst);
    } else {
      local_queue_.push(burst);
    }
  }

  void tick() {
    double local_used = 0.0;
    double global_used = 0.0;

    // Process Local Traffic
    while (!local_queue_.empty() && local_used + local_queue_.front().size_bytes <= local_bytes_per_tick_) {
      local_used += local_queue_.front().size_bytes;
      mux_.port(local_queue_.front().pc_id).enqueue(local_queue_.front());
      local_queue_.pop();
    }

    // Process Global Traffic (Bottleneck applied here)
    while (!global_queue_.empty() && global_used + global_queue_.front().size_bytes <= global_bytes_per_tick_) {
      global_used += global_queue_.front().size_bytes;
      mux_.port(global_queue_.front().pc_id).enqueue(global_queue_.front());
      global_queue_.pop();
    }
  }

 private:
  PseudoChannelMultiplexer& mux_;
  double clock_hz_;
  double local_bytes_per_tick_;
  double global_bytes_per_tick_;
  
  std::queue<MemoryBurst> local_queue_;
  std::queue<MemoryBurst> global_queue_;
};

} // namespace pim