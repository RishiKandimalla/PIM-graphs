#pragma once

#include "memory_request.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <queue>
#include <vector>

namespace pim {

/// Number of logical pseudo-channel ports multiplexed onto one Ramulator frontend.
inline constexpr int kNumPseudoChannels = 16;

/// Disjoint address region per PC so streams do not share lines (sequential pattern within region).
/// Keep 16 * kRegionSize + sequential footprint within the modeled device (HBM2 RoBaRaCoCh maps flat addr).
inline constexpr std::uint64_t kRegionSize = 1ull << 20;  // 1 MiB per PC (~16 MiB span across 16 PCs)

/// FIFO of bursts for one pseudo-channel.
class PseudoChannelPort {
 public:
  void enqueue(MemoryBurst b) { pending_.push(std::move(b)); }
  [[nodiscard]] bool empty() const { return pending_.empty(); }
  [[nodiscard]] MemoryBurst& front() { return pending_.front(); }
  void pop() { pending_.pop(); }
  [[nodiscard]] std::size_t size() const { return pending_.size(); }

 private:
  std::queue<MemoryBurst> pending_;
};

/// Round-robin arbiter over `kNumPseudoChannels` ports (tries each queue once per `pick_next`).
class PseudoChannelMultiplexer {
 public:
  PseudoChannelMultiplexer() { ports_.fill(PseudoChannelPort{}); }

  [[nodiscard]] PseudoChannelPort& port(int pc_id) { return ports_.at(static_cast<std::size_t>(pc_id)); }
  [[nodiscard]] const PseudoChannelPort& port(int pc_id) const {
    return ports_.at(static_cast<std::size_t>(pc_id));
  }

  /// Returns the next PC id that has a pending burst, or -1 if all empty.
  [[nodiscard]] int pick_next_round_robin() {
    for (int k = 0; k < kNumPseudoChannels; k++) {
      int pc = (rr_cursor_ + k) % kNumPseudoChannels;
      if (!ports_[static_cast<std::size_t>(pc)].empty()) {
        return pc;
      }
    }
    return -1;
  }

  /// Advance RR cursor after successfully issuing from `pc_id`.
  void advance_after_issue(int pc_id) { rr_cursor_ = (pc_id + 1) % kNumPseudoChannels; }

  [[nodiscard]] int round_robin_cursor() const { return rr_cursor_; }

  [[nodiscard]] std::uint64_t total_pending_bursts() const {
    std::uint64_t n = 0;
    for (const auto& p : ports_) {
      n += p.size();
    }
    return n;
  }

  /// Enqueue `bursts_per_pc` sequential 64B reads per PC starting at `pc * kRegionSize`.
  void load_sequential_workload(std::uint32_t bursts_per_pc, std::uint32_t burst_bytes = 64) {
    for (int pc = 0; pc < kNumPseudoChannels; pc++) {
      std::uint64_t base = static_cast<std::uint64_t>(pc) * kRegionSize;
      for (std::uint32_t i = 0; i < bursts_per_pc; i++) {
        MemoryBurst b;
        b.addr = base + static_cast<std::uint64_t>(i) * burst_bytes;
        b.size_bytes = burst_bytes;
        b.pc_id = pc;
        ports_[static_cast<std::size_t>(pc)].enqueue(b);
      }
    }
  }

 private:
  std::array<PseudoChannelPort, kNumPseudoChannels> ports_;
  int rr_cursor_ = 0;
};

}  // namespace pim
