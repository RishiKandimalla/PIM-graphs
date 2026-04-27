#pragma once

#include "memory_request.hpp"
#include "pseudo_channel.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace pim {

class OffsetBuffer {
 public:
  explicit OffsetBuffer(std::size_t capacity_entries = 256) : cap_(capacity_entries) {
    data_.reserve(cap_);
  }

  [[nodiscard]] bool empty() const { return size_ == 0; }
  [[nodiscard]] bool full() const { return size_ >= cap_; }
  [[nodiscard]] std::size_t size() const { return size_; }

  void clear() {
    head_ = 0;
    size_ = 0;
  }

  void push(std::uint32_t v) {
    if (full()) {
      return;
    }
    if (data_.size() < cap_) {
      data_.resize(cap_);
    }
    data_[(head_ + size_) % cap_] = v;
    size_++;
  }

  [[nodiscard]] std::uint32_t front() const { return empty() ? 0u : data_[head_]; }

  void pop() {
    if (empty()) {
      return;
    }
    head_ = (head_ + 1) % cap_;
    size_--;
  }

 private:
  std::size_t cap_ = 0;
  std::vector<std::uint32_t> data_;
  std::size_t head_ = 0;
  std::size_t size_ = 0;
};

/// Fixed-capacity FIFO of gathered values (Piccolo Data Buffer).
/// No exceptions; pushes are ignored when full.
class DataBuffer {
 public:
  explicit DataBuffer(std::size_t capacity_elems = 256) : cap_(capacity_elems) {
    data_.reserve(cap_);
  }

  [[nodiscard]] bool empty() const { return size_ == 0; }
  [[nodiscard]] bool full() const { return size_ >= cap_; }
  [[nodiscard]] std::size_t size() const { return size_; }

  void clear() {
    head_ = 0;
    size_ = 0;
  }

  void push(std::uint32_t v) {
    if (full()) {
      return;
    }
    if (data_.size() < cap_) {
      data_.resize(cap_);
    }
    data_[(head_ + size_) % cap_] = v;
    size_++;
  }

  [[nodiscard]] std::uint32_t front() const { return empty() ? 0u : data_[head_]; }

  void pop() {
    if (empty()) {
      return;
    }
    head_ = (head_ + 1) % cap_;
    size_--;
  }

 private:
  std::size_t cap_ = 0;
  std::vector<std::uint32_t> data_;
  std::size_t head_ = 0;
  std::size_t size_ = 0;
};

/// Piccolo GATHER controller (OB preloaded by caller).
///
/// This is intentionally a small model: it issues element-sized reads into the existing mux
/// and relies on the simulator to call `on_read_complete()` when requests finish.
class PiccoloGatherController {
 public:
  PiccoloGatherController(int pc_id, PseudoChannelMultiplexer& mux, std::uint64_t base_addr,
                          std::uint32_t elem_bytes, std::uint32_t num_elems)
      : pc_id_(pc_id), mux_(mux), base_addr_(base_addr), elem_bytes_(elem_bytes), num_elems_(num_elems) {}

  OffsetBuffer ob;
  DataBuffer db;

  std::uint32_t max_outstanding = 8;

  /// Enqueue as many element reads as allowed this tick.
  void tick_issue() {
    while (issued_ < num_elems_ && outstanding_ < max_outstanding && !ob.empty() && !db.full()) {
      const std::uint32_t off = ob.front();
      ob.pop();
      MemoryBurst b;
      b.addr = base_addr_ + static_cast<std::uint64_t>(off) * elem_bytes_;
      b.size_bytes = elem_bytes_;
      b.pc_id = pc_id_;
      mux_.port(pc_id_).enqueue(b);
      issued_++;
      outstanding_++;
    }
  }

  /// Call this once per completed element read.
  void on_read_complete(std::uint32_t value = 0) {
    if (outstanding_ == 0) {
      return;
    }
    db.push(value);
    outstanding_--;
    completed_++;
  }

  [[nodiscard]] bool done() const { return completed_ >= num_elems_; }

 private:
  int pc_id_ = 0;
  PseudoChannelMultiplexer& mux_;
  std::uint64_t base_addr_ = 0;
  std::uint32_t elem_bytes_ = 4;
  std::uint32_t num_elems_ = 0;
  std::uint32_t issued_ = 0;
  std::uint32_t completed_ = 0;
  std::uint32_t outstanding_ = 0;
};

}  // namespace pim

