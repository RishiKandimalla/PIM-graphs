#pragma once

#include <cstdint>
#include <functional>

namespace pim {

/// One memory burst issued on behalf of a logical pseudo-channel (PC).
struct MemoryBurst {
  std::uint64_t addr = 0;
  std::uint32_t size_bytes = 64;
  int pc_id = 0;
  std::function<void()> on_complete;
};

}  // namespace pim
