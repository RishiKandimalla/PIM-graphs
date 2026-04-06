#pragma once

#include <cstdint>

namespace pim {

/// One memory burst issued on behalf of a logical pseudo-channel (PC).
struct MemoryBurst {
  std::uint64_t addr = 0;
  std::uint32_t size_bytes = 64;
  int pc_id = 0;
};

}  // namespace pim
