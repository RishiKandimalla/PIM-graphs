#pragma once

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace pim {

enum class PartitionMode {
  RowInterleaved,
  FeaturePartitioned,
};

struct FeaturePartitionConfig {
  std::uint32_t num_vaults = 16;
  std::uint32_t feature_dim = 0;
  /// Number of consecutive feature dimensions assigned as a unit.
  /// 1 means per-dimension striping.
  std::uint32_t chunk_size = 1;
  /// Hybrid threshold in dimensions: feature_dim <= threshold -> row interleaving.
  std::uint32_t hybrid_threshold_dim = 128;
};

struct FeaturePartitionResult {
  PartitionMode mode = PartitionMode::FeaturePartitioned;
  std::uint32_t num_vaults = 0;
  std::uint32_t feature_dim = 0;
  std::uint32_t chunk_size = 1;
  /// For each feature dimension d in [0, feature_dim), stores the assigned vault id.
  std::vector<std::uint32_t> dim_to_vault;
};

struct PartitionStats {
  std::uint64_t min_dims_per_vault = 0;
  std::uint64_t max_dims_per_vault = 0;
  double avg_dims_per_vault = 0.0;
  /// max / avg, 1.0 is perfectly balanced.
  double imbalance_ratio = 0.0;
};

[[nodiscard]] inline const char* to_string(PartitionMode mode) {
  switch (mode) {
    case PartitionMode::RowInterleaved:
      return "row_interleaved";
    case PartitionMode::FeaturePartitioned:
      return "feature_partitioned";
  }
  return "unknown";
}

[[nodiscard]] inline PartitionMode choose_partition_mode(const FeaturePartitionConfig& cfg) {
  if (cfg.feature_dim == 0) {
    throw std::invalid_argument("feature_dim must be > 0");
  }
  if (cfg.num_vaults == 0) {
    throw std::invalid_argument("num_vaults must be > 0");
  }
  return (cfg.feature_dim <= cfg.hybrid_threshold_dim) ? PartitionMode::RowInterleaved
                                                        : PartitionMode::FeaturePartitioned;
}

/// Baseline row-interleaved mapping: vault = dim % num_vaults.
[[nodiscard]] inline FeaturePartitionResult make_row_interleaved_partition(
    const FeaturePartitionConfig& cfg) {
  if (cfg.feature_dim == 0 || cfg.num_vaults == 0) {
    throw std::invalid_argument("feature_dim and num_vaults must be > 0");
  }
  FeaturePartitionResult out;
  out.mode = PartitionMode::RowInterleaved;
  out.num_vaults = cfg.num_vaults;
  out.feature_dim = cfg.feature_dim;
  out.chunk_size = 1;
  out.dim_to_vault.resize(cfg.feature_dim);
  for (std::uint32_t d = 0; d < cfg.feature_dim; d++) {
    out.dim_to_vault[d] = d % cfg.num_vaults;
  }
  return out;
}

/// Feature-dimension partitioning with optional chunking:
/// - Group dimensions into chunks of `chunk_size`
/// - Assign chunks round-robin across vaults
[[nodiscard]] inline FeaturePartitionResult make_feature_partition(
    const FeaturePartitionConfig& cfg) {
  if (cfg.feature_dim == 0 || cfg.num_vaults == 0) {
    throw std::invalid_argument("feature_dim and num_vaults must be > 0");
  }
  if (cfg.chunk_size == 0) {
    throw std::invalid_argument("chunk_size must be > 0");
  }

  FeaturePartitionResult out;
  out.mode = PartitionMode::FeaturePartitioned;
  out.num_vaults = cfg.num_vaults;
  out.feature_dim = cfg.feature_dim;
  out.chunk_size = cfg.chunk_size;
  out.dim_to_vault.resize(cfg.feature_dim);

  std::uint32_t chunk_id = 0;
  for (std::uint32_t start = 0; start < cfg.feature_dim; start += cfg.chunk_size) {
    const std::uint32_t end = std::min(cfg.feature_dim, start + cfg.chunk_size);
    const std::uint32_t vault = chunk_id % cfg.num_vaults;
    for (std::uint32_t d = start; d < end; d++) {
      out.dim_to_vault[d] = vault;
    }
    chunk_id++;
  }

  return out;
}

/// Hybrid entry point for Week-5:
/// small feature vectors use row interleaving; large vectors use feature partitioning.
[[nodiscard]] inline FeaturePartitionResult make_hybrid_partition(const FeaturePartitionConfig& cfg) {
  const PartitionMode mode = choose_partition_mode(cfg);
  if (mode == PartitionMode::RowInterleaved) {
    return make_row_interleaved_partition(cfg);
  }
  return make_feature_partition(cfg);
}

[[nodiscard]] inline std::vector<std::uint64_t> dims_per_vault(const FeaturePartitionResult& r) {
  std::vector<std::uint64_t> counts(r.num_vaults, 0);
  for (std::uint32_t v : r.dim_to_vault) {
    if (v >= r.num_vaults) {
      throw std::runtime_error("dim_to_vault contains out-of-range vault id");
    }
    counts[v]++;
  }
  return counts;
}

[[nodiscard]] inline PartitionStats compute_partition_stats(const FeaturePartitionResult& r) {
  PartitionStats s;
  const auto counts = dims_per_vault(r);
  if (counts.empty()) {
    return s;
  }

  s.min_dims_per_vault = *std::min_element(counts.begin(), counts.end());
  s.max_dims_per_vault = *std::max_element(counts.begin(), counts.end());
  std::uint64_t total = 0;
  for (std::uint64_t c : counts) {
    total += c;
  }
  s.avg_dims_per_vault = static_cast<double>(total) / static_cast<double>(counts.size());
  s.imbalance_ratio =
      (s.avg_dims_per_vault > 0.0) ? static_cast<double>(s.max_dims_per_vault) / s.avg_dims_per_vault : 0.0;
  return s;
}

}  // namespace pim

