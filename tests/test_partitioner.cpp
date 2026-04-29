#include "feature_partitioner.hpp"

#include <gtest/gtest.h>

namespace {

TEST(FeaturePartitioner, RowInterleavedWhenBelowThreshold) {
  pim::FeaturePartitionConfig cfg;
  cfg.num_vaults = 16;
  cfg.feature_dim = 64;
  cfg.hybrid_threshold_dim = 128;
  cfg.chunk_size = 4;

  const auto out = pim::make_hybrid_partition(cfg);
  EXPECT_EQ(out.mode, pim::PartitionMode::RowInterleaved);
  ASSERT_EQ(out.dim_to_vault.size(), cfg.feature_dim);
  for (std::uint32_t d = 0; d < cfg.feature_dim; d++) {
    EXPECT_EQ(out.dim_to_vault[d], d % cfg.num_vaults);
  }
}

TEST(FeaturePartitioner, FeaturePartitionWhenAboveThreshold) {
  pim::FeaturePartitionConfig cfg;
  cfg.num_vaults = 16;
  cfg.feature_dim = 256;
  cfg.hybrid_threshold_dim = 128;
  cfg.chunk_size = 8;

  const auto out = pim::make_hybrid_partition(cfg);
  EXPECT_EQ(out.mode, pim::PartitionMode::FeaturePartitioned);
  ASSERT_EQ(out.dim_to_vault.size(), cfg.feature_dim);
  for (std::uint32_t v : out.dim_to_vault) {
    EXPECT_LT(v, cfg.num_vaults);
  }
}

TEST(FeaturePartitioner, StatsAreBalancedForEvenCase) {
  pim::FeaturePartitionConfig cfg;
  cfg.num_vaults = 16;
  cfg.feature_dim = 256;
  cfg.chunk_size = 1;

  const auto out = pim::make_feature_partition(cfg);
  const auto stats = pim::compute_partition_stats(out);

  EXPECT_DOUBLE_EQ(stats.avg_dims_per_vault, 16.0);
  EXPECT_EQ(stats.min_dims_per_vault, 16u);
  EXPECT_EQ(stats.max_dims_per_vault, 16u);
  EXPECT_DOUBLE_EQ(stats.imbalance_ratio, 1.0);
}

}  // namespace

