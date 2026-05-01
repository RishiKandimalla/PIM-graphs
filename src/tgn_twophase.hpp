#pragma once



#include "array_feeder.hpp"
#include "gru.hpp"
#include "piccolo_controller.hpp"
#include "psau.hpp"
#include "systolic_array.hpp"

#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

namespace pim {

enum class VertexProgramPhase {
  IDLE,
  // ── Phase 1: GenUpdate (Message + Aggregate) ──
  GEN_UPDATE_GATHER,       // Piccolo gathers neighbor feature bytes per vault
  GEN_UPDATE_COMPUTE,      // Systolic arrays compute partial message vectors
  GEN_UPDATE_PSAU_DRAIN,   // PSAU reduction tree produces aggregated message
  // ── Phase 2: ApplyUpdate (Memory Update via GRU) ──
  APPLY_UPDATE_LOAD_H,     // Load previous hidden state h_v(t-1) from HBM
  APPLY_UPDATE_GRU,        // GRU cell sequences through gate computations
  WRITEBACK,               // h_v(t) is ready; caller issues the HBM write burst
  DONE,
};

inline const char* to_string(VertexProgramPhase p) {
  switch (p) {
    case VertexProgramPhase::IDLE:                   return "IDLE";
    case VertexProgramPhase::GEN_UPDATE_GATHER:      return "GEN_UPDATE_GATHER";
    case VertexProgramPhase::GEN_UPDATE_COMPUTE:     return "GEN_UPDATE_COMPUTE";
    case VertexProgramPhase::GEN_UPDATE_PSAU_DRAIN:  return "GEN_UPDATE_PSAU_DRAIN";
    case VertexProgramPhase::APPLY_UPDATE_LOAD_H:    return "APPLY_UPDATE_LOAD_H";
    case VertexProgramPhase::APPLY_UPDATE_GRU:       return "APPLY_UPDATE_GRU";
    case VertexProgramPhase::WRITEBACK:              return "WRITEBACK";
    case VertexProgramPhase::DONE:                   return "DONE";
  }
  return "UNKNOWN";
}


/// Per-vertex configuration passed to start_vertex().
struct VertexProgramConfig {
  /// Base HBM address of the vertex feature block (used by Piccolo).
  std::uint64_t vertex_addr        = 0;
  /// Address of h_v(t-1) in HBM (logic-layer SRAM cache or DRAM).
  std::uint64_t hidden_state_addr  = 0;
  /// Address where the new h_v(t) should be written back.
  std::uint64_t output_addr        = 0;
  /// Number of temporal neighbours to aggregate in GenUpdate.
  std::uint32_t num_neighbors      = 0;
  /// Byte offsets of each neighbour inside the vertex feature block.
  /// Pre-filled by the TGN event scheduler from the temporal edge list.
  std::vector<std::uint32_t> neighbor_offsets;
};


struct VertexProgramStats {
  std::uint64_t gather_cycles     = 0;  ///< Cycles spent in GEN_UPDATE_GATHER
  std::uint64_t compute_cycles    = 0;  ///< Cycles spent in GEN_UPDATE_COMPUTE
  std::uint64_t psau_cycles       = 0;  ///< Cycles spent draining the PSAU
  std::uint64_t load_h_cycles     = 0;  ///< Cycles waiting for h_v(t-1) read
  std::uint64_t gru_cycles        = 0;  ///< Cycles in APPLY_UPDATE_GRU
  std::uint64_t total_cycles      = 0;  ///< Total elapsed cycles this vertex
  std::uint32_t vertices_done     = 0;  ///< Number of vertices fully processed
};



/// Orchestrates the two-phase vertex program across all 16 vaults.
///
/// External connections (must be set before calling start_vertex):
///   - 16 PiccoloGatherController*  (one per vault)
///   - PartialSumAccumulationUnit&  (shared PSAU at the logic layer)
///   - 16 SystolicArray*            (one per vault, borrowed during GRU)
///   - 16 ArrayFeeder*              (one per vault, provides wavefront delay)
///
/// The class does NOT own any of the hardware objects it references.
class TGNVertexProgram {
 public:
  using Vec16 = std::array<float, 16>;

  // Number of PSAU pipeline stages. After the last valid input is presented,
  // the PSAU needs 4 more ticks to flush (psau.hpp §pipeline).
  static constexpr int kPSAUDepth = 4;


  TGNVertexProgram(std::array<PiccoloGatherController*, 16> piccolos,
                   PartialSumAccumulationUnit&               psau,
                   std::array<SystolicArray*, 16>            systolic_arrays,
                   std::array<ArrayFeeder*, 16>              feeders)
      : piccolos_(piccolos),
        psau_(psau),
        systolic_arrays_(systolic_arrays),
        feeders_(feeders) {
    for (int i = 0; i < 16; ++i) {
      gru_cells_[i] = std::make_unique<GRUCell>();
    }
  }


  /// Broadcast the same GRU weight matrices to all 16 vault GRU cells.
  /// In the physical design (TGN-PNM §3.4), weights are broadcast from a
  /// shared weight buffer at the logic layer to avoid 16× redundant storage.
  void load_gru_weights(const std::vector<float>& W_r,
                        const std::vector<float>& W_z,
                        const std::vector<float>& W_in,
                        const std::vector<float>& W_hn) {
    for (int i = 0; i < 16; ++i) {
      gru_cells_[i]->load_weights(W_r, W_z, W_in, W_hn);
    }
  }

  void load_gru_biases(const Vec16& b_r,  const Vec16& b_z,
                       const Vec16& b_in, const Vec16& b_hn) {
    for (int i = 0; i < 16; ++i) {
      gru_cells_[i]->load_biases(b_r, b_z, b_in, b_hn);
    }
  }

  // ── Vertex Processing API ────────────────────────────────────────────────

  /// Begin the two-phase program for a new target vertex.
  /// The caller must have filled cfg.neighbor_offsets with the byte offsets
  /// of each temporal neighbour's feature slice in the feature partition.
  ///
  /// GraphP note: offset computation is the responsibility of the
  /// "source-cut → feature-dimension-cut" mapping layer
  /// (see feature_partitioner.hpp).
  void start_vertex(const VertexProgramConfig& cfg) {
    cfg_   = cfg;
    phase_ = VertexProgramPhase::GEN_UPDATE_GATHER;

    // Reset counters
    gather_cycles_  = 0;
    compute_cycles_ = 0;
    psau_cycles_    = 0;
    load_h_cycles_  = 0;
    gru_cycles_     = 0;
    total_cycles_   = 0;
    psau_flush_count_ = 0;

    aggregated_message_.fill(0.0f);
    prev_hidden_state_.fill(0.0f);
    new_hidden_state_.fill(0.0f);
    h_load_done_   = false;
    writeback_ready_ = false;

    // ── Piccolo Offset Buffer pre-load ──────────────────────────────────
    // the feature-dimension partition means each vault
    // owns a contiguous slice of feature dimensions for ALL nodes. The
    // Piccolo controller's Offset Buffer is pre-populated with the byte
    // offsets corresponding to each temporal neighbour's feature slice.
    // This is analogous to TGN-PNM "bank-level temporal filter" which
    // sends index lists to the FIM unit rather than raw addresses.
    for (int v = 0; v < 16; ++v) {
      if (piccolos_[v] == nullptr) continue;
      piccolos_[v]->reset_for_new_batch(cfg_.num_neighbors);
      piccolos_[v]->set_base_addr(cfg_.vertex_addr);
      for (std::uint32_t off : cfg_.neighbor_offsets) {
        piccolos_[v]->ob.push(off);
      }
    }
  }

  /// Register a callback that the simulator calls when the h_v(t-1) read
  /// completes.  The callback writes the payload into prev_hidden_state_ and
  /// sets h_load_done_ = true.
  std::function<void(const Vec16&)> make_load_h_callback() {
    return [this](const Vec16& payload) {
      prev_hidden_state_ = payload;
      h_load_done_       = true;
    };
  }

  /// Advance the vertex program by one simulation cycle.
  ///
  /// sa_outputs  Current bottom-row outputs from all 16 systolic arrays (passed in from ramulator_hbm.cpp main loop).
  /// sa_valid    True if the SA outputs carry real data this cycle.
  /// @returns true when the program has reached DONE for the current vertex.
  bool tick(const std::array<Vec16, 16>& sa_outputs, bool sa_valid) {
    if (phase_ == VertexProgramPhase::IDLE ||
        phase_ == VertexProgramPhase::DONE) {
      return phase_ == VertexProgramPhase::DONE;
    }

    total_cycles_++;

    switch (phase_) {
      case VertexProgramPhase::GEN_UPDATE_GATHER:
        tick_gen_update_gather();
        break;
      case VertexProgramPhase::GEN_UPDATE_COMPUTE:
        tick_gen_update_compute(sa_outputs);
        break;
      case VertexProgramPhase::GEN_UPDATE_PSAU_DRAIN:
        tick_psau_drain(sa_outputs, sa_valid);
        break;
      case VertexProgramPhase::APPLY_UPDATE_LOAD_H:
        tick_load_hidden_state();
        break;
      case VertexProgramPhase::APPLY_UPDATE_GRU:
        tick_gru();
        break;
      case VertexProgramPhase::WRITEBACK:
        tick_writeback();
        break;
      default:
        break;
    }

    return phase_ == VertexProgramPhase::DONE;
  }

  [[nodiscard]] VertexProgramPhase  get_phase()             const { return phase_; }
  [[nodiscard]] bool                is_done()               const { return phase_ == VertexProgramPhase::DONE; }
  [[nodiscard]] bool                writeback_ready()       const { return writeback_ready_; }
  [[nodiscard]] std::uint64_t       output_addr()           const { return cfg_.output_addr; }
  [[nodiscard]] const Vec16&        get_result()            const { return new_hidden_state_; }
  [[nodiscard]] const Vec16&        get_aggregated_message()const { return aggregated_message_; }

  [[nodiscard]] VertexProgramStats  get_stats() const {
    return {gather_cycles_, compute_cycles_, psau_cycles_,
            load_h_cycles_, gru_cycles_,    total_cycles_,
            stats_.vertices_done};
  }

  void increment_vertex_count() { stats_.vertices_done++; }

 private:

  /// GEN_UPDATE_GATHER
  /// Tick all 16 Piccolo controllers.  When every vault has finished
  /// gathering its neighbour feature slice, drain the Data Buffers into
  /// the ArrayFeeders to initiate the systolic array message transform.
  ///
  /// GraphP §3.2: "Each cube independently generates its local contribution
  /// to the update without coordinating with other cubes."  Here that means
  /// each Piccolo controller runs independently; the PSAU later combines
  /// the partial outputs.
  void tick_gen_update_gather() {
    gather_cycles_++;

    bool all_done = true;
    for (int v = 0; v < 16; ++v) {
      if (piccolos_[v] == nullptr) continue;
      piccolos_[v]->tick_issue();
      if (!piccolos_[v]->done()) {
        all_done = false;
      }
    }

    if (!all_done) return;

    // Piccolo returns 4-byte elements (floats) packed in the Data Buffer.
    // We drain the DB into a 16-element vector and push it through the
    // ArrayFeeder, which adds the spatial wavefront staggering needed by
    // the systolic array (array_feeder.hpp).
    for (int v = 0; v < 16; ++v) {
      if (feeders_[v] == nullptr || piccolos_[v] == nullptr) continue;

      std::vector<float> gathered_vec(16, 0.0f);
      for (int d = 0; d < 16 && !piccolos_[v]->db.empty(); ++d) {
        std::uint32_t raw = piccolos_[v]->db.front();
        piccolos_[v]->db.pop();
        // Bit-cast uint32 → float (same reinterpret_cast used in simd_core.hpp)
        float fval;
        __builtin_memcpy(&fval, &raw, sizeof(float));
        gathered_vec[d] = fval;
      }
      feeders_[v]->push_vector(gathered_vec);
    }

    compute_cycles_ = 0;
    phase_          = VertexProgramPhase::GEN_UPDATE_COMPUTE;
  }

  /// GEN_UPDATE_COMPUTE
  /// Tick all 16 systolic arrays through the feeder for kDimension cycles.
  /// At the end of the drain, the bottom-row outputs of each SA represent
  /// that vault's partial message vector (one element per output-dimension).
  ///
  /// The systolic array executes the dense feature
  /// transformation W_msg · h_u in a weight-stationary dataflow. - TGN
  void tick_gen_update_compute(const std::array<Vec16, 16>& sa_outputs) {
    compute_cycles_++;

    // Drain the ArrayFeeders into each systolic array
    Vec16 zero_psums{}; zero_psums.fill(0.0f);
    for (int v = 0; v < 16; ++v) {
      if (systolic_arrays_[v] == nullptr || feeders_[v] == nullptr) continue;
      auto acts = feeders_[v]->get_next_activations();
      systolic_arrays_[v]->tick(acts, zero_psums);
    }

    // After kDimension cycles the wavefront has fully traversed all SA rows
    if (compute_cycles_ >= SystolicArray::kDimension) {
      // Latch SA outputs as PSAU inputs
      partial_sum_inputs_ = sa_outputs;
      psau_flush_count_   = 0;
      psau_cycles_        = 0;
      phase_              = VertexProgramPhase::GEN_UPDATE_PSAU_DRAIN;
    }
  }

  /// GEN_UPDATE_PSAU_DRAIN
  /// Feed the 16 partial vectors into the PSAU and wait for the valid
  /// output to emerge from the 4-stage pipeline.
  ///
  /// PSAU architecture (psau.hpp): 4-stage binary adder tree
  ///   Stage 1: 16 → 8   Stage 2: 8 → 4   Stage 3: 4 → 2   Stage 4: 2 → 1
  /// The output is valid kPSAUDepth = 4 cycles after the last input.
  ///
  /// This exactly matches "Inter-Vault Reduction" which uses a
  /// pipelined adder tree to collapse vault-level partial sums in O(log N)
  /// depth rather than O(N) sequential adds.
  void tick_psau_drain(const std::array<Vec16, 16>& /*sa_outputs*/, bool sa_valid) {
    psau_cycles_++;

    Vec16 psau_out{};
    // On the first few ticks provide the real partial inputs; after that
    // provide bubbles so the pipeline flushes.
    bool input_valid = (psau_flush_count_ == 0);
    bool output_valid = psau_.tick(partial_sum_inputs_, input_valid, psau_out);
    psau_flush_count_++;

    if (output_valid) {
      aggregated_message_ = psau_out;
      h_load_done_        = false;
      load_h_cycles_      = 0;
      phase_              = VertexProgramPhase::APPLY_UPDATE_LOAD_H;
    } else if (psau_flush_count_ > kPSAUDepth + 2) {
      // Safety valve: the PSAU must have drained by now; take current output
      aggregated_message_ = psau_out;
      h_load_done_        = false;
      load_h_cycles_      = 0;
      phase_              = VertexProgramPhase::APPLY_UPDATE_LOAD_H;
    }
  }

  /// APPLY_UPDATE_LOAD_H
  /// Wait for the external memory system to satisfy the h_v(t-1) read.
  /// The ramulator_hbm.cpp tick loop issues a MemoryBurst to cfg_.hidden_state_addr
  /// and the on_complete callback fires make_load_h_callback() which sets
  /// h_load_done_ = true and stores the payload in prev_hidden_state_.
  ///
  /// This models the "memory access for temporal state" described in
  /// "The hidden state of each vertex is stored in the HBM
  /// logic-layer SRAM, avoiding a TSV round-trip for every update." - TGN
  void tick_load_hidden_state() {
    load_h_cycles_++;
    if (!h_load_done_) return;

    // Start all 16 GRU cells in parallel. Each vault runs its own GRUCell
    // because the feature-dimension partition means each vault owns a
    // different slice of the 16-dim weight space.
    for (int v = 0; v < 16; ++v) {
      if (gru_cells_[v]) {
        gru_cells_[v]->start(prev_hidden_state_, aggregated_message_);
      }
    }
    gru_cycles_ = 0;
    phase_      = VertexProgramPhase::APPLY_UPDATE_GRU;
  }

  /// APPLY_UPDATE_GRU
  /// Tick all 16 GRU cells each cycle until every vault has finished.
  ///
  /// "The SIMD unit handles time encoding and the element-wise
  /// gating operations; the systolic array handles all linear projections." - TGN
  /// Each GRUCell borrows the vault's systolic array for its GEMM phases.
  void tick_gru() {
    gru_cycles_++;
    int done_count = 0;

    for (int v = 0; v < 16; ++v) {
      if (!gru_cells_[v] || systolic_arrays_[v] == nullptr || feeders_[v] == nullptr) {
        done_count++;
        continue;
      }
      if (gru_cells_[v]->is_done()) {
        done_count++;
      } else {
        gru_cells_[v]->tick(*systolic_arrays_[v], *feeders_[v]);
        if (gru_cells_[v]->is_done()) done_count++;
      }
    }

    if (done_count < 16) return;

    // ── All vaults have produced h_v(t) ──────────────────────────────────
    // GraphP ApplyUpdate is owned by the vertex's home vault (vault 0
    // in our model for simplicity). In a full implementation the 16 partial
    // hidden states would be re-reduced through the PSAU; here they are
    // averaged to produce the final 16-dim output.
    new_hidden_state_.fill(0.0f);
    for (int v = 0; v < 16; ++v) {
      if (!gru_cells_[v]) continue;
      const auto& h = gru_cells_[v]->get_h_new();
      for (int d = 0; d < 16; ++d) {
        new_hidden_state_[d] += h[d] / 16.0f;
      }
    }
    phase_ = VertexProgramPhase::WRITEBACK;
  }

  /// WRITEBACK
  /// Signal the simulator that h_v(t) is ready to be written to HBM.
  /// The actual MemoryBurst is issued by ramulator_hbm.cpp using
  /// cfg_.output_addr (similar to the PSAU writeback already in the
  /// existing run() loop).
  void tick_writeback() {
    writeback_ready_ = true;
    phase_           = VertexProgramPhase::DONE;
    stats_.vertices_done++;
  }

  // ── Hardware references (non-owning) ─────────────────────────────────────
  std::array<PiccoloGatherController*, 16> piccolos_;
  PartialSumAccumulationUnit&              psau_;
  std::array<SystolicArray*, 16>          systolic_arrays_;
  std::array<ArrayFeeder*, 16>            feeders_;

  // One GRU cell per vault (owned here)
  std::array<std::unique_ptr<GRUCell>, 16> gru_cells_;

  // ── Program state ─────────────────────────────────────────────────────────
  VertexProgramConfig  cfg_{};
  VertexProgramPhase   phase_ = VertexProgramPhase::IDLE;

  // Intermediate data vectors
  std::array<Vec16, 16> partial_sum_inputs_{};
  Vec16 aggregated_message_{};
  Vec16 prev_hidden_state_{};
  Vec16 new_hidden_state_{};

  // Phase cycle counters (for stats)
  std::uint64_t gather_cycles_  = 0;
  std::uint64_t compute_cycles_ = 0;
  std::uint64_t psau_cycles_    = 0;
  std::uint64_t load_h_cycles_  = 0;
  std::uint64_t gru_cycles_     = 0;
  std::uint64_t total_cycles_   = 0;

  // PSAU flush tracking
  int psau_flush_count_ = 0;

  // Load-H completion flag (set by make_load_h_callback())
  bool h_load_done_    = false;
  bool writeback_ready_ = false;

  // Summary stats
  VertexProgramStats stats_{};
};  

} // namespace pim