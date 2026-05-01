#pragma once


#include "array_feeder.hpp"
#include "systolic_array.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

namespace pim {

// ---------------------------------------------------------------------------
// SIMD-lane helper arithmetic (16-wide, purely combinational / 1 SIMD cycle)
// ---------------------------------------------------------------------------

/// Approximate sigmoid via 1/(1+exp(-x)).
/// In silicon this would be a 256-entry LUT inside each SIMD lane (cf. Newton
/// §4.3 which stores activation LUTs adjacent to the MAC pipeline).
inline std::array<float, 16> vec_sigmoid(const std::array<float, 16>& v) {
  std::array<float, 16> out{};
  for (int i = 0; i < 16; ++i) {
    out[i] = 1.0f / (1.0f + std::exp(-v[i]));
  }
  return out;
}

inline std::array<float, 16> vec_tanh(const std::array<float, 16>& v) {
  std::array<float, 16> out{};
  for (int i = 0; i < 16; ++i) {
    out[i] = std::tanh(v[i]);
  }
  return out;
}

/// Element-wise Hadamard product (MUL_VEC in the SIMD ISA).
inline std::array<float, 16> vec_hadamard(const std::array<float, 16>& a,
                                           const std::array<float, 16>& b) {
  std::array<float, 16> out{};
  for (int i = 0; i < 16; ++i) out[i] = a[i] * b[i];
  return out;
}

/// Element-wise addition (ADD_VEC in the SIMD ISA).
inline std::array<float, 16> vec_add(const std::array<float, 16>& a,
                                      const std::array<float, 16>& b) {
  std::array<float, 16> out{};
  for (int i = 0; i < 16; ++i) out[i] = a[i] + b[i];
  return out;
}

/// scalar − v  (used for (1 − z) ⊙ h).
inline std::array<float, 16> vec_scalar_sub(float scalar,
                                             const std::array<float, 16>& v) {
  std::array<float, 16> out{};
  for (int i = 0; i < 16; ++i) out[i] = scalar - v[i];
  return out;
}

// ---------------------------------------------------------------------------
// GRU state machine
// ---------------------------------------------------------------------------

/// Sequenced phases of a single GRU cell evaluation.
/// The systolic array is pipelined; each *_DRAIN state counts down the
/// wavefront latency before reading the bottom-row output.
enum class GRUPhase {
  IDLE,
  // Gate 1: reset
  LOAD_RESET_GATE,    // Load W_r into SA; push h_prev into feeder
  DRAIN_RESET_GATE,   // Count kGEMMLatency cycles; read SA bottom row
  // Gate 2: update
  LOAD_UPDATE_GATE,   // Load W_z into SA; push h_prev into feeder
  DRAIN_UPDATE_GATE,
  // Gate 3a: W_hn · h  (to be reset-gated)
  LOAD_NEW_GATE_H,    // Load W_hn into SA; push h_prev into feeder
  DRAIN_NEW_GATE_H,
  // Gate 3b: W_in · x  (add to reset-gated term)
  LOAD_NEW_GATE_X,    // Load W_in into SA; push msg into feeder
  DRAIN_NEW_GATE_X,
  // Final: element-wise epilogue in SIMD
  APPLY_ELEMENTWISE,
  DONE,
};

// ---------------------------------------------------------------------------
// GRUCell
// ---------------------------------------------------------------------------

/// Usage (per vault, per tick):
///   gru.start(h_prev, aggregated_message);
///   while (!gru.tick(systolic_array, array_feeder)) { /* next cycle */ }
///   Vec16 h_new = gru.get_h_new();
class GRUCell {
 public:
  using Vec16 = std::array<float, 16>;

  // Each GEMM: 15 cycles of feeder wavefront + 16 SA rows = 31 active cycles.
  // We conservatively add 1 to account for the final drain of the bottom row.
  static constexpr int kGEMMLatency = SystolicArray::kDimension * 2; // 32

  GRUCell() {
    // Default: identity-like weights (W = I), zero biases.
    W_r_  = std::vector<float>(16 * 16, 0.0f);
    W_z_  = std::vector<float>(16 * 16, 0.0f);
    W_in_ = std::vector<float>(16 * 16, 0.0f);
    W_hn_ = std::vector<float>(16 * 16, 0.0f);
    for (int i = 0; i < 16; ++i) {
      W_r_ [i * 16 + i] = 1.0f;
      W_z_ [i * 16 + i] = 1.0f;
      W_in_[i * 16 + i] = 1.0f;
      W_hn_[i * 16 + i] = 1.0f;
    }
    b_r_.fill(0.0f); b_z_.fill(0.0f); b_in_.fill(0.0f); b_hn_.fill(0.0f);
  }

  // --- Weight / bias loading ---

  /// Load all four gate weight matrices (each 16×16 = 256 floats, row-major).
  /// TGN-PNM pre-loads weights into the systolic array during the "Update"
  /// phase so that the reuse factor across all vertices in a batch is maximised.
  void load_weights(const std::vector<float>& W_r, const std::vector<float>& W_z,
                    const std::vector<float>& W_in, const std::vector<float>& W_hn) {
    W_r_ = W_r; W_z_ = W_z; W_in_ = W_in; W_hn_ = W_hn;
  }

  void load_biases(const Vec16& b_r, const Vec16& b_z,
                   const Vec16& b_in, const Vec16& b_hn) {
    b_r_ = b_r; b_z_ = b_z; b_in_ = b_in; b_hn_ = b_hn;
  }

  // --- Computation API ---

  /// Begin computing GRU(msg, h_prev).
  ///   h_prev : previous temporal hidden state h_v(t−1) stored in HBM
  ///   msg    : aggregated message m_v(t) output by the PSAU
  void start(const Vec16& h_prev, const Vec16& msg) {
    h_prev_ = h_prev;
    msg_    = msg;
    phase_       = GRUPhase::LOAD_RESET_GATE;
    drain_count_ = 0;
    done_        = false;
    sa_bottom_.fill(0.0f);
    h_new_.fill(0.0f);
  }

  /// Advance by one clock cycle.
  /// `sa`     — the vault's systolic array (borrowed during GRU computation)
  /// `feeder` — the vault's array feeder (provides wavefront alignment)
  /// Returns true when computation is complete; h_new_ is valid thereafter.
  bool tick(SystolicArray& sa, ArrayFeeder& feeder) {
    if (done_) return true;
    drain_count_++;

    switch (phase_) {

      // RESET GATE  r = σ(W_r · h_prev + b_r + msg)
      // We approximate concat([h;x]) by projecting h alone then adding msg
      // as an additive input bias, matching the TGN-PNM simplification where
      // the message is injected after the GEMM (§3.3).
      case GRUPhase::LOAD_RESET_GATE:
        sa.load_weights(W_r_);
        feeder.push_vector(std::vector<float>(h_prev_.begin(), h_prev_.end()));
        drain_count_ = 0;
        phase_ = GRUPhase::DRAIN_RESET_GATE;
        break;

      case GRUPhase::DRAIN_RESET_GATE: {
        auto acts = feeder.get_next_activations();
        Vec16 zero_psums{}; zero_psums.fill(0.0f);
        sa_bottom_ = sa.tick(acts, zero_psums);

        if (drain_count_ >= kGEMMLatency) {
          Vec16 raw;
          for (int i = 0; i < 16; ++i)
            raw[i] = sa_bottom_[i] + b_r_[i] + msg_[i]; // additive msg injection
          reset_gate_ = vec_sigmoid(raw);
          drain_count_ = 0;
          phase_ = GRUPhase::LOAD_UPDATE_GATE;
        }
        break;
      }

      // UPDATE GATE  z = σ(W_z · h_prev + b_z + msg)
      case GRUPhase::LOAD_UPDATE_GATE:
        sa.load_weights(W_z_);
        feeder.push_vector(std::vector<float>(h_prev_.begin(), h_prev_.end()));
        drain_count_ = 0;
        phase_ = GRUPhase::DRAIN_UPDATE_GATE;
        break;

      case GRUPhase::DRAIN_UPDATE_GATE: {
        auto acts = feeder.get_next_activations();
        Vec16 zero_psums{}; zero_psums.fill(0.0f);
        sa_bottom_ = sa.tick(acts, zero_psums);

        if (drain_count_ >= kGEMMLatency) {
          Vec16 raw;
          for (int i = 0; i < 16; ++i)
            raw[i] = sa_bottom_[i] + b_z_[i] + msg_[i];
          update_gate_ = vec_sigmoid(raw);
          drain_count_ = 0;
          phase_ = GRUPhase::LOAD_NEW_GATE_H;
        }
        break;
      }

      // NEW GATE (hidden branch)  W_hn · h_prev  → apply reset gate

      case GRUPhase::LOAD_NEW_GATE_H:
        sa.load_weights(W_hn_);
        feeder.push_vector(std::vector<float>(h_prev_.begin(), h_prev_.end()));
        drain_count_ = 0;
        phase_ = GRUPhase::DRAIN_NEW_GATE_H;
        break;

      case GRUPhase::DRAIN_NEW_GATE_H: {
        auto acts = feeder.get_next_activations();
        Vec16 zero_psums{}; zero_psums.fill(0.0f);
        sa_bottom_ = sa.tick(acts, zero_psums);

        if (drain_count_ >= kGEMMLatency) {
          Vec16 hn_raw;
          for (int i = 0; i < 16; ++i)
            hn_raw[i] = sa_bottom_[i] + b_hn_[i];
          // r ⊙ (W_hn · h): SIMD Hadamard — 1 MUL_VEC instruction
          reset_gated_h_ = vec_hadamard(reset_gate_, hn_raw);
          drain_count_ = 0;
          phase_ = GRUPhase::LOAD_NEW_GATE_X;
        }
        break;
      }

      // -----------------------------------------------------------------------
      // NEW GATE (input branch)  n = tanh(W_in · msg + b_in + r⊙(W_hn·h))
      // -----------------------------------------------------------------------
      case GRUPhase::LOAD_NEW_GATE_X:
        sa.load_weights(W_in_);
        feeder.push_vector(std::vector<float>(msg_.begin(), msg_.end()));
        drain_count_ = 0;
        phase_ = GRUPhase::DRAIN_NEW_GATE_X;
        break;

      case GRUPhase::DRAIN_NEW_GATE_X: {
        auto acts = feeder.get_next_activations();
        Vec16 zero_psums{}; zero_psums.fill(0.0f);
        sa_bottom_ = sa.tick(acts, zero_psums);

        if (drain_count_ >= kGEMMLatency) {
          Vec16 n_raw;
          for (int i = 0; i < 16; ++i)
            n_raw[i] = sa_bottom_[i] + b_in_[i] + reset_gated_h_[i];
          new_gate_ = vec_tanh(n_raw);
          drain_count_ = 0;
          phase_ = GRUPhase::APPLY_ELEMENTWISE;
        }
        break;
      }

      // -----------------------------------------------------------------------
      // HIDDEN STATE UPDATE  h' = (1−z)⊙h + z⊙n
      // Three pure SIMD instructions (ADD_VEC, MUL_VEC × 2) executed in the
      // SIMD lane in a single pipeline cycle (TGN-PNM §3.3, "element-wise
      // operations are issued to the SIMD unit").
      // -----------------------------------------------------------------------
      case GRUPhase::APPLY_ELEMENTWISE: {
        auto one_minus_z = vec_scalar_sub(1.0f, update_gate_);   // 1−z
        auto term1       = vec_hadamard(one_minus_z, h_prev_);   // (1−z)⊙h
        auto term2       = vec_hadamard(update_gate_, new_gate_); // z⊙n
        h_new_           = vec_add(term1, term2);
        done_  = true;
        phase_ = GRUPhase::DONE;
        return true;
      }

      case GRUPhase::DONE:
        return true;

      default:
        break;
    }
    return false;
  }


  [[nodiscard]] bool is_done() const { return done_; }
  [[nodiscard]] const Vec16& get_h_new() const { return h_new_; }
  [[nodiscard]] GRUPhase get_phase()const { return phase_; }
  [[nodiscard]] int drain_count() const { return drain_count_; }

 private:
  // Weight matrices (16×16, row-major)
  std::vector<float> W_r_, W_z_, W_in_, W_hn_;
  // Bias vectors
  Vec16 b_r_{}, b_z_{}, b_in_{}, b_hn_{};

  // Runtime state
  Vec16 h_prev_{}, msg_{};
  Vec16 reset_gate_{}, update_gate_{}, reset_gated_h_{}, new_gate_{};
  Vec16 sa_bottom_{};   // last bottom-row readout from the systolic array
  Vec16 h_new_{};

  GRUPhase phase_      = GRUPhase::IDLE;
  int      drain_count_ = 0;
  bool     done_        = false;
};

} // namespace pim