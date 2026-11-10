// Error-state EKF: IMU propagation + position fixes (docs/derivation.md §3–6).
//
// A port of python/navekf_py/eskf.py, which is the reference; its module docstring is the
// contract for the step order. run() walks the IMU timestamps t_k and at each k:
//
//   1. every fix with t_fix == t_k: update_position (innovation, S, NIS, gate; Joseph update),
//      then inject_and_reset;
//   2. the hook sees the posterior at t_k;
//   3. if k < K−1: propagate with sample k over dt = (t_{k+1} − t_k)·1e-9:
//      a. F, G from the start-of-step nominal state and sample,
//      b. the nominal step of conventions §6,
//      c. Φ, Q_d from F, G Q_c Gᵀ, dt,
//      d. P ← Φ P Φᵀ + Q_d, P ← ½(P + Pᵀ), optional min-eigenvalue check.
//
// Nothing in propagate(), update_position(), inject_and_reset() or run() touches the heap
// (tests/cpp/test_no_alloc.cpp), except to build the message of an exception.
#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>

#include "navekf/imu_model.hpp"
#include "navekf/types.hpp"

namespace navekf {

/// chi2.ppf(0.99, 3), exactly as Python computes it (derivation §5 rounds it to 11.345).
inline constexpr double kGate99_3dof = 11.344866730144373;

enum class Reset : std::uint8_t { kJr, kIdentity };

// ------------------------------------------------------------- the pure functions

/// dt = (t_{k+1} − t_k) · 1e-9: integer subtraction first, then one multiply (conventions §1).
inline double step_dt(const std::int64_t t_k, const std::int64_t t_next) {
  return static_cast<double>(t_next - t_k) * 1e-9;
}

/// The one integrator (conventions §6).
State propagate_nominal(const State& x, const ImuSample& imu, double dt, double g = kGMagnitude);

/// x ⊞ δx: p + δp, v + δv, normalize(q ⊗ Exp(δθ)), b + δb.
State boxplus(const State& x, const ErrorState& dx);
/// x ⊖ x̂: the error with x = x̂ ⊞ δx; orientation Log(q̂* ⊗ q).
ErrorState boxminus(const State& x, const State& x_hat);

struct Jacobians {
  Mat15 F;
  Mat15x12 G;
};
/// Continuous-time F, G at the start of the step (derivation §3).
Jacobians jacobians(const State& x, const ImuSample& imu);

struct PositionModel {
  Vec3 z_hat;
  Mat3x15 H;
};
/// ẑ = p̂ + R̂ l and H = [I 0 −R̂[l]× 0 0] (derivation §5).
PositionModel position_model(const State& x, const Vec3& lever_arm);

/// G_r = blockdiag(I, I, J_r(δθ̂), I, I) (kJr) or I (kIdentity), derivation §6.
Mat15 reset_jacobian(const Vec3& dth, Reset method);

// ------------------------------------------------------------------- the filter

struct EskfConfig {
  ImuNoise noise;
  double g = kGMagnitude;
  Vec3 lever_arm = Vec3::Zero();  ///< antenna in B [m]
  Discretization discretization = Discretization::kFirst;
  Reset reset = Reset::kJr;
  std::optional<double> gate = kGate99_3dof;  ///< NIS threshold; nullopt disables the gate
  bool check_psd = false;                     ///< min-eigenvalue check of P after every step
  double psd_tol = 0.0;                       ///< fail if min eig(P) < −psd_tol · max eig(P)
};

struct UpdateResult {
  std::int64_t t_ns = 0;
  Vec3 y = Vec3::Zero();  ///< innovation z − ẑ
  Mat3 S = Mat3::Zero();  ///< innovation covariance
  double nis = 0.0;       ///< yᵀ S⁻¹ y
  bool accepted = false;  ///< NIS ≤ gate
};

class Eskf {
 public:
  Eskf(const State& x0, const Mat15& P0, const EskfConfig& cfg, std::int64_t t0_ns = 0);

  /// One IMU step. Without t_next_ns the new time is imu.t_ns + round(dt·1e9).
  void propagate(const ImuSample& imu, double dt,
                 std::optional<std::int64_t> t_next_ns = std::nullopt);

  /// Position update with NIS gate and Joseph form, then injection and reset (unless
  /// inject = false). A rejected fix leaves x̂ and P bit-identical and counts in rejections().
  UpdateResult update_position(const PositionFix& fix, bool inject = true);

  /// x̂ ← x̂ ⊞ δx̂, P ← G_r P G_rᵀ, δx̂ ← 0.
  void inject_and_reset();

  const State& x() const { return x_; }
  const Mat15& P() const { return P_; }
  const ErrorState& dx() const { return dx_; }
  std::int64_t t_ns() const { return t_ns_; }
  const EskfConfig& config() const { return cfg_; }
  std::size_t rejections() const { return rejections_; }
  /// Smallest eigenvalue of P at the last check (only with check_psd).
  std::optional<double> min_eig() const { return min_eig_; }

 private:
  void check();

  EskfConfig cfg_;
  State x_;
  Mat15 P_;
  ErrorState dx_ = ErrorState::Zero();
  Mat12 Qc_;
  std::int64_t t_ns_ = 0;
  std::size_t rejections_ = 0;
  std::optional<double> min_eig_;
};

// ----------------------------------------------------------------- the run loop

/// Row kinds of the per-step log (conventions §8), as in Python's StepLog.
inline constexpr int kRowImu = 0;
inline constexpr int kRowUpdate = 1;

/// Filter a whole IMU stream from imu[0].t_ns (the state's time), in the step order above.
///
/// hook(k, filter) sees the posterior at t_k. on_row(filter, kind, update) fires after every
/// propagate (kRowImu, update == nullptr) and every update (kRowUpdate). `fixes` must be
/// sorted by time; fixes before imu[0] are skipped and one off the IMU grid throws.
template <class Hook, class OnRow>
void run(Eskf& filter, std::span<const ImuSample> imu, std::span<const PositionFix> fixes,
         Hook hook, OnRow on_row) {
  for (std::size_t i = 1; i < fixes.size(); ++i) {
    if (fixes[i].t_ns < fixes[i - 1].t_ns) {
      throw std::invalid_argument("run: fixes must be sorted by time");
    }
  }
  std::size_t j = 0;
  while (j < fixes.size() && !imu.empty() && fixes[j].t_ns < imu[0].t_ns) {
    ++j;
  }
  const std::size_t n = imu.size();
  for (std::size_t k = 0; k < n; ++k) {
    while (j < fixes.size() && fixes[j].t_ns <= imu[k].t_ns) {
      if (fixes[j].t_ns != imu[k].t_ns) {
        throw std::invalid_argument("fix at t_ns=" + std::to_string(fixes[j].t_ns) +
                                    " is not on an IMU timestamp");
      }
      const UpdateResult u = filter.update_position(fixes[j]);
      on_row(static_cast<const Eskf&>(filter), kRowUpdate, &u);
      ++j;
    }
    hook(k, static_cast<const Eskf&>(filter));
    if (k + 1 < n) {
      filter.propagate(imu[k], step_dt(imu[k].t_ns, imu[k + 1].t_ns), imu[k + 1].t_ns);
      on_row(static_cast<const Eskf&>(filter), kRowImu, static_cast<const UpdateResult*>(nullptr));
    }
  }
}

}  // namespace navekf
