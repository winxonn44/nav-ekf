// IMU noise model and the discretization of the error dynamics (derivation §3–4).
#pragma once

#include <cstdint>

#include "navekf/types.hpp"

namespace navekf {

/// Continuous-time noise densities, per axis (conventions §4).
struct ImuNoise {
  double sigma_a = 0.0;   ///< accel white noise      [m/s²/√Hz]
  double sigma_g = 0.0;   ///< gyro white noise (ARW) [rad/s/√Hz]
  double sigma_ba = 0.0;  ///< accel bias random walk [m/s³/√Hz]
  double sigma_bg = 0.0;  ///< gyro bias random walk  [rad/s²/√Hz]

  /// Q_c = diag(σ_a² I, σ_g² I, σ_ba² I, σ_bg² I), 12×12.
  Mat12 Qc() const;
};

enum class Discretization : std::uint8_t { kFirst, kSecond, kVanLoan };

struct Discrete {
  Mat15 Phi;
  Mat15 Qd;
};

/// Φ, Q_d over one step, with GQG = G Q_c Gᵀ.
///   kFirst:   Φ = I + F dt, Q_d = GQG dt.
///   kSecond:  Φ = I + F dt + ½F²dt², Q_d = GQG dt + ½(F GQG + GQG Fᵀ)dt² + ⅓ F GQG Fᵀ dt³.
///   kVanLoan: exact, from the matrix exponential of the 30×30 block matrix.
Discrete discretize(const Mat15& F, const Mat15& GQG, double dt, Discretization method);

}  // namespace navekf
