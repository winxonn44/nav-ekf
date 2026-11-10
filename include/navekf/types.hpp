// Plain data types (docs/conventions.md §1, §5). Fixed-size Eigen only: nothing here allocates.
#pragma once

#include <Eigen/Core>

#include <cstdint>

namespace navekf {

using Vec3 = Eigen::Vector3d;
/// A quaternion stored [w, x, y, z] (conventions §3). Deliberately not Eigen::Quaterniond,
/// which stores [x, y, z, w]; so3::to_eigen / so3::from_eigen are the only crossings.
using Quat = Eigen::Vector4d;
using Mat3 = Eigen::Matrix3d;
using Vec15 = Eigen::Matrix<double, 15, 1>;
using Mat12 = Eigen::Matrix<double, 12, 12>;
using Mat15 = Eigen::Matrix<double, 15, 15>;
using Mat15x12 = Eigen::Matrix<double, 15, 12>;
using Mat3x15 = Eigen::Matrix<double, 3, 15>;
using Mat15x3 = Eigen::Matrix<double, 15, 3>;

inline constexpr int kNErr = 15;
inline constexpr double kGMagnitude = 9.81;  // m/s², conventions §4

// Error-state block offsets (conventions §5).
inline constexpr int kIP = 0;
inline constexpr int kIV = 3;
inline constexpr int kITh = 6;
inline constexpr int kIBa = 9;
inline constexpr int kIBg = 12;

/// Nominal state x̂ (conventions §5).
struct State {
  Vec3 p = Vec3::Zero();                       ///< position of B in W [m]
  Vec3 v = Vec3::Zero();                       ///< velocity in W [m/s]
  Quat q = (Quat() << 1, 0, 0, 0).finished();  ///< q_W_B, [w, x, y, z]
  Vec3 ba = Vec3::Zero();                      ///< accelerometer bias [m/s²]
  Vec3 bg = Vec3::Zero();                      ///< gyro bias [rad/s]
};

/// Error state δx: [δp, δv, δθ, δb_a, δb_g].
using ErrorState = Vec15;

/// One IMU reading, held over [t_ns, t_ns_next) (zero-order hold, conventions §6).
struct ImuSample {
  std::int64_t t_ns = 0;
  Vec3 acc = Vec3::Zero();  ///< specific force a_m in B [m/s²]
  Vec3 gyr = Vec3::Zero();  ///< angular rate ω_m in B [rad/s]
};

/// A position fix of the GPS antenna in W (derivation §5).
struct PositionFix {
  std::int64_t t_ns = 0;
  Vec3 p_W = Vec3::Zero();  ///< measured antenna position [m]
  double sigma = 1.0;       ///< per-axis std [m]; R_meas = σ² I
};

}  // namespace navekf
