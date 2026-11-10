// SO(3) and quaternion helpers (docs/conventions.md §3), mirroring python/navekf_py/so3.py:
// same formulas, same branches, same order of floating-point operations.
#pragma once

#include <Eigen/Geometry>

#include <cmath>

#include "navekf/types.hpp"

namespace navekf::so3 {

inline constexpr double kSmallAngle = 1e-8;  // conventions §3.1: below this use the Taylor branches

/// [a]× with [a]× b = a × b.
inline Mat3 skew(const Vec3& a) {
  Mat3 m;
  m << 0.0, -a.z(), a.y(),  //
      a.z(), 0.0, -a.x(),   //
      -a.y(), a.x(), 0.0;
  return m;
}

/// Hamilton product p ⊗ q, [w, x, y, z].
inline Quat quat_mul(const Quat& p, const Quat& q) {
  const double pw = p(0), px = p(1), py = p(2), pz = p(3);
  const double qw = q(0), qx = q(1), qy = q(2), qz = q(3);
  return {pw * qw - px * qx - py * qy - pz * qz,  //
          pw * qx + px * qw + py * qz - pz * qy,  //
          pw * qy - px * qz + py * qw + pz * qx,  //
          pw * qz + px * qy - py * qx + pz * qw};
}

inline Quat quat_conj(const Quat& q) { return {q(0), -q(1), -q(2), -q(3)}; }

inline Quat quat_normalize(const Quat& q) {
  const double n = std::sqrt(q(0) * q(0) + q(1) * q(1) + q(2) * q(2) + q(3) * q(3));
  return {q(0) / n, q(1) / n, q(2) / n, q(3) / n};
}

/// Sign-fixed copy with w ≥ 0. Only for comparing and logging (conventions §3).
inline Quat quat_canonical(const Quat& q) { return q(0) < 0.0 ? Quat(-q) : q; }

/// Rotation matrix R(q): v_A = R(q_A_B) v_B.
inline Mat3 quat_to_rot(const Quat& q) {
  const double w = q(0), x = q(1), y = q(2), z = q(3);
  Mat3 r;
  r << 1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y),  //
      2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x),   //
      2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y);
  return r;
}

/// The only way into Eigen's quaternion type: its constructor takes (w, x, y, z), its
/// storage (.coeffs()) is [x, y, z, w]. Never hand .coeffs() or .data() across a boundary.
inline Eigen::Quaterniond to_eigen(const Quat& q) { return {q(0), q(1), q(2), q(3)}; }
inline Quat from_eigen(const Eigen::Quaterniond& q) { return {q.w(), q.x(), q.y(), q.z()}; }

/// Exp(φ): rotation vector -> unit quaternion. Regular branch [cos(θ/2), sin(θ/2) φ/θ] (not
/// re-normalized); for θ < 1e-8 the Taylor form [1 − θ²/8, φ/2 (1 − θ²/24)], then normalized.
inline Quat Exp(const Vec3& phi) {
  const double th2 = phi(0) * phi(0) + phi(1) * phi(1) + phi(2) * phi(2);
  const double th = std::sqrt(th2);
  if (th < kSmallAngle) {
    const double c = 1.0 - th2 / 24.0;
    return quat_normalize(
        Quat(1.0 - th2 / 8.0, 0.5 * phi(0) * c, 0.5 * phi(1) * c, 0.5 * phi(2) * c));
  }
  const double s = std::sin(0.5 * th) / th;
  return {std::cos(0.5 * th), s * phi(0), s * phi(1), s * phi(2)};
}

/// Log(q): unit quaternion -> rotation vector with ‖φ‖ ≤ π. Flips q -> −q when w < 0, then
/// φ = 2 atan2(‖v‖, w) v/‖v‖; for ‖v‖ < 1e-8 the Taylor form φ = (2/w)(1 − ‖v‖²/(3w²)) v.
inline Vec3 Log(const Quat& q_in) {
  const Quat q = quat_canonical(q_in);
  const double w = q(0);
  const double n2 = q(1) * q(1) + q(2) * q(2) + q(3) * q(3);
  const double n = std::sqrt(n2);
  const double scale = n < kSmallAngle ? 2.0 / w * (1.0 - n2 / (3.0 * w * w))  //
                                       : 2.0 * std::atan2(n, w) / n;
  return {scale * q(1), scale * q(2), scale * q(3)};
}

/// Exp(φ) as a rotation matrix, via the quaternion.
inline Mat3 exp_rot(const Vec3& phi) { return quat_to_rot(Exp(phi)); }

/// Right Jacobian (derivation §6): J_r(φ) = I − (1 − cos θ)/θ² [φ]× + (θ − sin θ)/θ³ [φ]×²;
/// small-angle I − ½[φ]×. It satisfies Exp(φ + ε) ≈ Exp(φ) Exp(J_r(φ) ε).
inline Mat3 right_jacobian(const Vec3& phi) {
  const double th = std::sqrt(phi(0) * phi(0) + phi(1) * phi(1) + phi(2) * phi(2));
  const Mat3 k = skew(phi);
  if (th < kSmallAngle) {
    return Mat3::Identity() - 0.5 * k;
  }
  const Mat3 k2 = k * k;
  return Mat3::Identity() - (1.0 - std::cos(th)) / std::pow(th, 2.0) * k +
         (th - std::sin(th)) / std::pow(th, 3.0) * k2;
}

}  // namespace navekf::so3
