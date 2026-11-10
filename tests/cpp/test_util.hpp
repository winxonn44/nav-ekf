// Shared helpers for the GoogleTest suite. No std:: random distributions: they differ between
// libc++ and libstdc++, so draws here come from a small hand-written generator.
#pragma once

#include <Eigen/Core>

#include <cmath>
#include <cstdint>
#include <functional>
#include <vector>

#include "navekf/eskf.hpp"
#include "navekf/imu_model.hpp"
#include "navekf/so3.hpp"
#include "navekf/types.hpp"

namespace navekf::test {

/// SplitMix64 with Box-Muller normals. Same stream on every platform (up to libm round-off).
class Rng {
 public:
  explicit Rng(const std::uint64_t seed) : s_(seed) {}

  std::uint64_t next() {
    s_ += 0x9E3779B97F4A7C15ULL;
    std::uint64_t z = s_;
    z = (z ^ (z >> 30U)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27U)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31U);
  }

  /// Uniform in (0, 1].
  double uniform() { return (static_cast<double>(next() >> 11U) + 1.0) * 0x1.0p-53; }

  double normal() {
    const double u1 = uniform();
    const double u2 = uniform();
    return std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * kPi * u2);
  }

  Vec3 vec3() {
    const double a = normal();
    const double b = normal();
    const double c = normal();
    return Vec3(a, b, c);
  }

  static constexpr double kPi = 3.14159265358979323846;

 private:
  std::uint64_t s_;
};

inline ImuNoise default_noise() {  // configs/sim.yaml
  ImuNoise n;
  n.sigma_a = 2e-3;
  n.sigma_g = 1.7e-4;
  n.sigma_ba = 3e-3;
  n.sigma_bg = 2e-5;
  return n;
}

inline Vec3 lever_arm() { return Vec3(0.3, -0.2, 0.5); }

/// consistency.P0_DEFAULT: p 2 m, v 0.2 m/s, roll/pitch 0.02 rad, yaw 0.05 rad, b_a 0.05, b_g 2e-3.
inline Mat15 default_P0() {
  Vec15 s;
  s << 2, 2, 2, 0.2, 0.2, 0.2, 0.02, 0.02, 0.05, 0.05, 0.05, 0.05, 2e-3, 2e-3, 2e-3;
  return s.cwiseProduct(s).asDiagonal();
}

inline State random_state(const std::uint64_t seed = 1) {
  Rng rng(seed);
  State x;
  x.p = rng.vec3();
  x.v = rng.vec3();
  const double w = rng.normal();
  const Vec3 v = rng.vec3();
  x.q = so3::quat_normalize(Quat(w, v(0), v(1), v(2)));
  x.ba = 0.05 * rng.vec3();
  x.bg = 0.01 * rng.vec3();
  return x;
}

inline ImuSample sample_imu() {
  ImuSample s;
  s.acc = Vec3(0.8, -0.5, 9.6);
  s.gyr = Vec3(0.4, -0.7, 0.9);
  return s;
}

template <class A, class B>
double max_abs(const A& a, const B& b) {
  return (a - b).cwiseAbs().maxCoeff();
}

/// Central differences of fun: R^n -> R^m at 0.
inline Eigen::MatrixXd fd_jacobian(
    const std::function<Eigen::VectorXd(const Eigen::VectorXd&)>& fun, const Eigen::Index n,
    const double eps = 1e-6) {
  Eigen::MatrixXd out;
  for (Eigen::Index j = 0; j < n; ++j) {
    Eigen::VectorXd e = Eigen::VectorXd::Zero(n);
    e(j) = eps;
    const Eigen::VectorXd plus = fun(e);
    const Eigen::VectorXd minus = fun(-e);
    if (j == 0) {
      out.resize(plus.size(), n);
    }
    out.col(j) = (plus - minus) / (2.0 * eps);
  }
  return out;
}

/// A consistent stream (conventions §7): samples invert the §6 integrator between true states.
struct ConsistentRun {
  std::vector<ImuSample> imu;
  std::vector<State> truth;  // position = the §6 recursion p_{k+1} = p_k + (v_k + v_{k+1})/2 · dt
  std::vector<PositionFix> fixes;
};

/// Figure-8 with changing yaw, pitch and roll; noise-free fixes on a 10 Hz schedule.
inline ConsistentRun simulate_consistent(const double duration_s, const Vec3& lever,
                                         const double sigma = 1.0) {
  constexpr std::int64_t kPeriodNs = 5000000;       // 200 Hz
  constexpr std::int64_t kGpsPeriodNs = 100000000;  // 10 Hz
  const double dt = step_dt(0, kPeriodNs);
  const double w = 2.0 * Rng::kPi / 30.0;
  const auto count =
      static_cast<std::size_t>(std::llround(duration_s * 1e9 / static_cast<double>(kPeriodNs))) + 1;

  ConsistentRun out;
  out.truth.resize(count);
  for (std::size_t k = 0; k < count; ++k) {
    const double t = static_cast<double>(k) * dt;
    const double yaw = 0.3 + 0.1 * t + 0.5 * std::sin(0.21 * t);
    const double pitch = 0.1 * std::sin(0.35 * t);
    const double roll = 0.15 * std::sin(0.5 * t);
    State& s = out.truth[k];
    s.v = Vec3(20.0 * w * std::cos(w * t), 20.0 * w * std::cos(2.0 * w * t),
               1.5 * 0.35 * std::cos(0.35 * t));
    s.q = so3::quat_normalize(
        so3::quat_mul(so3::quat_mul(so3::Exp(Vec3(0, 0, yaw)), so3::Exp(Vec3(0, pitch, 0))),
                      so3::Exp(Vec3(roll, 0, 0))));
    if (k > 0) {
      s.p = out.truth[k - 1].p + 0.5 * (out.truth[k - 1].v + s.v) * dt;
    }
  }

  out.imu.resize(count);
  const Vec3 g_W(0, 0, -kGMagnitude);
  for (std::size_t k = 0; k + 1 < count; ++k) {
    const State& a = out.truth[k];
    const State& b = out.truth[k + 1];
    ImuSample& s = out.imu[k];
    s.t_ns = static_cast<std::int64_t>(k) * kPeriodNs;
    s.gyr = so3::Log(so3::quat_mul(so3::quat_conj(a.q), b.q)) / dt;
    const Vec3 a_W = (b.v - a.v) / dt;
    s.acc = so3::quat_to_rot(a.q).transpose() * (a_W - g_W);
  }
  out.imu[count - 1] = out.imu[count - 2];
  out.imu[count - 1].t_ns = static_cast<std::int64_t>(count - 1) * kPeriodNs;

  for (std::size_t k = 0; k < count; ++k) {
    const std::int64_t t_ns = out.imu[k].t_ns;
    if (t_ns % kGpsPeriodNs == 0) {
      PositionFix f;
      f.t_ns = t_ns;
      f.p_W = out.truth[k].p + so3::quat_to_rot(out.truth[k].q) * lever;
      f.sigma = sigma;
      out.fixes.push_back(f);
    }
  }
  return out;
}

}  // namespace navekf::test
