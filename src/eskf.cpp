#include "navekf/eskf.hpp"

#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>
#include <Eigen/LU>

#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <utility>

#include "navekf/so3.hpp"

namespace navekf {

namespace {

template <int N>
Eigen::Matrix<double, N, N> sym(const Eigen::Matrix<double, N, N>& a) {
  return 0.5 * (a + a.transpose());
}

}  // namespace

State propagate_nominal(const State& x, const ImuSample& imu, const double dt, const double g) {
  const Vec3 a_hat = imu.acc - x.ba;
  const Vec3 w_hat = imu.gyr - x.bg;
  const Mat3 R = so3::quat_to_rot(x.q);
  // Column by column, as Python's _matvec, so the operation order is fixed.
  const Vec3 a_W = R.col(0) * a_hat(0) + R.col(1) * a_hat(1) + R.col(2) * a_hat(2) + Vec3(0, 0, -g);
  State out;
  out.p = x.p + x.v * dt + 0.5 * a_W * dt * dt;
  out.v = x.v + a_W * dt;
  out.q = so3::quat_normalize(so3::quat_mul(x.q, so3::Exp(w_hat * dt)));
  out.ba = x.ba;
  out.bg = x.bg;
  return out;
}

State boxplus(const State& x, const ErrorState& dx) {
  State out;
  out.p = x.p + dx.segment<3>(kIP);
  out.v = x.v + dx.segment<3>(kIV);
  out.q = so3::quat_normalize(so3::quat_mul(x.q, so3::Exp(dx.segment<3>(kITh))));
  out.ba = x.ba + dx.segment<3>(kIBa);
  out.bg = x.bg + dx.segment<3>(kIBg);
  return out;
}

ErrorState boxminus(const State& x, const State& x_hat) {
  ErrorState out;
  out.segment<3>(kIP) = x.p - x_hat.p;
  out.segment<3>(kIV) = x.v - x_hat.v;
  out.segment<3>(kITh) = so3::Log(so3::quat_mul(so3::quat_conj(x_hat.q), x.q));
  out.segment<3>(kIBa) = x.ba - x_hat.ba;
  out.segment<3>(kIBg) = x.bg - x_hat.bg;
  return out;
}

Jacobians jacobians(const State& x, const ImuSample& imu) {
  const Vec3 a_hat = imu.acc - x.ba;
  const Vec3 w_hat = imu.gyr - x.bg;
  const Mat3 R = so3::quat_to_rot(x.q);
  const Mat3 I3 = Mat3::Identity();
  Jacobians out;
  out.F.setZero();
  out.F.block<3, 3>(kIP, kIV) = I3;
  out.F.block<3, 3>(kIV, kITh) = -R * so3::skew(a_hat);
  out.F.block<3, 3>(kIV, kIBa) = -R;
  out.F.block<3, 3>(kITh, kITh) = -so3::skew(w_hat);
  out.F.block<3, 3>(kITh, kIBg) = -I3;
  out.G.setZero();
  out.G.block<3, 3>(kIV, 0) = -R;
  out.G.block<3, 3>(kITh, 3) = -I3;
  out.G.block<3, 3>(kIBa, 6) = I3;
  out.G.block<3, 3>(kIBg, 9) = I3;
  return out;
}

PositionModel position_model(const State& x, const Vec3& lever_arm) {
  const Mat3 R = so3::quat_to_rot(x.q);
  PositionModel out;
  out.z_hat = x.p + R * lever_arm;
  out.H.setZero();
  out.H.block<3, 3>(0, kIP) = Mat3::Identity();
  out.H.block<3, 3>(0, kITh) = -R * so3::skew(lever_arm);
  return out;
}

Mat15 reset_jacobian(const Vec3& dth, const Reset method) {
  Mat15 gr = Mat15::Identity();
  if (method == Reset::kJr) {
    gr.block<3, 3>(kITh, kITh) = so3::right_jacobian(dth);
  }
  return gr;
}

Eskf::Eskf(State x0, const Mat15& P0, const EskfConfig& cfg, const std::int64_t t0_ns)
    : cfg_(cfg), x_(std::move(x0)), P_(sym<kNErr>(P0)), Qc_(cfg.noise.Qc()), t_ns_(t0_ns) {
  x_.q = so3::quat_normalize(x_.q);
  check();
}

void Eskf::propagate(const ImuSample& imu, const double dt,
                     const std::optional<std::int64_t> t_next_ns) {
  const Jacobians J = jacobians(x_, imu);
  x_ = propagate_nominal(x_, imu, dt, cfg_.g);
  const Mat15 GQG = J.G * Qc_ * J.G.transpose();
  const Discrete d = discretize(J.F, GQG, dt, cfg_.discretization);
  const Mat15 P_pred = d.Phi * P_ * d.Phi.transpose() + d.Qd;
  P_ = sym<kNErr>(P_pred);
  // nearbyint is round-half-to-even in the default mode, like Python's round().
  t_ns_ = t_next_ns.has_value() ? *t_next_ns
                                : imu.t_ns + static_cast<std::int64_t>(std::nearbyint(dt * 1e9));
  check();
}

UpdateResult Eskf::update_position(const PositionFix& fix, const bool inject) {
  const Mat3 R_meas = fix.sigma * fix.sigma * Mat3::Identity();
  const PositionModel m = position_model(x_, cfg_.lever_arm);
  const Vec3 y = fix.p_W - m.z_hat;
  const Mat15x3 PHt = P_ * m.H.transpose();
  const Mat3 S_raw = m.H * PHt + R_meas;
  const Mat3 S = sym<3>(S_raw);
  const Eigen::PartialPivLU<Mat3> lu(S);  // LU with partial pivoting, as np.linalg.solve
  const Vec3 Sinv_y = lu.solve(y);
  const double nis = y(0) * Sinv_y(0) + y(1) * Sinv_y(1) + y(2) * Sinv_y(2);
  // Written as Python writes it, so a NaN NIS is rejected.
  const bool accepted = !cfg_.gate.has_value() || nis <= *cfg_.gate;

  t_ns_ = fix.t_ns;
  dx_.setZero();
  if (accepted) {
    const Mat15x3 K = lu.solve(PHt.transpose()).transpose();  // P Hᵀ S⁻¹ (S, P symmetric)
    dx_ = K * y;
    const Mat15 IKH = Mat15::Identity() - K * m.H;
    const Mat15 P_joseph = IKH * P_ * IKH.transpose() + K * R_meas * K.transpose();
    P_ = sym<kNErr>(P_joseph);
    if (inject) {
      inject_and_reset();
    }
  } else {
    ++rejections_;
  }
  check();
  UpdateResult out;
  out.t_ns = fix.t_ns;
  out.y = y;
  out.S = S;
  out.nis = nis;
  out.accepted = accepted;
  return out;
}

void Eskf::inject_and_reset() {
  const Vec3 dth = dx_.segment<3>(kITh);
  x_ = boxplus(x_, dx_);
  if (cfg_.reset == Reset::kJr) {  // kIdentity skips the product: bit-exact "no reset"
    const Mat15 Gr = reset_jacobian(dth, cfg_.reset);
    const Mat15 P_reset = Gr * P_ * Gr.transpose();
    P_ = sym<kNErr>(P_reset);
  }
  dx_.setZero();
}

void Eskf::check() {
  if (!cfg_.check_psd) {
    return;
  }
  const Eigen::SelfAdjointEigenSolver<Mat15> es(P_, Eigen::EigenvaluesOnly);
  const double lo = es.eigenvalues()(0);
  const double hi = es.eigenvalues()(kNErr - 1);
  min_eig_ = lo;
  if (lo < -cfg_.psd_tol * hi) {
    throw std::runtime_error("P lost positive semi-definiteness at t_ns=" + std::to_string(t_ns_) +
                             ": min eig " + std::to_string(lo));
  }
}

}  // namespace navekf
