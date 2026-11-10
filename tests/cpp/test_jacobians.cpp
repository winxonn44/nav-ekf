// F, G, H, reset Jacobians and the discretizations, against finite differences of the real code.
#include <gtest/gtest.h>
#include <unsupported/Eigen/MatrixFunctions>

#include <array>
#include <cmath>
#include <string>

#include "navekf/eskf.hpp"
#include "navekf/imu_model.hpp"
#include "navekf/so3.hpp"

#include "test_util.hpp"

namespace navekf {
namespace {

using test::default_noise;
using test::fd_jacobian;
using test::lever_arm;
using test::max_abs;
using test::random_state;
using test::sample_imu;

Mat15 gqg(const Mat15x12& G) { return G * default_noise().Qc() * G.transpose(); }

Eigen::MatrixXd discrete_error_jacobian(const State& x, const ImuSample& imu, const double dt) {
  const State x1 = propagate_nominal(x, imu, dt);
  return fd_jacobian(
      [&](const Eigen::VectorXd& e) -> Eigen::VectorXd {
        const Vec15 ev = e;
        return boxminus(propagate_nominal(boxplus(x, ev), imu, dt), x1);
      },
      15);
}

TEST(Jacobians, BoxplusBoxminusRoundTrip) {
  const State x = random_state();
  test::Rng rng(2);
  Vec15 dx;
  for (int i = 0; i < 15; ++i) {
    dx(i) = 0.3 * rng.normal();
  }
  EXPECT_LT(max_abs(boxminus(boxplus(x, dx), x), dx), 1e-13);
}

TEST(Jacobians, FStructureAndValues) {
  const State x = random_state();
  const ImuSample imu = sample_imu();
  const Jacobians J = jacobians(x, imu);
  const Mat3 R = so3::quat_to_rot(x.q);
  Mat15 expected = Mat15::Zero();
  expected.block<3, 3>(kIP, kIV) = Mat3::Identity();
  expected.block<3, 3>(kIV, kITh) = -R * so3::skew(imu.acc - x.ba);
  expected.block<3, 3>(kIV, kIBa) = -R;
  expected.block<3, 3>(kITh, kITh) = -so3::skew(imu.gyr - x.bg);
  expected.block<3, 3>(kITh, kIBg) = -Mat3::Identity();
  EXPECT_EQ(J.F, expected);
  const Mat15 Q = gqg(J.G);
  EXPECT_LT(max_abs(Mat3(Q.block<3, 3>(kIV, kIV)),
                    default_noise().sigma_a * default_noise().sigma_a * Mat3::Identity()),
            1e-20);  // R σ² I Rᵀ = σ² I
}

// (J(dt) − I)/dt → F at first order in dt; ‖Φ₁ − J‖ = O(dt²).
TEST(Jacobians, FIsTheLimitOfTheDiscreteStepJacobian) {
  const State x = random_state();
  const ImuSample imu = sample_imu();
  const Jacobians J = jacobians(x, imu);
  std::array<double, 3> err_f{};
  std::array<double, 3> err_phi{};
  const std::array<double, 3> dts{2e-3, 1e-3, 5e-4};
  for (std::size_t i = 0; i < dts.size(); ++i) {
    const double dt = dts.at(i);
    const Eigen::MatrixXd Jd = discrete_error_jacobian(x, imu, dt);
    const Discrete d = discretize(J.F, gqg(J.G), dt, Discretization::kFirst);
    err_f.at(i) = ((Jd - Mat15::Identity()) / dt - J.F).cwiseAbs().maxCoeff();
    err_phi.at(i) = (d.Phi - Jd).cwiseAbs().maxCoeff();
  }
  EXPECT_LT(err_f[2], 0.01 * J.F.cwiseAbs().maxCoeff());
  EXPECT_NEAR(err_f[0] / err_f[1], 2.0, 0.1);
  EXPECT_NEAR(err_f[1] / err_f[2], 2.0, 0.1);
  EXPECT_NEAR(err_phi[0] / err_phi[1], 4.0, 0.2);
  EXPECT_NEAR(err_phi[1] / err_phi[2], 4.0, 0.2);
}

// ∂δx_{k+1}/∂(n_a, n_g) = G[:, :6]·dt + O(dt²): perturb the readings by −n (true = measured − n).
TEST(Jacobians, GMatchesFiniteDifferencesOfTheNoiseInputs) {
  const State x = random_state();
  const ImuSample imu = sample_imu();
  const double dt = 1e-3;
  const Jacobians J = jacobians(x, imu);
  const State x1 = propagate_nominal(x, imu, dt);
  const Eigen::MatrixXd Jn = fd_jacobian(
      [&](const Eigen::VectorXd& n) -> Eigen::VectorXd {
        ImuSample perturbed = imu;
        perturbed.acc -= n.head<3>();
        perturbed.gyr -= n.tail<3>();
        return boxminus(propagate_nominal(x, perturbed, dt), x1);
      },
      6);
  const Eigen::MatrixXd G6 = J.G.leftCols<6>();
  EXPECT_LT((Eigen::MatrixXd(Jn / dt) - G6).cwiseAbs().maxCoeff(), 5.0 * dt);
}

TEST(Jacobians, HMatchesFiniteDifferencesWithLeverArm) {
  const State x = random_state();
  const PositionModel m = position_model(x, lever_arm());
  const Eigen::MatrixXd fd = fd_jacobian(
      [&](const Eigen::VectorXd& e) -> Eigen::VectorXd {
        const Vec15 ev = e;
        return position_model(boxplus(x, ev), lever_arm()).z_hat;
      },
      15);
  EXPECT_LT((m.H - fd).cwiseAbs().maxCoeff(), 1e-9);
  const Mat3 lever_block = m.H.block<3, 3>(0, kITh);
  EXPECT_GT(lever_block.cwiseAbs().maxCoeff(), 0.1);  // the lever-arm block is exercised
  EXPECT_LT(max_abs(m.z_hat, Vec3(x.p + so3::quat_to_rot(x.q) * lever_arm())), 1e-15);
}

class ResetJacobian : public ::testing::TestWithParam<Vec3> {};

// New error after injecting δx̂: ε ↦ (x̂ ⊞ (δx̂ + ε)) ⊖ (x̂ ⊞ δx̂); its Jacobian is G_r.
TEST_P(ResetJacobian, MatchesFiniteDifferences) {
  const State x = random_state();
  Vec15 dx_hat;
  dx_hat << 0.1, 0.2, -0.3, 0.01, 0.0, 0.02, GetParam(), 1e-3, 1e-3, 1e-3, -1e-4, -1e-4, -1e-4;
  const State x_plus = boxplus(x, dx_hat);
  const Eigen::MatrixXd fd = fd_jacobian(
      [&](const Eigen::VectorXd& e) -> Eigen::VectorXd {
        const Vec15 ev = e;
        return boxminus(boxplus(x, Vec15(dx_hat + ev)), x_plus);
      },
      15);
  const Mat15 Gr = reset_jacobian(GetParam(), Reset::kJr);
  EXPECT_LT((Gr - fd).cwiseAbs().maxCoeff(), 1e-9);
  EXPECT_EQ(reset_jacobian(GetParam(), Reset::kIdentity), Mat15(Mat15::Identity()));
}

INSTANTIATE_TEST_SUITE_P(Angles, ResetJacobian,
                         ::testing::Values(Vec3(0.1, -0.2, 0.15), Vec3(1e-3, 2e-3, -1e-3),
                                           Vec3(0.0, 0.0, 0.0)),
                         [](const auto& param) { return std::to_string(param.index); });

struct OrderCase {
  Discretization method;
  double order;
};

class DiscretizationOrder : public ::testing::TestWithParam<OrderCase> {};

// ‖Φ − expm(F dt)‖ and ‖Q_d − Q_d^VanLoan‖ shrink as dt^order.
TEST_P(DiscretizationOrder, ErrorShrinksAtTheRightRate) {
  const Jacobians J = jacobians(random_state(), sample_imu());
  const Mat15 Q = gqg(J.G);
  std::array<double, 3> e_phi{};
  std::array<double, 3> e_q{};
  const std::array<double, 3> dts{4e-2, 2e-2, 1e-2};
  for (std::size_t i = 0; i < dts.size(); ++i) {
    const Discrete d = discretize(J.F, Q, dts.at(i), GetParam().method);
    const Discrete ref = discretize(J.F, Q, dts.at(i), Discretization::kVanLoan);
    e_phi.at(i) = max_abs(d.Phi, ref.Phi);
    e_q.at(i) = max_abs(d.Qd, ref.Qd);
  }
  const double expected = std::pow(2.0, GetParam().order);
  EXPECT_NEAR(e_phi[0] / e_phi[1], expected, 0.15 * expected);
  EXPECT_NEAR(e_phi[1] / e_phi[2], expected, 0.1 * expected);
  EXPECT_NEAR(e_q[1] / e_q[2], expected, 0.15 * expected);
}

INSTANTIATE_TEST_SUITE_P(Methods, DiscretizationOrder,
                         ::testing::Values(OrderCase{Discretization::kFirst, 2.0},
                                           OrderCase{Discretization::kSecond, 3.0}),
                         [](const auto& param) { return std::to_string(param.index); });

TEST(Discretization, VanLoanMatchesExpmAndQuadrature) {
  const Jacobians J = jacobians(random_state(), sample_imu());
  const Mat15 Q = gqg(J.G);
  const double dt = 5e-3;
  const Discrete d = discretize(J.F, Q, dt, Discretization::kVanLoan);
  const Mat15 Fdt = J.F * dt;
  const Mat15 expm = Fdt.exp();
  EXPECT_LT(max_abs(d.Phi, expm), 1e-13);

  // Simpson's rule on ∫ Φ(τ) GQG Φ(τ)ᵀ dτ.
  Mat15 integral = Mat15::Zero();
  constexpr int kPanels = 200;
  for (int i = 0; i <= kPanels; ++i) {
    const Mat15 Ft = J.F * (dt * i / kPanels);
    const Mat15 phi = Ft.exp();
    const double w = (i == 0 || i == kPanels) ? 1.0 : ((i % 2 == 1) ? 4.0 : 2.0);
    integral += w * phi * Q * phi.transpose();
  }
  integral *= dt / (3.0 * kPanels);
  EXPECT_LT(max_abs(d.Qd, integral), 1e-9 * Q.cwiseAbs().maxCoeff() * dt + 1e-22);
  EXPECT_LT(max_abs(d.Qd, Mat15(d.Qd.transpose())), 1e-22);
}

TEST(ImuModel, QcLayout) {
  const ImuNoise n = default_noise();
  const Mat12 Q = n.Qc();
  EXPECT_EQ(Q(0, 0), n.sigma_a * n.sigma_a);
  EXPECT_EQ(Q(5, 5), n.sigma_g * n.sigma_g);
  EXPECT_EQ(Q(8, 8), n.sigma_ba * n.sigma_ba);
  EXPECT_EQ(Q(11, 11), n.sigma_bg * n.sigma_bg);
  EXPECT_EQ(Q.sum(), Q.trace());
}

}  // namespace
}  // namespace navekf
