#include <Eigen/Geometry>
#include <gtest/gtest.h>

#include <cmath>

#include "navekf/so3.hpp"

#include "test_util.hpp"

namespace navekf {
namespace {

using test::max_abs;
using test::Rng;

Vec3 random_rotvec(Rng& rng, const double max_norm) {
  Vec3 phi = rng.vec3();
  phi *= (max_norm * rng.uniform()) / phi.norm();
  return phi;
}

// Conventions §3: Eigen's constructor takes (w, x, y, z) but it stores [x, y, z, w].
TEST(So3, test_eigen_quaternion_order) {
  const Eigen::Quaterniond e(1.0, 2.0, 3.0, 4.0);
  EXPECT_EQ(e.w(), 1.0);
  EXPECT_EQ(e.x(), 2.0);
  EXPECT_EQ(e.coeffs()(0), 2.0);  // x first...
  EXPECT_EQ(e.coeffs()(3), 1.0);  // ...w last: the trap

  const Quat q(1.0, 2.0, 3.0, 4.0);
  const Eigen::Quaterniond converted = so3::to_eigen(q);
  EXPECT_EQ(converted.w(), 1.0);
  EXPECT_EQ(converted.z(), 4.0);
  EXPECT_EQ(so3::from_eigen(converted), q);
}

TEST(So3, SkewIsCrossProduct) {
  Rng rng(1);
  const Vec3 a = rng.vec3();
  const Vec3 b = rng.vec3();
  EXPECT_LT(max_abs(so3::skew(a) * b, a.cross(b)), 1e-15);
  EXPECT_LT(max_abs(so3::skew(a).transpose(), -so3::skew(a)), 0.0 + 1e-300);
}

TEST(So3, ProductAndRotationMatchEigen) {
  Rng rng(2);
  for (int i = 0; i < 100; ++i) {
    const Quat a = so3::Exp(random_rotvec(rng, 3.0));
    const Quat b = so3::Exp(random_rotvec(rng, 3.0));
    const Eigen::Quaterniond ea = so3::to_eigen(a);
    const Eigen::Quaterniond eb = so3::to_eigen(b);
    EXPECT_LT(max_abs(so3::quat_mul(a, b), so3::from_eigen(ea * eb)), 1e-15);
    EXPECT_LT(max_abs(so3::quat_to_rot(a), ea.toRotationMatrix()), 1e-15);
    EXPECT_LT(
        max_abs(so3::quat_to_rot(so3::quat_mul(a, b)), so3::quat_to_rot(a) * so3::quat_to_rot(b)),
        1e-14);
  }
}

TEST(So3, ExpMatchesRodrigues) {
  Rng rng(3);
  for (int i = 0; i < 100; ++i) {
    const Vec3 phi = random_rotvec(rng, 3.1);
    const double th = phi.norm();
    const Mat3 u = so3::skew(phi / th);
    const Mat3 rodrigues = Mat3::Identity() + std::sin(th) * u + (1.0 - std::cos(th)) * u * u;
    EXPECT_LT(max_abs(so3::exp_rot(phi), rodrigues), 1e-14);
  }
}

TEST(So3, LogExpRoundTrip) {
  Rng rng(4);
  for (int i = 0; i < 1000; ++i) {
    const Vec3 phi = random_rotvec(rng, 3.14);
    EXPECT_LT(max_abs(so3::Log(so3::Exp(phi)), phi), 1e-13);
  }
  for (const double norm : {1e-12, 1e-9, 0.99e-8, 1.01e-8, 1e-6}) {
    const Vec3 phi = norm * Vec3(1.0, -2.0, 2.0) / 3.0;
    EXPECT_LT(max_abs(so3::Log(so3::Exp(phi)), phi), 1e-20) << "norm " << norm;
  }
  EXPECT_EQ(so3::Log(so3::Exp(Vec3::Zero())), Vec3::Zero());
}

TEST(So3, ExpLogRoundTripUpToSign) {
  Rng rng(5);
  for (int i = 0; i < 100; ++i) {
    const Quat q =
        so3::quat_normalize(Quat(rng.normal(), rng.normal(), rng.normal(), rng.normal()));
    EXPECT_LT(max_abs(so3::Exp(so3::Log(q)), so3::quat_canonical(q)), 1e-14);
  }
}

TEST(So3, LogIsTheShortRotation) {
  const Vec3 phi(0.1, -0.2, 0.3);
  const Quat q = so3::Exp(phi);
  const Vec3 flipped = so3::Log(Quat(-q));  // q and −q are the same rotation
  EXPECT_LT(max_abs(flipped, phi), 1e-15);
}

TEST(So3, SmallAngleBranchesAreContinuous) {
  const Vec3 dir = Vec3(1.0, 2.0, -2.0) / 3.0;
  EXPECT_LT(max_abs(so3::Exp(0.999999e-8 * dir), so3::Exp(1.000001e-8 * dir)), 1e-13);
  // J_r jumps by up to ½θ here, as it does in Python: just above the threshold 1 − cos θ is
  // exactly 0 in double, so the regular branch loses its −½[φ]× term. Same branches on purpose.
  EXPECT_LT(max_abs(so3::right_jacobian(0.999999e-8 * dir), so3::right_jacobian(1.000001e-8 * dir)),
            1e-8);
  EXPECT_NEAR(so3::Exp(1e-12 * dir).norm(), 1.0, 1e-15);
}

TEST(So3, RightJacobianMatchesFiniteDifferences) {
  Rng rng(6);
  for (int i = 0; i < 20; ++i) {
    const Vec3 phi = random_rotvec(rng, 2.5);
    const Quat base = so3::Exp(phi);
    const Eigen::MatrixXd fd = test::fd_jacobian(
        [&](const Eigen::VectorXd& e) -> Eigen::VectorXd {
          const Vec3 eps = e;
          return so3::Log(so3::quat_mul(so3::quat_conj(base), so3::Exp(phi + eps)));
        },
        3);
    EXPECT_LT(max_abs(so3::right_jacobian(phi), fd), 1e-8);
  }
}

TEST(So3, RightJacobianSmallAngleForm) {
  const Vec3 phi(1e-10, -2e-10, 0.0);
  EXPECT_EQ(so3::right_jacobian(phi), Mat3(Mat3::Identity() - 0.5 * so3::skew(phi)));
}

}  // namespace
}  // namespace navekf
