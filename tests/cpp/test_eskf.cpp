// The filter: IMU sign, the noise-free exactness run, the gate, update and hygiene.
#include <Eigen/LU>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include "navekf/eskf.hpp"
#include "navekf/so3.hpp"

#include "test_util.hpp"

namespace navekf {
namespace {

using test::ConsistentRun;
using test::default_noise;
using test::default_P0;
using test::lever_arm;
using test::max_abs;
using test::random_state;
using test::sample_imu;
using test::simulate_consistent;

EskfConfig base_config() {
  EskfConfig c;
  c.noise = default_noise();
  return c;
}

const auto no_hook = [](std::size_t, const Eskf&) {};
const auto no_row = [](const Eskf&, int, const UpdateResult*) {};

// conventions §4: a still, level IMU reads +g on z, and integrates to no motion at all.
TEST(Propagate, StaticImuReadsPlusG) {
  ImuSample still;
  still.acc = Vec3(0.0, 0.0, kGMagnitude);
  State x;
  for (int i = 0; i < 2000; ++i) {
    x = propagate_nominal(x, still, 0.005);
  }
  EXPECT_LT(x.p.cwiseAbs().maxCoeff(), 1e-12);
  EXPECT_LT(x.v.cwiseAbs().maxCoeff(), 1e-12);
  EXPECT_LT(max_abs(x.q, Quat(1, 0, 0, 0)), 1e-12);

  // The wrong sign falls at 2g: v_z = −2 g t.
  ImuSample wrong = still;
  wrong.acc = -still.acc;
  State y;
  y = propagate_nominal(y, wrong, 1.0);
  EXPECT_NEAR(y.v.z(), -2.0 * kGMagnitude, 1e-12);
}

TEST(Propagate, OneStepMatchesConventionsSection6) {
  const State x = random_state();
  const ImuSample imu = sample_imu();
  const double dt = 0.005;
  const Mat3 R = so3::quat_to_rot(x.q);
  const Vec3 a_W = R * (imu.acc - x.ba) + Vec3(0, 0, -kGMagnitude);
  const State n = propagate_nominal(x, imu, dt);
  EXPECT_LT(max_abs(n.p, Vec3(x.p + x.v * dt + 0.5 * a_W * dt * dt)), 1e-14);
  EXPECT_LT(max_abs(n.v, Vec3(x.v + a_W * dt)), 1e-14);
  EXPECT_LT(max_abs(n.q, so3::quat_mul(x.q, so3::Exp((imu.gyr - x.bg) * dt))), 1e-15);
  EXPECT_EQ(n.ba, x.ba);
  EXPECT_EQ(n.bg, x.bg);
}

TEST(Propagate, StepDtIsExactOnTheIntegerDifference) {
  EXPECT_EQ(step_dt(1000000000000000000LL, 1000000000005000000LL), 5000000.0 * 1e-9);
}

// The C++ twin of test_noise_free_filter_tracks_truth_exactly: a consistent stream, no noise,
// updates on with a lever arm. State = truth to 1e-9 and innovations ≈ 0.
TEST(Filter, NoiseFreeTracksTruthExactly) {
  const ConsistentRun r = simulate_consistent(20.0, lever_arm());
  EskfConfig cfg = base_config();
  cfg.lever_arm = lever_arm();
  cfg.check_psd = true;
  Eskf f(r.truth[0], default_P0(), cfg);
  ASSERT_EQ(r.fixes.size(), 201U);

  double max_y = 0.0;
  double max_err = 0.0;
  std::size_t updates = 0;
  std::size_t rows = 0;
  run(
      f, std::span<const ImuSample>(r.imu), std::span<const PositionFix>(r.fixes),
      [&](const std::size_t k, const Eskf& e) {
        max_err = std::max(max_err, boxminus(r.truth[k], e.x()).cwiseAbs().maxCoeff());
      },
      [&](const Eskf&, const int kind, const UpdateResult* u) {
        ++rows;
        if (kind == kRowUpdate) {
          ++updates;
          EXPECT_TRUE(u->accepted);
          max_y = std::max(max_y, u->y.cwiseAbs().maxCoeff());
        } else {
          EXPECT_EQ(u, nullptr);
        }
      });
  EXPECT_EQ(updates, 201U);
  EXPECT_EQ(rows, r.imu.size() - 1 + r.fixes.size());
  EXPECT_LT(max_err, 1e-9);
  EXPECT_LT(max_y, 1e-9);
  EXPECT_EQ(f.rejections(), 0U);
  ASSERT_TRUE(f.min_eig().has_value());
  EXPECT_GT(*f.min_eig(), 0.0);
}

TEST(Filter, QuaternionStaysNormalized) {
  Eskf f(random_state(), default_P0(), base_config());
  const ImuSample imu = sample_imu();
  for (int i = 0; i < 50; ++i) {
    f.propagate(imu, 0.005);
    PositionFix fix;
    fix.t_ns = f.t_ns();
    fix.p_W = f.x().p + Vec3::Constant(0.1);
    (void)f.update_position(fix);
  }
  EXPECT_NEAR(f.x().q.norm(), 1.0, 1e-15);
}

TEST(Filter, TimeAdvancesFromNextStampOrFromDt) {
  Eskf f(random_state(), default_P0(), base_config());
  ImuSample imu = sample_imu();
  imu.t_ns = 1000;
  f.propagate(imu, 0.005);
  EXPECT_EQ(f.t_ns(), 1000 + 5000000);
  f.propagate(imu, 0.005, 123456789);
  EXPECT_EQ(f.t_ns(), 123456789);
}

// -------------------------------------------------------------------------- gate

TEST(Gate, ConstantIsExact) { EXPECT_EQ(kGate99_3dof, 11.344866730144373); }

TEST(Gate, RejectsA20mOutlierAndLeavesTheFilterBitIdentical) {
  ConsistentRun r = simulate_consistent(10.0, Vec3::Zero());
  constexpr std::size_t kBad = 80;
  r.fixes[kBad].p_W.x() += 20.0;
  EskfConfig cfg = base_config();
  Eskf f(r.truth[0], default_P0(), cfg);

  std::vector<UpdateResult> ups;
  State before;
  Mat15 P_before = Mat15::Zero();
  std::size_t fix_index = 0;
  run(f, std::span<const ImuSample>(r.imu), std::span<const PositionFix>(r.fixes), no_hook,
      [&](const Eskf& e, const int kind, const UpdateResult* u) {
        if (kind == kRowImu) {
          before = e.x();
          P_before = e.P();
          return;
        }
        ups.push_back(*u);
        if (fix_index == kBad) {
          EXPECT_EQ(e.x().p, before.p);
          EXPECT_EQ(e.x().v, before.v);
          EXPECT_EQ(e.x().q, before.q);
          EXPECT_EQ(e.x().ba, before.ba);
          EXPECT_EQ(e.x().bg, before.bg);
          EXPECT_EQ(e.P(), P_before);
          EXPECT_EQ(e.dx(), Vec15(Vec15::Zero()));
        }
        ++fix_index;
      });

  ASSERT_EQ(ups.size(), r.fixes.size());
  EXPECT_FALSE(ups[kBad].accepted);
  EXPECT_GT(ups[kBad].nis, 100.0);
  EXPECT_EQ(f.rejections(), 1U);
  for (std::size_t i = 0; i < ups.size(); ++i) {
    if (i != kBad) {
      EXPECT_TRUE(ups[i].accepted) << "fix " << i;
    }
  }
}

TEST(Gate, OffAcceptsTheOutlier) {
  const State x = random_state();
  EskfConfig cfg = base_config();
  cfg.gate = std::nullopt;
  Eskf f(x, default_P0(), cfg);
  PositionFix fix;
  fix.p_W = x.p + Vec3::Constant(20.0);
  const UpdateResult u = f.update_position(fix);
  EXPECT_TRUE(u.accepted);
  EXPECT_EQ(f.rejections(), 0U);
}

// The comparison is NIS ≤ gate, as in Python: a NIS exactly at the gate passes.
TEST(Gate, BoundaryIsInclusive) {
  const State x = random_state();
  PositionFix fix;
  fix.p_W = x.p + Vec3(3.0, -2.0, 1.0);

  EskfConfig cfg = base_config();
  cfg.gate = std::nullopt;
  Eskf probe(x, default_P0(), cfg);
  const double nis = probe.update_position(fix).nis;

  cfg.gate = nis;
  Eskf at(x, default_P0(), cfg);
  EXPECT_TRUE(at.update_position(fix).accepted);

  cfg.gate = std::nextafter(nis, 0.0);
  Eskf below(x, default_P0(), cfg);
  EXPECT_FALSE(below.update_position(fix).accepted);
}

TEST(Gate, NanNisIsRejected) {
  const State x = random_state();
  Eskf f(x, default_P0(), base_config());
  PositionFix fix;
  fix.p_W = Vec3::Constant(std::numeric_limits<double>::quiet_NaN());
  EXPECT_FALSE(f.update_position(fix).accepted);
  EXPECT_EQ(f.x().p, x.p);
}

// The known lock-out, ported on purpose (E6 fixes it): once the estimate is far outside the
// gate, every later fix is rejected, so the covariance never gets a chance to pull it back.
TEST(Gate, PostOutageLockOutIsPortedAsIs) {
  ConsistentRun r = simulate_consistent(10.0, Vec3::Zero());
  for (PositionFix& fix : r.fixes) {
    if (fix.t_ns >= 5000000000) {
      fix.p_W.x() += 30.0;
    }
  }
  EskfConfig cfg = base_config();
  Eskf f(r.truth[0], default_P0(), cfg);
  std::size_t accepted_after = 0;
  std::size_t rejected_after = 0;
  run(f, std::span<const ImuSample>(r.imu), std::span<const PositionFix>(r.fixes), no_hook,
      [&](const Eskf&, const int kind, const UpdateResult* u) {
        if (kind == kRowUpdate && u->t_ns >= 5000000000) {
          (u->accepted ? accepted_after : rejected_after) += 1;
        }
      });
  EXPECT_EQ(accepted_after, 0U);
  EXPECT_EQ(rejected_after, 51U);
}

// ------------------------------------------------------------------------ update

// One update by hand (inverse-based, standard form) vs the code (solve, Joseph); identity reset.
TEST(Update, JosephFormMatchesTheStandardFormAndStaysSymmetric) {
  const State x = random_state();
  test::Rng rng(3);
  Mat15 A;
  for (int r = 0; r < 15; ++r) {
    for (int c = 0; c < 15; ++c) {
      A(r, c) = rng.normal();
    }
  }
  const Mat15 P = A * A.transpose() * 0.01 + Mat15::Identity() * 1e-3;

  EskfConfig cfg = base_config();
  cfg.lever_arm = lever_arm();
  cfg.reset = Reset::kIdentity;
  cfg.gate = std::nullopt;
  Eskf f(x, P, cfg);
  PositionFix fix;
  fix.p_W = x.p + so3::quat_to_rot(x.q) * lever_arm() + Vec3(0.3, -0.1, 0.2);
  fix.sigma = 0.5;
  const UpdateResult u = f.update_position(fix, /*inject=*/false);

  const Mat3x15 H = position_model(x, lever_arm()).H;
  const Mat3 S = H * P * H.transpose() + 0.25 * Mat3::Identity();
  const Mat15x3 K = P * H.transpose() * S.inverse();
  EXPECT_LT(max_abs(u.S, S), 1e-12);
  EXPECT_LT(max_abs(f.dx(), Vec15(K * u.y)), 1e-9);
  EXPECT_LT(max_abs(f.P(), Mat15((Mat15::Identity() - K * H) * P)), 1e-8);
  EXPECT_EQ(f.P(), Mat15(f.P().transpose()));
  EXPECT_NEAR(u.nis, u.y.dot(S.inverse() * u.y), 1e-11 * u.nis);
  EXPECT_EQ(f.x().p, x.p);  // inject = false leaves the nominal state alone
}

TEST(Reset, InjectionFollowsBoxplusAndClearsTheError) {
  const State x = random_state();
  EskfConfig cfg = base_config();
  cfg.gate = std::nullopt;
  Eskf f(x, default_P0(), cfg);
  PositionFix fix;
  fix.p_W = x.p + Vec3(1.0, 0.0, 0.0);
  (void)f.update_position(fix, /*inject=*/false);
  const Vec15 dx = f.dx();
  ASSERT_GT(dx.cwiseAbs().maxCoeff(), 0.0);
  f.inject_and_reset();
  EXPECT_LT(max_abs(boxminus(f.x(), boxplus(x, dx)), Vec15(Vec15::Zero())), 1e-15);
  EXPECT_EQ(f.dx(), Vec15(Vec15::Zero()));
}

TEST(Reset, IdentityLeavesPBitExact) {
  const State x = random_state();
  EskfConfig cfg = base_config();
  cfg.gate = std::nullopt;
  cfg.reset = Reset::kIdentity;
  Eskf f(x, default_P0(), cfg);
  PositionFix fix;
  fix.p_W = x.p + Vec3(1.0, 0.5, 0.0);
  (void)f.update_position(fix, /*inject=*/false);
  const Mat15 P = f.P();
  f.inject_and_reset();
  EXPECT_EQ(f.P(), P);
}

// ----------------------------------------------------------------------- hygiene

TEST(Hygiene, CheckPsdThrowsOnABrokenP) {
  Mat15 P = default_P0();
  P(0, 0) = -1.0;
  EskfConfig cfg = base_config();
  cfg.check_psd = true;
  EXPECT_THROW(Eskf(random_state(), P, cfg), std::runtime_error);
}

TEST(Run, FixOffTheImuGridThrows) {
  const ConsistentRun r = simulate_consistent(1.0, Vec3::Zero());
  Eskf f(r.truth[0], default_P0(), base_config());
  std::vector<PositionFix> fixes(1);
  fixes[0].t_ns = 12345;
  EXPECT_THROW(run(f, std::span<const ImuSample>(r.imu), std::span<const PositionFix>(fixes),
                   no_hook, no_row),
               std::invalid_argument);
}

TEST(Run, UnsortedFixesThrow) {
  const ConsistentRun r = simulate_consistent(1.0, Vec3::Zero());
  Eskf f(r.truth[0], default_P0(), base_config());
  std::vector<PositionFix> fixes(2);
  fixes[0].t_ns = 10000000;
  fixes[1].t_ns = 5000000;
  EXPECT_THROW(run(f, std::span<const ImuSample>(r.imu), std::span<const PositionFix>(fixes),
                   no_hook, no_row),
               std::invalid_argument);
}

TEST(Run, FixesBeforeTheFirstSampleAreSkipped) {
  ConsistentRun r = simulate_consistent(1.0, Vec3::Zero());
  std::vector<PositionFix> fixes(1);
  fixes[0].t_ns = r.imu[0].t_ns - 1;
  Eskf f(r.truth[0], default_P0(), base_config());
  std::size_t updates = 0;
  run(f, std::span<const ImuSample>(r.imu), std::span<const PositionFix>(fixes), no_hook,
      [&](const Eskf&, const int kind, const UpdateResult*) { updates += kind == kRowUpdate; });
  EXPECT_EQ(updates, 0U);
}

}  // namespace
}  // namespace navekf
