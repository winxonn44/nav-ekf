#include <gtest/gtest.h>

#include <array>
#include <cstdio>
#include <stdexcept>
#include <string>

#include "navekf/config.hpp"

namespace navekf {
namespace {

const char* const kInitial = R"(
initial:
  p: [1.0, 2.0, 3.0]
  v: [0.0, 0.0, 0.0]
  q_wxyz: [1.0, 0.0, 0.0, 0.0]
  ba: [0.0, 0.0, 0.0]
  bg: [0.0, 0.0, 0.0]
  P0_diag: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
)";

TEST(Config, DefaultsMatchPython) {
  const RunConfig c = parse_config(kInitial);
  EXPECT_EQ(c.eskf.g, 9.81);
  EXPECT_EQ(c.eskf.discretization, Discretization::kFirst);
  EXPECT_EQ(c.eskf.reset, Reset::kJr);
  ASSERT_TRUE(c.eskf.gate.has_value());
  EXPECT_EQ(c.eskf.gate.value_or(-1.0), kGate99_3dof);
  EXPECT_FALSE(c.eskf.check_psd);
  EXPECT_EQ(c.x0.p, Vec3(1.0, 2.0, 3.0));
  EXPECT_EQ(c.P0(14, 14), 15.0);
  EXPECT_EQ(c.P0(0, 1), 0.0);
  EXPECT_FALSE(c.t0_ns.has_value());
}

TEST(Config, ReadsEveryFilterOption) {
  const std::string text = std::string(R"(
g: 9.80665
imu_noise: {sigma_a: 0.1, sigma_g: 0.2, sigma_ba: 0.3, sigma_bg: 0.4}
gps: {rate_hz: 10.0, lever_arm: [0.3, -0.2, 0.5]}
trajectory: {period_s: 30.0}
filter: {discretization: vanloan, reset: identity, gate: null, check_psd: true, psd_tol: 1.0e-9}
)") + kInitial + "  t0_ns: 1403636579758555392\n";
  const RunConfig c = parse_config(text);
  EXPECT_EQ(c.eskf.g, 9.80665);
  EXPECT_EQ(c.eskf.noise.sigma_bg, 0.4);
  EXPECT_EQ(c.eskf.lever_arm, Vec3(0.3, -0.2, 0.5));
  EXPECT_EQ(c.eskf.discretization, Discretization::kVanLoan);
  EXPECT_EQ(c.eskf.reset, Reset::kIdentity);
  EXPECT_FALSE(c.eskf.gate.has_value());
  EXPECT_TRUE(c.eskf.check_psd);
  EXPECT_EQ(c.eskf.psd_tol, 1.0e-9);
  EXPECT_EQ(c.t0_ns, 1403636579758555392LL);  // int64 ns survive exactly
}

TEST(Config, FullPrecisionNumbersRoundTripBitForBit) {
  for (const double value : {0.1, 1.0 / 3.0, 2.7182818284590451e-5, -1.2345678901234567e+200,
                             4.9406564584124654e-324, 11.344866730144373}) {
    std::array<char, 64> buf{};
    (void)std::snprintf(buf.data(), buf.size(), "%.17g", value);
    const RunConfig c =
        parse_config(std::string("filter: {gate: ") + buf.data() + "}\n" + kInitial);
    ASSERT_TRUE(c.eskf.gate.has_value());
    EXPECT_EQ(c.eskf.gate.value_or(-1.0), value) << buf.data();
  }
}

TEST(Config, FullP0IsRowByRow) {
  std::string list = "[";
  for (int i = 0; i < 225; ++i) {
    list += std::to_string(i) + (i < 224 ? "," : "]");
  }
  const std::string text = std::string(R"(
initial:
  p: [0, 0, 0]
  v: [0, 0, 0]
  q_wxyz: [1, 0, 0, 0]
  ba: [0, 0, 0]
  bg: [0, 0, 0]
  P0: )") + list + "\n";
  const RunConfig c = parse_config(text);
  EXPECT_EQ(c.P0(0, 1), 1.0);
  EXPECT_EQ(c.P0(1, 0), 15.0);
  EXPECT_EQ(c.P0(14, 14), 224.0);
}

TEST(Config, BadInputIsAnError) {
  EXPECT_THROW(parse_config("g: 9.81\n"), std::runtime_error);  // no `initial`
  EXPECT_THROW(parse_config(std::string("filter: {gat: 1.0}\n") + kInitial), std::runtime_error);
  EXPECT_THROW(parse_config(std::string("filter: {reset: none}\n") + kInitial), std::runtime_error);
  EXPECT_THROW(parse_config(std::string("filter: {discretization: rk4}\n") + kInitial),
               std::runtime_error);
  EXPECT_THROW(parse_config(std::string("imu_noise: {sigma_q: 1.0}\n") + kInitial),
               std::runtime_error);
  EXPECT_THROW(parse_config("initial: {p: [0, 0]}\n"), std::runtime_error);
  EXPECT_THROW(load_config("/nonexistent/config.yaml"), std::runtime_error);
}

}  // namespace
}  // namespace navekf
