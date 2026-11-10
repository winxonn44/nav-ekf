#include "navekf/config.hpp"

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdlib>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <string_view>

namespace navekf {

namespace {

void check_keys(const YAML::Node& node, const std::string& where,
                std::initializer_list<std::string_view> allowed) {
  if (!node.IsMap()) {
    throw std::runtime_error("config: `" + where + "` must be a mapping");
  }
  for (const auto& kv : node) {
    const auto key = kv.first.as<std::string>();
    if (std::find(allowed.begin(), allowed.end(), key) == allowed.end()) {
      throw std::runtime_error("config: unknown key `" + key + "` in `" + where + "`");
    }
  }
}

/// Through strtod rather than yaml-cpp's stream extraction: correctly rounded on every platform,
/// and subnormals parse instead of failing the stream.
double read_double(const YAML::Node& node, const std::string& name) {
  if (node.IsScalar()) {
    const std::string& text = node.Scalar();
    char* end = nullptr;
    const double value = std::strtod(text.c_str(), &end);
    if (end != text.c_str() && *end == '\0') {
      return value;
    }
  }
  throw std::runtime_error("config: `" + name + "` must be a number");
}

template <int N>
Eigen::Matrix<double, N, 1> read_vec(const YAML::Node& node, const std::string& name) {
  if (!node.IsSequence() || node.size() != static_cast<std::size_t>(N)) {
    throw std::runtime_error("config: `" + name + "` must be a list of " + std::to_string(N) +
                             " numbers");
  }
  Eigen::Matrix<double, N, 1> v;
  for (int i = 0; i < N; ++i) {
    v(i) = read_double(node[static_cast<std::size_t>(i)], name);
  }
  return v;
}

Discretization parse_discretization(const std::string& s) {
  if (s == "first") {
    return Discretization::kFirst;
  }
  if (s == "second") {
    return Discretization::kSecond;
  }
  if (s == "vanloan") {
    return Discretization::kVanLoan;
  }
  throw std::runtime_error("config: discretization must be first, second or vanloan, not `" + s +
                           "`");
}

Reset parse_reset(const std::string& s) {
  if (s == "jr") {
    return Reset::kJr;
  }
  if (s == "identity") {
    return Reset::kIdentity;
  }
  throw std::runtime_error("config: reset must be jr or identity, not `" + s + "`");
}

RunConfig from_node(const YAML::Node& root) {
  if (!root.IsMap()) {
    throw std::runtime_error("config: the top level must be a mapping");
  }
  RunConfig out;
  EskfConfig& e = out.eskf;

  if (root["g"]) {
    e.g = read_double(root["g"], "g");
  }
  if (const YAML::Node n = root["imu_noise"]) {
    check_keys(n, "imu_noise", {"sigma_a", "sigma_g", "sigma_ba", "sigma_bg"});
    if (n["sigma_a"]) {
      e.noise.sigma_a = read_double(n["sigma_a"], "imu_noise.sigma_a");
    }
    if (n["sigma_g"]) {
      e.noise.sigma_g = read_double(n["sigma_g"], "imu_noise.sigma_g");
    }
    if (n["sigma_ba"]) {
      e.noise.sigma_ba = read_double(n["sigma_ba"], "imu_noise.sigma_ba");
    }
    if (n["sigma_bg"]) {
      e.noise.sigma_bg = read_double(n["sigma_bg"], "imu_noise.sigma_bg");
    }
  }
  if (const YAML::Node n = root["gps"]) {
    if (n["lever_arm"]) {
      e.lever_arm = read_vec<3>(n["lever_arm"], "gps.lever_arm");
    }
  }
  if (const YAML::Node n = root["filter"]) {
    check_keys(n, "filter", {"discretization", "reset", "gate", "check_psd", "psd_tol"});
    if (n["discretization"]) {
      e.discretization = parse_discretization(n["discretization"].as<std::string>());
    }
    if (n["reset"]) {
      e.reset = parse_reset(n["reset"].as<std::string>());
    }
    if (n["gate"]) {
      if (n["gate"].IsNull()) {
        e.gate = std::nullopt;
      } else {
        e.gate = read_double(n["gate"], "filter.gate");
      }
    }
    if (n["check_psd"]) {
      e.check_psd = n["check_psd"].as<bool>();
    }
    if (n["psd_tol"]) {
      e.psd_tol = read_double(n["psd_tol"], "filter.psd_tol");
    }
  }

  const YAML::Node n = root["initial"];
  if (!n) {
    throw std::runtime_error("config: `initial` is required");
  }
  check_keys(n, "initial", {"p", "v", "q_wxyz", "ba", "bg", "P0_diag", "P0", "t0_ns"});
  const auto require = [&n](const char* key) {
    if (!n[key]) {
      throw std::runtime_error(std::string("config: `initial.") + key + "` is required");
    }
    return n[key];
  };
  out.x0.p = read_vec<3>(require("p"), "initial.p");
  out.x0.v = read_vec<3>(require("v"), "initial.v");
  out.x0.q = read_vec<4>(require("q_wxyz"), "initial.q_wxyz");
  out.x0.ba = read_vec<3>(require("ba"), "initial.ba");
  out.x0.bg = read_vec<3>(require("bg"), "initial.bg");
  if (n["P0_diag"] && n["P0"]) {
    throw std::runtime_error("config: give one of `initial.P0_diag` and `initial.P0`");
  }
  if (n["P0_diag"]) {
    out.P0 = read_vec<kNErr>(n["P0_diag"], "initial.P0_diag").asDiagonal();
  } else {
    const Eigen::Matrix<double, kNErr * kNErr, 1> v =
        read_vec<kNErr * kNErr>(require("P0"), "initial.P0");
    for (int r = 0; r < kNErr; ++r) {
      for (int c = 0; c < kNErr; ++c) {
        out.P0(r, c) = v((r * kNErr) + c);
      }
    }
  }
  if (n["t0_ns"]) {
    out.t0_ns = n["t0_ns"].as<std::int64_t>();
  }
  return out;
}

}  // namespace

RunConfig parse_config(const std::string& yaml_text) {
  try {
    return from_node(YAML::Load(yaml_text));
  } catch (const YAML::Exception& e) {
    throw std::runtime_error(std::string("config: ") + e.what());
  }
}

RunConfig load_config(const std::string& path) {
  try {
    return from_node(YAML::LoadFile(path));
  } catch (const YAML::Exception& e) {
    throw std::runtime_error("config " + path + ": " + e.what());
  }
}

}  // namespace navekf
