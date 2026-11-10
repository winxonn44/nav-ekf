// YAML run configuration. The top-level keys `g`, `imu_noise` and `gps.lever_arm` have the
// same names and meaning as in configs/sim.yaml; `filter` and `initial` are new:
//
//   filter:
//     discretization: first      # first | second | vanloan
//     reset: jr                  # jr | identity
//     gate: 11.344866730144373   # null disables the gate; omitted = this default
//     check_psd: false
//     psd_tol: 0.0
//   initial:                     # required
//     p: [0, 0, 0]
//     v: [0, 0, 0]
//     q_wxyz: [1, 0, 0, 0]
//     ba: [0, 0, 0]
//     bg: [0, 0, 0]
//     P0_diag: [...]             # 15 numbers, or
//     P0: [...]                  # 225 numbers, row by row
//     t0_ns: 0                   # optional; default is the first IMU timestamp
//
// Unknown keys inside `filter`, `initial` and `imu_noise` are errors (a typo must not silently
// fall back to a default). Other top-level keys are ignored, so a simulator config can be extended.
#pragma once

#include <cstdint>
#include <optional>
#include <string>

#include "navekf/eskf.hpp"
#include "navekf/types.hpp"

namespace navekf {

struct RunConfig {
  EskfConfig eskf;
  State x0;
  Mat15 P0 = Mat15::Identity();
  std::optional<std::int64_t> t0_ns;
};

/// Parse YAML text. Throws std::runtime_error with the offending key on bad input.
RunConfig parse_config(const std::string& yaml_text);
/// Read and parse a YAML file.
RunConfig load_config(const std::string& path);

}  // namespace navekf
