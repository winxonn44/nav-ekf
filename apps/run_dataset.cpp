// Runs the filter over an IMU CSV and a GPS CSV; writes a TUM trajectory and the per-step log.
//
//   run_dataset --imu imu.csv --gps gps.csv --config cfg.yaml --tum out.tum --log out.csv
//   [--parity]
//
// Inputs (one header line, then rows):
//   imu.csv  t_ns,ax,ay,az,wx,wy,wz       specific force [m/s²] and body rate [rad/s]
//   gps.csv  t_ns,x,y,z,sigma             antenna position in W [m], per-axis std [m]
//
// Outputs (docs/conventions.md §8):
//   out.tum  t x y z qx qy qz qw          one pose per IMU timestamp (posterior), w ≥ 0
//   out.csv  t_ns,kind,<state 16>,<diag P 15>,nis,accepted   one row per IMU step or update
//
// --parity replaces the diagonal of P by its upper triangle (row by row) and logs q exactly as
// the filter holds it, with no w ≥ 0 sign fix. tests/test_cpp_parity.py compares that to Python.
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "navekf/config.hpp"
#include "navekf/eskf.hpp"
#include "navekf/so3.hpp"
#include "navekf/types.hpp"

namespace {

using navekf::Eskf;
using navekf::ImuSample;
using navekf::kNErr;
using navekf::PositionFix;
using navekf::Quat;
using navekf::UpdateResult;

struct FileCloser {
  void operator()(std::FILE* f) const { std::fclose(f); }
};
using File = std::unique_ptr<std::FILE, FileCloser>;

File open_out(const std::string& path) {
  File f(std::fopen(path.c_str(), "w"));
  if (!f) {
    throw std::runtime_error("cannot write " + path);
  }
  return f;
}

std::string csv_error(const char* what, const std::string& where, const std::string& line) {
  std::string msg(what);
  msg += where;
  msg += ": ";
  msg += line;
  return msg;
}

/// Splits "t_ns,a,b,..." into an integer time and exactly N doubles (strtod: correctly rounded).
template <std::size_t N>
std::int64_t parse_row(const std::string& line, std::array<double, N>& out,
                       const std::string& where) {
  const char* p = line.c_str();
  char* end = nullptr;
  const std::int64_t t = std::strtoll(p, &end, 10);
  if (end == p) {
    throw std::runtime_error("bad row in " + where + ": " + line);
  }
  p = end;
  for (std::size_t i = 0; i < N; ++i) {
    if (*p != ',') {
      throw std::runtime_error(csv_error("too few columns in ", where, line));
    }
    ++p;
    out.at(i) = std::strtod(p, &end);
    if (end == p) {
      throw std::runtime_error(csv_error("bad number in ", where, line));
    }
    p = end;
  }
  return t;
}

template <std::size_t N, class Row>
void read_csv(const std::string& path, const Row& row) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("cannot read " + path);
  }
  std::string line;
  std::getline(in, line);  // header
  while (std::getline(in, line)) {
    if (line.empty()) {
      continue;
    }
    std::array<double, N> v{};
    const std::int64_t t = parse_row<N>(line, v, path);
    row(t, v);
  }
}

std::vector<ImuSample> read_imu(const std::string& path) {
  std::vector<ImuSample> out;
  read_csv<6>(path, [&out](const std::int64_t t, const std::array<double, 6>& v) {
    ImuSample s;
    s.t_ns = t;
    s.acc = navekf::Vec3(v[0], v[1], v[2]);
    s.gyr = navekf::Vec3(v[3], v[4], v[5]);
    out.push_back(s);
  });
  return out;
}

std::vector<PositionFix> read_gps(const std::string& path) {
  std::vector<PositionFix> out;
  read_csv<4>(path, [&out](const std::int64_t t, const std::array<double, 4>& v) {
    PositionFix f;
    f.t_ns = t;
    f.p_W = navekf::Vec3(v[0], v[1], v[2]);
    f.sigma = v[3];
    out.push_back(f);
  });
  // Stable, like Python's sorted(): fixes with equal times keep their file order.
  std::stable_sort(out.begin(), out.end(),
                   [](const PositionFix& a, const PositionFix& b) { return a.t_ns < b.t_ns; });
  return out;
}

void write_header(std::FILE* f, const bool parity) {
  std::fputs("t_ns,kind,px,py,pz,vx,vy,vz,qw,qx,qy,qz,bax,bay,baz,bgx,bgy,bgz", f);
  if (parity) {
    for (int r = 0; r < kNErr; ++r) {
      for (int c = r; c < kNErr; ++c) {
        std::fprintf(f, ",P_%d_%d", r, c);
      }
    }
  } else {
    for (int i = 0; i < kNErr; ++i) {
      std::fprintf(f, ",P_%d_%d", i, i);
    }
  }
  std::fputs(",nis,accepted\n", f);
}

void write_row(std::FILE* f, const Eskf& e, const int kind, const UpdateResult* u,
               const bool parity) {
  const navekf::State& x = e.x();
  const Quat q = parity ? x.q : navekf::so3::quat_canonical(x.q);
  std::fprintf(f, "%lld,%d", static_cast<long long>(e.t_ns()), kind);
  for (const double v : {x.p(0), x.p(1), x.p(2), x.v(0), x.v(1), x.v(2), q(0), q(1), q(2), q(3),
                         x.ba(0), x.ba(1), x.ba(2), x.bg(0), x.bg(1), x.bg(2)}) {
    std::fprintf(f, ",%.17g", v);
  }
  if (parity) {
    for (int r = 0; r < kNErr; ++r) {
      for (int c = r; c < kNErr; ++c) {
        std::fprintf(f, ",%.17g", e.P()(r, c));
      }
    }
  } else {
    for (int i = 0; i < kNErr; ++i) {
      std::fprintf(f, ",%.17g", e.P()(i, i));
    }
  }
  if (u != nullptr) {
    std::fprintf(f, ",%.17g,%d\n", u->nis, u->accepted ? 1 : 0);
  } else {
    std::fputs(",nan,1\n", f);
  }
}

/// Exact %.9f seconds from integer nanoseconds (no floating-point round trip), TUM order.
void write_tum(std::FILE* f, const std::int64_t t_ns, const navekf::State& x) {
  const Quat q = navekf::so3::quat_canonical(x.q);
  const std::int64_t mag = t_ns < 0 ? -t_ns : t_ns;
  std::fprintf(f, "%s%lld.%09lld %.9f %.9f %.9f %.9f %.9f %.9f %.9f\n", t_ns < 0 ? "-" : "",
               static_cast<long long>(mag / 1000000000), static_cast<long long>(mag % 1000000000),
               x.p(0), x.p(1), x.p(2), q(1), q(2), q(3), q(0));
}

struct Args {
  std::string imu, gps, config, tum, log;
  bool parity = false;
};

int usage() {
  std::fputs(
      "usage: run_dataset --imu imu.csv --gps gps.csv --config cfg.yaml "
      "--tum out.tum --log out.csv [--parity]\n",
      stderr);
  return 2;
}

std::optional<Args> parse_args(const std::span<char* const> argv) {
  Args a;
  for (std::size_t i = 1; i < argv.size(); ++i) {
    const std::string_view key = argv[i];
    if (key == "--parity") {
      a.parity = true;
      continue;
    }
    if (i + 1 >= argv.size()) {
      return std::nullopt;
    }
    const std::string value = argv[++i];
    if (key == "--imu") {
      a.imu = value;
    } else if (key == "--gps") {
      a.gps = value;
    } else if (key == "--config") {
      a.config = value;
    } else if (key == "--tum") {
      a.tum = value;
    } else if (key == "--log") {
      a.log = value;
    } else {
      return std::nullopt;
    }
  }
  if (a.imu.empty() || a.gps.empty() || a.config.empty() || a.tum.empty() || a.log.empty()) {
    return std::nullopt;
  }
  return a;
}

int run_app(const Args& a) {
  const navekf::RunConfig cfg = navekf::load_config(a.config);
  const std::vector<ImuSample> imu = read_imu(a.imu);
  const std::vector<PositionFix> fixes = read_gps(a.gps);
  if (imu.empty()) {
    throw std::runtime_error(a.imu + " has no samples");
  }

  const File tum = open_out(a.tum);
  const File log = open_out(a.log);
  write_header(log.get(), a.parity);

  Eskf filter(cfg.x0, cfg.P0, cfg.eskf, cfg.t0_ns.value_or(imu.front().t_ns));
  navekf::run(
      filter, std::span<const ImuSample>(imu), std::span<const PositionFix>(fixes),
      [&](const std::size_t k, const Eskf& e) { write_tum(tum.get(), imu[k].t_ns, e.x()); },
      [&](const Eskf& e, const int kind, const UpdateResult* u) {
        write_row(log.get(), e, kind, u, a.parity);
      });

  std::fprintf(stderr, "run_dataset: %zu IMU samples, %zu fixes, %zu rejected\n", imu.size(),
               fixes.size(), filter.rejections());
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const std::optional<Args> a =
        parse_args(std::span<char* const>(argv, static_cast<std::size_t>(argc)));
    if (!a.has_value()) {
      return usage();
    }
    return run_app(*a);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "run_dataset: %s\n", e.what());
    return 1;
  }
}
