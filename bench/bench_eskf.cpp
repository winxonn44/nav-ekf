// µs per propagate and per update. Real numbers (docs/results.md):
//   navekf_bench --benchmark_repetitions=20 --benchmark_report_aggregates_only=true
#include <benchmark/benchmark.h>

#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <new>

#include "navekf/eskf.hpp"
#include "navekf/so3.hpp"

namespace {
std::atomic<std::size_t> g_new_calls{0};
}  // namespace

// Counts heap use for the "allocs_per_iter" counter (Eigen's own malloc is covered by
// tests/cpp/test_no_alloc.cpp).
void* operator new(std::size_t n) {
  g_new_calls.fetch_add(1, std::memory_order_relaxed);
  void* p = std::malloc(n == 0 ? 1 : n);
  if (p == nullptr) {
    throw std::bad_alloc();
  }
  return p;
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }

namespace navekf {
namespace {

constexpr double kDt = 0.005;
constexpr int kResetEvery = 1024;  // keeps P bounded over millions of iterations

Mat15 initial_P() {
  Vec15 s;
  s << 2, 2, 2, 0.2, 0.2, 0.2, 0.02, 0.02, 0.05, 0.05, 0.05, 0.05, 2e-3, 2e-3, 2e-3;
  return s.cwiseProduct(s).asDiagonal();
}

EskfConfig make_config(const Discretization d) {
  EskfConfig c;
  c.noise.sigma_a = 2e-3;
  c.noise.sigma_g = 1.7e-4;
  c.noise.sigma_ba = 3e-3;
  c.noise.sigma_bg = 2e-5;
  c.lever_arm = Vec3(0.3, -0.2, 0.5);
  c.discretization = d;
  return c;
}

State make_state() {
  State x;
  x.q = so3::Exp(Vec3(0.2, -0.1, 0.7));
  return x;
}

ImuSample make_imu() {
  ImuSample s;
  s.acc = Vec3(0.8, -0.5, 9.6);
  s.gyr = Vec3(0.4, -0.7, 0.9);
  return s;
}

void report_allocs(benchmark::State& state, const std::size_t before) {
  state.counters["allocs_per_iter"] = benchmark::Counter(
      static_cast<double>(g_new_calls.load() - before), benchmark::Counter::kAvgIterations);
}

void BM_Propagate(benchmark::State& state, const Discretization d) {
  const Eskf start(make_state(), initial_P(), make_config(d));
  Eskf f = start;
  const ImuSample imu = make_imu();
  int count = 0;
  const std::size_t before = g_new_calls.load();
  for (auto _ : state) {
    f.propagate(imu, kDt);
    benchmark::DoNotOptimize(f);
    if (++count == kResetEvery) {
      f = start;
      count = 0;
    }
  }
  report_allocs(state, before);
}

BENCHMARK_CAPTURE(BM_Propagate, first, Discretization::kFirst)->Unit(benchmark::kMicrosecond);
BENCHMARK_CAPTURE(BM_Propagate, second, Discretization::kSecond)->Unit(benchmark::kMicrosecond);
BENCHMARK_CAPTURE(BM_Propagate, vanloan, Discretization::kVanLoan)->Unit(benchmark::kMicrosecond);

/// update_position including injection and reset. `offset_m` picks the path: a small innovation
/// is accepted (full Joseph update + reset), a huge one is rejected by the gate.
void BM_Update(benchmark::State& state, const double offset_m) {
  const Eskf start(make_state(), initial_P(), make_config(Discretization::kFirst));
  Eskf f = start;
  PositionFix fix;
  fix.sigma = 1.0;
  int count = 0;
  double sign = 1.0;
  const std::size_t before = g_new_calls.load();
  for (auto _ : state) {
    fix.p_W = f.x().p + Vec3(sign * offset_m, -0.5 * sign * offset_m, 0.25 * offset_m);
    sign = -sign;
    UpdateResult u = f.update_position(fix);
    benchmark::DoNotOptimize(u);
    benchmark::DoNotOptimize(f);
    if (++count == kResetEvery) {
      f = start;
      count = 0;
    }
  }
  report_allocs(state, before);
}

BENCHMARK_CAPTURE(BM_Update, accepted, 0.5)->Unit(benchmark::kMicrosecond);
BENCHMARK_CAPTURE(BM_Update, rejected, 1000.0)->Unit(benchmark::kMicrosecond);

}  // namespace
}  // namespace navekf

BENCHMARK_MAIN();
