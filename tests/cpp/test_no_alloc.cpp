// Proves that the filter loop does not touch the heap.
//
// Two detectors, each with a positive control so that a silent detector fails the suite:
//   * Eigen's runtime check: this binary compiles the filter sources with
//     EIGEN_RUNTIME_NO_MALLOC and assertions on, so any Eigen heap allocation aborts while
//     malloc is disallowed (Eigen allocates with malloc, which operator new would not see).
//   * replaced global operator new, which counts everything else (std::vector, std::string).
#ifndef EIGEN_RUNTIME_NO_MALLOC
#error "test_no_alloc.cpp must be built with EIGEN_RUNTIME_NO_MALLOC"
#endif

#include <Eigen/Core>
#include <gtest/gtest.h>

#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <new>
#include <span>
#include <string>
#include <vector>

#include "navekf/eskf.hpp"
#include "navekf/so3.hpp"

#include "test_util.hpp"

namespace {
std::atomic<std::size_t> g_new_calls{0};

void* counted_malloc(const std::size_t n) {
  g_new_calls.fetch_add(1, std::memory_order_relaxed);
  void* p = std::malloc(n == 0 ? 1 : n);
  if (p == nullptr) {
    throw std::bad_alloc();
  }
  return p;
}

void* counted_aligned(const std::size_t n, const std::align_val_t a) {
  g_new_calls.fetch_add(1, std::memory_order_relaxed);
  const auto align = static_cast<std::size_t>(a);
  // aligned_alloc needs a size that is a multiple of the alignment.
  void* p = std::aligned_alloc(align, ((n == 0 ? 1 : n) + align - 1) / align * align);
  if (p == nullptr) {
    throw std::bad_alloc();
  }
  return p;
}

template <class Alloc>
void* nothrow_alloc(const Alloc& alloc) noexcept {
  try {
    return alloc();
  } catch (const std::bad_alloc&) {
    return nullptr;
  }
}
}  // namespace

// The whole replaceable family, nothrow forms included: a nothrow new left to the runtime is
// freed by the delete below, a mismatch (std::get_temporary_buffer uses nothrow new).
void* operator new(std::size_t n) { return counted_malloc(n); }
void* operator new[](std::size_t n) { return counted_malloc(n); }
void* operator new(std::size_t n, std::align_val_t a) { return counted_aligned(n, a); }
void* operator new[](std::size_t n, std::align_val_t a) { return counted_aligned(n, a); }
void* operator new(std::size_t n, const std::nothrow_t&) noexcept {
  return nothrow_alloc([n] { return counted_malloc(n); });
}
void* operator new[](std::size_t n, const std::nothrow_t&) noexcept {
  return nothrow_alloc([n] { return counted_malloc(n); });
}
void* operator new(std::size_t n, std::align_val_t a, const std::nothrow_t&) noexcept {
  return nothrow_alloc([n, a] { return counted_aligned(n, a); });
}
void* operator new[](std::size_t n, std::align_val_t a, const std::nothrow_t&) noexcept {
  return nothrow_alloc([n, a] { return counted_aligned(n, a); });
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }
void operator delete(void* p, std::align_val_t) noexcept { std::free(p); }
void operator delete[](void* p, std::align_val_t) noexcept { std::free(p); }
void operator delete(void* p, std::size_t, std::align_val_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t, std::align_val_t) noexcept { std::free(p); }
void operator delete(void* p, const std::nothrow_t&) noexcept { std::free(p); }
void operator delete[](void* p, const std::nothrow_t&) noexcept { std::free(p); }
void operator delete(void* p, std::align_val_t, const std::nothrow_t&) noexcept { std::free(p); }
void operator delete[](void* p, std::align_val_t, const std::nothrow_t&) noexcept { std::free(p); }

namespace navekf {
namespace {

struct Options {
  Discretization discretization;
  Reset reset;
  bool check_psd;
};

class NoAlloc : public ::testing::TestWithParam<Options> {};

TEST_P(NoAlloc, FilterLoopDoesNotTouchTheHeap) {
  const test::ConsistentRun r = test::simulate_consistent(1.0, test::lever_arm());
  std::vector<PositionFix> fixes = r.fixes;
  fixes[3].p_W.x() += 50.0;  // an outlier, so the rejection path runs too

  EskfConfig cfg;
  cfg.noise = test::default_noise();
  cfg.lever_arm = test::lever_arm();
  cfg.discretization = GetParam().discretization;
  cfg.reset = GetParam().reset;
  cfg.check_psd = GetParam().check_psd;
  Eskf f(r.truth[0], test::default_P0(), cfg);

  const std::size_t before = g_new_calls.load();
  Eigen::internal::set_is_malloc_allowed(false);
  std::size_t rows = 0;
  std::size_t rejected = 0;
  run(
      f, std::span<const ImuSample>(r.imu), std::span<const PositionFix>(fixes),
      [](std::size_t, const Eskf&) {},
      [&rows, &rejected](const Eskf&, const int kind, const UpdateResult* u) {
        ++rows;
        if (kind == kRowUpdate && !u->accepted) {
          ++rejected;
        }
      });
  // The other public entry point: an update without injection, then the injection.
  PositionFix fix;
  fix.t_ns = f.t_ns();
  fix.p_W = f.x().p + so3::quat_to_rot(f.x().q) * cfg.lever_arm + Vec3::Constant(0.1);
  (void)f.update_position(fix, /*inject=*/false);
  f.inject_and_reset();
  Eigen::internal::set_is_malloc_allowed(true);
  const std::size_t after = g_new_calls.load();

  EXPECT_EQ(after - before, 0U);
  EXPECT_EQ(rows, r.imu.size() - 1 + fixes.size());
  EXPECT_EQ(rejected, 1U);
}

INSTANTIATE_TEST_SUITE_P(
    AllOptions, NoAlloc,
    ::testing::Values(Options{Discretization::kFirst, Reset::kJr, false},
                      Options{Discretization::kFirst, Reset::kJr, true},
                      Options{Discretization::kFirst, Reset::kIdentity, false},
                      Options{Discretization::kSecond, Reset::kJr, false},
                      Options{Discretization::kSecond, Reset::kIdentity, false},
                      Options{Discretization::kVanLoan, Reset::kJr, false},
                      Options{Discretization::kVanLoan, Reset::kIdentity, false}),
    [](const auto& param) { return std::to_string(param.index); });

TEST(NoAllocControl, OperatorNewCounterSeesAnAllocation) {
  const std::size_t before = g_new_calls.load();
  int* p = new int[100];
  volatile int* escape = p;  // keeps the compiler from eliding the allocation
  escape[0] = 1;
  delete[] p;
  EXPECT_GT(g_new_calls.load(), before);
}

TEST(NoAllocControlDeathTest, EigenCheckCatchesAHeapMatrix) {
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  EXPECT_DEATH(
      {
        Eigen::internal::set_is_malloc_allowed(false);
        const Eigen::MatrixXd m = Eigen::MatrixXd::Identity(100, 100);
        std::exit(m(0, 0) == 1.0 ? 0 : 1);
      },
      "");
}

}  // namespace
}  // namespace navekf
