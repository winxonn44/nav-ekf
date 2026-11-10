cmake_minimum_required(VERSION 3.25)  # FetchContent SYSTEM

project(navekf VERSION 0.3.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Build type" FORCE)
endif()

option(NAVEKF_BUILD_APPS "Build apps/run_dataset" ON)
option(NAVEKF_BUILD_TESTS "Build the GoogleTest suite" ON)
option(NAVEKF_BUILD_BENCH "Build the Google Benchmark suite" ON)
option(NAVEKF_WERROR "Treat warnings as errors (CI)" OFF)
set(NAVEKF_SANITIZE "" CACHE STRING "Comma-separated -fsanitize= list, e.g. address,undefined")

# Clang and GCC must not fuse a*b+c differently from Python's order of operations.
add_compile_options(-ffp-contract=off)

if(NAVEKF_SANITIZE)
  add_compile_options(-fsanitize=${NAVEKF_SANITIZE} -fno-omit-frame-pointer
                      -fno-sanitize-recover=all)
  add_link_options(-fsanitize=${NAVEKF_SANITIZE})
endif()

# Warnings apply to our targets only, never to fetched dependencies.
add_library(navekf_warnings INTERFACE)
target_compile_options(navekf_warnings INTERFACE
  -Wall -Wextra -Wpedantic -Wconversion -Wshadow
  $<$<BOOL:${NAVEKF_WERROR}>:-Werror>)

# Dependencies, pinned. FetchContent downloads them at configure time; to reuse
# a local checkout pass -DFETCHCONTENT_SOURCE_DIR_<NAME>=/path on the command line.
include(FetchContent)
set(FETCHCONTENT_QUIET ON)

# Header-only: SOURCE_SUBDIR points nowhere so that Eigen's own (slow, test-heavy)
# CMakeLists is not added.
FetchContent_Declare(eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG 3.4.0
  GIT_SHALLOW TRUE
  SOURCE_SUBDIR do-not-add-eigen-cmake)
FetchContent_MakeAvailable(eigen)
add_library(navekf_eigen INTERFACE)
target_include_directories(navekf_eigen SYSTEM INTERFACE ${eigen_SOURCE_DIR})

set(YAML_CPP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(YAML_CPP_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(YAML_CPP_BUILD_CONTRIB OFF CACHE BOOL "" FORCE)
set(YAML_CPP_INSTALL OFF CACHE BOOL "" FORCE)
set(YAML_BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
FetchContent_Declare(yaml-cpp
  GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
  GIT_TAG 0.8.0
  GIT_SHALLOW TRUE
  SYSTEM)
# yaml-cpp 0.8.0 asks for CMake 3.4, which CMake 4 refuses without this.
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
FetchContent_MakeAvailable(yaml-cpp)
unset(CMAKE_POLICY_VERSION_MINIMUM)

if(NAVEKF_BUILD_TESTS)
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
  FetchContent_Declare(googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.17.0
    GIT_SHALLOW TRUE
    SYSTEM)
  FetchContent_MakeAvailable(googletest)
endif()

if(NAVEKF_BUILD_BENCH)
  set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
  set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
  set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
  set(BENCHMARK_ENABLE_WERROR OFF CACHE BOOL "" FORCE)
  FetchContent_Declare(benchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG v1.9.4
    GIT_SHALLOW TRUE
    SYSTEM)
  FetchContent_MakeAvailable(benchmark)
endif()

# The filter: no YAML in its public interface, so config.cpp is the only TU that sees yaml-cpp.
set(NAVEKF_FILTER_SOURCES src/imu_model.cpp src/eskf.cpp)
add_library(navekf STATIC ${NAVEKF_FILTER_SOURCES} src/config.cpp)
add_library(navekf::navekf ALIAS navekf)
target_include_directories(navekf PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(navekf PUBLIC cxx_std_20)
target_link_libraries(navekf PUBLIC navekf_eigen PRIVATE yaml-cpp::yaml-cpp navekf_warnings)

if(NAVEKF_BUILD_APPS)
  add_executable(run_dataset apps/run_dataset.cpp)
  target_link_libraries(run_dataset PRIVATE navekf navekf_warnings)
endif()

if(NAVEKF_BUILD_TESTS)
  enable_testing()
  include(GoogleTest)

  add_executable(navekf_tests
    tests/cpp/test_so3.cpp
    tests/cpp/test_jacobians.cpp
    tests/cpp/test_eskf.cpp
    tests/cpp/test_config.cpp)
  target_link_libraries(navekf_tests PRIVATE navekf GTest::gtest_main navekf_warnings)
  gtest_discover_tests(navekf_tests DISCOVERY_MODE PRE_TEST NO_PRETTY_VALUES)

  # Rebuilds the filter sources with Eigen's runtime malloc check switched on
  # (and with assertions on, because that check is an eigen_assert).
  add_executable(navekf_alloc_tests tests/cpp/test_no_alloc.cpp ${NAVEKF_FILTER_SOURCES})
  target_include_directories(navekf_alloc_tests PRIVATE include)
  target_compile_definitions(navekf_alloc_tests PRIVATE EIGEN_RUNTIME_NO_MALLOC)
  target_compile_options(navekf_alloc_tests PRIVATE -UNDEBUG)
  target_link_libraries(navekf_alloc_tests PRIVATE navekf_eigen GTest::gtest_main navekf_warnings)
  gtest_discover_tests(navekf_alloc_tests DISCOVERY_MODE PRE_TEST NO_PRETTY_VALUES)
endif()

if(NAVEKF_BUILD_BENCH)
  add_executable(navekf_bench bench/bench_eskf.cpp)
  target_link_libraries(navekf_bench PRIVATE navekf benchmark::benchmark navekf_warnings)
endif()
