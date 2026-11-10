# Decisions

One line per choice that the spec left open, newest checkpoint last. Format:
**choice**: why. (Where it lives.)

## E2: Python ESKF

- **Batch axis everywhere** (`P` is `(..., 15, 15)`): the 50-run Monte Carlo is one batched filter (~6 s for 50 × 120 s at 200 Hz). The single-run path is the same code; `test_batched_filter_equals_single_runs` pins them to round-off. (`eskf.py`)
- **Step order: updates at `t_k`, then the hook/log, then propagate with sample `k`**: one fixed sequence for E3 to copy, written at the top of `eskf.py`.
- **`F, G` from the start-of-step nominal state**, computed *before* `propagate.propagate` advances it: matches the zero-order hold in conventions §6 and derivation §4.
- **The filter calls `propagate.propagate` for the nominal step**; it never re-implements §6.
- **`G Q_c Gᵀ` built each step from `G`**, though with isotropic noise it is constant (`R σ² I Rᵀ = σ² I`): it stays correct for anisotropic noise later, at negligible cost.
- **`"second"` discretization = `Φ₂` plus the `Q_d` you get by integrating `Φ(τ) Q Φ(τ)ᵀ` with `Φ(τ) ≈ I + Fτ`** (`Q dt + ½(FQ + QFᵀ)dt² + ⅓FQFᵀdt³`). The derivation names a second-order `Φ` but no `Q_d` to go with it. The error is still `O(dt³)` against Van Loan (tested).
- **Van Loan via `scipy.linalg.expm` on the stacked 30×30 matrices**: exact reference, tested against `expm(F dt)` and Simpson quadrature.
- **Fixes must land on IMU timestamps; `run` raises otherwise**: true for the simulator (200 Hz / 10 Hz). Splitting a step at the fix time is left to E4 (datasets).
- **Gate = `chi2.ppf(0.99, 3)` = 11.3449** (derivation rounds it to 11.345); `gate=None` disables it. NIS is computed and logged for every fix, rejected or not, and the NIS statistics include rejected fixes (that is the χ²(3) population).
- **A rejected fix leaves `x̂` and `P` bit-identical** (masked with `np.where`, not "update with K = 0"). Each rejection is appended to `ESKF.rejections` (t, run, NIS) and logged at INFO level.
- **`K = solve(S, H P)ᵀ` and `NIS = yᵀ solve(S, y)`**, `S` symmetrized first; Joseph form; `P` symmetrized after every propagate, update and reset.
- **`inject_and_reset` runs right after each accepted update** (`update_position(..., inject=True)` by default); `inject=False` is for tests that look at `δx̂`.
- **Identity reset skips the `G_r P G_rᵀ` product** rather than multiplying by `I`, so `"identity"` is bit-exact "no reset".
- **`so3.right_jacobian` now broadcasts** (needed by the batched reset); the single-vector path and its branch are unchanged.
- **Min-eigenvalue check is opt-in** (`check_psd`, `psd_tol`); it raises `FloatingPointError`. Off in the Monte Carlo, on in the exactness test.
- **Per-step log = Python lists, turned into arrays on demand** (`StepLog.arrays()`): one row per IMU step (posterior after propagation, `nis = nan`) and one per update (after reset). Off in the Monte Carlo (`log=False`).
- **Init from truth draws the initial error exactly from `P₀`**: `x̂₀ = truth ⊞ (−δx₀)`, `δx₀ ~ N(0, P₀)`, from `SeedSequence(seed + i, spawn_key=(2,))` (independent of the simulator's IMU and GPS streams, children 0 and 1). The true biases start at 0 and the *estimate* is perturbed; for the error statistics this is the same as random true biases.
- **Honest `P₀` (σ)**: p 2 m, v 0.2 m/s, roll/pitch 0.02 rad, yaw 0.05 rad, b_a 0.05 m/s², b_g 2e-3 rad/s. "Overconfident" = the filter starts with `P₀/100` (σ/10) while the error is still drawn from the honest `P₀`. (`consistency.P0_DEFAULT`)
- **Static init returns the state; `static_init_P0` gives a matching `P₀` with the tilt / accel-bias correlation** `δθ ≈ (1/g)[û]× δb_a`. Without that term, an accel bias looks like a tilt the filter is sure about. `b_a = 0`, `v = 0`, yaw given, no Earth rate.
- **NEES on the GPS-period grid (10 Hz), on the posterior** (after any update at that time), including ticks inside the dropout. Truth position on the consistent stream is `p_ref` (the §6 recursion), not the analytic `p` (they differ by 3.7e-6 m).
- **Headline MC on the consistent stream**: there the filter's process model is exact, so any NEES excess is the filter's fault. The analytic stream is reported as an ablation (the integrator error is then unmodeled).
- **E2 run = 120 s with a 30 s dropout at [60, 90) s** (`configs/e2_consistency.yaml`); `configs/sim.yaml` (60 s, 10 s dropout) drives the fast smoke test.
- **"In band" is the share of time in band, averaged over independent batches, with the gate off.** The band is per time step, so a consistent filter is inside 95% of the time in expectation, whatever the correlation over time; correlation only widens the spread between batches. The slow test uses 6 batches × 50 runs on seeds used nowhere else and requires ≥ 91% (NEES, 3 standard errors below 95%, from a batch std of about 3 points) and ≥ 93% (NIS). This replaced a single-batch ≥ 70% threshold that had been set after seeing seed 0 read 76%. Measured: 96.5% ± 0.9% over 12 batches (results.md). The gate is left off because it deliberately truncates the innovations; its cost is reported separately.
- **Default `pytest` deselects `slow`** (`addopts = "-m 'not slow'"`); run the 50-run test with `pytest -m slow`.

## E3: C++ port

- **`Quat` is an `Eigen::Vector4d` holding `[w, x, y, z]`, not `Eigen::Quaterniond`**: the storage order in conventions §3 then holds in C++ too; `so3::to_eigen` / `from_eigen` are the only crossings, pinned by `test_eigen_quaternion_order`. (`types.hpp`, `so3.hpp`)
- **Formulas written out in Python's order of operations, with `-ffp-contract=off` on every target**: `§6`, `quat_mul`, `Exp`, `Log` and `J_r` match term by term. Matrix products use Eigen and BLAS on each side, so those agree to round-off only. (`so3.hpp`, `eskf.cpp`, `CMakeLists.txt`)
- **`S` is solved with `PartialPivLU`**: the same algorithm as `np.linalg.solve`. (`eskf.cpp`)
- **A rejected fix returns before `K` and the Joseph update are computed**: Python computes them and masks them out; the state and `P` are bit-identical either way, and it is what makes a rejected update 8× cheaper. (`eskf.cpp`)
- **A rejection counter instead of Python's list of `Rejection`s**: a growing list would allocate in the filter loop; the log row carries `t_ns`, NIS and `accepted`. (`eskf.hpp`)
- **The gate is the literal `11.344866730144373`, as specified**: scipy 1.18.1's `chi2.ppf(0.99, 3)` returns 11.34486673014437, 2 ulp (3.6e-15) lower, and this can vary by scipy version. The parity runs hand Python's value to C++ through the config, so they compare the filters and not that digit. (`eskf.hpp`, `parity.py`)
- **`--parity` logs `q` raw and the upper triangle of `P`; the default log and the TUM file log `w ≥ 0` and `diag P`**: conventions §3 asks for `w ≥ 0` when logging, while Python's `StepLog` logs `q` raw. Comparing raw values also avoids false sign flips near `w ≈ 0`. (`apps/run_dataset.cpp`)
- **`run()` takes two callbacks**: `hook(k, filter)` is Python's hook (posterior at `t_k`, drives the TUM file) and `on_row` replaces `StepLog` (one call per IMU step or update). `run` requires sorted fixes and throws otherwise; the CLI sorts them with a stable sort, like Python's `sorted`. (`eskf.hpp`)
- **The no-heap proof uses two detectors, each with a positive control**: Eigen's `EIGEN_RUNTIME_NO_MALLOC` with assertions forced on (Eigen allocates with `malloc`, which `operator new` would not see), and a replaced global `operator new`. It covers all 3 discretizations, both resets, `check_psd`, and accepted and rejected fixes. (`tests/cpp/test_no_alloc.cpp`)
- **Config numbers go through `strtod`, not yaml-cpp's stream extraction**: correctly rounded on every platform, and the stream version refuses subnormals. Unknown keys in `filter`, `initial` and `imu_noise` are errors. (`config.cpp`)
- **Eigen is fetched header-only** (`SOURCE_SUBDIR` pointing nowhere) so its CMake project is never added; **yaml-cpp 0.8.0 builds with `CMAKE_POLICY_VERSION_MINIMUM=3.5`** because CMake 4 refuses its `VERSION 3.4` otherwise. (`CMakeLists.txt`)
- **C++ tests draw from a hand-written SplitMix64 with Box-Muller**: `std::` distributions differ between libc++ and libstdc++. (`tests/cpp/test_util.hpp`)
- **Golden files are generated in pytest's temp directory, never stored**: about 90 MB of text for the 10 scenarios. (`tests/test_cpp_parity.py`)
- **TSan is not run**: the filter, the CLI and the tests start no threads. (`.github/workflows/core.yml`)
- **`J_r`'s behaviour just above the small-angle threshold is ported unchanged**: for `θ` slightly over 1e-8, `1 − cos θ` is exactly 0 in `double`, so the regular branch drops the `−½[φ]×` term, a jump of about `½θ` (≈ 5e-9). Same in both languages and harmless at this size. (`so3.hpp`)
