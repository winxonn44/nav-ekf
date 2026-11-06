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
- **"In band" is reported as % of time, and the slow test asks for ≥ 70% (NEES) / ≥ 90% (NIS) plus a time-averaged mean inside the band**. The band is per time step, and the errors are correlated over tens of seconds, so one 50-run batch can sit near an edge for a while. Ten independent batches (500 runs) quantify that spread (results.md).
- **Default `pytest` deselects `slow`** (`addopts = "-m 'not slow'"`); run the 50-run test with `pytest -m slow`.
