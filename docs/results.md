# Results

Every number here says which simulator stream produced it (conventions §7).
Regenerate with the command given in each section.

## E1: simulator and IMU dead reckoning

Code: `python/navekf_py/` (`sim.py`, `propagate.py`, `so3.py`). Tests: `pytest` (53 tests, ~3 s on an M4 Air).

### Exactness and convergence

| Check | Stream | Setup | Result | Required |
|---|---|---|---|---|
| Noise-free DR vs `q, v, p_ref` | consistent | figure-8, 60 s at 200 Hz (12 001 samples) | max errors: `p` 1.7e-11 m, `v` 6.3e-13 m/s, `q` 3.3e-14 rad | < 1e-9 |
| `p_ref` (§6 recursion) vs analytic `p` | consistent | same | 3.7e-6 m | (trapezoid error, for reference) |
| Noise-free DR vs analytic truth, max ‖δp‖ over 30 s | analytic | 100 / 200 / 400 Hz | 0.177 / 0.088 / 0.044 m, fitted slope **1.00** in `dt` | O(dt) |

`tests/test_propagate.py::test_consistent_stream_dead_reckoning_is_exact`,
`::test_analytic_stream_error_is_first_order`.

### Drift laws (derivation §7)

Still, level IMU, 60 s at 100 Hz, **100 Monte Carlo runs per source**, one source
on at a time, run `i` seeded with `0 + i`. The error is the per-axis horizontal
RMS position error. Biases are constant per run, drawn `N(0, value²)` per axis.

![E1 drift](figures/e1_drift.png)

| source | value | RMS/theory at 10 s | at 60 s | min–max (t ≥ 1 s) | fitted slope | theory |
|---|---|---|---|---|---|---|
| accel white noise `σ_a` | 2e-3 m/s²/√Hz | 1.042 | 1.039 | 0.965–1.055 | 1.512 | 1.5 |
| accel bias `b_a` | 1e-2 m/s² | 0.983 | 0.983 | 0.983–0.983 | 2.000 | 2.0 |
| gyro white noise `σ_g` | 1.7e-4 rad/s/√Hz | 1.022 | 1.107 | 1.004–1.132 | 2.485 | 2.5 |
| gyro bias `b_g` | 5e-4 rad/s | 0.982 | 0.983 | 0.969–0.983 | 3.003 | 3.0 |

All ratios are inside the ±20% sampling band (at 100 runs one point's 1σ is
about 7%), and all slopes are within 0.02 of theory. The bias rows have a
constant ratio because their error is deterministic given the drawn bias:
0.983 is just the sample RMS of the 200 drawn values per axis. The gyro-bias
ratio dips below that at small `t` (0.969 at 1 s) because the start-of-step
hold lags the growing tilt. The discrete sums give `⅙ g b t³ (1 − ~1.5 dt/t)`,
which is −1.5% at `t = 1 s` and 100 Hz. The tests (`tests/test_drift.py`) assert ratio ∈ [0.8, 1.2] for
`t ≥ 1 s` and slope within ±0.1.

Regenerate: `python scripts/plot_drift.py` (~2 s).
