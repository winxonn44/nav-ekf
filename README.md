# nav-ekf

An error-state extended Kalman filter (ESKF) that fuses a 6-axis IMU with
position fixes. It is prototyped in Python as a reference oracle, ported to
C++20/Eigen, and evaluated on simulation, EuRoC and KITTI.

- [`docs/conventions.md`](docs/conventions.md): frames (ENU/FLU), Hamilton `[w,x,y,z]`
  quaternions, local errors, the one integrator, the simulator contract.
- [`docs/derivation.md`](docs/derivation.md): error dynamics, Jacobians, discretization,
  update, reset, drift laws, consistency bands. Checked by `scripts/check_derivation.py`.
- [`docs/results.md`](docs/results.md): numbers and figures per checkpoint.

## Status

| | Checkpoint | Status |
|---|---|---|
| E0 | Conventions and derivation, numerically checked | done |
| E1 | Python simulator (figure-8, IMU + GPS) and IMU dead reckoning; drift laws reproduced | done |
| E2 | Python ESKF with GPS updates; NEES/NIS consistency (50 runs, 30 s dropout) | done |
| E3 | C++20/Eigen port, matching Python to 1e-9 | – |
| E4 | EuRoC | – |
| E5 | KITTI | – |
| E6 | Robustness (dropouts, outliers, gating) | – |
| E7 | Tie-in with VoroNav | – |

## Layout

```
python/navekf_py/   so3.py, types.py, sim.py, propagate.py, drift.py, frames.py, io.py, eskf.py, consistency.py
configs/sim.yaml    simulator config (trajectory, IMU noise densities, GPS)
configs/e2_consistency.yaml   E2 Monte Carlo: 120 s with a 30 s GPS dropout
tests/              pytest suite
scripts/            check_derivation.py, plot_drift.py, run_consistency.py
docs/DECISIONS.md   one line per design choice
```

## Running

Python ≥ 3.11 with numpy, scipy, matplotlib and pyyaml:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest                          # 80 tests, ~5 s (slow Monte Carlo deselected)
pytest -m slow                  # the 50-run NEES/NIS test, ~8 s
ruff check .
python scripts/check_derivation.py
python scripts/plot_drift.py    # docs/figures/e1_drift.png
python scripts/run_consistency.py   # docs/figures/e2_*.png + E2 tables, ~4 min
```

`pytest` finds the package through `pythonpath = ["python"]` in `pyproject.toml`,
so an existing environment with those packages also works without installing.
Plain `pytest` deselects tests marked `slow` (`addopts` in `pyproject.toml`); `pytest -m slow`
runs them. CI runs the default suite.
