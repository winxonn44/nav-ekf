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
| E2 | Python ESKF with GPS updates; NEES/NIS consistency | – |
| E3 | C++20/Eigen port, matching Python to 1e-9 | – |
| E4 | EuRoC | – |
| E5 | KITTI | – |
| E6 | Robustness (dropouts, outliers, gating) | – |
| E7 | Tie-in with VoroNav | – |

## Layout

```
python/navekf_py/   so3.py, types.py, sim.py, propagate.py, drift.py, frames.py, io.py
configs/sim.yaml    simulator config (trajectory, IMU noise densities, GPS)
tests/              pytest suite
scripts/            check_derivation.py, plot_drift.py
```

## Running

Python ≥ 3.11 with numpy, scipy, matplotlib and pyyaml:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest                          # 53 tests, ~3 s
ruff check .
python scripts/check_derivation.py
python scripts/plot_drift.py    # docs/figures/e1_drift.png
```

`pytest` finds the package through `pythonpath = ["python"]` in `pyproject.toml`,
so an existing environment with those packages also works without installing.
A `slow` marker is registered for future long Monte Carlo tests (`pytest -m "not slow"`);
the E1 drift tests take ~1.5 s, so they don't use it.
