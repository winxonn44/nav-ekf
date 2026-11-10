"""Cross-language parity scenarios (E3): export inputs and Python's per-step filter output.

Each scenario writes four files to a directory:

* ``imu.csv`` (``t_ns,ax,ay,az,wx,wy,wz``) and ``gps.csv`` (``t_ns,x,y,z,sigma``);
* ``config.yaml``, the filter settings and the initial state for ``apps/run_dataset``;
* ``python_log.csv``, one row per IMU step or update, in the column layout of
  ``run_dataset --parity``: ``t_ns, kind, state (16, q raw), upper triangle of P (120), nis, accepted``.

Floats are written with ``%.17g`` and times as integer ns, so nothing is lost on the way.
``ESKF`` itself is not modified: :class:`RecordingESKF` only copies ``P`` out after each step.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import eskf, sim, so3
from .consistency import P0_DEFAULT
from .eskf import ESKF, EskfConfig
from .types import ImuNoise, ImuStream, NominalState, PositionFix

NOISE = ImuNoise(sigma_a=2e-3, sigma_g=1.7e-4, sigma_ba=3e-3, sigma_bg=2e-5)  # configs/sim.yaml
LEVER = np.array([0.3, -0.2, 0.5])
_IU = np.triu_indices(15)


@dataclass
class Case:
    imu: ImuStream
    fixes: list[PositionFix]
    x0: NominalState
    P0: np.ndarray
    cfg: EskfConfig
    expect_rejections: bool = False  # the scenario must exercise the gate


class RecordingESKF(ESKF):
    """``ESKF`` plus a copy of the full ``P`` for every row of the per-step log."""

    def __init__(self, *args, **kwargs):
        self.P_rows: list[np.ndarray] = []
        super().__init__(*args, **kwargs)

    def propagate(self, *args, **kwargs):
        super().propagate(*args, **kwargs)
        self.P_rows.append(self.P[_IU])

    def update_position(self, *args, **kwargs):
        out = super().update_position(*args, **kwargs)
        self.P_rows.append(self.P[_IU])
        return out


# ------------------------------------------------------------------- scenarios


def consistent_fixes(r: sim.SimResult, lever: np.ndarray, sigma: float = 1.0, rate_hz: float = 10.0) -> list[PositionFix]:
    """Noise-free fixes consistent with the §6 reference (as in ``tests/test_eskf.py``)."""
    period = int(round(1e9 / rate_hz))
    return [
        PositionFix(t, r.p_ref_consistent[k] + so3.quat_to_rot(r.truth.q[k]) @ lever, sigma)
        for k, t in enumerate(r.truth.t_ns)
        if int(t) % period == 0
    ]


def _shifted(fixes: list[PositionFix], at_s: dict[float, list[float]]) -> list[PositionFix]:
    """Copies of ``fixes`` with the given position offsets added at the given times [s]."""
    wanted = {int(round(t * 1e9)): np.array(d) for t, d in at_s.items()}
    out = [PositionFix(f.t_ns, f.p_W + wanted.get(int(f.t_ns), 0.0), f.sigma) for f in fixes]
    assert len({int(f.t_ns) for f in out} & set(wanted)) == len(wanted), "an outlier time has no fix"
    return out


def golden() -> Case:
    """``test_noise_free_filter_tracks_truth_exactly``: 20 s, consistent stream, lever arm."""
    r = sim.simulate(sim.SimConfig(duration_s=20.0))
    t0 = r.truth.state(0)
    t0.p = r.p_ref_consistent[0].copy()
    cfg = EskfConfig(noise=NOISE, lever_arm=LEVER, check_psd=True)
    return Case(r.imu_consistent, consistent_fixes(r, LEVER), t0, P0_DEFAULT, cfg)


def _noisy_run(duration_s: float, seed: int, dropouts: list[list[float]]) -> sim.SimResult:
    gps = sim.GpsConfig(rate_hz=10.0, sigma=1.0, lever_arm=LEVER.tolist(), dropouts=dropouts)
    return sim.simulate(sim.SimConfig(duration_s=duration_s, imu_noise=NOISE, seed=seed, gps=gps))


def _perturbed_start(truth: NominalState, P0: np.ndarray, seed: int) -> NominalState:
    """``truth ⊞ (−δx₀)`` with ``δx₀ ~ N(0, P₀)``, as ``ESKF.from_truth``."""
    dx0 = np.linalg.cholesky(P0) @ np.random.default_rng(seed).standard_normal(15)
    return eskf.boxplus(truth, -dx0)


def noisy() -> Case:
    """Noise, a perturbed start, a 10 s dropout and three outliers: the gate must decide alike."""
    r = _noisy_run(40.0, seed=5, dropouts=[[15.0, 25.0]])
    fixes = _shifted(r.gps, {8.0: [20.0, 0.0, 0.0], 12.5: [0.0, -30.0, 5.0], 30.0: [15.0, 15.0, -15.0]})
    x0 = _perturbed_start(r.truth.state(0), P0_DEFAULT, seed=105)
    return Case(r.imu_consistent, fixes, x0, P0_DEFAULT, EskfConfig(noise=NOISE, lever_arm=LEVER), expect_rejections=True)


def lockout() -> Case:
    """The known post-outage lock-out: after a dropout every fix is far away and is rejected."""
    r = _noisy_run(10.0, seed=6, dropouts=[[3.0, 5.0]])
    offset = np.array([40.0, 0.0, 0.0])
    fixes = [PositionFix(f.t_ns, f.p_W + (offset if int(f.t_ns) >= 5 * 10**9 else 0.0), f.sigma) for f in r.gps]
    cfg = EskfConfig(noise=NOISE, lever_arm=LEVER)
    return Case(r.imu_consistent, fixes, r.truth.state(0), P0_DEFAULT, cfg, expect_rejections=True)


def option(discretization: str, reset: str, gate: float | None = eskf.GATE_99_3DOF) -> Callable[[], Case]:
    """A 10 s noisy run on the analytic stream with a fully correlated ``P₀`` and one outlier."""

    def build() -> Case:
        r = _noisy_run(10.0, seed=7, dropouts=[])
        x0 = r.truth.state(0)
        P0 = eskf.static_init_P0(x0, 5.0, NOISE, sigma_ba0=0.05, sigma_yaw=0.05, sigma_p=1.0)
        fixes = _shifted(r.gps, {6.0: [50.0, 0.0, 0.0]})
        cfg = EskfConfig(noise=NOISE, lever_arm=LEVER, discretization=discretization, reset=reset, gate=gate)
        return Case(r.imu_analytic, fixes, x0, P0, cfg, expect_rejections=gate is not None)

    return build


SCENARIOS: dict[str, Callable[[], Case]] = {
    "golden_noise_free": golden,
    "noisy_dropout_outliers": noisy,
    "lockout": lockout,
    "first_jr": option("first", "jr"),
    "first_identity": option("first", "identity"),
    "second_jr": option("second", "jr"),
    "second_identity": option("second", "identity"),
    "vanloan_jr": option("vanloan", "jr"),
    "vanloan_identity": option("vanloan", "identity"),
    "gate_off": option("first", "jr", gate=None),
}


# ---------------------------------------------------------------------- export


def _yaml_list(a: np.ndarray) -> str:
    return "[" + ", ".join(f"{float(v):.17g}" for v in np.ravel(a)) + "]"


def config_yaml(case: Case) -> str:
    c, x = case.cfg, case.x0
    gate = "null" if c.gate is None else f"{c.gate:.17g}"
    n = c.noise
    return (
        f"g: {c.g:.17g}\n"
        f"imu_noise: {{sigma_a: {n.sigma_a:.17g}, sigma_g: {n.sigma_g:.17g}, "
        f"sigma_ba: {n.sigma_ba:.17g}, sigma_bg: {n.sigma_bg:.17g}}}\n"
        f"gps: {{lever_arm: {_yaml_list(c.lever_arm)}}}\n"
        f"filter: {{discretization: {c.discretization}, reset: {c.reset}, gate: {gate}, "
        f"check_psd: {'true' if c.check_psd else 'false'}, psd_tol: {c.psd_tol:.17g}}}\n"
        "initial:\n"
        f"  p: {_yaml_list(x.p)}\n"
        f"  v: {_yaml_list(x.v)}\n"
        f"  q_wxyz: {_yaml_list(x.q)}\n"
        f"  ba: {_yaml_list(x.ba)}\n"
        f"  bg: {_yaml_list(x.bg)}\n"
        f"  P0: {_yaml_list(case.P0)}\n"
    )


def log_header() -> str:
    names = ["t_ns", "kind", "px", "py", "pz", "vx", "vy", "vz", "qw", "qx", "qy", "qz", "bax", "bay", "baz", "bgx", "bgy", "bgz"]
    names += [f"P_{r}_{c}" for r, c in zip(*_IU, strict=True)]
    return ",".join([*names, "nis", "accepted"])


def run_python(case: Case) -> np.ndarray:
    """Run the reference filter; returns the rows of ``python_log.csv`` as a float array."""
    f = RecordingESKF(case.x0, case.P0, case.cfg, t0_ns=int(case.imu.t_ns[0]))
    eskf.run(f, case.imu, case.fixes)
    L = f.log.arrays()
    return np.column_stack(
        [L["t_ns"], L["kind"], L["x"], np.stack(f.P_rows), L["nis"], L["accepted"]],
    )


def write_imu_csv(path: str | Path, imu: ImuStream) -> None:
    with open(path, "w") as fh:
        fh.write("t_ns,ax,ay,az,wx,wy,wz\n")
        for t, a, w in zip(imu.t_ns, imu.acc, imu.gyr, strict=True):
            fh.write(f"{int(t)},{a[0]:.17g},{a[1]:.17g},{a[2]:.17g},{w[0]:.17g},{w[1]:.17g},{w[2]:.17g}\n")


def write_gps_csv(path: str | Path, fixes: list[PositionFix]) -> None:
    with open(path, "w") as fh:
        fh.write("t_ns,x,y,z,sigma\n")
        for f in fixes:
            fh.write(f"{int(f.t_ns)},{f.p_W[0]:.17g},{f.p_W[1]:.17g},{f.p_W[2]:.17g},{f.sigma:.17g}\n")


def write_case(case: Case, out_dir: str | Path) -> Path:
    """Write ``imu.csv``, ``gps.csv`` and ``config.yaml`` (the inputs of ``apps/run_dataset``) to ``out_dir``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_imu_csv(out / "imu.csv", case.imu)
    write_gps_csv(out / "gps.csv", case.fixes)
    (out / "config.yaml").write_text(config_yaml(case))
    return out


def write_python_log(path: str | Path, rows: np.ndarray) -> None:
    fmt = ["%d", "%d"] + ["%.17g"] * (rows.shape[1] - 3) + ["%d"]
    np.savetxt(path, rows, fmt=fmt, delimiter=",", header=log_header(), comments="")


def export(name: str, out_dir: str | Path) -> tuple[Case, np.ndarray]:
    """Write a scenario's four files to ``out_dir``; returns the case and Python's rows."""
    case = SCENARIOS[name]()
    out = write_case(case, out_dir)
    rows = run_python(case)
    write_python_log(out / "python_log.csv", rows)
    return case, rows
