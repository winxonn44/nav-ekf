"""Dead-reckoning drift laws (docs/derivation.md §7) by Monte Carlo.

A still, level IMU is integrated with the §6 integrator while exactly one
error source is on. All runs go through :func:`propagate.dead_reckon` in one
batch (run axis second), with run ``i`` drawing its noise from
``default_rng(seed + i)`` (conventions §8).

The reported error is the per-axis horizontal RMS position error,
``sqrt(mean over runs of (e_x² + e_y²)/2)``, which is the quantity in the §7 table.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .propagate import dead_reckon
from .sim import imu_noise, static_imu
from .types import G_MAGNITUDE, ImuNoise, ImuStream, NominalState

# Default magnitudes: consumer/tactical-grade, small enough that tilts stay linear.
DEFAULT_VALUES = {
    "accel_noise": 2.0e-3,  # σ_a  [m/s²/√Hz]
    "accel_bias": 1.0e-2,  # std of a constant b_a per axis [m/s²]
    "gyro_noise": 1.7e-4,  # σ_g  [rad/s/√Hz]
    "gyro_bias": 5.0e-4,  # std of a constant b_g per axis [rad/s] (~100 °/h)
}
SLOPES = {"accel_noise": 1.5, "accel_bias": 2.0, "gyro_noise": 2.5, "gyro_bias": 3.0}
LABELS = {
    "accel_noise": r"accel white noise $\sigma_a$",
    "accel_bias": r"accel bias $b_a$",
    "gyro_noise": r"gyro white noise $\sigma_g$",
    "gyro_bias": r"gyro bias $b_g$",
}


def theory(source: str, t: np.ndarray, value: float, g: float = G_MAGNITUDE) -> np.ndarray:
    """Per-axis horizontal RMS position error from derivation §7."""
    t = np.asarray(t, dtype=float)
    if source == "accel_noise":
        return value * t**1.5 / np.sqrt(3.0)
    if source == "accel_bias":
        return 0.5 * value * t**2
    if source == "gyro_noise":
        return g * value * t**2.5 / np.sqrt(20.0)
    if source == "gyro_bias":
        return g * value * t**3 / 6.0
    raise ValueError(source)


@dataclass
class DriftResult:
    source: str
    value: float
    n_runs: int
    t: np.ndarray  # (K,) seconds
    rms: np.ndarray  # (K,) per-axis horizontal RMS position error [m]
    theory: np.ndarray  # (K,)


def run(
    source: str,
    n_runs: int = 100,
    duration_s: float = 60.0,
    rate_hz: float = 100.0,
    value: float | None = None,
    seed: int = 0,
    g: float = G_MAGNITUDE,
) -> DriftResult:
    value = DEFAULT_VALUES[source] if value is None else value
    clean = static_imu(duration_s, rate_hz, g)
    K = len(clean)
    dt = float(clean.t_ns[1] - clean.t_ns[0]) * 1e-9

    noise = ImuNoise(
        sigma_a=value if source == "accel_noise" else 0.0,
        sigma_g=value if source == "gyro_noise" else 0.0,
    )
    acc = np.empty((K, n_runs, 3))
    gyr = np.empty((K, n_runs, 3))
    for i in range(n_runs):
        rng = np.random.default_rng(seed + i)
        ba0 = value * rng.standard_normal(3) if source == "accel_bias" else None
        bg0 = value * rng.standard_normal(3) if source == "gyro_bias" else None
        acc_err, gyr_err, _, _ = imu_noise(K, dt, noise, rng, ba0, bg0)
        acc[:, i] = clean.acc + acc_err
        gyr[:, i] = clean.gyr + gyr_err

    x0 = NominalState(
        np.zeros((n_runs, 3)), np.zeros((n_runs, 3)), np.tile([1.0, 0.0, 0.0, 0.0], (n_runs, 1)),
        np.zeros((n_runs, 3)), np.zeros((n_runs, 3)),
    )  # fmt: skip
    out = dead_reckon(x0, ImuStream(clean.t_ns, acc, gyr, kind=f"static+{source}"), g)
    e = out.p[:, :, :2]  # truth is p = 0
    rms = np.sqrt(np.mean(e**2, axis=(1, 2)))
    t = clean.t_ns * 1e-9
    return DriftResult(source, value, n_runs, t, rms, theory(source, t, value, g))


def loglog_slope(t: np.ndarray, y: np.ndarray, t_min: float = 1.0) -> float:
    """Least-squares slope of ``log y`` vs ``log t`` on log-spaced samples with ``t ≥ t_min``."""
    tq = np.geomspace(t_min, t[-1], 30)
    idx = np.searchsorted(t, tq).clip(0, len(t) - 1)
    return float(np.polyfit(np.log(t[idx]), np.log(y[idx]), 1)[0])
