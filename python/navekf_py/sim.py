"""Trajectory / IMU / GPS simulator (docs/conventions.md §4, §7).

The trajectory is a smooth figure-8 (Lissajous) in ENU with changing yaw and
some roll and pitch, all with analytic derivatives. An optional static segment
at the start is joined to the motion through a smooth time warp ``τ(t)``:
``τ = 0`` while static, then ``τ̇`` ramps 0 -> 1 with a quintic smootherstep, so
position, velocity and acceleration stay continuous.

Two IMU streams come from the same trajectory (conventions §7):

* **consistent**: samples invert the §6 integrator between true states at
  ``t_k`` and ``t_{k+1}``. Noise-free dead reckoning then reproduces ``q, v``
  and the reference position ``p_ref`` (defined by the §6 recursion) to ~1e-9.
* **analytic**: the exact instantaneous ``a_m(t_k), ω_m(t_k)``. Dead reckoning
  converges to the true trajectory as ``O(dt)``.

``Truth`` always holds the analytic trajectory. The same noise realization
(white noise plus bias random walks from :class:`ImuNoise`) is added to both
streams. GPS fixes come at ``gps_rate_hz`` with an optional lever arm and
scripted dropout windows.

Random numbers: ``SeedSequence(seed).spawn(2)`` gives independent generators
for the IMU noise and the GPS noise, so changing one never changes the other.
Monte Carlo run ``i`` should use ``seed + i`` (conventions §8).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from . import so3
from .types import G_MAGNITUDE, ImuNoise, ImuStream, PositionFix, Truth, gravity_w

NS_PER_S = 1_000_000_000


@dataclass
class TrajectoryConfig:
    """Figure-8: ``x = A_x sin(ωτ)``, ``y = A_y sin(2ωτ)``, ``z = z0 + A_z sin(ω_z τ)``.

    Attitude is ZYX Euler (``R_W_B = Rz(ψ) Ry(θ) Rx(φ)``):
    ``ψ = ψ0 + r τ + Ψ sin(ω_ψ τ)``, ``θ = Θ sin(ω_θ τ)``, ``φ = Φ sin(ω_φ τ)``.
    Roll and pitch are zero at ``τ = 0``, so the static segment is level.
    """

    period_s: float = 30.0  # figure-8 period, ω = 2π / period
    amp_x: float = 20.0
    amp_y: float = 10.0
    amp_z: float = 1.5
    z0: float = 0.0
    omega_z: float = 0.35  # rad/s
    yaw0: float = 0.3  # rad
    yaw_rate: float = 0.1  # rad/s
    yaw_amp: float = 0.5  # rad
    yaw_omega: float = 0.21  # rad/s
    pitch_amp: float = 0.10  # rad
    pitch_omega: float = 0.35  # rad/s
    roll_amp: float = 0.15  # rad
    roll_omega: float = 0.5  # rad/s
    static_s: float = 5.0  # length of the initial static segment
    ramp_s: float = 5.0  # time for τ̇ to ramp 0 -> 1 after it (must be > 0 if static_s > 0)


@dataclass
class GpsConfig:
    rate_hz: float = 10.0
    sigma: float = 1.0  # m, per axis
    lever_arm: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])  # antenna in B [m]
    dropouts: list[list[float]] = field(default_factory=list)  # [[t_start_s, t_end_s], ...]


@dataclass
class SimConfig:
    duration_s: float = 60.0
    imu_rate_hz: float = 200.0
    g: float = G_MAGNITUDE
    seed: int = 0
    trajectory: TrajectoryConfig = field(default_factory=TrajectoryConfig)
    imu_noise: ImuNoise = field(default_factory=ImuNoise)
    ba0: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])  # initial accel bias [m/s²]
    bg0: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])  # initial gyro bias [rad/s]
    gps: GpsConfig = field(default_factory=GpsConfig)

    def to_dict(self) -> dict:
        return asdict(self)


def load_config(path: str | Path) -> SimConfig:
    """Read a YAML config (see configs/sim.yaml). Missing keys keep their defaults."""
    import yaml

    d = yaml.safe_load(Path(path).read_text()) or {}
    return config_from_dict(d)


def config_from_dict(d: dict) -> SimConfig:
    d = dict(d)
    sub = {
        "trajectory": TrajectoryConfig(**d.pop("trajectory", {})),
        "imu_noise": ImuNoise(**d.pop("imu_noise", {})),
        "gps": GpsConfig(**d.pop("gps", {})),
    }
    return SimConfig(**d, **sub)


# ---------------------------------------------------------------- trajectory


def _time_warp(t: np.ndarray, static_s: float, ramp_s: float):
    """``τ(t), τ̇(t), τ̈(t)``: 0 while static, quintic ramp of ``τ̇``, then ``τ̇ = 1``."""
    t = np.asarray(t, dtype=float)
    if static_s == 0.0 and ramp_s == 0.0:
        return t, np.ones_like(t), np.zeros_like(t)
    if ramp_s <= 0.0:
        raise ValueError("ramp_s must be > 0 when static_s > 0")
    u = np.clip((t - static_s) / ramp_s, 0.0, 1.0)
    s = u**3 * (10 - 15 * u + 6 * u**2)  # τ̇ (smootherstep)
    ds = 30 * u**2 * (1 - u) ** 2 / ramp_s  # τ̈
    tau_ramp = ramp_s * u**4 * (2.5 - 3 * u + u**2)  # ∫ s dt over the ramp
    tau = np.where(t <= static_s + ramp_s, tau_ramp, 0.5 * ramp_s + (t - static_s - ramp_s))
    return tau, s, np.where((u > 0) & (u < 1), ds, 0.0)


def _euler_zyx_quat(yaw, pitch, roll):
    """``q_W_B = q_z(ψ) ⊗ q_y(θ) ⊗ q_x(φ)``."""
    cy, sy = np.cos(0.5 * yaw), np.sin(0.5 * yaw)
    cp, sp = np.cos(0.5 * pitch), np.sin(0.5 * pitch)
    cr, sr = np.cos(0.5 * roll), np.sin(0.5 * roll)
    return np.stack(
        [
            cy * cp * cr + sy * sp * sr,
            cy * cp * sr - sy * sp * cr,
            cy * sp * cr + sy * cp * sr,
            sy * cp * cr - cy * sp * sr,
        ],
        -1,
    )


def evaluate(cfg: TrajectoryConfig, t: np.ndarray) -> dict[str, np.ndarray]:
    """Analytic ``p, v, a`` (in W), ``q_W_B`` and body rate ``w_B`` at times ``t`` [s]."""
    t = np.atleast_1d(np.asarray(t, dtype=float))
    tau, td, tdd = _time_warp(t, cfg.static_s, cfg.ramp_s)
    w = 2 * np.pi / cfg.period_s
    wz = cfg.omega_z

    # position as a function of τ, and its first two τ-derivatives
    ax, ay, az = cfg.amp_x, cfg.amp_y, cfg.amp_z
    P = np.stack([ax * np.sin(w * tau), ay * np.sin(2 * w * tau), cfg.z0 + az * np.sin(wz * tau)], -1)
    dP = np.stack([ax * w * np.cos(w * tau), 2 * ay * w * np.cos(2 * w * tau), az * wz * np.cos(wz * tau)], -1)
    ddP = np.stack([-ax * w**2 * np.sin(w * tau), -4 * ay * w**2 * np.sin(2 * w * tau), -az * wz**2 * np.sin(wz * tau)], -1)
    v = dP * td[:, None]
    a = ddP * (td**2)[:, None] + dP * tdd[:, None]

    yaw = cfg.yaw0 + cfg.yaw_rate * tau + cfg.yaw_amp * np.sin(cfg.yaw_omega * tau)
    pitch = cfg.pitch_amp * np.sin(cfg.pitch_omega * tau)
    roll = cfg.roll_amp * np.sin(cfg.roll_omega * tau)
    dyaw = (cfg.yaw_rate + cfg.yaw_amp * cfg.yaw_omega * np.cos(cfg.yaw_omega * tau)) * td
    dpitch = cfg.pitch_amp * cfg.pitch_omega * np.cos(cfg.pitch_omega * tau) * td
    droll = cfg.roll_amp * cfg.roll_omega * np.cos(cfg.roll_omega * tau) * td

    # ZYX Euler rates -> body rates
    sr, cr, sp, cp = np.sin(roll), np.cos(roll), np.sin(pitch), np.cos(pitch)
    w_B = np.stack([droll - sp * dyaw, cr * dpitch + sr * cp * dyaw, -sr * dpitch + cr * cp * dyaw], -1)
    q = so3.quat_normalize(_euler_zyx_quat(yaw, pitch, roll))
    return {"p": P, "v": v, "a": a, "q": q, "w_B": w_B}


def specific_force(q_W_B: np.ndarray, a_W: np.ndarray, g: float = G_MAGNITUDE) -> np.ndarray:
    """Noise-free accelerometer reading ``R_W_Bᵀ (a_W − g_W)`` (conventions §4)."""
    R = so3.quat_to_rot(q_W_B)
    return np.einsum("...ji,...j->...i", R, a_W - gravity_w(g))


# --------------------------------------------------------------------- noise


def imu_noise(
    n: int,
    dt: float,
    noise: ImuNoise,
    rng: np.random.Generator,
    ba0: np.ndarray | None = None,
    bg0: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Additive IMU errors for ``n`` samples at period ``dt`` (conventions §4).

    Returns ``(acc_err, gyr_err, ba, bg)``, each ``(n, 3)``, with
    ``acc_err[k] = ba[k] + σ_a/√dt · N`` and ``ba[k+1] = ba[k] + σ_ba √dt · N``
    (same for the gyro). Draw order is fixed: accel white, gyro white, accel
    bias steps, gyro bias steps.
    """
    ba0 = np.zeros(3) if ba0 is None else np.asarray(ba0, dtype=float)
    bg0 = np.zeros(3) if bg0 is None else np.asarray(bg0, dtype=float)
    wa = rng.standard_normal((n, 3)) * (noise.sigma_a / np.sqrt(dt))
    wg = rng.standard_normal((n, 3)) * (noise.sigma_g / np.sqrt(dt))
    sa = rng.standard_normal((n - 1, 3)) * (noise.sigma_ba * np.sqrt(dt))
    sg = rng.standard_normal((n - 1, 3)) * (noise.sigma_bg * np.sqrt(dt))
    ba = ba0 + np.concatenate([np.zeros((1, 3)), np.cumsum(sa, 0)])
    bg = bg0 + np.concatenate([np.zeros((1, 3)), np.cumsum(sg, 0)])
    return ba + wa, bg + wg, ba, bg


# ----------------------------------------------------------------- simulator


@dataclass
class SimResult:
    config: SimConfig
    truth: Truth  # analytic trajectory at the IMU timestamps
    p_ref_consistent: np.ndarray  # (K, 3) §7 reference position for the consistent stream
    imu_consistent: ImuStream
    imu_analytic: ImuStream
    gps: list[PositionFix]


def imu_timestamps(duration_s: float, rate_hz: float) -> np.ndarray:
    """``t_k = k · period_ns`` for ``k = 0..K−1`` (int64 ns); the rate must divide 1e9 ns."""
    period_ns = NS_PER_S / rate_hz
    if abs(period_ns - round(period_ns)) > 1e-6:
        raise ValueError(f"IMU rate {rate_hz} Hz does not give an integer period in ns")
    period_ns = int(round(period_ns))
    K = int(round(duration_s * NS_PER_S / period_ns)) + 1
    return np.arange(K, dtype=np.int64) * np.int64(period_ns)


def consistent_imu(t_ns: np.ndarray, q: np.ndarray, v: np.ndarray, g: float = G_MAGNITUDE):
    """Invert the §6 integrator between consecutive true states (conventions §7).

    Returns ``(acc, gyr)`` of shape ``(K, 3)``; the last sample repeats the
    previous one (it is never integrated). ``dt`` is formed exactly as
    :func:`navekf_py.propagate.step_dt` does.
    """
    dt = (np.diff(t_ns).astype(np.int64)).astype(float) * 1e-9
    gyr = so3.log(so3.quat_mul(so3.quat_conj(q[:-1]), q[1:])) / dt[:, None]
    a_W = (v[1:] - v[:-1]) / dt[:, None]
    acc = specific_force(q[:-1], a_W, g)
    return np.vstack([acc, acc[-1:]]), np.vstack([gyr, gyr[-1:]])


def consistent_reference_position(t_ns: np.ndarray, p0: np.ndarray, v: np.ndarray) -> np.ndarray:
    """``p_{k+1} = p_k + (v_k + v_{k+1})/2 · dt``, the position §6 reproduces from the consistent stream."""
    dt = (np.diff(t_ns).astype(np.int64)).astype(float) * 1e-9
    steps = 0.5 * (v[:-1] + v[1:]) * dt[:, None]
    return np.vstack([p0, p0 + np.cumsum(steps, 0)])


def gps_fixes(cfg: SimConfig, rng: np.random.Generator, t_end_ns: int, tcfg: TrajectoryConfig | None = None) -> list[PositionFix]:
    """Antenna fixes ``z = p + R l + n``, ``n ~ N(0, σ² I)``, skipping dropout windows.

    Fix times are multiples of the GPS period (so they land on IMU timestamps
    whenever the IMU rate is a multiple of the GPS rate). One noise vector is
    drawn per scheduled fix, including dropped ones, so dropouts don't shift
    the noise of later fixes.
    """
    gc = cfg.gps
    period_ns = int(round(NS_PER_S / gc.rate_hz))
    t_ns = np.arange(0, t_end_ns + 1, period_ns, dtype=np.int64)
    tr = evaluate(tcfg or cfg.trajectory, t_ns * 1e-9)
    lever = np.asarray(gc.lever_arm, dtype=float)
    z = tr["p"] + np.einsum("kij,j->ki", so3.quat_to_rot(tr["q"]), lever)
    z = z + gc.sigma * rng.standard_normal(z.shape)
    t_s = t_ns * 1e-9
    keep = np.ones(len(t_ns), bool)
    for t0, t1 in gc.dropouts:
        keep &= ~((t_s >= t0) & (t_s < t1))
    return [PositionFix(t_ns[i], z[i], gc.sigma) for i in np.flatnonzero(keep)]


def simulate(cfg: SimConfig) -> SimResult:
    t_ns = imu_timestamps(cfg.duration_s, cfg.imu_rate_hz)
    tr = evaluate(cfg.trajectory, t_ns * 1e-9)
    imu_seq, gps_seq = np.random.SeedSequence(cfg.seed).spawn(2)

    # clean streams
    acc_c, gyr_c = consistent_imu(t_ns, tr["q"], tr["v"], cfg.g)
    acc_a = specific_force(tr["q"], tr["a"], cfg.g)
    gyr_a = tr["w_B"]

    # one noise realization, added to both streams
    dt = float(np.int64(t_ns[1]) - np.int64(t_ns[0])) * 1e-9
    acc_err, gyr_err, ba, bg = imu_noise(len(t_ns), dt, cfg.imu_noise, np.random.default_rng(imu_seq), cfg.ba0, cfg.bg0)

    truth = Truth(t_ns, tr["p"], tr["v"], tr["a"], tr["q"], tr["w_B"], ba, bg)
    return SimResult(
        config=cfg,
        truth=truth,
        p_ref_consistent=consistent_reference_position(t_ns, tr["p"][0], tr["v"]),
        imu_consistent=ImuStream(t_ns, acc_c + acc_err, gyr_c + gyr_err, kind="consistent"),
        imu_analytic=ImuStream(t_ns, acc_a + acc_err, gyr_a + gyr_err, kind="analytic"),
        gps=gps_fixes(cfg, np.random.default_rng(gps_seq), int(t_ns[-1])),
    )


def static_imu(duration_s: float, rate_hz: float, g: float = G_MAGNITUDE) -> ImuStream:
    """Noise-free samples of a still, level IMU: ``a_m = [0, 0, +g]``, ``ω_m = 0``."""
    t_ns = imu_timestamps(duration_s, rate_hz)
    acc = np.tile(specific_force(np.array([1.0, 0, 0, 0]), np.zeros(3), g), (len(t_ns), 1))
    return ImuStream(t_ns, acc, np.zeros((len(t_ns), 3)), kind="static")
