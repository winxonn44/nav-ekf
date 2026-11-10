"""EuRoC MAV reader and E4 run setup (docs/data.md, docs/results.md).

Layout read here: ``<root>/<sequence>/mav0/{imu0,state_groundtruth_estimate0}/{data.csv,sensor.yaml}``.

Frames, per docs/conventions.md §2, each one checked by :func:`data_report` rather than assumed:

* The filter's ``B`` is the IMU sensor frame ``S``. ``T_BS`` in ``imu0/sensor.yaml`` is applied here if it is a pure
  rotation; any translation raises, because it would need angular-rate terms (it is the identity in EuRoC).
* The ground truth is the batch estimate of the pose of ``S`` in EuRoC's reference frame ``R``, used directly as the
  filter's ``W`` (z up, gravity along −z). It is not ENU in any geographic sense, and nothing here needs it to be.

On this MAV the IMU's x axis points roughly up, so ``B`` is not FLU; roll/pitch/yaw are near gimbal lock.
:func:`static_init_tilt_corrected` therefore starts from the ground-truth attitude and applies the minimal rotation
that puts the measured gravity on the z axis. That keeps the ground-truth heading without any Euler angles.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from . import so3
from .eskf import IBG, EskfConfig, static_init_P0
from .types import G_MAGNITUDE, ImuNoise, ImuStream, NominalState, PositionFix

NS_PER_S = 1_000_000_000
FIX_NOISE_STREAM, P0_STREAM = 0, 1  # SeedSequence spawn keys: the two uses of randomness per sequence


@dataclass
class GroundTruth:
    t_ns: np.ndarray  # (K,) int64
    p: np.ndarray  # (K, 3) p_R_S
    q: np.ndarray  # (K, 4) q_R_S [w, x, y, z]
    v: np.ndarray  # (K, 3) v_R
    bg: np.ndarray  # (K, 3) batch-estimated gyro bias
    ba: np.ndarray  # (K, 3) batch-estimated accel bias


@dataclass
class Sequence:
    name: str
    imu: ImuStream
    gt: GroundTruth
    noise: ImuNoise  # the sensor.yaml densities
    rate_hz: float


@dataclass
class StillSegment:
    t_start_ns: int
    t_end_ns: int

    @property
    def duration_s(self) -> float:
        return (self.t_end_ns - self.t_start_ns) / NS_PER_S


@dataclass
class InitConfig:
    window_s: float = 0.5
    min_still_s: float = 1.0
    acc_std_max: float = 0.1
    gyro_std_max: float = 0.01
    sigma_ba0: float = 0.1
    sigma_bg0: float = 2.0e-3
    sigma_yaw: float = 0.02
    sigma_v: float = 0.01


@dataclass
class FixConfig:
    rate_hz: float = 10.0
    sigma: float = 0.10
    seed: int = 0


# ------------------------------------------------------------------------- reader


def _read_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _t_bs(sensor: dict) -> np.ndarray:
    T = sensor["T_BS"]
    return np.array(T["data"], dtype=float).reshape(T["rows"], T["cols"])


def _read_csv(path: Path, columns: int) -> tuple[np.ndarray, np.ndarray]:
    """``(t_ns int64, values (K, columns))``. Time is parsed as an integer: 19 digits do not fit a double."""
    t = np.loadtxt(path, delimiter=",", comments="#", usecols=0, dtype=np.int64, ndmin=1)
    a = np.loadtxt(path, delimiter=",", comments="#", usecols=range(1, columns + 1), dtype=float, ndmin=2)
    d = np.diff(t)
    if np.any(d <= 0):
        raise ValueError(f"{path}: timestamps must strictly increase ({int(np.sum(d <= 0))} rows do not)")
    return t, a


def noise_from_sensor(sensor: dict) -> ImuNoise:
    return ImuNoise(
        sigma_a=float(sensor["accelerometer_noise_density"]),
        sigma_g=float(sensor["gyroscope_noise_density"]),
        sigma_ba=float(sensor["accelerometer_random_walk"]),
        sigma_bg=float(sensor["gyroscope_random_walk"]),
    )


def load_sequence(root: str | Path, name: str) -> Sequence:
    """Read one extracted sequence and convert it to filter conventions (docstring above)."""
    mav = Path(root) / name / "mav0"
    imu_yaml = _read_yaml(mav / "imu0" / "sensor.yaml")
    T_BS = _t_bs(imu_yaml)
    if np.any(np.abs(T_BS[:3, 3]) > 1e-9):
        raise ValueError(f"{name}: IMU T_BS has a translation, which this loader does not handle")
    R_BS = T_BS[:3, :3]

    t, a = _read_csv(mav / "imu0" / "data.csv", 6)  # columns: w_x, w_y, w_z, a_x, a_y, a_z
    imu = ImuStream(t, (R_BS @ a[:, 3:6][..., None])[..., 0], (R_BS @ a[:, 0:3][..., None])[..., 0], kind="euroc")

    gt_yaml = _read_yaml(mav / "state_groundtruth_estimate0" / "sensor.yaml")
    if not np.allclose(_t_bs(gt_yaml), np.eye(4)):
        raise ValueError(f"{name}: ground truth T_BS is not the identity, which this loader does not handle")
    tg, g = _read_csv(mav / "state_groundtruth_estimate0" / "data.csv", 16)
    gt = GroundTruth(tg, g[:, 0:3], so3.quat_normalize(g[:, 3:7]), g[:, 7:10], g[:, 10:13], g[:, 13:16])
    return Sequence(name, imu, gt, noise_from_sensor(imu_yaml), float(imu_yaml["rate_hz"]))


# ------------------------------------------------------------------------ stillness


def find_still_segments(imu: ImuStream, rate_hz: float, cfg: InitConfig) -> list[StillSegment]:
    """Runs of consecutive ``window_s`` windows that are still by the thresholds in ``cfg``, at least ``min_still_s`` long."""
    n = int(round(cfg.window_s * rate_hz))
    if n < 2:
        raise ValueError("window_s is too short for this rate")
    still = np.array(
        [
            np.all(imu.acc[i : i + n].std(0) < cfg.acc_std_max) and np.all(imu.gyr[i : i + n].std(0) < cfg.gyro_std_max)
            for i in range(0, len(imu) - n + 1, n)
        ]
    )
    segments = []
    i = 0
    while i < len(still):
        if still[i]:
            j = i
            while j < len(still) and still[j]:
                j += 1
            start, end = i * n, min(j * n, len(imu) - 1)
            seg = StillSegment(int(imu.t_ns[start]), int(imu.t_ns[end]))
            if seg.duration_s >= cfg.min_still_s:
                segments.append(seg)
            i = j
        else:
            i += 1
    return segments


def select_still_segment(seq: Sequence, cfg: InitConfig) -> StillSegment:
    """The first still segment ending inside ground-truth coverage (the heading and position come from ground truth there)."""
    for s in find_still_segments(seq.imu, seq.rate_hz, cfg):
        if seq.gt.t_ns[0] <= s.t_end_ns <= seq.gt.t_ns[-1]:
            return s
    raise ValueError(f"{seq.name}: no still segment of at least {cfg.min_still_s} s inside ground-truth coverage")


def static_init_tilt_corrected(
    imu: ImuStream, seg: StillSegment, q_ref: np.ndarray, p0: np.ndarray
) -> tuple[NominalState, float]:
    """Initial state from a still segment; returns the state and the tilt correction applied [rad].

    Starts at the reference attitude ``q_ref`` (heading from ground truth) and applies the smallest rotation that takes
    the body-frame averaged specific force onto world +z. Unlike ZYX roll/pitch, this has no singularity at pitch = ±90°,
    where this IMU sits. ``b_a = 0`` (not separable from tilt while still), ``b_g`` = gyro mean, ``v = 0``.
    """
    m = (imu.t_ns >= seg.t_start_ns) & (imu.t_ns < seg.t_end_ns)
    a = np.mean(imu.acc[m], axis=0)
    w = np.mean(imu.gyr[m], axis=0)
    u_W = so3.quat_to_rot(q_ref) @ (a / np.linalg.norm(a))
    z = np.array([0.0, 0.0, 1.0])
    axis = np.cross(u_W, z)
    sin_a, cos_a = np.linalg.norm(axis), float(u_W @ z)
    angle = float(np.arctan2(sin_a, cos_a))
    if sin_a < 1e-12:
        q = so3.quat_normalize(q_ref.copy())
    else:
        q = so3.quat_normalize(so3.quat_mul(so3.exp(axis / sin_a * angle), q_ref))
    return NominalState(np.asarray(p0, dtype=float).copy(), np.zeros(3), q, np.zeros(3), w), angle


# ------------------------------------------------------------------- fixes and gaps


def gt_position_at(gt: GroundTruth, t_ns: np.ndarray) -> np.ndarray:
    """Ground-truth position linearly interpolated at ``t_ns`` (clipped to the ground-truth span)."""
    t = np.asarray(t_ns, dtype=np.int64)
    # relative to the first sample, so the interpolation variable fits a double exactly
    x = (t - gt.t_ns[0]).astype(float)
    xp = (gt.t_ns - gt.t_ns[0]).astype(float)
    return np.column_stack([np.interp(x, xp, gt.p[:, i]) for i in range(3)])


def synthesize_fixes(seq: Sequence, k0: int, cfg: FixConfig, seq_index: int) -> list[PositionFix]:
    """Position fixes from ground truth plus white noise, every ``rate/fix_rate`` IMU samples starting at IMU sample ``k0``.

    The fix times are IMU timestamps, as ``eskf.run`` requires: ground truth is interpolated there (it is within 256 ns of
    an IMU sample in EuRoC). These are synthetic fixes: their only error is the injected noise, which is the point to keep
    in mind when reading the ATE.
    """
    step = seq.rate_hz / cfg.rate_hz
    if abs(step - round(step)) > 1e-9:
        raise ValueError(f"fix rate {cfg.rate_hz} Hz does not divide the IMU rate {seq.rate_hz} Hz")
    k = np.arange(k0, len(seq.imu), int(round(step)))
    t = seq.imu.t_ns[k]
    t = t[t <= seq.gt.t_ns[-1]]
    rng = np.random.default_rng(np.random.SeedSequence(cfg.seed + seq_index, spawn_key=(FIX_NOISE_STREAM,)))
    p = gt_position_at(seq.gt, t) + cfg.sigma * rng.standard_normal((len(t), 3))
    return [PositionFix(np.int64(ti), pi, cfg.sigma) for ti, pi in zip(t, p, strict=True)]


def outage_windows(t0_ns: int, t_end_ns: int, duration_s: float, warmup_s: float, recovery_s: float) -> list[tuple[int, int]]:
    """Reproducible ``[start, end)`` outage windows in ns: first at ``t0 + warmup``, then every ``duration + recovery``."""
    out = []
    dur, gap = int(round(duration_s * NS_PER_S)), int(round(recovery_s * NS_PER_S))
    start = t0_ns + int(round(warmup_s * NS_PER_S))
    while start + dur + gap <= t_end_ns:
        out.append((start, start + dur))
        start += dur + gap
    return out


def drop_fixes(fixes: list[PositionFix], window: tuple[int, int]) -> list[PositionFix]:
    return [f for f in fixes if not window[0] <= int(f.t_ns) < window[1]]


# ------------------------------------------------------------------------- run setup


@dataclass
class RunSetup:
    """Everything a run of one sequence needs: the filter inputs and the ground truth it is scored against."""

    seq: Sequence
    seq_index: int
    still: StillSegment
    k0: int  # first IMU sample the filter uses (t ≥ end of the still segment)
    imu: ImuStream  # imu samples k0 ... up to and including the last sample ≤ ground-truth end
    fixes: list[PositionFix]
    x0: NominalState
    P0: np.ndarray
    tilt_correction_rad: float
    cfg: EskfConfig


def build_P0(x0: NominalState, T_s: float, noise: ImuNoise, init: InitConfig, sigma_p: float, g: float) -> np.ndarray:
    """``static_init_P0`` with the gyro bias replaced by ``sigma_bg0``.

    ``static_init_P0`` only knows the averaging noise (≈1e-4 rad/s here). The still-time gyro means differ from the
    ground-truth batch biases by up to 1.4e-3 rad/s, so the averaging noise would be overconfident on this data.
    """
    P = static_init_P0(x0, T_s, noise, init.sigma_ba0, init.sigma_yaw, sigma_p, init.sigma_v, g)
    P[IBG, IBG] = init.sigma_bg0**2 * np.eye(3)
    return P


def prepare_run(seq: Sequence, seq_index: int, init: InitConfig, fixes_cfg: FixConfig, cfg: EskfConfig) -> RunSetup:
    """Pick the still segment, build ``x0``/``P0``, and synthesize the fixes for the whole sequence."""
    still = select_still_segment(seq, init)
    k0 = int(np.searchsorted(seq.imu.t_ns, still.t_end_ns))
    last = int(np.searchsorted(seq.imu.t_ns, seq.gt.t_ns[-1], side="right"))
    imu = ImuStream(seq.imu.t_ns[k0:last].copy(), seq.imu.acc[k0:last].copy(), seq.imu.gyr[k0:last].copy(), kind="euroc")

    k_ref = int(np.clip(np.searchsorted(seq.gt.t_ns, still.t_end_ns), 0, len(seq.gt.t_ns) - 1))
    rng = np.random.default_rng(np.random.SeedSequence(fixes_cfg.seed + seq_index, spawn_key=(P0_STREAM,)))
    p0 = seq.gt.p[k_ref] + fixes_cfg.sigma * rng.standard_normal(3)
    x0, tilt = static_init_tilt_corrected(seq.imu, still, seq.gt.q[k_ref], p0)
    P0 = build_P0(x0, still.duration_s, seq.noise, init, fixes_cfg.sigma, cfg.g)
    fixes = synthesize_fixes(seq, k0, fixes_cfg, seq_index)
    return RunSetup(seq, seq_index, still, k0, imu, fixes, x0, P0, tilt, cfg)


# --------------------------------------------------------------------------- checks


def data_report(seq: Sequence, init: InitConfig, g_nominal: float = G_MAGNITUDE) -> dict:
    """Measured facts about the raw sequence (conventions and assumptions, checked rather than assumed)."""
    imu, gt = seq.imu, seq.gt
    dt_imu = np.diff(imu.t_ns)
    # ground truth against the nearest IMU sample
    i = np.clip(np.searchsorted(imu.t_ns, gt.t_ns), 1, len(imu) - 1)
    j = np.where(np.abs(imu.t_ns[i - 1] - gt.t_ns) < np.abs(imu.t_ns[i] - gt.t_ns), i - 1, i)
    gt_offset_ns = int(np.max(np.abs(gt.t_ns - imu.t_ns[j])))

    # body rate from ground truth vs the (bias-free) gyro, averaged over the same 5 ms
    dt = np.diff(gt.t_ns) * 1e-9
    dq = so3.quat_mul(so3.quat_conj(gt.q[:-1]), gt.q[1:])
    w_gt = so3.log(dq) / dt[..., None]
    x = (gt.t_ns - imu.t_ns[0]).astype(float)
    xi = (imu.t_ns - imu.t_ns[0]).astype(float)
    w_m = np.column_stack([np.interp(x, xi, imu.gyr[:, c]) for c in range(3)]) - gt.bg
    w_mid = 0.5 * (w_m[:-1] + w_m[1:])
    H = w_mid.T @ w_gt
    U, _, Vt = np.linalg.svd(H)
    D = np.diag([1.0, 1.0, float(np.linalg.det(Vt.T @ U.T))])
    gyro_rotation = so3.log(so3.rot_to_quat(Vt.T @ D @ U.T))

    # gravity implied by ground truth: g_W = dv/dt - R(a_m - b_a), averaged over the whole sequence
    R = so3.quat_to_rot(gt.q)
    a_m = np.column_stack([np.interp(x, xi, imu.acc[:, c]) for c in range(3)]) - gt.ba
    f_W = (R @ a_m[..., None])[..., 0]
    g_implied = np.mean(np.diff(gt.v, axis=0) / dt[..., None] - 0.5 * (f_W[:-1] + f_W[1:]), axis=0)

    still = find_still_segments(imu, seq.rate_hz, init)
    t0 = int(imu.t_ns[0])
    return {
        "imu_samples": len(imu),
        "imu_duration_s": (int(imu.t_ns[-1]) - t0) / NS_PER_S,
        "imu_dt_min_ms": float(dt_imu.min()) * 1e-6,
        "imu_dt_max_ms": float(dt_imu.max()) * 1e-6,
        "imu_gaps": int(np.sum(dt_imu > 1.5 * np.median(dt_imu))),
        "gt_samples": len(gt.t_ns),
        "gt_gaps": int(np.sum(np.diff(gt.t_ns) > 1.5 * np.median(np.diff(gt.t_ns)))),
        "gt_starts_after_imu_s": (int(gt.t_ns[0]) - t0) / NS_PER_S,
        "gt_ends_before_imu_s": (int(imu.t_ns[-1]) - int(gt.t_ns[-1])) / NS_PER_S,
        "gt_to_imu_timestamp_max_ns": gt_offset_ns,
        "gyro_vs_gt_rotation_deg": float(np.degrees(np.linalg.norm(gyro_rotation))),
        "gyro_vs_gt_rate_rms_rad_s": float(np.sqrt(np.mean((w_mid - w_gt) ** 2))),
        "g_implied": float(np.linalg.norm(g_implied)),
        "g_nominal": g_nominal,
        "gravity_tilt_deg": float(np.degrees(np.arctan2(np.hypot(g_implied[0], g_implied[1]), -g_implied[2]))),
        "gt_bias_ba": np.mean(gt.ba, axis=0).tolist(),
        "gt_bias_bg": np.mean(gt.bg, axis=0).tolist(),
        "still_segments_s": [[(s.t_start_ns - t0) / NS_PER_S, (s.t_end_ns - t0) / NS_PER_S] for s in still],
        "begins_still": bool(still) and still[0].t_start_ns == t0,
    }
