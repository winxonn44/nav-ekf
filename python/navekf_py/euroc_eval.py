"""E4 evaluation: run a :class:`~navekf_py.euroc.RunSetup` through either filter, score it, and call evo.

``run_filter`` returns the per-step log as one float array with the columns of ``run_dataset`` (conventions §8):
``t_ns, kind, p (3), v (3), q (4), b_a (3), b_g (3), diag P (15), nis, accepted``. evo is imported only by
:func:`evo_metrics`, so the rest of the package and its tests do not need it (see ``[project.optional-dependencies]``).
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import eskf, parity, so3
from .euroc import NS_PER_S, GroundTruth, RunSetup, gt_position_at
from .io import write_tum
from .propagate import propagate
from .types import ImuSample, NominalState, PositionFix

# columns of the log array
C_T, C_KIND, C_P, C_V, C_Q, C_BA, C_BG, C_PD, C_NIS, C_ACC = (
    0, 1, slice(2, 5), slice(5, 8), slice(8, 12), slice(12, 15), slice(15, 18), slice(18, 33), 33, 34,
)  # fmt: skip


def as_case(setup: RunSetup, fixes: list[PositionFix]) -> parity.Case:
    return parity.Case(setup.imu, fixes, setup.x0, setup.P0, setup.cfg)


def write_inputs(setup: RunSetup, fixes: list[PositionFix], work: Path) -> tuple[Path, Path]:
    """``gps.csv`` and ``config.yaml`` for one run; the shared ``imu.csv`` is written once by :func:`write_imu`."""
    work.mkdir(parents=True, exist_ok=True)
    gps, cfg = work / "gps.csv", work / "config.yaml"
    parity.write_gps_csv(gps, fixes)
    cfg.write_text(parity.config_yaml(as_case(setup, fixes)))
    return gps, cfg


def write_imu(setup: RunSetup, work: Path) -> Path:
    work.mkdir(parents=True, exist_ok=True)
    path = work / "imu.csv"
    if not path.exists():
        parity.write_imu_csv(path, setup.imu)
    return path


def run_cpp(binary: Path, imu_csv: Path, gps_csv: Path, config: Path, work: Path, *, parity_mode: bool = False) -> np.ndarray:
    log, tum = work / "log.csv", work / "traj.tum"
    cmd = [
        str(binary),
        "--imu",
        str(imu_csv),
        "--gps",
        str(gps_csv),
        "--config",
        str(config),
        "--tum",
        str(tum),
        "--log",
        str(log),
    ]
    if parity_mode:
        cmd.append("--parity")
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    rows = np.loadtxt(log, delimiter=",", skiprows=1, ndmin=2)
    log.unlink()
    tum.unlink()
    return rows


def run_python(setup: RunSetup, fixes: list[PositionFix]) -> np.ndarray:
    f = eskf.ESKF(setup.x0, setup.P0, setup.cfg, t0_ns=int(setup.imu.t_ns[0]))
    eskf.run(f, setup.imu, fixes)
    L = f.log.arrays()
    return np.column_stack([L["t_ns"], L["kind"], L["x"], L["P_diag"], L["nis"], L["accepted"]])


def run_filter(setup: RunSetup, fixes: list[PositionFix], work: Path, binary: Path | None) -> np.ndarray:
    """C++ if ``binary`` is given, else the Python reference."""
    if binary is None:
        return run_python(setup, fixes)
    imu = write_imu(setup, work)
    gps, cfg = write_inputs(setup, fixes, work)
    return run_cpp(binary, imu, gps, cfg, work)


def posterior(rows: np.ndarray) -> np.ndarray:
    """One row per IMU time: the last row at that time (the update row where there is one)."""
    t = rows[:, C_T]
    return rows[np.append(t[1:] != t[:-1], True)]


# ------------------------------------------------------------------------ outages


@dataclass
class Outage:
    """One outage window of one run, on the IMU-time grid from the last fix before the outage."""

    duration_s: float
    window: tuple[int, int]
    t_since_fix_s: np.ndarray
    err_norm: np.ndarray  # ‖p̂ − p‖ [m]
    err_axes: np.ndarray  # (K, 3)
    sigma_axes: np.ndarray  # (K, 3) √diag P_pp
    env_norm: np.ndarray  # 3 √tr P_pp [m]
    left_norm_s: float | None  # first time since the fix with ‖e‖ > 3√trP, or None
    left_axis_s: float | None  # first time since the fix with some |e_i| > 3σ_i, or None
    fraction_outside: float  # share of outage steps with some |e_i| > 3σ_i
    last_fix_ns: int
    rejected_before: int  # gate rejections before the outage
    rejected_after: int  # gate rejections after it
    first_accept_after_s: float | None  # time from the end of the window to the first accepted fix; None = locked out
    fixes_after: int
    floor_end_m: float = float("nan")  # truth_start_replay, filled in by the caller


def score_outage(rows: np.ndarray, gt: GroundTruth, duration_s: float, window: tuple[int, int],
                 fixes: list[PositionFix]) -> Outage:  # fmt: skip
    t_fix = np.array([int(f.t_ns) for f in fixes], dtype=np.int64)
    before = t_fix[t_fix < window[0]]
    if len(before) == 0:
        raise ValueError("outage window has no fix before it")
    t_last = int(before[-1])

    post = posterior(rows)
    t = post[:, C_T].astype(np.int64)
    m = (t >= t_last) & (t < window[1])
    tp = t[m]
    e = post[m][:, C_P] - gt_position_at(gt, tp)
    sigma = np.sqrt(post[m][:, C_PD][:, 0:3])
    env = 3.0 * np.sqrt(np.sum(post[m][:, C_PD][:, 0:3], axis=1))
    err = np.linalg.norm(e, axis=1)
    tau = (tp - t_last) / NS_PER_S

    def first(mask: np.ndarray) -> float | None:
        return float(tau[np.argmax(mask)]) if np.any(mask) else None

    out_axis = np.any(np.abs(e) > 3.0 * sigma, axis=1)

    updates = rows[rows[:, C_KIND] == 1]
    t_up = updates[:, C_T].astype(np.int64)
    accepted = updates[:, C_ACC] > 0.5
    after = t_up >= window[1]
    rejected_before = int(np.sum((t_up < window[0]) & ~accepted))
    rejected_after = int(np.sum(after & ~accepted))
    first_acc = t_up[after & accepted]
    first_accept = (int(first_acc[0]) - window[1]) / NS_PER_S if len(first_acc) else None
    return Outage(duration_s, window, tau, err, e, sigma, env, first(err > env), first(out_axis), float(np.mean(out_axis)),
                  t_last, rejected_before, rejected_after, first_accept, int(np.sum(after)))  # fmt: skip


def truth_start_replay(setup: RunSetup, gt: GroundTruth, t_start_ns: int, t_end_ns: int) -> float:
    """Position error [m] of §6 dead reckoning over ``[t_start, t_end]``, started from ground truth with its batch biases.

    The floor for this IMU and integrator: no filter state error is involved. Ground truth is not independent of the IMU
    (its attitude comes from these gyros), so this is a floor against that reference, not a calibration of the sensor.
    """
    imu = setup.imu
    k0, k1 = int(np.searchsorted(imu.t_ns, t_start_ns)), int(np.searchsorted(imu.t_ns, t_end_ns))
    i = int(np.clip(np.searchsorted(gt.t_ns, imu.t_ns[k0]), 0, len(gt.t_ns) - 1))
    x = NominalState(gt_position_at(gt, imu.t_ns[[k0]])[0], gt.v[i].copy(), gt.q[i].copy(), gt.ba[i].copy(), gt.bg[i].copy())
    for k in range(k0, k1):
        sample = ImuSample(imu.t_ns[k], imu.acc[k], imu.gyr[k])
        x = propagate(x, sample, float(imu.t_ns[k + 1] - imu.t_ns[k]) * 1e-9, setup.cfg.g)
    return float(np.linalg.norm(x.p - gt_position_at(gt, imu.t_ns[[k1]])[0]))


def seed_spread(per_seed: list[list[dict]]) -> list[dict]:
    """Summarize the same outage windows run under several fix-noise seeds; ``per_seed[0]`` is the reported baseline.

    Each record has ``sequence, duration_s, end_error_m, left_norm, left_axis, locked``. Returns one row per
    ``(sequence, duration)`` and one pooled row per duration (``sequence = "all"``). The statistic of one seed is the median
    end error over the row's windows, or a count of windows; the row gives seed 0, the median over seeds, and the range.
    """
    keys = sorted({(r["sequence"], r["duration_s"]) for r in per_seed[0]})
    groups = [(lambda r, k=k: (r["sequence"], r["duration_s"]) == k, k) for k in keys]
    groups += [(lambda r, d=d: r["duration_s"] == d, ("all", d)) for d in sorted({k[1] for k in keys})]

    def stat(values: list[float]) -> dict:
        v = np.asarray(values, dtype=float)
        return {"seed0": float(v[0]), "median": float(np.median(v)), "min": float(v.min()), "max": float(v.max())}

    rows = []
    for match, (sequence, duration) in groups:
        chosen = [[r for r in records if match(r)] for records in per_seed]
        if len({len(c) for c in chosen}) != 1:
            raise ValueError("seeds disagree on the number of outage windows")
        row: dict = {"sequence": sequence, "duration_s": duration, "windows": len(chosen[0]), "seeds": len(per_seed)}
        row["end_error_m"] = stat([float(np.median([r["end_error_m"] for r in c])) for c in chosen])
        for name in ("left_norm", "left_axis", "locked"):
            row[name] = stat([sum(bool(r[name]) for r in c) for c in chosen])
        rows.append(row)
    return rows


def nis_summary(rows: np.ndarray) -> dict:
    u = rows[rows[:, C_KIND] == 1]
    return {
        "updates": len(u),
        "rejected": int(np.sum(u[:, C_ACC] < 0.5)),
        "nis_mean": float(np.mean(u[:, C_NIS])),
    }


# ---------------------------------------------------------------------- trajectories


def write_tum_files(rows: np.ndarray, gt: GroundTruth, out: Path) -> tuple[Path, Path]:
    """``estimate.tum`` (posterior at each IMU time, ``w ≥ 0``) and ``groundtruth.tum`` at the same times."""
    out.mkdir(parents=True, exist_ok=True)
    post = posterior(rows)
    t = post[:, C_T].astype(np.int64)
    est, ref = out / "estimate.tum", out / "groundtruth.tum"
    write_tum(est, t, post[:, C_P], so3.quat_canonical(post[:, C_Q]))
    idx = np.clip(np.searchsorted(gt.t_ns, t), 0, len(gt.t_ns) - 1)
    write_tum(ref, t, gt_position_at(gt, t), so3.quat_canonical(gt.q[idx]))
    return ref, est


def evo_metrics(ref_tum: Path, est_tum: Path, rpe_delta_frames: int = 200) -> dict:
    """ATE (SE(3)-aligned and unaligned) and RPE through evo's Python API. Needs ``pip install evo``.

    RPE is over ``rpe_delta_frames`` poses: evo has no seconds unit for RPE, and 200 poses are 1 s at 200 Hz.
    """
    import copy

    from evo.core import metrics, sync
    from evo.core.units import Unit
    from evo.tools import file_interface

    ref, est = sync.associate_trajectories(
        file_interface.read_tum_trajectory_file(str(ref_tum)),
        file_interface.read_tum_trajectory_file(str(est_tum)),
        max_diff=0.001,
    )

    def ape(a, b) -> float:
        m = metrics.APE(metrics.PoseRelation.translation_part)
        m.process_data((a, b))
        return float(m.get_statistic(metrics.StatisticsType.rmse))

    def rpe(a, b, relation) -> float:
        m = metrics.RPE(relation, delta=rpe_delta_frames, delta_unit=Unit.frames, all_pairs=False)
        m.process_data((a, b))
        return float(m.get_statistic(metrics.StatisticsType.rmse))

    aligned = copy.deepcopy(est)
    aligned.align(ref, correct_scale=False)
    return {
        "ate_se3_rmse_m": ape(ref, aligned),
        "ate_unaligned_rmse_m": ape(ref, est),
        "rpe_trans_rmse_m": rpe(ref, est, metrics.PoseRelation.translation_part),
        "rpe_rot_rmse_deg": rpe(ref, est, metrics.PoseRelation.rotation_angle_deg),
        "rpe_delta_frames": rpe_delta_frames,
        "poses": ref.num_poses,
    }
