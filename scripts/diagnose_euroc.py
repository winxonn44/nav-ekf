"""E4 diagnostics behind the outage section of docs/results.md (not part of the results table; needs no evo).

    python -I scripts/diagnose_euroc.py [--data-root ~/Documents/Code/datasets/euroc] [--binary build-release/run_dataset]

Prints, for every outage window of every sequence in configs/euroc.yaml:
* in-flight IMU noise amplitude against the sensor.yaml density (5-40 Hz band);
* how far the filter's own state is from ground truth, in its own sigmas, at the last fix before the outage;
* end-of-outage error of §6 dead reckoning under substitutions: filter state, then the state with ground-truth biases,
  ground-truth velocity and attitude, and everything from ground truth; plus a midpoint integrator and a run that uses the
  ground-truth attitude, both from ground truth.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy.signal import detrend, welch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))
import run_euroc as R  # noqa: E402
from navekf_py import euroc, so3  # noqa: E402
from navekf_py import euroc_eval as ev  # noqa: E402
from navekf_py.propagate import propagate  # noqa: E402
from navekf_py.types import ImuSample, NominalState  # noqa: E402


def band_asd(x: np.ndarray, fs: float, lo: float, hi: float) -> float:
    n = min(len(x), 1024)
    f, p = welch(detrend(x, axis=0), fs=fs, nperseg=n, axis=0)
    m = (f >= lo) & (f < hi)
    return float(np.sqrt(np.mean(p[m])))


def noise_table(seq: euroc.Sequence, init: euroc.InitConfig) -> str:
    still = euroc.select_still_segment(seq, init)
    t = (seq.imu.t_ns - seq.imu.t_ns[0]) * 1e-9
    s = (seq.imu.t_ns >= still.t_start_ns) & (seq.imu.t_ns < still.t_end_ns)
    fly = (t > 60.0) & (t < 90.0)
    fs = seq.rate_hz
    return (
        f"| {seq.name} | {band_asd(seq.imu.acc[s], fs, 5, 40):.1e} | {band_asd(seq.imu.acc[fly], fs, 5, 40):.1e} | {seq.noise.sigma_a:.1e} "  # noqa: E501
        f"| {band_asd(seq.imu.gyr[s], fs, 5, 40):.1e} | {band_asd(seq.imu.gyr[fly], fs, 5, 40):.1e} | {seq.noise.sigma_g:.1e} |"
    )


def truth_replays(setup: euroc.RunSetup, seq: euroc.Sequence, k0: int, k1: int, g: float) -> tuple[float, float]:
    """End error [m] over samples ``k0..k1`` from ground truth: midpoint integrator, §6 with ground-truth attitude."""
    gt, imu = seq.gt, setup.imu
    gi = np.clip(np.searchsorted(gt.t_ns, imu.t_ns), 0, len(gt.t_ns) - 1)
    a = imu.acc - gt.ba[gi]
    w = imu.gyr - gt.bg[gi]
    Rg = so3.quat_to_rot(gt.q)
    q2, v2, p2 = gt.q[gi[k0]].copy(), gt.v[gi[k0]].copy(), gt.p[gi[k0]].copy()
    v3, p3 = v2.copy(), p2.copy()
    gv = np.array([0.0, 0.0, -g])
    for k in range(k0, k1):
        dt = float(imu.t_ns[k + 1] - imu.t_ns[k]) * 1e-9
        wm, am = 0.5 * (w[k] + w[k + 1]), 0.5 * (a[k] + a[k + 1])
        qm = so3.quat_normalize(so3.quat_mul(q2, so3.exp(wm * dt * 0.5)))
        ag = so3.quat_to_rot(qm) @ am + gv
        p2 = p2 + v2 * dt + 0.5 * ag * dt * dt
        v2 = v2 + ag * dt
        q2 = so3.quat_normalize(so3.quat_mul(q2, so3.exp(wm * dt)))
        ag3 = Rg[gi[k]] @ a[k] + gv
        p3 = p3 + v3 * dt + 0.5 * ag3 * dt * dt
        v3 = v3 + ag3 * dt
    end = gt.p[gi[k1]]
    return float(np.linalg.norm(p2 - end)), float(np.linalg.norm(p3 - end))


def diagnose_window(setup: euroc.RunSetup, seq: euroc.Sequence, duration: float, window: tuple[int, int], g: float,
                    binary: Path, work: Path) -> tuple[np.ndarray, np.ndarray]:  # fmt: skip
    """``(z, end)``: the worst-axis normalized errors at the last fix (v, attitude, b_a, b_g) and the replay end errors."""
    fixes = euroc.drop_fixes(setup.fixes, window)
    rows = ev.run_filter(setup, fixes, work, binary)
    o = ev.score_outage(rows, seq.gt, duration, window, fixes)
    post = ev.posterior(rows)
    x = post[int(np.searchsorted(post[:, ev.C_T].astype(np.int64), o.last_fix_ns))]
    gt = seq.gt
    i = min(int(np.searchsorted(gt.t_ns, o.last_fix_ns)), len(gt.t_ns) - 1)
    d = np.sqrt(x[ev.C_PD])
    dth = so3.log(so3.quat_mul(so3.quat_conj(gt.q[i]), x[ev.C_Q]))
    z = np.array([
        np.max(np.abs(x[ev.C_V] - gt.v[i]) / d[3:6]),
        np.max(np.abs(dth) / d[6:9]),
        np.max(np.abs(x[ev.C_BA] - gt.ba[i]) / d[9:12]),
        np.max(np.abs(x[ev.C_BG] - gt.bg[i]) / d[12:15]),
    ])  # fmt: skip

    imu = setup.imu
    k0, k1 = int(np.searchsorted(imu.t_ns, o.last_fix_ns)), int(np.searchsorted(imu.t_ns, window[1]))
    n = 5
    P, V, Q = np.tile(x[ev.C_P], (n, 1)), np.tile(x[ev.C_V], (n, 1)), np.tile(x[ev.C_Q], (n, 1))
    BA, BG = np.tile(x[ev.C_BA], (n, 1)), np.tile(x[ev.C_BG], (n, 1))
    BA[1], BG[2] = gt.ba[i], gt.bg[i]
    V[3], Q[3] = gt.v[i], gt.q[i]
    P[4], V[4], Q[4], BA[4], BG[4] = gt.p[i], gt.v[i], gt.q[i], gt.ba[i], gt.bg[i]
    st = NominalState(P, V, Q, BA, BG)
    for k in range(k0, k1):
        st = propagate(st, ImuSample(imu.t_ns[k], imu.acc[k], imu.gyr[k]), float(imu.t_ns[k + 1] - imu.t_ns[k]) * 1e-9, g)
    end = euroc.gt_position_at(gt, imu.t_ns[[k1]])[0]
    e = np.linalg.norm(st.p - end, axis=1)
    return z, np.append(e, truth_replays(setup, seq, k0, k1, g))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, default=R.DEFAULT_DATA)
    ap.add_argument("--binary", type=Path, default=ROOT / "build-release" / "run_dataset")
    args = ap.parse_args()
    s = R.load_settings(ROOT / "configs" / "euroc.yaml")

    print("| sequence | accel ASD, still | accel ASD, flight | sensor.yaml | gyro ASD, still | gyro ASD, flight | sensor.yaml |")
    print("|---|---|---|---|---|---|---|")
    zs: list[np.ndarray] = []
    ends: dict[float, list[np.ndarray]] = {d: [] for d in s.durations}
    work = Path(tempfile.mkdtemp())
    for i, name in enumerate(s.sequences):
        seq = euroc.load_sequence(args.data_root, name)
        print(noise_table(seq, s.init))
        setup = euroc.prepare_run(seq, i, s.init, s.fixes, R.eskf_config(s, seq.noise))
        for d in s.durations:
            for w in euroc.outage_windows(int(setup.imu.t_ns[0]), int(setup.imu.t_ns[-1]), d, s.warmup_s, s.recovery_s):
                z, e = diagnose_window(setup, seq, d, w, s.g, args.binary, work / name)
                zs.append(z)
                ends[d].append(e)
    shutil.rmtree(work)

    z = np.array(zs)
    print(f"\nWorst-axis |error| / sigma at the last fix, {len(z)} windows, ground truth as the reference:")
    for j, label in enumerate(["velocity", "attitude", "accel bias", "gyro bias"]):
        print(
            f"  {label}: median {np.median(z[:, j]):.1f}, 90th percentile {np.percentile(z[:, j], 90):.1f}, max {z[:, j].max():.1f}"  # noqa: E501
        )

    labels = [
        "filter state",
        "filter state, GT b_a",
        "filter state, GT b_g",
        "GT v and q, filter biases",
        "all from GT (§6)",
        "all from GT, midpoint",
        "all from GT, GT attitude",
    ]
    print("\nMedian end-of-outage error [m] of the dead-reckoning replays:")
    print("| replay | " + " | ".join(f"{d:.0f} s" for d in s.durations) + " |")
    print("|---|" + "---|" * len(s.durations))
    for j, label in enumerate(labels):
        print(f"| {label} | " + " | ".join(f"{np.median(np.array(ends[d])[:, j]):.1f}" for d in s.durations) + " |")


if __name__ == "__main__":
    main()
