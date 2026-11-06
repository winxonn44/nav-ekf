"""NEES / NIS consistency and the batched Monte Carlo (docs/derivation.md §8).

Run ``i`` uses simulator seed ``seed + i`` (conventions §8). All runs share one
GPS schedule (same rate and dropouts), so the ``N`` runs are filtered as one
batch: ``P`` has shape ``(N, 15, 15)`` and every step is a handful of batched
NumPy calls. The math per run is exactly the single-run filter's.

NEES is evaluated on the GPS-period grid (every fix time, including those inside
a dropout) on the posterior, i.e. after any update at that time. Truth is
``q, v, b_a, b_g`` from the simulator and, for position, ``p_ref`` on the
consistent stream (the §6 recursion, 3.7e-6 m from the analytic ``p``) or the
analytic ``p`` on the analytic stream.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
from scipy.stats import chi2

from . import sim
from .eskf import ESKF, EskfConfig, boxminus, boxplus, run
from .types import ImuStream, NominalState, PositionFix


def nees(err: np.ndarray, P: np.ndarray) -> np.ndarray:
    """``eᵀ P⁻¹ e`` over the last axis (``solve``, no inverse)."""
    return np.sum(err * np.linalg.solve(P, err[..., None])[..., 0], -1)


def nis(y: np.ndarray, S: np.ndarray) -> np.ndarray:
    return nees(y, S)


def chi2_band(dof: int, n_runs: int, conf: float = 0.95) -> tuple[float, float]:
    """Two-sided band on the mean over ``n_runs`` of a χ²(dof) statistic: ``χ²(dof·N)`` quantiles / N."""
    a = (1.0 - conf) / 2.0
    lo, hi = chi2.ppf([a, 1.0 - a], dof * n_runs) / n_runs
    return float(lo), float(hi)


def P0_from_sigmas(s_p, s_v, s_th, s_ba, s_bg) -> np.ndarray:
    """Diagonal ``P₀`` from per-block std (scalars or 3-vectors), in the order of conventions §5."""
    d = np.concatenate([np.broadcast_to(np.asarray(s, dtype=float), 3) for s in (s_p, s_v, s_th, s_ba, s_bg)])
    return np.diag(d**2)


# honest initial uncertainty used by E2 (roll, pitch, yaw for δθ)
P0_DEFAULT = P0_from_sigmas(2.0, 0.2, [0.02, 0.02, 0.05], 0.05, 2e-3)


@dataclass
class McConfig:
    sim: sim.SimConfig
    n_runs: int = 50
    seed: int = 0
    stream: str = "consistent"  # "consistent" | "analytic"
    P0_true: np.ndarray = field(default_factory=lambda: P0_DEFAULT.copy())  # where the initial error is drawn from
    P0_filter_scale: float = 1.0  # filter starts with P0_true · scale (< 1: overconfident)
    discretization: str = "first"
    reset: str = "jr"
    gate: float | None = EskfConfig.__dataclass_fields__["gate"].default


@dataclass
class McResult:
    cfg: McConfig
    t_tick_ns: np.ndarray  # (T,)
    nees: np.ndarray  # (T, N)
    err: np.ndarray  # (T, N, 15) truth ⊖ estimate
    sigma: np.ndarray  # (T, N, 15) √diag P
    t_fix_ns: np.ndarray  # (M,)
    nis: np.ndarray  # (M, N), before the gate decision
    accepted: np.ndarray  # (M, N)
    n_rejected: int
    runtime_s: float = 0.0


def _stack_runs(cfg: McConfig):
    """Simulate run ``i`` with seed ``seed + i``; stack IMU ``(K, N, 3)``, truth ``(K, N, ·)``, fixes ``(N, 3)``."""
    runs = [sim.simulate(replace(cfg.sim, seed=cfg.seed + i)) for i in range(cfg.n_runs)]
    r0 = runs[0]
    imu_attr = "imu_consistent" if cfg.stream == "consistent" else "imu_analytic"
    acc = np.stack([getattr(r, imu_attr).acc for r in runs], 1)
    gyr = np.stack([getattr(r, imu_attr).gyr for r in runs], 1)
    imu = ImuStream(r0.truth.t_ns, acc, gyr, kind=cfg.stream)
    p = np.stack([r.p_ref_consistent if cfg.stream == "consistent" else r.truth.p for r in runs], 1)
    truth = NominalState(p, *(np.stack([getattr(r.truth, f) for r in runs], 1) for f in ("v", "q", "ba", "bg")))
    t_fix = [int(f.t_ns) for f in r0.gps]
    fixes = [PositionFix(np.int64(t), np.stack([r.gps[m].p_W for r in runs]), r0.gps[m].sigma) for m, t in enumerate(t_fix)]
    return imu, truth, fixes


def monte_carlo(cfg: McConfig) -> McResult:
    import time

    t_start = time.perf_counter()
    imu, truth, fixes = _stack_runs(cfg)
    N = cfg.n_runs
    ecfg = EskfConfig(noise=cfg.sim.imu_noise, g=cfg.sim.g, lever_arm=np.asarray(cfg.sim.gps.lever_arm),
                      discretization=cfg.discretization, reset=cfg.reset, gate=cfg.gate, log=False)  # fmt: skip
    truth0 = NominalState(*(a[0] for a in (truth.p, truth.v, truth.q, truth.ba, truth.bg)))
    # initial error of run i: its own stream, independent of the simulator's two (SeedSequence child 2)
    L = np.linalg.cholesky(cfg.P0_true)
    z = np.stack([np.random.default_rng(np.random.SeedSequence(cfg.seed + i, spawn_key=(2,))).standard_normal(15)
                  for i in range(N)])  # fmt: skip

    x0 = boxplus(truth0, -(z @ L.T))
    f = ESKF(x0, cfg.P0_true * cfg.P0_filter_scale, ecfg, t0_ns=int(imu.t_ns[0]))

    tick_ns = int(round(1e9 / cfg.sim.gps.rate_hz))
    ticks = np.flatnonzero(np.asarray(imu.t_ns) % tick_ns == 0)
    T = len(ticks)
    out_nees, out_err, out_sig = np.empty((T, N)), np.empty((T, N, 15)), np.empty((T, N, 15))
    tick_of = {int(k): i for i, k in enumerate(ticks)}

    def hook(k, e: ESKF, _updates):
        i = tick_of.get(k)
        if i is None:
            return
        xt = NominalState(truth.p[k], truth.v[k], truth.q[k], truth.ba[k], truth.bg[k])
        err = boxminus(xt, e.x)
        out_err[i], out_nees[i] = err, nees(err, e.P)
        out_sig[i] = np.sqrt(np.diagonal(e.P, axis1=-2, axis2=-1))

    ups = run(f, imu, fixes, hook)
    return McResult(
        cfg=cfg,
        t_tick_ns=np.asarray(imu.t_ns)[ticks],
        nees=out_nees,
        err=out_err,
        sigma=out_sig,
        t_fix_ns=np.array([int(u.t_ns) for u in ups], dtype=np.int64),
        nis=np.stack([u.nis for u in ups]),
        accepted=np.stack([u.accepted for u in ups]),
        n_rejected=len(f.rejections),
        runtime_s=time.perf_counter() - t_start,
    )


def summarize(r: McResult, conf: float = 0.95) -> dict:
    """Run-averaged NEES/NIS, their bands for this ``N``, and the share of time inside."""
    N = r.cfg.n_runs
    nees_m, nis_m = r.nees.mean(1), r.nis.mean(1)
    b15, b3 = chi2_band(15, N, conf), chi2_band(3, N, conf)
    return {
        "n_runs": N,
        "nees_band": b15,
        "nis_band": b3,
        "nees_mean": float(nees_m.mean()),
        "nis_mean": float(nis_m.mean()),
        "nees_in_band": float(np.mean((nees_m >= b15[0]) & (nees_m <= b15[1]))),
        "nis_in_band": float(np.mean((nis_m >= b3[0]) & (nis_m <= b3[1]))),
        "nees_max": float(nees_m.max()),
        "reject_rate": float(1.0 - r.accepted.mean()),
        "runtime_s": r.runtime_s,
    }
