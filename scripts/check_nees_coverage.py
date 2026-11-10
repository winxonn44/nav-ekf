"""E2: how often is the 50-run average NEES inside its 95% band, on average over batches?

    python scripts/check_nees_coverage.py [--batches 12] [--runs 50] [--seed 10000]

The band is per time step, so for a consistent filter the expected share of time
inside it is 95% at any fixed N, whatever the correlation over time. (Expectation is
linear: the expected share of time is the mean of the per-step probabilities.)
Correlation only widens the spread between batches, so a single batch can read low,
and the honest statistic is the mean over independent batches, with its standard error.

Runs the gate-off filter (the model under test) and the gate-on filter (the model plus
the 99% gate). About 6 s per 50-run batch per setting on an M-series Air, so about
2.5 min at the defaults. The default seeds are disjoint from the ones in run_consistency.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.stats import chi2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from navekf_py import consistency as c  # noqa: E402
from navekf_py import eskf, sim  # noqa: E402

PHASES = ((0, 10), (10, 60), (60, 90), (90, 120))


def se(x: np.ndarray) -> float:
    return float(np.std(x, ddof=1) / np.sqrt(len(x)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", type=int, default=12)
    ap.add_argument("--runs", type=int, default=50)
    ap.add_argument("--seed", type=int, default=10000)
    a = ap.parse_args()
    cfg = sim.load_config(ROOT / "configs" / "e2_consistency.yaml")
    lo, hi = c.chi2_band(15, a.runs)
    lo3, hi3 = c.chi2_band(3, a.runs)
    print(
        f"{a.batches} batches x {a.runs} runs, seeds from {a.seed}; bands NEES [{lo:.2f}, {hi:.2f}], NIS [{lo3:.2f}, {hi3:.2f}]"
    )
    for name, gate in (("gate off", None), ("gate on", eskf.GATE_99_3DOF)):
        rs = [c.monte_carlo(c.McConfig(sim=cfg, n_runs=a.runs, seed=a.seed + b * a.runs, gate=gate)) for b in range(a.batches)]
        nees = np.stack([r.nees for r in rs])  # (B, T, N)
        nis = np.stack([r.nis for r in rs])
        m, mn = nees.mean(2), nis.mean(2)
        inside = (m >= lo) & (m <= hi)
        fr, frn = inside.mean(1), ((mn >= lo3) & (mn <= hi3)).mean(1)
        print(f"\n{name}")
        print(
            f"  NEES mean {nees.mean():.3f} (se {se(m.mean(1)):.3f}); time in band {fr.mean():.3f} "
            f"(se {se(fr):.3f}, batches {fr.min():.2f}-{fr.max():.2f});"
            f" above {(m > hi).mean():.3f}, below {(m < lo).mean():.3f}"
        )
        print(f"  NIS  mean {nis.mean():.3f} (se {se(mn.mean(1)):.3f}); time in band {frn.mean():.3f} (se {se(frn):.3f})")
        t = rs[0].t_tick_ns * 1e-9
        for t0, t1 in PHASES:
            k = (t >= t0) & (t < t1)
            print(
                f"  t in [{t0:3d}, {t1:3d}): NEES {nees[:, k].mean():.2f} "
                f"(se {se(nees[:, k].mean((1, 2))):.2f}), in band {inside[:, k].mean():.3f}"
            )
        q = chi2.cdf(nees, 15)
        print(
            f"  single-run NEES: P(> chi2 95%) = {(q > 0.95).mean():.4f}, "
            f"P(> chi2 99%) = {(q > 0.99).mean():.4f}, max {nees.max():.0f}"
        )


if __name__ == "__main__":
    main()
