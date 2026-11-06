"""E1 figure: dead-reckoning drift, Monte Carlo RMS vs derivation §7 theory.

    python scripts/plot_drift.py [--runs 100] [--out docs/figures/e1_drift.png]

Still, level IMU, 60 s at 100 Hz, one error source at a time (navekf_py.drift).
Prints a Markdown table of RMS/theory ratios and fitted slopes for docs/results.md.
Takes ~2 s.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from navekf_py import drift  # noqa: E402

# Validated categorical slots 1-4 (light surface), in fixed order.
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e6e5e0"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=ROOT / "docs" / "figures" / "e1_drift.png")
    args = ap.parse_args()

    results = [drift.run(s, n_runs=args.runs, seed=args.seed) for s in drift.SLOPES]

    fig, ax = plt.subplots(figsize=(7.5, 5.0), dpi=150)
    for r, c in zip(results, COLORS, strict=True):
        m = r.t >= 0.1
        ax.plot(r.t[m], r.rms[m], color=c, lw=2, label=f"{drift.LABELS[r.source]} (MC)")
        ax.plot(r.t[m], r.theory[m], color=c, lw=1.2, ls=(0, (4, 3)), alpha=0.9)
        slope = drift.loglog_slope(r.t, r.rms)
        dy = {"accel_bias": 6, "gyro_noise": -6}.get(r.source, 0)  # keep the two close labels apart
        ax.annotate(f"slope {slope:.2f}", (r.t[-1], r.rms[-1]), xytext=(6, dy), textcoords="offset points",
                    va="center", fontsize=8.5, color=INK)  # fmt: skip
    ax.plot([], [], color=MUTED, lw=1.2, ls=(0, (4, 3)), label="theory (derivation §7)")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.1, 120)
    ax.set_xlabel("time [s]", color=INK)
    ax.set_ylabel("horizontal RMS position error, per axis [m]", color=INK)
    ax.set_title(f"E1 dead-reckoning drift: {args.runs} Monte Carlo runs per source, 100 Hz, still and level IMU",
                 fontsize=10, color=INK, loc="left")  # fmt: skip
    ax.grid(True, which="major", color=GRID, lw=0.8)
    ax.tick_params(colors=MUTED, labelcolor=INK)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    print(f"wrote {args.out.relative_to(ROOT)}\n")

    print("| source | value | RMS/theory at 10 s | at 60 s | min–max (t ≥ 1 s) | slope | theory |")
    print("|---|---|---|---|---|---|---|")
    for r in results:
        i10, i60 = np.searchsorted(r.t, 10.0), len(r.t) - 1
        ratio = r.rms[r.t >= 1.0] / r.theory[r.t >= 1.0]
        print(
            f"| {r.source} | {r.value:g} | {r.rms[i10] / r.theory[i10]:.3f} | {r.rms[i60] / r.theory[i60]:.3f} "
            f"| {ratio.min():.3f}–{ratio.max():.3f} | {drift.loglog_slope(r.t, r.rms):.3f} | {drift.SLOPES[r.source]} |"
        )


if __name__ == "__main__":
    main()
