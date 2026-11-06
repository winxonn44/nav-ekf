"""E2: NEES/NIS consistency over a batched Monte Carlo, outage figure, ablations.

    python scripts/run_consistency.py [--runs 50] [--seed 0] [--batches 10] [--no-ablations]

Config: configs/e2_consistency.yaml (figure-8, 120 s at 200 Hz, 10 Hz GPS, one
30 s dropout [60, 90) s). Consistent IMU stream (conventions §7). Run i uses
seed + i. Writes docs/figures/e2_nees_nis.png and docs/figures/e2_outage.png and
prints the Markdown tables for docs/results.md. About 6 s per 50-run batch with
the default filter (Van Loan is slower). Everything (ablations, plus 10 batches
with the gate on and off) takes ~4 min on an M-series Air.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from navekf_py import consistency as c  # noqa: E402
from navekf_py import sim  # noqa: E402

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # E1's validated categorical slots 1-4
INK, MUTED, GRID, BAND, SHADE = "#1f1f1e", "#6b6a64", "#e6e5e0", "#d6e6f8", "#efeeea"


def style(ax):
    ax.grid(True, color=GRID, lw=0.8)
    ax.tick_params(colors=MUTED, labelcolor=INK)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)


def plot_nees_nis(r: c.McResult, s: dict, cfg: sim.SimConfig, out: Path) -> None:
    t, tf = r.t_tick_ns * 1e-9, r.t_fix_ns * 1e-9
    nis_m = r.nis.mean(1)
    gap = np.flatnonzero(np.diff(tf) > 1.5 / cfg.gps.rate_hz) + 1  # break the NIS line across dropouts
    tf, nis_m = np.insert(tf, gap, np.nan), np.insert(nis_m, gap, np.nan)
    fig, axes = plt.subplots(2, 1, figsize=(8.0, 6.0), dpi=150, sharex=True)
    for ax, x, y, band, dof, name in (
        (axes[0], t, r.nees.mean(1), s["nees_band"], 15, "NEES (15-dim)"),
        (axes[1], tf, nis_m, s["nis_band"], 3, "NIS (3-dim)"),
    ):
        for a, b in cfg.gps.dropouts:
            ax.axvspan(a, b, color=SHADE, lw=0, zorder=0)
        ax.axhspan(*band, color=BAND, lw=0, zorder=1, label=f"95% band, N={s['n_runs']}: [{band[0]:.2f}, {band[1]:.2f}]")
        ax.axhline(dof, color=MUTED, lw=1, ls=(0, (4, 3)), zorder=2)
        ax.plot(x, y, color=COLORS[0], lw=1.4, zorder=3, label=f"mean over {s['n_runs']} runs")
        ax.set_ylabel(f"average {name}", color=INK)
        inside = s["nees_in_band"] if dof == 15 else s["nis_in_band"]
        mean = s["nees_mean"] if dof == 15 else s["nis_mean"]
        ax.set_title(
            f"{name}: time-average {mean:.2f} (expected {dof}), {100 * inside:.0f}% of time in band",
            fontsize=9.5,
            color=INK,
            loc="left",
        )
        ax.legend(frameon=False, fontsize=8, loc="upper right")
        style(ax)
    a, b = cfg.gps.dropouts[0]
    axes[1].text((a + b) / 2, 3.0, f"GPS dropout\n{b - a:.0f} s, no fixes", ha="center", va="center", fontsize=8, color=MUTED)
    axes[1].set_xlabel("time [s]", color=INK)
    fig.suptitle(
        "E2 ESKF consistency: figure-8, consistent IMU stream, 10 Hz GPS (σ = 1 m)", fontsize=10.5, color=INK, x=0.01, ha="left"
    )
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def outage_stats(r: c.McResult, cfg: sim.SimConfig):
    a, b = cfg.gps.dropouts[0]
    t = r.t_tick_ns * 1e-9
    t_last = r.t_fix_ns[r.t_fix_ns * 1e-9 < a].max() * 1e-9
    sel = (t >= t_last - 1e-9) & (t < b - 1e-9)
    return t[sel] - t_last, r.err[sel][:, :, 0:3], r.sigma[sel][:, :, 0:3], sel


def plot_outage(r: c.McResult, cfg: sim.SimConfig, out: Path) -> None:
    dt, e, sg, _ = outage_stats(r, cfg)
    fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.8), dpi=150, sharex=True)
    for i, (ax, name) in enumerate(zip(axes, ("east", "north", "up"), strict=True)):
        ax.plot(dt, e[:, :, i], color=COLORS[0], lw=0.6, alpha=0.35)
        s3 = 3 * sg[:, :, i].mean(1)
        rms = np.sqrt((e[:, :, i] ** 2).mean(1))
        ax.plot(dt, s3, color=COLORS[1], lw=2, label="filter ±3σ (run mean)")
        ax.plot(dt, -s3, color=COLORS[1], lw=2)
        ax.plot(dt, 3 * rms, color=INK, lw=1.2, ls=(0, (4, 3)), label="±3 × empirical RMS")
        ax.plot(dt, -3 * rms, color=INK, lw=1.2, ls=(0, (4, 3)))
        ax.set_title(f"{name} error", fontsize=9.5, color=INK, loc="left")
        ax.set_xlabel("time since last fix [s]", color=INK)
        style(ax)
    axes[0].set_ylabel("position error [m] (each line one run)", color=INK)
    axes[0].legend(frameon=False, fontsize=8, loc="lower left")
    fig.suptitle(f"E2 GPS outage: {r.cfg.n_runs} runs, 30 s without fixes", fontsize=10.5, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def row(name: str, s: dict, r: c.McResult, base: c.McResult | None) -> str:
    t = r.t_tick_ns * 1e-9
    early = r.nees.mean(1)[t < 10].mean()
    d = "–" if base is None else f"{np.max(np.abs(r.nees.mean(1) - base.nees.mean(1))):.3g}"
    return (
        f"| {name} | {s['nees_mean']:.2f} | {100 * s['nees_in_band']:.0f}% | {early:.1f} | {s['nees_max']:.1f} | "
        f"{s['nis_mean']:.2f} | {100 * s['nis_in_band']:.0f}% | {100 * s['reject_rate']:.2f}% | {d} | {s['runtime_s']:.0f} s |"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batches", type=int, default=10, help="extra independent 50-run batches (seed + 50·j)")
    ap.add_argument("--no-ablations", action="store_true")
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "e2_consistency.yaml")
    args = ap.parse_args()
    cfg = sim.load_config(args.config)
    figs = ROOT / "docs" / "figures"
    figs.mkdir(parents=True, exist_ok=True)

    base_cfg = c.McConfig(sim=cfg, n_runs=args.runs, seed=args.seed)
    r = c.monte_carlo(base_cfg)
    s = c.summarize(r)
    plot_nees_nis(r, s, cfg, figs / "e2_nees_nis.png")
    plot_outage(r, cfg, figs / "e2_outage.png")
    print("wrote docs/figures/e2_nees_nis.png, docs/figures/e2_outage.png\n")
    print(f"headline (seed {args.seed}, N={args.runs}): " + ", ".join(f"{k}={v}" for k, v in s.items()) + "\n")

    # outage numbers
    dt, e, sg, sel = outage_stats(r, cfg)
    print("| time since last fix | RMS err E / N / U [m] | filter σ E / N / U [m] | runs inside ±3σ (all axes) | NEES (mean) |")
    print("|---|---|---|---|---|")
    nees_m = r.nees.mean(1)[sel]
    for target in (0.0, 10.0, 20.0, 30.0):
        i = min(int(np.argmin(np.abs(dt - target))), len(dt) - 1)
        rms = np.sqrt((e[i] ** 2).mean(0))
        sig = sg[i].mean(0)
        inside = np.mean(np.all(np.abs(e[i]) <= 3 * sg[i], axis=1))
        print(
            f"| {dt[i]:.1f} s | {' / '.join(f'{v:.2f}' for v in rms)} | {' / '.join(f'{v:.2f}' for v in sig)} "
            f"| {100 * inside:.0f}% | {nees_m[i]:.1f} |"
        )
    print()

    if not args.no_ablations:
        print(
            "| variant | NEES mean | NEES in band | NEES mean t<10 s | NEES max | NIS mean | NIS in band | rejected | max |ΔNEES| vs baseline | time |"  # noqa: E501
        )  # noqa: E501
        print("|---|---|---|---|---|---|---|---|---|---|")
        print(row("baseline: first-order, J_r reset, honest P₀", s, r, None))
        variants = [
            ("second-order Φ, Q_d", dict(discretization="second")),
            ("Van Loan (exact Φ, Q_d)", dict(discretization="vanloan")),
            ("identity reset (G_r = I)", dict(reset="identity")),
            ("NIS gate off", dict(gate=None)),
            ("overconfident P₀ (filter P₀ = P₀/100, σ/10)", dict(P0_filter_scale=0.01)),
            ("analytic IMU stream (integrator error unmodeled)", dict(stream="analytic")),
        ]
        for name, kw in variants:
            rv = c.monte_carlo(replace(base_cfg, **kw))
            print(row(name, c.summarize(rv), rv, r))
        print()

    if args.batches:
        print(
            f"| batch (N={args.runs}) | seeds | gate | NEES mean | NEES in band | NIS mean | NIS in band | rejected | runs locked out |"  # noqa: E501
        )
        print("|---|---|---|---|---|---|---|---|---|")
        pooled = {}
        for gate_name, gate in (("on", base_cfg.gate), ("off", None)):
            allnees, allnis = [], []
            for j in range(args.batches):
                sd = args.seed + args.runs * j
                rb = r if (j == 0 and gate_name == "on") else c.monte_carlo(replace(base_cfg, seed=sd, gate=gate))
                sb = c.summarize(rb)
                allnees.append(rb.nees)
                allnis.append(rb.nis)
                locked = int(np.sum((~rb.accepted).sum(0) > 100))  # > 10 s of consecutive-ish rejections
                print(
                    f"| {j} | {sd}–{sd + args.runs - 1} | {gate_name} | {sb['nees_mean']:.2f} | {100 * sb['nees_in_band']:.0f}% "
                    f"| {sb['nis_mean']:.2f} | {100 * sb['nis_in_band']:.0f}% | {100 * sb['reject_rate']:.2f}% | {locked} |"
                )
            pooled[gate_name] = (np.concatenate(allnees, 1).mean(1), np.concatenate(allnis, 1).mean(1))
        Nall = args.runs * args.batches
        b15, b3 = c.chi2_band(15, Nall), c.chi2_band(3, Nall)
        for gate_name, (nees_all, nis_all) in pooled.items():
            print(
                f"\npooled N={Nall}, gate {gate_name}: NEES mean {nees_all.mean():.3f}, band [{b15[0]:.2f}, {b15[1]:.2f}], "
                f"{100 * np.mean((nees_all >= b15[0]) & (nees_all <= b15[1])):.0f}% in band; NIS mean {nis_all.mean():.3f}, "
                f"band [{b3[0]:.2f}, {b3[1]:.2f}], {100 * np.mean((nis_all >= b3[0]) & (nis_all <= b3[1])):.0f}% in band"
            )


if __name__ == "__main__":
    main()
