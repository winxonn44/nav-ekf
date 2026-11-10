"""E4: regenerate the EuRoC results table and figures from the extracted data (docs/data.md).

    python -I scripts/run_euroc.py [--data-root ~/Documents/Code/datasets/euroc] [--binary build-release/run_dataset]
                                   [--sequences MH_01_easy MH_02_easy] [--write-results docs/results.md]

``--scale-imu-white F`` multiplies sigma_a and sigma_g by F: a diagnostic ablation for docs/results.md, never the reported
baseline (it refuses ``--write-results``). ``--seed-spread`` also reruns every outage under ``outages.fix_seeds`` fix-noise
seeds (seed 0 is the baseline and must reproduce it exactly).

Config: configs/euroc.yaml. Writes TUM trajectories and results.json to results/euroc/ (git-ignored), the figures to
docs/figures/e4_*.png, and prints the Markdown tables; ``--write-results`` also replaces them between the
``<!-- e4:tables:... -->`` markers in docs/results.md. Runs are sequential: one full run plus one run per outage window for
each sequence, in C++ (under a minute in total on a fanless M4), plus a Python/C++ comparison on the first sequence. Needs evo.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from navekf_py import eskf, euroc, parity  # noqa: E402
from navekf_py import euroc_eval as ev  # noqa: E402
from navekf_py.types import ImuNoise  # noqa: E402

DEFAULT_DATA = Path.home() / "Documents/Code/datasets/euroc"
IX_STATE, IX_P = slice(2, 18), slice(18, 138)
FIG_DIR = ROOT / "docs" / "figures"
SEED_STEP = 100  # spread seeds are seed + j * SEED_STEP, so that sequence i's seed + i never collides across j


@dataclass
class Settings:
    sequences: list[str]
    g: float
    noise: ImuNoise
    fixes: euroc.FixConfig
    init: euroc.InitConfig
    durations: list[float]
    warmup_s: float
    recovery_s: float
    fix_seeds: int
    filter: dict


def load_settings(path: Path) -> Settings:
    c = yaml.safe_load(path.read_text())
    return Settings(
        sequences=c["sequences"],
        g=float(c["g"]),
        noise=ImuNoise(**{k: float(v) for k, v in c["imu_noise"].items()}),
        fixes=euroc.FixConfig(float(c["fixes"]["rate_hz"]), float(c["fixes"]["sigma"]), int(c["fixes"]["seed"])),
        init=euroc.InitConfig(**{k: float(v) for k, v in c["init"].items()}),
        durations=[float(d) for d in c["outages"]["durations_s"]],
        warmup_s=float(c["outages"]["warmup_s"]),
        recovery_s=float(c["outages"]["recovery_s"]),
        fix_seeds=int(c["outages"]["fix_seeds"]),
        filter=c["filter"],
    )


def eskf_config(s: Settings, seq_noise: ImuNoise, scale_white: float = 1.0) -> eskf.EskfConfig:
    if s.noise != seq_noise:
        raise SystemExit(
            f"configs/euroc.yaml imu_noise {s.noise} differs from sensor.yaml {seq_noise}; E4 is the untuned baseline"
        )
    noise = ImuNoise(s.noise.sigma_a * scale_white, s.noise.sigma_g * scale_white, s.noise.sigma_ba, s.noise.sigma_bg)
    gate = s.filter["gate"]
    return eskf.EskfConfig(noise=noise, g=s.g, discretization=s.filter["discretization"], reset=s.filter["reset"],
                           gate=None if gate is None else float(gate), log=True)  # fmt: skip


def run_sequence(seq: euroc.Sequence, index: int, s: Settings, binary: Path, out: Path, work: Path,
                 scale_white: float = 1.0) -> tuple[dict, list[ev.Outage]]:  # fmt: skip
    setup = euroc.prepare_run(seq, index, s.init, s.fixes, eskf_config(s, seq.noise, scale_white))
    report = euroc.data_report(seq, s.init, s.g)
    work = work / seq.name
    rows = ev.run_filter(setup, setup.fixes, work, binary)
    ref, est = ev.write_tum_files(rows, seq.gt, out / seq.name)
    result = {
        "name": seq.name,
        "report": report,
        "still_used_s": [
            (setup.still.t_start_ns - int(seq.imu.t_ns[0])) / 1e9,
            (setup.still.t_end_ns - int(seq.imu.t_ns[0])) / 1e9,
        ],
        "tilt_correction_deg": float(np.degrees(setup.tilt_correction_rad)),
        "filter_start_s": (int(setup.imu.t_ns[0]) - int(seq.imu.t_ns[0])) / 1e9,
        "filter_duration_s": (int(setup.imu.t_ns[-1]) - int(setup.imu.t_ns[0])) / 1e9,
        "fixes": ev.nis_summary(rows),
        "evo": ev.evo_metrics(ref, est),
        "outages": [],
    }
    outages = []
    for duration in s.durations:
        windows = euroc.outage_windows(int(setup.imu.t_ns[0]), int(setup.imu.t_ns[-1]), duration, s.warmup_s, s.recovery_s)
        for window in windows:
            fixes = euroc.drop_fixes(setup.fixes, window)
            rows = ev.run_filter(setup, fixes, work, binary)
            o = ev.score_outage(rows, seq.gt, duration, window, fixes)
            o.floor_end_m = ev.truth_start_replay(setup, seq.gt, o.last_fix_ns, window[1])
            outages.append(o)
    result["outages"] = [
        {
            "duration_s": o.duration_s,
            "floor_end_m": o.floor_end_m,
            "start_s": (o.window[0] - int(seq.imu.t_ns[0])) / 1e9,
            "end_error_m": float(o.err_norm[-1]),
            "end_env_m": float(o.env_norm[-1]),
            "left_norm_s": o.left_norm_s,
            "left_axis_s": o.left_axis_s,
            "fraction_outside": o.fraction_outside,
            "rejected_before": o.rejected_before,
            "rejected_after": o.rejected_after,
            "first_accept_after_s": o.first_accept_after_s,
            "fixes_after": o.fixes_after,
        }
        for o in outages
    ]
    shutil.rmtree(work, ignore_errors=True)
    return result, outages


def outage_records(seq: euroc.Sequence, index: int, s: Settings, binary: Path, work: Path, fix_seed: int) -> list[dict]:
    """Every outage window of one sequence under one fix-noise seed (windows depend only on the config)."""
    setup = euroc.prepare_run(seq, index, s.init, replace(s.fixes, seed=fix_seed), eskf_config(s, seq.noise))
    work = work / f"{seq.name}_seed{fix_seed}"
    records = []
    for duration in s.durations:
        for window in euroc.outage_windows(int(setup.imu.t_ns[0]), int(setup.imu.t_ns[-1]), duration, s.warmup_s, s.recovery_s):
            fixes = euroc.drop_fixes(setup.fixes, window)
            o = ev.score_outage(ev.run_filter(setup, fixes, work, binary), seq.gt, duration, window, fixes)
            records.append({
                "sequence": seq.name,
                "duration_s": duration,
                "end_error_m": float(o.err_norm[-1]),
                "left_norm": o.left_norm_s is not None,
                "left_axis": o.left_axis_s is not None,
                "locked": o.first_accept_after_s is None,
            })  # fmt: skip
    shutil.rmtree(work, ignore_errors=True)
    return records


def parity_check(seq: euroc.Sequence, index: int, s: Settings, binary: Path, work: Path) -> dict:
    """Python vs C++ at every step, full upper triangle of ``P``: the full-fix run and one 30 s outage run."""
    setup = euroc.prepare_run(seq, index, s.init, s.fixes, eskf_config(s, seq.noise))
    window = euroc.outage_windows(int(setup.imu.t_ns[0]), int(setup.imu.t_ns[-1]), 30.0, s.warmup_s, s.recovery_s)[0]
    out = {}
    for label, fixes in (("full", setup.fixes), ("outage_30s", euroc.drop_fixes(setup.fixes, window))):
        case = ev.as_case(setup, fixes)
        d = work / f"parity_{label}"
        parity.write_case(case, d)
        cpp = ev.run_cpp(binary, d / "imu.csv", d / "gps.csv", d / "config.yaml", d, parity_mode=True)
        py = parity.run_python(case)
        if cpp.shape != py.shape or not (
            np.array_equal(cpp[:, 0], py[:, 0]) and np.array_equal(cpp[:, 1], py[:, 1]) and np.array_equal(cpp[:, -1], py[:, -1])
        ):
            raise SystemExit("Python and C++ disagree on rows, timestamps or gate decisions")
        u = py[:, 1] == 1
        out[label] = {
            "rows": len(py),
            "max_abs_dstate": float(np.abs(cpp[:, IX_STATE] - py[:, IX_STATE]).max()),
            "max_abs_dP": float(np.abs(cpp[:, IX_P] - py[:, IX_P]).max()),
            "max_rel_dP": float(np.abs(cpp[:, IX_P] - py[:, IX_P]).max() / np.abs(py[:, IX_P]).max()),
            "max_abs_dNIS": float(np.abs(cpp[u, -2] - py[u, -2]).max()),
            "rejected": int(np.sum(py[u, -1] == 0)),
        }
        shutil.rmtree(d)
    return out


# -------------------------------------------------------------------------- tables


def table_data(results: list[dict]) -> str:
    lines = [
        "| sequence | IMU [s] | GT starts after IMU [s] | begins still | still used [s] | filter runs [s] | tilt correction [°] | gravity tilt of GT frame [°] | implied g |",  # noqa: E501
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        d = r["report"]
        lines.append(
            f"| {r['name']} | {d['imu_duration_s']:.1f} | {d['gt_starts_after_imu_s']:.2f} | {'yes' if d['begins_still'] else 'no'} | "  # noqa: E501
            f"{r['still_used_s'][0]:.1f}–{r['still_used_s'][1]:.1f} | {r['filter_duration_s']:.1f} | "
            f"{r['tilt_correction_deg']:.2f} | {d['gravity_tilt_deg']:.3f} | {d['g_implied']:.4f} |"
        )
    return "\n".join(lines)


def table_accuracy(results: list[dict]) -> str:
    lines = [
        "| sequence | ATE, SE(3) aligned [m] | ATE, no alignment [m] | RPE trans (1 s) [m] | RPE rot (1 s) [°] | fixes | rejected | mean NIS |",  # noqa: E501
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        e, f = r["evo"], r["fixes"]
        lines.append(
            f"| {r['name']} | {e['ate_se3_rmse_m']:.3f} | {e['ate_unaligned_rmse_m']:.3f} | {e['rpe_trans_rmse_m']:.3f} | "
            f"{e['rpe_rot_rmse_deg']:.3f} | {f['updates']} | {f['rejected']} | {f['nis_mean']:.2f} |"
        )
    return "\n".join(lines)


def table_outages(results: list[dict], durations: list[float]) -> str:
    lines = [
        "| sequence | outage [s] | windows | end error, median [m] | end error, max [m] | 3σ radius at end, median [m] | "
        "floor (from truth, no filter), median [m] | left ‖e‖ envelope | left per-axis 3σ | locked out |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        for d in durations:
            w = [o for o in r["outages"] if o["duration_s"] == d]
            if not w:
                lines.append(f"| {r['name']} | {d:.0f} | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |")
                continue
            end = [o["end_error_m"] for o in w]
            env = [o["end_env_m"] for o in w]
            left_n = sum(o["left_norm_s"] is not None for o in w)
            left_a = sum(o["left_axis_s"] is not None for o in w)
            floor = np.median([o["floor_end_m"] for o in w])
            lock = sum(o["first_accept_after_s"] is None for o in w)
            lines.append(
                f"| {r['name']} | {d:.0f} | {len(w)} | {np.median(end):.2f} | {np.max(end):.2f} | {np.median(env):.2f} | {floor:.2f} | "  # noqa: E501
                f"{left_n}/{len(w)} | {left_a}/{len(w)} | {lock}/{len(w)} |"
            )
    return "\n".join(lines)


def table_spread(rows: list[dict]) -> str:
    def count(c: dict) -> str:
        return f"{c['seed0']:.0f} / {c['median']:.1f} / {c['min']:.0f}–{c['max']:.0f}"

    lines = [
        "| sequence | outage [s] | windows | end error, median over windows: seed 0 / median over seeds [m] | "
        "range over seeds [m] | left per-axis 3σ: seed 0 / median / range | "
        "left ‖e‖ envelope: seed 0 / median / range | locked out: seed 0 / median / range |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in sorted(rows, key=lambda x: (x["sequence"] != "all", x["sequence"], x["duration_s"])):
        e = r["end_error_m"]
        name = "**all**" if r["sequence"] == "all" else r["sequence"]
        lines.append(
            f"| {name} | {r['duration_s']:.0f} | {r['windows']} | {e['seed0']:.1f} / {e['median']:.1f} | "
            f"{e['min']:.1f}–{e['max']:.1f} | {count(r['left_axis'])} | {count(r['left_norm'])} | {count(r['locked'])} |"
        )
    return "\n".join(lines)


def table_parity(p: dict, name: str) -> str:
    lines = [f"Parity on {name} (all {p['full']['rows']} log rows of the full-fix run and all rows of one 30 s outage run):", ""]
    lines.append(
        "| run | rows | max \\|Δstate\\| | max \\|ΔP\\| | max \\|ΔP\\| / max \\|P\\| | max \\|ΔNIS\\| | gate rejections |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for k, v in p.items():
        lines.append(
            f"| {k} | {v['rows']} | {v['max_abs_dstate']:.2e} | {v['max_abs_dP']:.2e} | {v['max_rel_dP']:.2e} | {v['max_abs_dNIS']:.2e} | {v['rejected']} |"  # noqa: E501
        )
    return "\n".join(lines)


def replace_marked(path: Path, tables: dict[str, str]) -> None:
    text = path.read_text()
    for key, body in tables.items():
        start, end = f"<!-- e4:tables:{key}:start -->", f"<!-- e4:tables:{key}:end -->"
        if start not in text or end not in text:
            raise SystemExit(f"{path}: missing marker {start}")
        head, rest = text.split(start, 1)
        _, tail = rest.split(end, 1)
        text = f"{head}{start}\n{body}\n{end}{tail}"
    path.write_text(text)


# ------------------------------------------------------------------------- figures


def plot_outage_grid(all_outages: dict[str, list[ev.Outage]], durations: list[float], path: Path) -> None:
    names = list(all_outages)
    ink, faint = "#2a6fbb", "#a9c4e0"
    env_color = "#c0392b"
    fig, axes = plt.subplots(len(names), len(durations), figsize=(11.5, 2.3 * len(names)), squeeze=False, sharex="col")
    for r, name in enumerate(names):
        for c, d in enumerate(durations):
            ax = axes[r, c]
            ws = [o for o in all_outages[name] if o.duration_s == d]
            for i, o in enumerate(ws):
                ax.plot(o.t_since_fix_s, o.err_norm, color=ink if i == 0 else faint, lw=1.2 if i == 0 else 0.9,
                        label="‖e‖ (first window in dark)" if i == 0 else None)  # fmt: skip
                ax.plot(o.t_since_fix_s, o.env_norm, color=env_color, lw=0.9, ls="--", label="3√tr $P_{pp}$" if i == 0 else None)
            if not ws:
                ax.text(0.5, 0.5, "no window fits", transform=ax.transAxes, ha="center", color="0.4")
            ax.set_xlim(0, d)
            ax.set_ylim(bottom=0)
            ax.grid(alpha=0.25)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            if r == 0:
                ax.set_title(f"{d:.0f} s outage", fontsize=11)
            if c == 0:
                ax.set_ylabel(f"{name.replace('_', ' ')}\nposition error [m]", fontsize=9)
            if r == len(names) - 1:
                ax.set_xlabel("time since last fix [s]")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "euroc.yaml")
    ap.add_argument("--binary", type=Path, default=ROOT / "build-release" / "run_dataset")
    ap.add_argument("--sequences", nargs="*", help="default: all in the config")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "euroc")
    ap.add_argument("--write-results", type=Path, default=None, help="replace the marked tables in this file")
    ap.add_argument("--scale-imu-white", type=float, default=1.0, help="diagnostic ablation: multiply sigma_a and sigma_g")
    ap.add_argument("--seed-spread", action="store_true", help="also rerun every outage under outages.fix_seeds fix-noise seeds")
    args = ap.parse_args()
    if args.scale_imu_white != 1.0 and (args.write_results or args.seed_spread):
        raise SystemExit("--scale-imu-white is a diagnostic; it cannot write the reported tables or run the seed spread")

    if not args.binary.exists():
        raise SystemExit(f"C++ binary not found: {args.binary}. Build it (README) or pass --binary.")
    s = load_settings(args.config)
    names = args.sequences or s.sequences
    args.out.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    results, all_outages = [], {}
    parity_result = None
    spread_rows = None
    with tempfile.TemporaryDirectory(dir=args.out) as tmp:
        work = Path(tmp)
        for name in names:
            i = s.sequences.index(name)
            seq = euroc.load_sequence(args.data_root, name)
            r, outages = run_sequence(seq, i, s, args.binary, args.out, work, args.scale_imu_white)
            print(
                f"{name}: filter {r['filter_duration_s']:.1f} s, ATE {r['evo']['ate_se3_rmse_m']:.3f} m, {len(outages)} outage runs",  # noqa: E501
                flush=True,
            )
            results.append(r)
            all_outages[name] = outages
            if parity_result is None and args.scale_imu_white == 1.0:
                parity_result = parity_check(seq, i, s, args.binary, work)
        if args.seed_spread:
            sequences = [(name, s.sequences.index(name), euroc.load_sequence(args.data_root, name)) for name in names]
            per_seed = []
            for j in range(s.fix_seeds):
                fix_seed = s.fixes.seed + j * SEED_STEP
                per_seed.append([r for _, i, seq in sequences for r in outage_records(seq, i, s, args.binary, work, fix_seed)])
                print(f"seed {fix_seed}: {len(per_seed[-1])} outage runs", flush=True)
            baseline = [o["end_error_m"] for r in results for o in r["outages"]]
            if not np.array_equal(sorted(baseline), sorted(r["end_error_m"] for r in per_seed[0])):
                raise SystemExit("seed-spread seed 0 does not reproduce the reported baseline")
            spread_rows = ev.seed_spread(per_seed)
            (args.out / "seed_spread.json").write_text(json.dumps({"rows": spread_rows, "per_seed": per_seed}, indent=1))

    (args.out / "results.json").write_text(json.dumps({"results": results, "parity": parity_result}, indent=1))
    if args.scale_imu_white == 1.0:
        plot_outage_grid(all_outages, s.durations, FIG_DIR / "e4_outage_drift.png")

    tables = {
        "data": table_data(results),
        "accuracy": table_accuracy(results),
        "outages": table_outages(results, s.durations),
    }
    if parity_result is not None:
        tables["parity"] = table_parity(parity_result, names[0])
    if spread_rows is not None:
        tables["spread"] = table_spread(spread_rows)
    for t in tables.values():
        print("\n" + t)
    if args.write_results:
        replace_marked(args.write_results, tables)


if __name__ == "__main__":
    main()
