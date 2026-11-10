"""Write the E3 parity inputs and Python's per-step output for one or all scenarios.

    python scripts/export_parity.py OUT_DIR [--scenario NAME]

Each scenario goes to OUT_DIR/NAME (imu.csv, gps.csv, config.yaml, python_log.csv). The pytest
parity suite (tests/test_cpp_parity.py) does this by itself in a temporary directory; this script
is for looking at the files or running build/run_dataset on them by hand.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from navekf_py import parity  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--scenario", choices=sorted(parity.SCENARIOS), help="default: all of them")
    args = ap.parse_args()
    for name in [args.scenario] if args.scenario else parity.SCENARIOS:
        _, rows = parity.export(name, args.out_dir / name)
        print(f"{name}: {len(rows)} rows -> {args.out_dir / name}")


if __name__ == "__main__":
    main()
