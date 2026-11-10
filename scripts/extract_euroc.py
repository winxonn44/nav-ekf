"""Extract only what E4 needs from the EuRoC machine-hall archive (see docs/data.md).

Usage (``-I`` so nothing next to the data can shadow an import):

    python -I scripts/extract_euroc.py ~/Documents/Code/datasets/euroc/machine_hall.zip \
        ~/Documents/Code/datasets/euroc --tmp /some/empty/dir

The outer zip holds ``machine_hall/MH_0x_*/MH_0x_*.zip``. Each nested zip is copied to ``--tmp``, the
whitelisted members are written to ``<out>/<sequence>/mav0/...``, and the nested copy is deleted before the
next one. No camera images, no ``.bag`` files. The outer zip is opened read-only. Only exact whitelisted
names are ever written, so a hostile member name cannot escape the output directory.
"""

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

MEMBERS = (
    "mav0/imu0/data.csv",
    "mav0/imu0/sensor.yaml",
    "mav0/state_groundtruth_estimate0/data.csv",
    "mav0/state_groundtruth_estimate0/sensor.yaml",
    "mav0/body.yaml",
)
REQUIRED = set(MEMBERS) - {"mav0/body.yaml"}
MIN_FREE_BYTES = 10 * 1024**3


def extract_sequence(outer: zipfile.ZipFile, name: str, out: Path, tmp: Path) -> list[str]:
    seq = Path(name).stem
    nested_path = tmp / f"{seq}.zip"
    written = []
    try:
        with outer.open(name) as src, open(nested_path, "wb") as dst:
            shutil.copyfileobj(src, dst, 1 << 20)
        with zipfile.ZipFile(nested_path) as nested:
            names = {i.filename.removeprefix("./"): i for i in nested.infolist()}
            missing = REQUIRED - names.keys()
            if missing:
                raise RuntimeError(f"{seq}: missing {sorted(missing)}")
            for member in MEMBERS:
                if member not in names:
                    continue
                target = out / seq / member
                target.parent.mkdir(parents=True, exist_ok=True)
                with nested.open(names[member]) as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst, 1 << 20)
                written.append(member)
    finally:
        nested_path.unlink(missing_ok=True)
    return written


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("archive", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--tmp", type=Path, required=True, help="empty directory for the nested zip, one at a time")
    args = ap.parse_args()

    args.tmp.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.archive) as outer:
        nested = sorted(n for n in outer.namelist() if n.endswith(".zip") and "/._" not in n)
        for name in nested:
            size = outer.getinfo(name).file_size
            free = shutil.disk_usage(args.tmp).free
            if free - size < MIN_FREE_BYTES:
                print(f"stopping before {name}: {free / 1e9:.1f} GB free, need {size / 1e9:.1f} GB + 10 GiB", file=sys.stderr)
                return 1
            written = extract_sequence(outer, name, args.out, args.tmp)
            print(f"{Path(name).stem}: {', '.join(written)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
