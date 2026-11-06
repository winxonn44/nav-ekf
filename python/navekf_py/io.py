"""Trajectory files (docs/conventions.md §8).

TUM format: one row per pose, ``t x y z qx qy qz qw`` with ``t`` in seconds
written as ``%.9f``. Note the quaternion order on disk is ``[x, y, z, w]``,
while this repo stores ``[w, x, y, z]`` in memory; only these two functions
reorder it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _ns_to_str(t_ns: int) -> str:
    """Exact ``%.9f`` seconds from integer nanoseconds (no float round trip)."""
    t_ns = int(t_ns)
    sign = "-" if t_ns < 0 else ""
    s, ns = divmod(abs(t_ns), 1_000_000_000)
    return f"{sign}{s}.{ns:09d}"


def _str_to_ns(s: str) -> int:
    neg = s.startswith("-")
    whole, _, frac = s.lstrip("-").partition(".")
    ns = int(whole) * 1_000_000_000 + int((frac + "000000000")[:9])
    return -ns if neg else ns


def write_tum(path: str | Path, t_ns: np.ndarray, p: np.ndarray, q_wxyz: np.ndarray) -> None:
    """Write poses ``(t_ns, p_W, q_W_B [w,x,y,z])`` as a TUM file."""
    with open(path, "w") as f:
        for t, pk, qk in zip(t_ns, p, q_wxyz, strict=True):
            w, x, y, z = qk
            f.write(f"{_ns_to_str(t)} {pk[0]:.9f} {pk[1]:.9f} {pk[2]:.9f} {x:.9f} {y:.9f} {z:.9f} {w:.9f}\n")


def read_tum(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read a TUM file; returns ``(t_ns int64, p (K,3), q (K,4) [w,x,y,z])``."""
    t, rows = [], []
    for line in Path(path).read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        cols = line.split()
        t.append(_str_to_ns(cols[0]))
        rows.append([float(c) for c in cols[1:8]])
    a = np.array(rows).reshape(-1, 7)
    q = np.column_stack([a[:, 6], a[:, 3], a[:, 4], a[:, 5]])
    return np.array(t, dtype=np.int64), a[:, 0:3], q
