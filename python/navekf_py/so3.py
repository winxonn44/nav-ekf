"""SO(3) and quaternion helpers. Conventions: docs/conventions.md §3.

Quaternions are Hamilton, stored ``[w, x, y, z]``. Every function except
``rot_to_quat`` broadcasts over leading axes, so the same
code serves a single state (shape ``(3,)`` / ``(4,)``) and a batch of Monte Carlo
runs (shape ``(N, 3)`` / ``(N, 4)``). Element-wise formulas are written out
explicitly (no BLAS calls) so that the order of floating-point operations is
fixed and easy to mirror in C++.
"""

from __future__ import annotations

import numpy as np

SMALL_ANGLE = 1e-8  # conventions §3.1: below this ‖φ‖ use the Taylor branches


def skew(a: np.ndarray) -> np.ndarray:
    """``[a]×`` with ``[a]× b = a × b``. Shape ``(..., 3) -> (..., 3, 3)``."""
    a = np.asarray(a, dtype=float)
    ax, ay, az = a[..., 0], a[..., 1], a[..., 2]
    z = np.zeros_like(ax)
    return np.stack([np.stack([z, -az, ay], -1), np.stack([az, z, -ax], -1), np.stack([-ay, ax, z], -1)], -2)


def quat_mul(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Hamilton product ``p ⊗ q`` (``[w, x, y, z]``)."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    pw, px, py, pz = p[..., 0], p[..., 1], p[..., 2], p[..., 3]
    qw, qx, qy, qz = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack(
        [
            pw * qw - px * qx - py * qy - pz * qz,
            pw * qx + px * qw + py * qz - pz * qy,
            pw * qy - px * qz + py * qw + pz * qx,
            pw * qz + px * qy - py * qx + pz * qw,
        ],
        -1,
    )


def quat_conj(q: np.ndarray) -> np.ndarray:
    """Conjugate ``q*`` (the inverse of a unit quaternion)."""
    q = np.asarray(q, dtype=float)
    return np.concatenate([q[..., :1], -q[..., 1:]], -1)


def quat_normalize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    return q / np.sqrt(q[..., 0:1] ** 2 + q[..., 1:2] ** 2 + q[..., 2:3] ** 2 + q[..., 3:4] ** 2)


def quat_canonical(q: np.ndarray) -> np.ndarray:
    """Sign-fixed copy with ``w ≥ 0``. Only for comparing and logging (§3)."""
    q = np.asarray(q, dtype=float)
    return np.where(q[..., 0:1] < 0, -q, q)


def quat_to_rot(q: np.ndarray) -> np.ndarray:
    """Rotation matrix ``R(q)`` of a unit quaternion: ``v_A = R(q_A_B) v_B``."""
    q = np.asarray(q, dtype=float)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack(
        [
            np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
            np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
            np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
        ],
        -2,
    )


def rot_to_quat(R: np.ndarray) -> np.ndarray:
    """Unit quaternion (``w ≥ 0``) of a single rotation matrix (Shepperd's method)."""
    R = np.asarray(R, dtype=float)
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    i = int(np.argmax([tr, R[0, 0], R[1, 1], R[2, 2]]))
    if i == 0:
        s = 2.0 * np.sqrt(1.0 + tr)
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    elif i == 1:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    elif i == 2:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    return quat_canonical(quat_normalize(np.array(q)))


def exp(phi: np.ndarray) -> np.ndarray:
    """``Exp(φ)``: rotation vector -> unit quaternion (conventions §3.1).

    ``[cos(θ/2), sin(θ/2) φ/θ]`` with ``θ = ‖φ‖``; for ``θ < 1e-8`` the Taylor form
    ``[1 − θ²/8, φ/2 (1 − θ²/24)]``, then normalized. The regular branch is not
    re-normalized.
    """
    phi = np.asarray(phi, dtype=float)
    th2 = phi[..., 0:1] ** 2 + phi[..., 1:2] ** 2 + phi[..., 2:3] ** 2
    th = np.sqrt(th2)
    small = th < SMALL_ANGLE
    th_safe = np.where(small, 1.0, th)
    q_reg = np.concatenate([np.cos(0.5 * th), np.sin(0.5 * th) / th_safe * phi], -1)
    q_small = quat_normalize(np.concatenate([1.0 - th2 / 8.0, 0.5 * phi * (1.0 - th2 / 24.0)], -1))
    return np.where(small, q_small, q_reg)


def log(q: np.ndarray) -> np.ndarray:
    """``Log(q)``: unit quaternion -> rotation vector with ``‖φ‖ ≤ π``.

    Flips ``q -> −q`` when ``w < 0``, then ``φ = 2 atan2(‖v‖, w) v/‖v‖``; for
    ``‖v‖ < 1e-8`` the Taylor form ``φ = (2/w)(1 − ‖v‖²/(3w²)) v``.
    """
    q = quat_canonical(q)
    w = q[..., 0:1]
    v = q[..., 1:]
    n2 = v[..., 0:1] ** 2 + v[..., 1:2] ** 2 + v[..., 2:3] ** 2
    n = np.sqrt(n2)
    small = n < SMALL_ANGLE
    n_safe = np.where(small, 1.0, n)
    w_safe = np.where(small, w, 1.0)  # w ≈ 1 whenever ‖v‖ is tiny
    scale = np.where(small, 2.0 / w_safe * (1.0 - n2 / (3.0 * w_safe * w_safe)), 2.0 * np.arctan2(n, w) / n_safe)
    return scale * v


def exp_rot(phi: np.ndarray) -> np.ndarray:
    """``Exp(φ)`` as a rotation matrix (Rodrigues), via the quaternion."""
    return quat_to_rot(exp(phi))


def right_jacobian(phi: np.ndarray) -> np.ndarray:
    """Right Jacobian of SO(3) (derivation §6), for a single ``φ`` of shape ``(3,)``.

    ``J_r(φ) = I − (1 − cos θ)/θ² [φ]× + (θ − sin θ)/θ³ [φ]×²``; small-angle ``I − ½[φ]×``.
    It satisfies ``Exp(φ + ε) ≈ Exp(φ) Exp(J_r(φ) ε)``.
    """
    phi = np.asarray(phi, dtype=float)
    if phi.ndim > 1:  # batched (E2 Monte Carlo): same formula and branch, per run
        th = np.sqrt(phi[..., 0] ** 2 + phi[..., 1] ** 2 + phi[..., 2] ** 2)[..., None, None]
        K = skew(phi)
        small = th < SMALL_ANGLE
        ts = np.where(small, 1.0, th)
        reg = np.eye(3) - (1 - np.cos(ts)) / ts**2 * K + (ts - np.sin(ts)) / ts**3 * (K @ K)
        return np.where(small, np.eye(3) - 0.5 * K, reg)
    th = float(np.linalg.norm(phi))
    K = skew(phi)
    if th < SMALL_ANGLE:
        return np.eye(3) - 0.5 * K
    return np.eye(3) - (1 - np.cos(th)) / th**2 * K + (th - np.sin(th)) / th**3 * (K @ K)
