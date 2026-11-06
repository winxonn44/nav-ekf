"""Numerical checks for docs/derivation.md. Stand-alone (NumPy + SciPy only),
runs in about a second:

    python scripts/check_derivation.py

Checks F, H and the reset Jacobian against finite differences of the
nonlinear model, compares the discretizations with Van Loan, and prints the
chi-square bands. It exits non-zero if any check fails. E1/E2 turn these into
pytest tests against the real code; this script only validates the math.
"""

from __future__ import annotations

import sys

import numpy as np
from scipy.linalg import expm
from scipy.stats import chi2

G_W = np.array([0.0, 0.0, -9.81])
FAILED: list[str] = []


def skew(a):
    return np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])


def so3_exp(phi):
    """Rotation matrix of rotation vector phi (Rodrigues)."""
    th = np.linalg.norm(phi)
    K = skew(phi)
    if th < 1e-8:
        return np.eye(3) + K + 0.5 * K @ K
    return np.eye(3) + np.sin(th) / th * K + (1 - np.cos(th)) / th**2 * K @ K


def so3_log(R):
    c = np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)
    th = np.arccos(c)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if th < 1e-8:
        return 0.5 * w
    return th / (2 * np.sin(th)) * w


def right_jacobian(phi):
    th = np.linalg.norm(phi)
    K = skew(phi)
    if th < 1e-8:
        return np.eye(3) - 0.5 * K
    return np.eye(3) - (1 - np.cos(th)) / th**2 * K + (th - np.sin(th)) / th**3 * K @ K


def check(name, err, tol):
    ok = err <= tol
    print(f"  {'ok  ' if ok else 'FAIL'} {name}: {err:.2e} (tol {tol:.0e})")
    if not ok:
        FAILED.append(name)


# ---- a random but realistic nominal state and IMU sample -------------------
rng = np.random.default_rng(0)
R0 = so3_exp(rng.normal(size=3))
p0, v0 = rng.normal(size=3), rng.normal(size=3)
ba0, bg0 = 0.05 * rng.normal(size=3), 0.01 * rng.normal(size=3)
a_m = R0.T @ (np.array([0.3, -0.2, 0.1]) - G_W) + ba0  # a moving, tilted IMU
w_m = np.array([0.4, -0.3, 0.8]) + bg0
a_hat, w_hat = a_m - ba0, w_m - bg0


def step(p, v, R, ba, bg, dt):
    """conventions.md §6: the one integrator."""
    aW = R @ (a_m - ba) + G_W
    return p + v * dt + 0.5 * aW * dt**2, v + aW * dt, R @ so3_exp((w_m - bg) * dt), ba, bg


def inject(dx):
    return (p0 + dx[0:3], v0 + dx[3:6], R0 @ so3_exp(dx[6:9]), ba0 + dx[9:12], bg0 + dx[12:15])


def error(true, nom):
    p, v, R, ba, bg = true
    pn, vn, Rn, ban, bgn = nom
    return np.concatenate([p - pn, v - vn, so3_log(Rn.T @ R), ba - ban, bg - bgn])


def F_matrix():
    F = np.zeros((15, 15))
    F[0:3, 3:6] = np.eye(3)
    F[3:6, 6:9] = -R0 @ skew(a_hat)
    F[3:6, 9:12] = -R0
    F[6:9, 6:9] = -skew(w_hat)
    F[6:9, 12:15] = -np.eye(3)
    return F


def G_matrix():
    G = np.zeros((15, 12))
    G[3:6, 0:3] = -R0
    G[6:9, 3:6] = -np.eye(3)
    G[9:12, 6:9] = np.eye(3)
    G[12:15, 9:12] = np.eye(3)
    return G


print("§3 F by finite differences of the integrator")
F = F_matrix()
for dt in (1e-4, 1e-5):
    nominal_next = step(p0, v0, R0, ba0, bg0, dt)
    eps = 1e-6
    J = np.zeros((15, 15))
    for i in range(15):
        e = np.zeros(15)
        e[i] = eps
        plus = error(step(*inject(e), dt), nominal_next)
        minus = error(step(*inject(-e), dt), nominal_next)
        J[:, i] = (plus - minus) / (2 * eps)
    F_num = (J - np.eye(15)) / dt
    # residual is O(dt * |F|^2): shrinks 10x when dt does
    check(f"max|F_num - F| at dt={dt:g}", np.abs(F_num - F).max(), 50 * dt)

print("§5 H by finite differences (lever arm on)")
lever = np.array([0.1, -0.05, 0.2])
H = np.zeros((3, 15))
H[:, 0:3] = np.eye(3)
H[:, 6:9] = -R0 @ skew(lever)
z0 = p0 + R0 @ lever
H_num = np.zeros((3, 15))
for i in range(15):
    e = np.zeros(15)
    e[i] = 1e-6
    zp = inject(e)[0] + inject(e)[2] @ lever
    zm = inject(-e)[0] + inject(-e)[2] @ lever
    H_num[:, i] = (zp - zm) / 2e-6
check("max|H_num - H|", np.abs(H_num - H).max(), 1e-8)

print("§4 discretization vs exact (expm / Van Loan)")
G = G_matrix()
sig = dict(a=2e-3, g=1.7e-4, ba=3e-3, bg=2e-5)  # EuRoC-like densities
Qc = np.diag(np.repeat([sig["a"] ** 2, sig["g"] ** 2, sig["ba"] ** 2, sig["bg"] ** 2], 3))
GQG = G @ Qc @ G.T
check("velocity block of GQcGt equals sigma_a^2 I", np.abs(GQG[3:6, 3:6] - sig["a"] ** 2 * np.eye(3)).max(), 1e-18)
prev = None
for dt in (0.02, 0.01, 0.005):
    A = np.zeros((30, 30))
    A[:15, :15], A[:15, 15:], A[15:, 15:] = -F, GQG, F.T
    B = expm(A * dt)
    Phi = B[15:, 15:].T
    Qd = Phi @ B[:15, 15:]
    e1 = np.abs(np.eye(15) + F * dt - Phi).max()
    e2 = np.abs(np.eye(15) + F * dt + 0.5 * F @ F * dt**2 - Phi).max()
    eq = np.abs(GQG * dt - Qd).max() / np.abs(Qd).max()
    check(f"Van Loan Phi == expm(F dt) at dt={dt}", np.abs(Phi - expm(F * dt)).max(), 1e-12)
    print(f"       dt={dt:<6} |Phi1-Phi|={e1:.2e}  |Phi2-Phi|={e2:.2e}  rel|Qd1-Qd|={eq:.2e}")
    if prev is not None:  # orders: halving dt cuts e1 by ~4 and e2 by ~8
        check(f"Phi1 error ratio (expect ~4) dt={dt}", abs(prev[0] / e1 - 4), 0.5)
        check(f"Phi2 error ratio (expect ~8) dt={dt}", abs(prev[1] / e2 - 8), 1.0)
    prev = (e1, e2)
check("Qd symmetric", np.abs(Qd - Qd.T).max(), 1e-20)

print("§6 reset Jacobian J_r by finite differences")
dth_hat = np.array([0.02, -0.01, 0.015])
Jr_num = np.zeros((3, 3))
for i in range(3):
    e = np.zeros(3)
    e[i] = 1e-7
    f = lambda d: so3_log(so3_exp(-dth_hat) @ so3_exp(d))  # noqa: E731
    Jr_num[:, i] = (f(dth_hat + e) - f(dth_hat - e)) / 2e-7
check("max|Jr_num - J_r|", np.abs(Jr_num - right_jacobian(dth_hat)).max(), 1e-7)
print(f"       first-order I - ½[δθ]× differs by {np.abs(Jr_num - (np.eye(3) - 0.5 * skew(dth_hat))).max():.1e}"
      f" at |δθ|={np.linalg.norm(dth_hat):.3f} rad")

print("§8 chi-square bands (95% two-sided, on the N-run mean)")
for n in (1, 50):
    lo, hi = chi2.ppf([0.025, 0.975], 15 * n) / n
    lo3, hi3 = chi2.ppf([0.025, 0.975], 3 * n) / n
    print(f"       N={n:<3} NEES(15): [{lo:.2f}, {hi:.2f}]   NIS(3): [{lo3:.2f}, {hi3:.2f}]")
print(f"       NIS gate, 99% chi2(3): {chi2.ppf(0.99, 3):.3f}")

print("\nall checks passed" if not FAILED else f"\nFAILED: {FAILED}")
sys.exit(1 if FAILED else 0)
