"""Error-state EKF: IMU propagation + position fixes (docs/derivation.md §3–6).

The nominal state is propagated by the one integrator (:func:`propagate.propagate`,
conventions §6); the 15×15 error covariance ``P`` follows it. Every array may
carry leading batch axes (one entry per Monte Carlo run): a single run has
``P.shape == (15, 15)``, a batch of ``N`` runs ``(N, 15, 15)``. The math per run is
the same either way.

**Step order** (what the E3 C++ port must reproduce, in this order):

``run`` walks the IMU timestamps ``t_k``. At each ``k``:

1. every position fix with ``t_fix == t_k``: :meth:`ESKF.update_position`
   (innovation, ``S``, NIS, gate; Joseph update) then :meth:`ESKF.inject_and_reset`;
2. the per-step hook / log sees the posterior at ``t_k``;
3. if ``k < K−1``: :meth:`ESKF.propagate` with sample ``k`` over
   ``dt = (t_{k+1} − t_k)·1e-9``:

   a. ``F, G`` from the **start-of-step** nominal state and sample (derivation §4);
   b. nominal step ``x̂ ← propagate.propagate(x̂, imu_k, dt, g)``;
   c. ``Φ, Q_d`` from ``F, G Q_c Gᵀ, dt`` (``"first"`` default);
   d. ``P ← Φ P Φᵀ + Q_d``; ``P ← ½(P + Pᵀ)``; optional min-eigenvalue check.

Update (derivation §5), with ``l`` the lever arm and ``R = R(q̂)``:
``ẑ = p̂ + R l``, ``y = z − ẑ``, ``H = [I 0 −R[l]× 0 0]``, ``S = H P Hᵀ + R_meas``
(symmetrized), ``NIS = yᵀ S⁻¹ y`` (via ``solve``), reject if ``NIS > gate``; else
``K = solve(S, H P)ᵀ``, ``δx̂ = K y``,
``P ← (I − K H) P (I − K H)ᵀ + K R_meas Kᵀ``, symmetrize.

Injection and reset (derivation §6): ``x̂ ← x̂ ⊞ δx̂`` (``q`` normalized), then
``P ← G_r P G_rᵀ`` with ``G_r = blockdiag(I, I, J_r(δθ̂), I, I)`` (``"jr"``,
default) or ``G_r = I`` (``"identity"``); symmetrize; ``δx̂ ← 0``.

Fixes must land on IMU timestamps (true for the simulator: 200 Hz IMU, 10 Hz
GPS); :func:`run` raises otherwise. Datasets (E4+) will need a split step.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import expm
from scipy.stats import chi2

from . import so3
from .propagate import propagate as nominal_step
from .propagate import step_dt
from .types import G_MAGNITUDE, ImuNoise, ImuSample, ImuStream, NominalState, PositionFix

log = logging.getLogger(__name__)

N_ERR = 15
GATE_99_3DOF = float(chi2.ppf(0.99, 3))  # 11.3449 (derivation §5 rounds to 11.345)
DISCRETIZATIONS = ("first", "second", "vanloan")
RESETS = ("jr", "identity")

# error-state blocks (conventions §5)
IP, IV, ITH, IBA, IBG = slice(0, 3), slice(3, 6), slice(6, 9), slice(9, 12), slice(12, 15)


# ----------------------------------------------------------- ⊞ / ⊖ on the state


def boxplus(x: NominalState, dx: np.ndarray) -> NominalState:
    """``x ⊞ δx``: ``p + δp``, ``v + δv``, ``normalize(q ⊗ Exp(δθ))``, ``b + δb`` (local error)."""
    dx = np.asarray(dx, dtype=float)
    q = so3.quat_normalize(so3.quat_mul(x.q, so3.exp(dx[..., ITH])))
    return NominalState(x.p + dx[..., IP], x.v + dx[..., IV], q, x.ba + dx[..., IBA], x.bg + dx[..., IBG])


def boxminus(x: NominalState, x_hat: NominalState) -> np.ndarray:
    """``x ⊖ x̂`` (15): the error with ``x = x̂ ⊞ δx``; orientation ``Log(q̂* ⊗ q)`` (derivation §8)."""
    dth = so3.log(so3.quat_mul(so3.quat_conj(x_hat.q), x.q))
    return np.concatenate([x.p - x_hat.p, x.v - x_hat.v, dth, x.ba - x_hat.ba, x.bg - x_hat.bg], -1)


# --------------------------------------------------------------- the matrices


def jacobians(x: NominalState, imu: ImuSample) -> tuple[np.ndarray, np.ndarray]:
    """Continuous-time ``F`` (15×15) and ``G`` (15×12) at the start of the step (derivation §3)."""
    a_hat = imu.acc - x.ba
    w_hat = imu.gyr - x.bg
    R = so3.quat_to_rot(x.q)
    batch = np.broadcast_shapes(np.shape(a_hat)[:-1], np.shape(R)[:-2])
    I3 = np.eye(3)
    F = np.zeros((*batch, N_ERR, N_ERR))
    F[..., IP, IV] = I3
    F[..., IV, ITH] = -R @ so3.skew(a_hat)
    F[..., IV, IBA] = -R
    F[..., ITH, ITH] = -so3.skew(w_hat)
    F[..., ITH, IBG] = -I3
    G = np.zeros((*batch, N_ERR, 12))
    G[..., IV, 0:3] = -R
    G[..., ITH, 3:6] = -I3
    G[..., IBA, 6:9] = I3
    G[..., IBG, 9:12] = I3
    return F, G


def discretize(F: np.ndarray, GQG: np.ndarray, dt: float, method: str = "first") -> tuple[np.ndarray, np.ndarray]:
    """``Φ, Q_d`` over one step (derivation §4). ``GQG = G Q_c Gᵀ``.

    * ``"first"``:   ``Φ = I + F dt``,  ``Q_d = GQG dt``.
    * ``"second"``:  ``Φ = I + F dt + ½F²dt²``, ``Q_d = GQG dt + ½(F GQG + GQG Fᵀ)dt² + ⅓ F GQG Fᵀ dt³``
      (the integral of ``Φ(τ) GQG Φ(τ)ᵀ`` with ``Φ(τ) ≈ I + Fτ``).
    * ``"vanloan"``: exact, from ``expm`` of the 30×30 block matrix.
    """
    I = np.eye(N_ERR)
    if method == "first":
        return I + F * dt, GQG * dt
    if method == "second":
        Phi = I + F * dt + 0.5 * (F @ F) * (dt * dt)
        FQ = F @ GQG
        Qd = GQG * dt + 0.5 * (FQ + np.swapaxes(FQ, -1, -2)) * (dt * dt) + (FQ @ np.swapaxes(F, -1, -2)) * (dt**3 / 3.0)
        return Phi, Qd
    if method == "vanloan":
        batch = F.shape[:-2]
        A = np.zeros((*batch, 2 * N_ERR, 2 * N_ERR))
        A[..., :N_ERR, :N_ERR] = -F
        A[..., :N_ERR, N_ERR:] = GQG
        A[..., N_ERR:, N_ERR:] = np.swapaxes(F, -1, -2)
        B = expm(A * dt)
        Phi = np.swapaxes(B[..., N_ERR:, N_ERR:], -1, -2)
        return Phi, Phi @ B[..., :N_ERR, N_ERR:]
    raise ValueError(f"unknown discretization {method!r}; use one of {DISCRETIZATIONS}")


def position_model(x: NominalState, lever_arm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Predicted antenna position ``ẑ = p̂ + R̂ l`` and ``H = [I 0 −R̂[l]× 0 0]`` (derivation §5)."""
    R = so3.quat_to_rot(x.q)
    l = np.asarray(lever_arm, dtype=float)
    z_hat = x.p + R @ l
    H = np.zeros((*R.shape[:-2], 3, N_ERR))
    H[..., :, IP] = np.eye(3)
    H[..., :, ITH] = -R @ so3.skew(l)
    return z_hat, H


def reset_jacobian(dth: np.ndarray, method: str = "jr") -> np.ndarray:
    """``G_r = blockdiag(I, I, J_r(δθ̂), I, I)`` (``"jr"``) or ``I`` (``"identity"``), derivation §6."""
    dth = np.asarray(dth, dtype=float)
    Gr = np.broadcast_to(np.eye(N_ERR), (*dth.shape[:-1], N_ERR, N_ERR)).copy()
    if method == "jr":
        Gr[..., ITH, ITH] = so3.right_jacobian(dth)
    elif method != "identity":
        raise ValueError(f"unknown reset {method!r}; use one of {RESETS}")
    return Gr


def sym(P: np.ndarray) -> np.ndarray:
    return 0.5 * (P + np.swapaxes(P, -1, -2))


# -------------------------------------------------------------------- the filter


@dataclass
class EskfConfig:
    noise: ImuNoise
    g: float = G_MAGNITUDE
    lever_arm: np.ndarray = field(default_factory=lambda: np.zeros(3))  # antenna in B [m]
    discretization: str = "first"  # "first" | "second" | "vanloan"
    reset: str = "jr"  # "jr" | "identity"
    gate: float | None = GATE_99_3DOF  # NIS threshold; None disables the gate
    check_psd: bool = False  # min-eigenvalue check of P after every step
    psd_tol: float = 0.0  # fail if min eig(P) < −psd_tol · max eig(P)
    log: bool = True  # keep the per-step log (conventions §8)

    def __post_init__(self):
        if self.discretization not in DISCRETIZATIONS:
            raise ValueError(f"discretization must be one of {DISCRETIZATIONS}")
        if self.reset not in RESETS:
            raise ValueError(f"reset must be one of {RESETS}")
        self.lever_arm = np.asarray(self.lever_arm, dtype=float)


@dataclass
class UpdateResult:
    t_ns: np.int64
    y: np.ndarray  # (..., 3) innovation z − ẑ
    S: np.ndarray  # (..., 3, 3) innovation covariance
    nis: np.ndarray  # (...,) yᵀ S⁻¹ y
    accepted: np.ndarray  # (...,) bool, NIS ≤ gate


@dataclass
class Rejection:
    t_ns: int
    run: int | None  # index into the batch, None for a single run
    nis: float


class StepLog:
    """One row per IMU step or update (conventions §8): ``t_ns``, kind, state (16), ``diag P`` (15), NIS, accepted.

    ``kind`` is 0 for an IMU step (posterior after propagation) and 1 for an
    update (after injection and reset). IMU rows have ``nis = nan``, ``accepted = True``.
    """

    def __init__(self):
        self.t_ns: list[int] = []
        self.kind: list[int] = []
        self.x: list[np.ndarray] = []
        self.P_diag: list[np.ndarray] = []
        self.nis: list[np.ndarray] = []
        self.accepted: list[np.ndarray] = []

    def append(self, t_ns, kind, x: NominalState, P, nis=np.nan, accepted=True):
        self.t_ns.append(int(t_ns))
        self.kind.append(kind)
        self.x.append(x.as_vector())
        self.P_diag.append(np.diagonal(P, axis1=-2, axis2=-1).copy())
        self.nis.append(np.asarray(nis, dtype=float))
        self.accepted.append(np.asarray(accepted, dtype=bool))

    def __len__(self) -> int:
        return len(self.t_ns)

    def arrays(self) -> dict[str, np.ndarray]:
        n = len(self)
        if n == 0:
            return {}
        shape = np.shape(self.x[0])[:-1]
        return {
            "t_ns": np.asarray(self.t_ns, dtype=np.int64),
            "kind": np.asarray(self.kind, dtype=np.int8),
            "x": np.stack(self.x),
            "P_diag": np.stack(self.P_diag),
            "nis": np.stack([np.broadcast_to(v, shape) for v in self.nis]),
            "accepted": np.stack([np.broadcast_to(v, shape) for v in self.accepted]),
        }


class ESKF:
    """Error-state EKF. ``x`` is the nominal state, ``P`` the 15×15 error covariance."""

    def __init__(self, x0: NominalState, P0: np.ndarray, cfg: EskfConfig, t0_ns: int = 0):
        self.cfg = cfg
        self.x = x0.copy()
        self.x.q = so3.quat_normalize(self.x.q)
        batch = np.shape(self.x.p)[:-1]
        self.P = sym(np.broadcast_to(np.asarray(P0, dtype=float), (*batch, N_ERR, N_ERR)).copy())
        self.dx = np.zeros((*batch, N_ERR))
        self.t_ns = np.int64(t0_ns)
        self.Qc = cfg.noise.Qc()
        self.log = StepLog() if cfg.log else None
        self.rejections: list[Rejection] = []
        self.min_eig: float | None = None
        self._check()

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return np.shape(self.x.p)[:-1]

    # ---------------------------------------------------------------- init

    @classmethod
    def from_truth(cls, truth_state: NominalState, P0: np.ndarray, cfg: EskfConfig, t0_ns: int = 0,
                   rng: np.random.Generator | None = None) -> ESKF:  # fmt: skip
        """Start at truth with covariance ``P0``. With ``rng``, the estimate is ``truth ⊞ (−δx₀)``,
        ``δx₀ ~ N(0, P0)``, so the initial error ``truth ⊖ x̂₀ = δx₀`` is exactly a draw from ``P0``."""
        x0 = truth_state.copy()
        if rng is not None:
            L = np.linalg.cholesky(np.asarray(P0, dtype=float))
            batch = np.shape(truth_state.p)[:-1]
            dx0 = (L @ rng.standard_normal((*batch, N_ERR, 1)))[..., 0]
            x0 = boxplus(truth_state, -dx0)
        return cls(x0, P0, cfg, t0_ns)

    # ----------------------------------------------------------- propagate

    def propagate(self, imu: ImuSample, dt: float, t_next_ns: int | None = None) -> None:
        """One IMU step (steps 3a–3d of the module docstring)."""
        F, G = jacobians(self.x, imu)
        self.x = nominal_step(self.x, imu, dt, self.cfg.g)
        GQG = G @ self.Qc @ np.swapaxes(G, -1, -2)
        Phi, Qd = discretize(F, GQG, dt, self.cfg.discretization)
        self.P = sym(Phi @ self.P @ np.swapaxes(Phi, -1, -2) + Qd)
        self.t_ns = np.int64(imu.t_ns + round(dt * 1e9)) if t_next_ns is None else np.int64(t_next_ns)
        self._check()
        if self.log is not None:
            self.log.append(self.t_ns, 0, self.x, self.P)

    # -------------------------------------------------------------- update

    def update_position(self, fix: PositionFix, inject: bool = True) -> UpdateResult:
        """Position update with NIS gate and Joseph form; then injection and reset (unless ``inject=False``)."""
        R_meas = fix.R_meas
        z_hat, H = position_model(self.x, self.cfg.lever_arm)
        y = np.asarray(fix.p_W, dtype=float) - z_hat
        Ht = np.swapaxes(H, -1, -2)
        PHt = self.P @ Ht
        S = sym(H @ PHt + R_meas)
        nis = np.sum(y * np.linalg.solve(S, y[..., None])[..., 0], -1)
        accepted = np.ones(np.shape(nis), bool) if self.cfg.gate is None else nis <= self.cfg.gate

        K = np.swapaxes(np.linalg.solve(S, np.swapaxes(PHt, -1, -2)), -1, -2)  # P Hᵀ S⁻¹ (S, P symmetric)
        dx = (K @ y[..., None])[..., 0]
        IKH = np.eye(N_ERR) - K @ H
        P_new = sym(IKH @ self.P @ np.swapaxes(IKH, -1, -2) + K @ R_meas @ np.swapaxes(K, -1, -2))

        acc = np.asarray(accepted)
        self.P = np.where(acc[..., None, None], P_new, self.P)
        self.dx = np.where(acc[..., None], dx, 0.0)
        self.t_ns = np.int64(fix.t_ns)
        for i in np.flatnonzero(~acc.ravel()):
            nis_i = float(np.ravel(nis)[i])
            run = None if acc.ndim == 0 else int(i)
            self.rejections.append(Rejection(int(fix.t_ns), run, nis_i))
            log.info("rejected fix t_ns=%d run=%s NIS=%.2f > %.3f", int(fix.t_ns), run, nis_i, self.cfg.gate)
        if inject:
            self.inject_and_reset(mask=acc)
        self._check()
        if self.log is not None:
            self.log.append(self.t_ns, 1, self.x, self.P, nis, accepted)
        return UpdateResult(np.int64(fix.t_ns), y, S, nis, accepted)

    def inject_and_reset(self, mask: np.ndarray | None = None) -> None:
        """``x̂ ← x̂ ⊞ δx̂``, ``P ← G_r P G_rᵀ``, ``δx̂ ← 0``. Runs with ``mask`` False are left untouched."""
        x_new = boxplus(self.x, self.dx)
        Gr = reset_jacobian(self.dx[..., ITH], self.cfg.reset)
        P_new = sym(Gr @ self.P @ np.swapaxes(Gr, -1, -2)) if self.cfg.reset == "jr" else self.P
        if mask is None or np.all(mask):
            self.x, self.P = x_new, P_new
        else:
            m = np.asarray(mask)
            new, old = x_new, self.x
            self.x = NominalState(
                *(np.where(m[..., None], getattr(new, f), getattr(old, f)) for f in ("p", "v", "q", "ba", "bg"))
            )
            self.P = np.where(m[..., None, None], P_new, self.P)
        self.dx = np.zeros_like(self.dx)

    # ------------------------------------------------------------- hygiene

    def _check(self) -> None:
        if not self.cfg.check_psd:
            return
        ev = np.linalg.eigvalsh(self.P)
        lo, hi = ev[..., 0], ev[..., -1]
        self.min_eig = float(np.min(lo))
        if np.any(lo < -self.cfg.psd_tol * hi):
            raise FloatingPointError(f"P lost positive semi-definiteness at t_ns={int(self.t_ns)}: min eig {self.min_eig:.3e}")


# -------------------------------------------------------------- static init


def static_init(imu: ImuStream, t_end_ns: int, yaw: float, p0: np.ndarray, g: float = G_MAGNITUDE) -> NominalState:
    """Initial state from a still segment ``t < t_end_ns`` (single run).

    Roll and pitch from the averaged specific force ``ā`` (``a_m = Rᵀ(0, 0, g)``
    for a still IMU): ``roll = atan2(ā_y, ā_z)``, ``pitch = atan2(−ā_x, √(ā_y² + ā_z²))``.
    Gyro bias = averaged gyro (flat, non-rotating Earth). Yaw is given; ``v = 0``,
    ``b_a = 0`` (not separable from tilt while still).
    """
    m = np.asarray(imu.t_ns) < t_end_ns
    a = np.mean(imu.acc[m], axis=0)
    w = np.mean(imu.gyr[m], axis=0)
    roll = np.arctan2(a[1], a[2])
    pitch = np.arctan2(-a[0], np.hypot(a[1], a[2]))
    q = so3.quat_mul(
        so3.quat_mul(so3.exp(np.array([0.0, 0.0, yaw])), so3.exp(np.array([0.0, pitch, 0.0]))),
        so3.exp(np.array([roll, 0.0, 0.0])),
    )  # fmt: skip  q = q_z(ψ) ⊗ q_y(θ) ⊗ q_x(φ)
    return NominalState(np.asarray(p0, dtype=float).copy(), np.zeros(3), so3.quat_normalize(q), np.zeros(3), w)


def static_init_P0(x0: NominalState, T_s: float, noise: ImuNoise, sigma_ba0: float, sigma_yaw: float,
                   sigma_p: float, sigma_v: float = 0.01, g: float = G_MAGNITUDE) -> np.ndarray:  # fmt: skip
    """Covariance that goes with :func:`static_init`, including the tilt / accel-bias correlation.

    An accel bias ``b`` shifts the averaged gravity direction, so the tilt error is
    ``δθ ≈ (1/g)[û]× δb_a`` (``û`` = body-frame up), plus averaging noise ``σ_a/(g√T)``
    on tilt and ``σ_g/√T`` on the gyro bias; yaw (about ``û``) gets ``sigma_yaw``.
    """
    u = so3.quat_to_rot(x0.q).T @ np.array([0.0, 0.0, 1.0])
    Uperp = np.eye(3) - np.outer(u, u)
    M = so3.skew(u) / g
    P = np.zeros((N_ERR, N_ERR))
    P[IP, IP] = sigma_p**2 * np.eye(3)
    P[IV, IV] = sigma_v**2 * np.eye(3)
    P[IBA, IBA] = sigma_ba0**2 * np.eye(3)
    P[ITH, ITH] = sigma_ba0**2 * M @ M.T + (noise.sigma_a**2 / (g * g * T_s)) * Uperp + sigma_yaw**2 * np.outer(u, u)
    P[ITH, IBA] = sigma_ba0**2 * M
    P[IBA, ITH] = P[ITH, IBA].T
    P[IBG, IBG] = (noise.sigma_g**2 / T_s + noise.sigma_bg**2 * T_s / 3.0) * np.eye(3)
    return P


# -------------------------------------------------------------- the run loop


def run(eskf: ESKF, imu: ImuStream, fixes: Sequence[PositionFix],
        hook: Callable[[int, ESKF, list[UpdateResult]], None] | None = None) -> list[UpdateResult]:  # fmt: skip
    """Filter a whole IMU stream from ``imu.t_ns[0]`` (the state's time), in the step order of the module docstring.

    ``hook(k, eskf, updates_at_k)`` is called with the posterior at ``t_k``. Fixes before
    ``t_ns[0]`` are skipped; a fix that does not land on an IMU timestamp raises.
    """
    t = np.asarray(imu.t_ns, dtype=np.int64)
    fixes = sorted(fixes, key=lambda f: int(f.t_ns))
    j = 0
    while j < len(fixes) and fixes[j].t_ns < t[0]:
        j += 1
    out: list[UpdateResult] = []
    K = len(t)
    for k in range(K):
        here: list[UpdateResult] = []
        while j < len(fixes) and fixes[j].t_ns <= t[k]:
            if fixes[j].t_ns != t[k]:
                raise ValueError(f"fix at t_ns={int(fixes[j].t_ns)} is not on an IMU timestamp")
            here.append(eskf.update_position(fixes[j]))
            j += 1
        out.extend(here)
        if hook is not None:
            hook(k, eskf, here)
        if k < K - 1:
            eskf.propagate(imu[k], step_dt(t, k), t_next_ns=t[k + 1])
    return out
