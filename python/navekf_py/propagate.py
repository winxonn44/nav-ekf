"""The one integrator (docs/conventions.md §6) and IMU dead reckoning.

This is the reference that the E2 filter and the E3 C++ port must match, so
it follows §6 line by line, in the same order:

    â   = a_m − b̂_a
    ω̂   = ω_m − b̂_g
    R   = R(q̂)                      (start of step, zero-order hold)
    a_W = R · â + g_W
    p  ← p + v·dt + ½·a_W·dt²        (evaluated as p + v*dt + 0.5*a_W*dt*dt)
    v  ← v + a_W·dt
    q  ← normalize(q ⊗ Exp(ω̂·dt))
    b_a, b_g unchanged

``R · â`` is summed column by column (``R[:,0]â₀ + R[:,1]â₁ + R[:,2]â₂``) rather
than through BLAS, so the operation order is fixed. All arrays may carry
leading batch axes (one entry per Monte Carlo run).
"""

from __future__ import annotations

import numpy as np

from . import so3
from .types import G_MAGNITUDE, ImuSample, ImuStream, NominalState, gravity_w


def _matvec(R: np.ndarray, a: np.ndarray) -> np.ndarray:
    return R[..., :, 0] * a[..., 0:1] + R[..., :, 1] * a[..., 1:2] + R[..., :, 2] * a[..., 2:3]


def propagate(state: NominalState, imu: ImuSample, dt: float, g: float = G_MAGNITUDE) -> NominalState:
    """Advance the nominal state over one IMU step of length ``dt`` seconds."""
    a_hat = imu.acc - state.ba
    w_hat = imu.gyr - state.bg
    R = so3.quat_to_rot(state.q)
    a_W = _matvec(R, a_hat) + gravity_w(g)
    p = state.p + state.v * dt + 0.5 * a_W * dt * dt
    v = state.v + a_W * dt
    q = so3.quat_normalize(so3.quat_mul(state.q, so3.exp(w_hat * dt)))
    return NominalState(p, v, q, state.ba, state.bg)


def step_dt(t_ns: np.ndarray, k: int) -> float:
    """``dt = (t_{k+1} − t_k) · 1e-9``: integer subtraction first, then one multiply (§1)."""
    return float(np.int64(t_ns[k + 1]) - np.int64(t_ns[k])) * 1e-9


def dead_reckon(initial_state: NominalState, imu_stream: ImuStream, g: float = G_MAGNITUDE) -> NominalState:
    """Integrate an IMU stream from ``initial_state`` (the state at ``t_ns[0]``).

    Sample ``k`` is held over ``[t_k, t_{k+1})``, so ``K`` samples give ``K`` states,
    one per timestamp; the last sample is not used. Returns a ``NominalState``
    whose arrays have a leading time axis: ``p[k]`` is the estimate at ``t_ns[k]``.
    Batched streams (``acc`` of shape ``(K, N, 3)``) integrate ``N`` runs at once.
    """
    K = len(imu_stream)
    x = initial_state
    batch = np.broadcast_shapes(np.shape(x.p)[:-1], np.shape(imu_stream.acc)[1:-1])
    out = NominalState(
        np.empty((K, *batch, 3)), np.empty((K, *batch, 3)), np.empty((K, *batch, 4)),
        np.empty((K, *batch, 3)), np.empty((K, *batch, 3)),
    )  # fmt: skip

    def store(k: int, s: NominalState) -> None:
        out.p[k], out.v[k], out.q[k], out.ba[k], out.bg[k] = s.p, s.v, s.q, s.ba, s.bg

    store(0, x)
    for k in range(K - 1):
        x = propagate(x, imu_stream[k], step_dt(imu_stream.t_ns, k), g)
        store(k + 1, x)
    return out
