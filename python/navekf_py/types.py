"""Plain data types shared by the simulator, the integrator and (later) the filter.

Frames and units follow docs/conventions.md: ENU world ``W``, FLU body ``B``,
SI units, ``int64`` nanosecond timestamps, quaternions ``[w, x, y, z]``.
Arrays may carry leading batch axes (Monte Carlo runs); the last axis is the
vector axis.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np

G_MAGNITUDE = 9.81  # m/s², conventions §4 default


def gravity_w(g: float = G_MAGNITUDE) -> np.ndarray:
    """``g_W = [0, 0, −g]`` in ENU."""
    return np.array([0.0, 0.0, -g])


@dataclass
class NominalState:
    """Nominal state ``x̂`` (conventions §5): ``p_W, v_W, q_W_B, b_a, b_g``."""

    p: np.ndarray  # (..., 3) position of B in W [m]
    v: np.ndarray  # (..., 3) velocity in W [m/s]
    q: np.ndarray  # (..., 4) q_W_B, [w, x, y, z]
    ba: np.ndarray  # (..., 3) accelerometer bias [m/s²]
    bg: np.ndarray  # (..., 3) gyro bias [rad/s]

    def as_vector(self) -> np.ndarray:
        """The 16-number layout of conventions §5: ``[p, v, q, b_a, b_g]``."""
        return np.concatenate([self.p, self.v, self.q, self.ba, self.bg], -1)

    @classmethod
    def from_vector(cls, x: np.ndarray) -> NominalState:
        x = np.asarray(x, dtype=float)
        return cls(x[..., 0:3], x[..., 3:6], x[..., 6:10], x[..., 10:13], x[..., 13:16])

    def copy(self) -> NominalState:
        return NominalState(self.p.copy(), self.v.copy(), self.q.copy(), self.ba.copy(), self.bg.copy())


@dataclass
class ImuSample:
    """One IMU reading, held over ``[t_ns, t_ns_next)`` (zero-order hold, §6)."""

    t_ns: np.int64
    acc: np.ndarray  # (..., 3) specific force a_m in B [m/s²]
    gyr: np.ndarray  # (..., 3) angular rate ω_m in B [rad/s]


@dataclass
class ImuStream:
    """A sequence of IMU samples stored as arrays.

    ``kind`` records which simulator stream this is ("consistent" or "analytic",
    conventions §7) or the dataset it came from; every result should report it.
    """

    t_ns: np.ndarray  # (K,) int64
    acc: np.ndarray  # (K, ..., 3)
    gyr: np.ndarray  # (K, ..., 3)
    kind: str = "unknown"

    def __len__(self) -> int:
        return len(self.t_ns)

    def __getitem__(self, k: int) -> ImuSample:
        return ImuSample(self.t_ns[k], self.acc[k], self.gyr[k])

    def __iter__(self) -> Iterator[ImuSample]:
        for k in range(len(self)):
            yield self[k]


@dataclass
class PositionFix:
    """A position fix of the GPS antenna in ``W`` (derivation §5)."""

    t_ns: np.int64
    p_W: np.ndarray  # (3,) measured antenna position [m]
    sigma: float  # per-axis std [m]; R_meas = sigma² I

    @property
    def R_meas(self) -> np.ndarray:
        return self.sigma**2 * np.eye(3)


@dataclass
class ImuNoise:
    """Continuous-time noise densities, per axis (conventions §4).

    Sampled at period ``dt``: white noise has per-sample std ``σ/√dt``, and a
    bias steps by ``σ_b √dt N(0, 1)`` per sample.
    """

    sigma_a: float = 0.0  # accel white noise      [m/s²/√Hz]
    sigma_g: float = 0.0  # gyro white noise (ARW) [rad/s/√Hz]
    sigma_ba: float = 0.0  # accel bias random walk [m/s³/√Hz]
    sigma_bg: float = 0.0  # gyro bias random walk  [rad/s²/√Hz]

    def Qc(self) -> np.ndarray:
        """``Q_c = diag(σ_a² I, σ_g² I, σ_ba² I, σ_bg² I)`` (12×12), for E2."""
        s = [self.sigma_a**2, self.sigma_g**2, self.sigma_ba**2, self.sigma_bg**2]
        return np.diag(np.repeat(s, 3))

    @property
    def is_zero(self) -> bool:
        return self.sigma_a == self.sigma_g == self.sigma_ba == self.sigma_bg == 0.0


@dataclass
class Truth:
    """Ground truth sampled at the IMU timestamps (always the analytic trajectory)."""

    t_ns: np.ndarray  # (K,) int64
    p: np.ndarray  # (K, 3)
    v: np.ndarray  # (K, 3)
    a: np.ndarray  # (K, 3) acceleration in W
    q: np.ndarray  # (K, 4) q_W_B
    w_B: np.ndarray  # (K, 3) body angular rate
    ba: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))  # (K, 3) true accel bias
    bg: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))  # (K, 3) true gyro bias

    def state(self, k: int) -> NominalState:
        """True state at sample ``k``, a natural initial state for dead reckoning."""
        return NominalState(self.p[k].copy(), self.v[k].copy(), self.q[k].copy(), self.ba[k].copy(), self.bg[k].copy())
