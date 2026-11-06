"""Frame changes at the I/O boundary (docs/conventions.md §2.2).

The filter is ENU world / FLU body. VoroNav and TartanGround use NED world /
FRD body; convert with these helpers and nowhere else.
"""

from __future__ import annotations

import numpy as np

R_ENU_NED = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])  # its own inverse
R_FRD_FLU = np.diag([1.0, -1.0, -1.0])  # FRD <-> FLU body change, also its own inverse


def ned_pose_to_enu(R_NED_FRD: np.ndarray, p_NED: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``R_ENU_FLU = R_ENU_NED · R_NED_FRD · diag(1, −1, −1)``, ``p_ENU = R_ENU_NED · p_NED``."""
    return R_ENU_NED @ R_NED_FRD @ R_FRD_FLU, R_ENU_NED @ p_NED


def enu_pose_to_ned(R_ENU_FLU: np.ndarray, p_ENU: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of :func:`ned_pose_to_enu`."""
    return R_ENU_NED @ R_ENU_FLU @ R_FRD_FLU, R_ENU_NED @ p_ENU
