"""A small synthetic sequence in EuRoC's on-disk layout, generated from the simulator (for CI; no dataset needed).

The IMU is the §7 consistent stream at 200 Hz and ground truth is the matching §6 reference, so a noise-free run of the
whole chain (loader, static init, fixes, filter) must track ground truth tightly. Options exercise what the real data does:
ground truth starting after the IMU, a body frame with x up (near gimbal lock for ZYX Euler angles), and a non-identity
``T_BS`` on the IMU.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import sim, so3
from .types import ImuNoise

R_BS_X_UP = np.array([[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])  # S's x axis along B's z axis


@dataclass
class FixtureConfig:
    name: str = "MH_00_fixture"
    duration_s: float = 40.0
    static_s: float = 5.0
    gt_delay_s: float = 0.5  # ground truth starts this long after the IMU
    seed: int = 0
    noise: ImuNoise = field(default_factory=ImuNoise)  # default: noise-free
    x_up: bool = False  # IMU frame with x up, ground truth in that frame (as on the EuRoC MAV)
    imu_R_BS: np.ndarray | None = None  # write the samples in a sensor frame and record this T_BS


def _yaml_t_bs(R: np.ndarray) -> str:
    T = np.eye(4)
    T[:3, :3] = R
    return "T_BS:\n  cols: 4\n  rows: 4\n  data: [" + ", ".join(f"{v:.17g}" for v in T.ravel()) + "]\n"


def write_fixture(root: str | Path, cfg: FixtureConfig | None = None) -> str:
    """Write the sequence under ``root/<name>/mav0`` and return ``name``."""
    cfg = cfg or FixtureConfig()
    traj = sim.TrajectoryConfig(static_s=cfg.static_s, ramp_s=5.0)
    r = sim.simulate(sim.SimConfig(duration_s=cfg.duration_s, imu_noise=cfg.noise, seed=cfg.seed, trajectory=traj))

    acc, gyr, q = r.imu_consistent.acc, r.imu_consistent.gyr, r.truth.q
    if cfg.x_up:
        R_BS = R_BS_X_UP
        acc = (R_BS.T @ acc[..., None])[..., 0]
        gyr = (R_BS.T @ gyr[..., None])[..., 0]
        q = so3.quat_mul(q, so3.rot_to_quat(R_BS))
    R_file = np.eye(3)
    if cfg.imu_R_BS is not None:
        R_file = np.asarray(cfg.imu_R_BS, dtype=float)
        acc = (R_file.T @ acc[..., None])[..., 0]
        gyr = (R_file.T @ gyr[..., None])[..., 0]

    mav = Path(root) / cfg.name / "mav0"
    (mav / "imu0").mkdir(parents=True, exist_ok=True)
    (mav / "state_groundtruth_estimate0").mkdir(parents=True, exist_ok=True)

    t = r.truth.t_ns + 1_403_636_579_000_000_000  # EuRoC-sized timestamps: 19 digits, more than a double holds
    with open(mav / "imu0" / "data.csv", "w") as f:
        f.write(
            "#timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],w_RS_S_z [rad s^-1],a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]\n"  # noqa: E501
        )
        for ti, w, a in zip(t, gyr, acc, strict=True):
            f.write(f"{ti},{w[0]:.17g},{w[1]:.17g},{w[2]:.17g},{a[0]:.17g},{a[1]:.17g},{a[2]:.17g}\n")

    delay = int(round(cfg.gt_delay_s * 200))
    with open(mav / "state_groundtruth_estimate0" / "data.csv", "w") as f:
        f.write(
            "#timestamp, p_RS_R_x [m], p_RS_R_y [m], p_RS_R_z [m], q_RS_w [], q_RS_x [], q_RS_y [], q_RS_z [], v_RS_R_x [m s^-1], v_RS_R_y [m s^-1], v_RS_R_z [m s^-1], b_w_RS_S_x [rad s^-1], b_w_RS_S_y [rad s^-1], b_w_RS_S_z [rad s^-1], b_a_RS_S_x [m s^-2], b_a_RS_S_y [m s^-2], b_a_RS_S_z [m s^-2]\n"  # noqa: E501
        )
        for k in range(delay, len(t) - delay):
            p, qk, v = r.p_ref_consistent[k], q[k], r.truth.v[k]
            f.write(
                f"{t[k]},{p[0]:.17g},{p[1]:.17g},{p[2]:.17g},{qk[0]:.17g},{qk[1]:.17g},{qk[2]:.17g},{qk[3]:.17g},{v[0]:.17g},{v[1]:.17g},{v[2]:.17g},0,0,0,0,0,0\n"
            )

    n = cfg.noise
    (mav / "imu0" / "sensor.yaml").write_text(
        "#Synthetic IMU yaml\nsensor_type: imu\n"
        + _yaml_t_bs(R_file)
        + f"rate_hz: 200\ngyroscope_noise_density: {n.sigma_g:.17g}\n"
        f"gyroscope_random_walk: {n.sigma_bg:.17g}\naccelerometer_noise_density: {n.sigma_a:.17g}\naccelerometer_random_walk: {n.sigma_ba:.17g}\n"  # noqa: E501
    )
    (mav / "state_groundtruth_estimate0" / "sensor.yaml").write_text("sensor_type: visual-inertial\n" + _yaml_t_bs(np.eye(3)))
    (mav / "body.yaml").write_text("comment: synthetic\n")
    return cfg.name
