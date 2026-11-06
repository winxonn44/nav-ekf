"""The convention tests named in docs/conventions.md."""

import numpy as np

from navekf_py import frames, io, sim, so3
from navekf_py.types import gravity_w


def test_static_imu_reads_plus_g():
    """§4: a still, level IMU reads a_m = [0, 0, +9.81] (specific force, not acceleration)."""
    np.testing.assert_array_equal(sim.specific_force(np.array([1.0, 0, 0, 0]), np.zeros(3)), [0, 0, 9.81])
    np.testing.assert_array_equal(gravity_w(), [0, 0, -9.81])
    # A pure yaw leaves it unchanged; the simulator's static segment reads the same.
    np.testing.assert_allclose(sim.specific_force(so3.exp([0, 0, 1.2]), np.zeros(3)), [0, 0, 9.81], atol=1e-15)
    r = sim.simulate(sim.SimConfig(duration_s=4.0))
    for stream in (r.imu_analytic, r.imu_consistent):
        np.testing.assert_allclose(stream.acc[:-1], np.tile([0, 0, 9.81], (len(stream) - 1, 1)), atol=1e-12)
        np.testing.assert_allclose(stream.gyr, 0.0, atol=1e-12)


def test_static_imu_tilted_sees_gravity_in_body():
    """Pitched nose-down by 90° (FLU): body x points down, so the up-reaction reads along body −x."""
    q = so3.exp([0, np.pi / 2, 0])  # +90° about body y (left): x -> −z, nose goes down
    np.testing.assert_allclose(so3.quat_to_rot(q) @ [1, 0, 0], [0, 0, -1], atol=1e-15)
    np.testing.assert_allclose(sim.specific_force(q, np.zeros(3)), [-9.81, 0, 0], atol=1e-14)


def test_ned_enu_roundtrip():
    """§2.2: NED/FRD <-> ENU/FLU, round trip and a 'facing north, level' pose."""
    np.testing.assert_array_equal(frames.R_ENU_NED @ frames.R_ENU_NED, np.eye(3))
    rng = np.random.default_rng(3)
    for _ in range(50):
        R_NED_FRD = so3.quat_to_rot(so3.quat_normalize(rng.standard_normal(4)))
        p_NED = rng.standard_normal(3)
        R_ENU_FLU, p_ENU = frames.ned_pose_to_enu(R_NED_FRD, p_NED)
        np.testing.assert_allclose(np.linalg.det(R_ENU_FLU), 1.0, atol=1e-14)
        R_back, p_back = frames.enu_pose_to_ned(R_ENU_FLU, p_ENU)
        np.testing.assert_allclose(R_back, R_NED_FRD, atol=1e-15)
        np.testing.assert_allclose(p_back, p_NED, atol=1e-15)

    # Facing north, level: NED/FRD is identity. In ENU/FLU, forward = north (+y),
    # left = west (−x), up = +z, i.e. a +90° yaw.
    R_ENU_FLU, p_ENU = frames.ned_pose_to_enu(np.eye(3), np.array([10.0, 20.0, -5.0]))  # 10 N, 20 E, 5 up
    np.testing.assert_array_equal(R_ENU_FLU, [[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    np.testing.assert_allclose(R_ENU_FLU, so3.quat_to_rot(so3.exp([0, 0, np.pi / 2])), atol=1e-15)
    np.testing.assert_array_equal(p_ENU, [20.0, 10.0, 5.0])


def test_tum_quaternion_order(tmp_path):
    """§3/§8: memory is [w,x,y,z]; TUM files are `t x y z qx qy qz qw`."""
    q = so3.exp([0.0, 0.0, 0.5])  # yaw 0.5 rad: [cos .25, 0, 0, sin .25]
    t_ns = np.array([1_403_636_579_758_555_392, 1_403_636_579_763_555_584], dtype=np.int64)  # EuRoC-sized
    p = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    path = tmp_path / "traj.tum"
    io.write_tum(path, t_ns, p, np.stack([q, q]))

    cols = path.read_text().splitlines()[0].split()
    assert cols[0] == "1403636579.758555392"  # exact ns, %.9f
    assert [float(c) for c in cols[1:4]] == [1.0, 2.0, 3.0]
    np.testing.assert_allclose([float(c) for c in cols[4:8]], [0.0, 0.0, np.sin(0.25), np.cos(0.25)], atol=1e-9)

    t2, p2, q2 = io.read_tum(path)
    np.testing.assert_array_equal(t2, t_ns)
    np.testing.assert_allclose(p2, p)
    np.testing.assert_allclose(q2[0], q, atol=1e-9)
