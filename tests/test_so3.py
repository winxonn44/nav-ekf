"""so3.py: round trips, finite differences, and agreement between representations."""

import numpy as np
import pytest

from navekf_py import so3


def rodrigues(phi):
    th = np.linalg.norm(phi)
    if th == 0:
        return np.eye(3)
    K = so3.skew(phi / th)
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


@pytest.fixture
def rng():
    return np.random.default_rng(1)


def random_rotvec(rng, max_angle=np.pi - 1e-6):
    u = rng.standard_normal(3)
    return u / np.linalg.norm(u) * rng.uniform(0, max_angle)


def test_skew_is_cross_product(rng):
    a, b = rng.standard_normal((2, 3))
    np.testing.assert_allclose(so3.skew(a) @ b, np.cross(a, b), atol=1e-15)
    np.testing.assert_array_equal(so3.skew(a), -so3.skew(a).T)


def test_skew_matches_conventions_layout():
    ax, ay, az = 1.0, 2.0, 3.0
    np.testing.assert_array_equal(so3.skew([ax, ay, az]), [[0, -az, ay], [az, 0, -ax], [-ay, ax, 0]])


@pytest.mark.parametrize("angle", [0.0, 1e-12, 1e-9, 1e-7, 1e-3, 0.5, 2.0, 3.0, np.pi - 1e-9])
def test_log_exp_roundtrip(rng, angle):
    for _ in range(20):
        u = rng.standard_normal(3)
        phi = u / np.linalg.norm(u) * angle
        np.testing.assert_allclose(so3.log(so3.exp(phi)), phi, atol=1e-14 * max(1.0, angle), rtol=1e-12)


def test_exp_log_roundtrip_quaternion(rng):
    for _ in range(200):
        q = so3.quat_normalize(rng.standard_normal(4))
        back = so3.exp(so3.log(q))
        np.testing.assert_allclose(so3.quat_canonical(back), so3.quat_canonical(q), atol=1e-14)


def test_log_norm_at_most_pi_and_sign_invariant(rng):
    for _ in range(200):
        q = so3.quat_normalize(rng.standard_normal(4))
        phi = so3.log(q)
        assert np.linalg.norm(phi) <= np.pi + 1e-15
        np.testing.assert_allclose(so3.log(-q), phi, atol=1e-15)


def test_exp_small_angle_branch_is_continuous():
    """Both branches give [1, φ/2] to machine precision on either side of the 1e-8 switch."""
    u = np.array([0.3, -0.5, 0.8]) / np.linalg.norm([0.3, -0.5, 0.8])
    for angle in (1e-8 * (1 - 1e-6), 1e-8 * (1 + 1e-6)):
        q = so3.exp(u * angle)
        assert q[0] == 1.0
        np.testing.assert_allclose(q[1:], 0.5 * u * angle, rtol=1e-15)


def test_quat_to_rot_matches_rodrigues(rng):
    for _ in range(50):
        phi = random_rotvec(rng)
        R = so3.quat_to_rot(so3.exp(phi))
        np.testing.assert_allclose(R, rodrigues(phi), atol=1e-14)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-14)
        assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-14)


def test_quat_mul_is_hamilton_and_composes_rotations(rng):
    i, j, k = np.eye(4)[1], np.eye(4)[2], np.eye(4)[3]
    np.testing.assert_array_equal(so3.quat_mul(i, j), k)  # ij = k (Hamilton)
    np.testing.assert_array_equal(so3.quat_mul(so3.quat_mul(i, j), k), [-1, 0, 0, 0])  # ijk = -1
    for _ in range(50):
        qa, qb = (so3.quat_normalize(rng.standard_normal(4)) for _ in range(2))
        np.testing.assert_allclose(so3.quat_to_rot(so3.quat_mul(qa, qb)), so3.quat_to_rot(qa) @ so3.quat_to_rot(qb), atol=1e-14)


def test_rotation_by_quaternion_sandwich(rng):
    q = so3.quat_normalize(rng.standard_normal(4))
    v = rng.standard_normal(3)
    vq = so3.quat_mul(so3.quat_mul(q, np.r_[0.0, v]), so3.quat_conj(q))
    np.testing.assert_allclose(vq[1:], so3.quat_to_rot(q) @ v, atol=1e-14)
    assert abs(vq[0]) < 1e-15


def test_rot_to_quat_roundtrip(rng):
    for _ in range(200):
        q = so3.quat_canonical(so3.quat_normalize(rng.standard_normal(4)))
        np.testing.assert_allclose(so3.rot_to_quat(so3.quat_to_rot(q)), q, atol=1e-14)


def test_exp_derivative_is_body_rate(rng):
    """d/dt (q ⊗ Exp(ω t)) at t=0 is ½ q ⊗ [0, ω] (conventions §4 kinematics)."""
    q = so3.quat_normalize(rng.standard_normal(4))
    w = rng.standard_normal(3)
    h = 1e-6
    num = (so3.quat_mul(q, so3.exp(w * h)) - so3.quat_mul(q, so3.exp(-w * h))) / (2 * h)
    np.testing.assert_allclose(num, 0.5 * so3.quat_mul(q, np.r_[0.0, w]), atol=1e-9)


@pytest.mark.parametrize("angle", [0.0, 1e-10, 0.02, 1.0, 2.5])
def test_right_jacobian_finite_difference(rng, angle):
    """Exp(φ + ε) ≈ Exp(φ) Exp(J_r(φ) ε)  ⇒  J_r = ∂ Log(Exp(φ)* ⊗ Exp(φ + ε)) / ∂ε."""
    u = rng.standard_normal(3)
    phi = u / np.linalg.norm(u) * angle
    eps = 1e-6
    J = np.zeros((3, 3))
    for i in range(3):
        e = np.zeros(3)
        e[i] = eps
        f = lambda d: so3.log(so3.quat_mul(so3.quat_conj(so3.exp(phi)), so3.exp(phi + d)))  # noqa: E731
        J[:, i] = (f(e) - f(-e)) / (2 * eps)
    np.testing.assert_allclose(J, so3.right_jacobian(phi), atol=1e-9)


def test_functions_broadcast_over_batch(rng):
    phis = rng.standard_normal((7, 3))
    phis[0] = 1e-12  # one small-angle row inside the batch
    qs = so3.exp(phis)
    for i in range(7):
        np.testing.assert_array_equal(qs[i], so3.exp(phis[i]))
        np.testing.assert_array_equal(so3.quat_to_rot(qs)[i], so3.quat_to_rot(qs[i]))
        np.testing.assert_array_equal(so3.log(qs)[i], so3.log(qs[i]))
