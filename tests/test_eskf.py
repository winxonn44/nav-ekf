"""eskf.py: Jacobians vs finite differences, discretizations, reset, exactness, static init, gate, hygiene."""

import numpy as np
import pytest
from scipy.linalg import expm

from navekf_py import eskf, sim, so3
from navekf_py.consistency import P0_DEFAULT
from navekf_py.eskf import ESKF, EskfConfig, boxminus, boxplus
from navekf_py.propagate import propagate
from navekf_py.types import ImuNoise, ImuSample, ImuStream, NominalState, PositionFix

NOISE = ImuNoise(sigma_a=2e-3, sigma_g=1.7e-4, sigma_ba=3e-3, sigma_bg=2e-5)  # configs/sim.yaml
LEVER = np.array([0.3, -0.2, 0.5])


def random_state(seed=1) -> NominalState:
    rng = np.random.default_rng(seed)
    return NominalState(rng.standard_normal(3), rng.standard_normal(3), so3.quat_normalize(rng.standard_normal(4)),
                        0.05 * rng.standard_normal(3), 0.01 * rng.standard_normal(3))  # fmt: skip


IMU = ImuSample(np.int64(0), np.array([0.8, -0.5, 9.6]), np.array([0.4, -0.7, 0.9]))


def fd_jacobian(fun, n, eps=1e-6):
    """Central differences of ``fun: R^n -> R^m`` at 0."""
    cols = []
    for j in range(n):
        e = np.zeros(n)
        e[j] = eps
        cols.append((fun(e) - fun(-e)) / (2 * eps))
    return np.stack(cols, -1)


def discrete_error_jacobian(x, imu, dt):
    """FD of the error map through the real §6 step: ``δx_k ↦ propagate(x̂ ⊞ δx_k) ⊖ propagate(x̂)``."""
    x1 = propagate(x, imu, dt)
    return fd_jacobian(lambda e: boxminus(propagate(boxplus(x, e), imu, dt), x1), 15)


# ------------------------------------------------------------------ Jacobians


def test_boxplus_boxminus_roundtrip():
    x = random_state()
    dx = np.random.default_rng(2).standard_normal(15) * 0.3
    np.testing.assert_allclose(boxminus(boxplus(x, dx), x), dx, atol=1e-13)


def test_F_is_the_limit_of_the_discrete_step_jacobian():
    """``(J(dt) − I)/dt → F`` at first order in ``dt``; ``‖Φ₁ − J‖ = O(dt²)`` (Φ from the code)."""
    x = random_state()
    F, G = eskf.jacobians(x, IMU)
    errs_F, errs_Phi = [], []
    for dt in (2e-3, 1e-3, 5e-4):
        J = discrete_error_jacobian(x, IMU, dt)
        Phi, _ = eskf.discretize(F, G @ NOISE.Qc() @ G.T, dt, "first")
        errs_F.append(np.abs((J - np.eye(15)) / dt - F).max())
        errs_Phi.append(np.abs(Phi - J).max())
    assert errs_F[-1] < 0.01 * np.abs(F).max()
    np.testing.assert_allclose(np.array(errs_F[:-1]) / errs_F[1:], 2.0, rtol=0.05)  # O(dt)
    np.testing.assert_allclose(np.array(errs_Phi[:-1]) / errs_Phi[1:], 4.0, rtol=0.05)  # O(dt²)


def test_F_structure_and_values():
    """Nonzero blocks only where derivation §3 puts them, with the right signs."""
    x = random_state()
    F, G = eskf.jacobians(x, IMU)
    R, a, w = so3.quat_to_rot(x.q), IMU.acc - x.ba, IMU.gyr - x.bg
    E = np.zeros((15, 15))
    E[0:3, 3:6] = np.eye(3)
    E[3:6, 6:9], E[3:6, 9:12] = -R @ so3.skew(a), -R
    E[6:9, 6:9], E[6:9, 12:15] = -so3.skew(w), -np.eye(3)
    np.testing.assert_array_equal(F, E)
    GQG = G @ NOISE.Qc() @ G.T
    np.testing.assert_allclose(GQG[3:6, 3:6], NOISE.sigma_a**2 * np.eye(3), atol=1e-20)  # R σ² I Rᵀ = σ² I


def test_G_matches_fd_of_the_noise_inputs():
    """``∂δx_{k+1}/∂(n_a, n_g) = G[:, :6]·dt + O(dt²)``: perturb the readings by −n (true = measured − n)."""
    x, dt = random_state(), 1e-3
    _, G = eskf.jacobians(x, IMU)
    x1 = propagate(x, IMU, dt)

    def f(n):
        imu = ImuSample(IMU.t_ns, IMU.acc - n[0:3], IMU.gyr - n[3:6])
        return boxminus(propagate(x, imu, dt), x1)

    Jn = fd_jacobian(f, 6)
    np.testing.assert_allclose(Jn / dt, G[:, :6], atol=5 * dt)


@pytest.mark.parametrize("method,order", [("first", 2), ("second", 3)])
def test_discretization_error_order(method, order):
    """``‖Φ − expm(F dt)‖`` and ``‖Q_d − Q_d^VanLoan‖`` shrink as ``dt^order``."""
    x = random_state()
    F, G = eskf.jacobians(x, IMU)
    GQG = G @ NOISE.Qc() @ G.T
    e_phi, e_q = [], []
    for dt in (4e-2, 2e-2, 1e-2):
        Phi, Qd = eskf.discretize(F, GQG, dt, method)
        Phi_x, Qd_x = eskf.discretize(F, GQG, dt, "vanloan")
        e_phi.append(np.abs(Phi - Phi_x).max())
        e_q.append(np.abs(Qd - Qd_x).max())
    assert e_phi[0] / e_phi[1] == pytest.approx(2**order, rel=0.15)
    assert e_phi[1] / e_phi[2] == pytest.approx(2**order, rel=0.1)
    assert e_q[1] / e_q[2] == pytest.approx(2**order, rel=0.15)


def test_van_loan_matches_expm_and_quadrature():
    x = random_state()
    F, G = eskf.jacobians(x, IMU)
    GQG = G @ NOISE.Qc() @ G.T
    dt = 5e-3
    Phi, Qd = eskf.discretize(F, GQG, dt, "vanloan")
    np.testing.assert_allclose(Phi, expm(F * dt), atol=1e-13)
    tau = np.linspace(0, dt, 201)  # Simpson's rule on ∫ Φ(τ) GQG Φ(τ)ᵀ dτ
    vals = np.stack([expm(F * t) @ GQG @ expm(F * t).T for t in tau])
    w = np.ones(201)
    w[1:-1:2], w[2:-1:2] = 4, 2
    Q_quad = np.tensordot(w, vals, 1) * (dt / 600)
    np.testing.assert_allclose(Qd, Q_quad, rtol=1e-9, atol=1e-22)
    np.testing.assert_allclose(Qd, Qd.T, atol=1e-22)


def test_H_matches_fd_with_lever_arm():
    x = random_state()
    z0, H = eskf.position_model(x, LEVER)
    J = fd_jacobian(lambda e: eskf.position_model(boxplus(x, e), LEVER)[0], 15)
    np.testing.assert_allclose(H, J, atol=1e-9)
    assert np.abs(H[:, 6:9]).max() > 0.1  # the lever-arm block is really exercised
    np.testing.assert_allclose(z0, x.p + so3.quat_to_rot(x.q) @ LEVER, atol=1e-15)


@pytest.mark.parametrize("dth", [[0.1, -0.2, 0.15], [1e-3, 2e-3, -1e-3], [0.0, 0.0, 0.0]])
def test_reset_jacobian_matches_fd(dth):
    """New error after injecting ``δx̂``: ``ε ↦ (x̂ ⊞ (δx̂ + ε)) ⊖ (x̂ ⊞ δx̂)``; its Jacobian is ``G_r``."""
    x = random_state()
    dx_hat = np.concatenate([[0.1, 0.2, -0.3], [0.01, 0.0, 0.02], dth, [1e-3] * 3, [-1e-4] * 3])
    x_plus = boxplus(x, dx_hat)
    J = fd_jacobian(lambda e: boxminus(boxplus(x, dx_hat + e), x_plus), 15)
    Gr = eskf.reset_jacobian(dx_hat[6:9], "jr")
    np.testing.assert_allclose(Gr, J, atol=1e-9)
    th = np.linalg.norm(dth)
    # identity reset is off by ~θ/2 on the attitude block; Solà's I − ½[δθ]× by O(θ²)
    assert np.abs(np.eye(15) - J).max() == pytest.approx(0.5 * np.abs(dth).max(), rel=0.2, abs=1e-9)
    sola = np.eye(3) - 0.5 * so3.skew(np.asarray(dth))
    assert np.abs(sola - J[6:9, 6:9]).max() <= 0.2 * th**2 + 1e-9


def test_batched_right_jacobian_matches_single():
    phis = np.array([[0.1, -0.2, 0.15], [1e-10, 0, 0], [0, 0, 0], [2.0, 1.0, -0.5]])
    Jb = so3.right_jacobian(phis)
    for i, p in enumerate(phis):
        np.testing.assert_allclose(Jb[i], so3.right_jacobian(p), atol=1e-15)


# ------------------------------------------------------- the filter, noise-free


def consistent_fixes(r: sim.SimResult, lever, sigma=1.0, rate_hz=10.0):
    """Noise-free fixes consistent with the §6 reference (``p_ref + R l``), on the GPS schedule."""
    period = int(round(1e9 / rate_hz))
    out = []
    for k, t in enumerate(r.truth.t_ns):
        if int(t) % period == 0:
            out.append(PositionFix(t, r.p_ref_consistent[k] + so3.quat_to_rot(r.truth.q[k]) @ lever, sigma))
    return out


def test_noise_free_filter_tracks_truth_exactly():
    """Consistent stream, no noise, updates on (lever arm ≠ 0): state = truth to 1e-9, innovations ≈ 0."""
    r = sim.simulate(sim.SimConfig(duration_s=20.0))  # noise-free figure-8 with the static start
    fixes = consistent_fixes(r, LEVER)
    t0 = r.truth.state(0)
    t0.p = r.p_ref_consistent[0].copy()
    f = ESKF(t0, P0_DEFAULT, EskfConfig(noise=NOISE, lever_arm=LEVER, check_psd=True))
    ups = eskf.run(f, r.imu_consistent, fixes)
    assert len(ups) == len(fixes) == 201
    assert all(bool(u.accepted) for u in ups)
    assert max(np.abs(u.y).max() for u in ups) < 1e-9
    L = f.log.arrays()
    k = (L["t_ns"] // (r.truth.t_ns[1] - r.truth.t_ns[0])).astype(int)
    truth = NominalState(r.p_ref_consistent[k], r.truth.v[k], r.truth.q[k], r.truth.ba[k], r.truth.bg[k])
    err = boxminus(truth, NominalState.from_vector(L["x"]))
    assert np.abs(err).max() < 1e-9
    assert len(L["t_ns"]) == len(r.truth.t_ns) - 1 + len(fixes)  # one row per IMU step or update (§8)
    assert L["P_diag"].shape == (len(L["t_ns"]), 15) and np.all(L["P_diag"] > 0)
    assert np.all(np.isnan(L["nis"][L["kind"] == 0])) and np.all(L["nis"][L["kind"] == 1] >= 0)


def test_batched_filter_equals_single_runs():
    """The Monte Carlo batch runs exactly the single-run filter (to round-off)."""
    runs = [sim.simulate(sim.SimConfig(duration_s=4.0, imu_noise=NOISE, seed=s)) for s in (3, 4)]
    cfg = EskfConfig(noise=NOISE, lever_arm=LEVER, log=False)
    singles = []
    for r in runs:
        f = ESKF(r.truth.state(0), P0_DEFAULT, cfg)
        eskf.run(f, r.imu_consistent, r.gps)
        singles.append(f)
    x0 = NominalState(*(np.stack([getattr(r.truth.state(0), a) for r in runs]) for a in ("p", "v", "q", "ba", "bg")))
    imu = ImuStream(runs[0].truth.t_ns, np.stack([r.imu_consistent.acc for r in runs], 1),
                    np.stack([r.imu_consistent.gyr for r in runs], 1))  # fmt: skip
    fixes = [PositionFix(a.t_ns, np.stack([a.p_W, b.p_W]), a.sigma) for a, b in zip(runs[0].gps, runs[1].gps, strict=True)]
    fb = ESKF(x0, P0_DEFAULT, cfg)
    eskf.run(fb, imu, fixes)
    for i, f in enumerate(singles):
        np.testing.assert_allclose(fb.x.as_vector()[i], f.x.as_vector(), atol=1e-12)
        np.testing.assert_allclose(fb.P[i], f.P, rtol=1e-10, atol=1e-18)


# --------------------------------------------------------------- static init


def test_static_init_on_the_simulator_static_segment():
    """E1's 5 s still, level start, with noise and a gyro bias: roll, pitch, b_g recovered."""
    bg0 = [1e-3, -2e-3, 5e-4]
    r = sim.simulate(sim.SimConfig(duration_s=6.0, imu_noise=NOISE, bg0=bg0, seed=11))
    T = 5.0
    x0 = eskf.static_init(r.imu_analytic, int(T * 1e9), yaw=0.3, p0=np.zeros(3))
    err = boxminus(r.truth.state(0), x0)
    # σ: tilt ≈ σ_a/(g√T) ⊕ (accel-bias walk mean σ_ba√(T/3))/g ≈ 4e-4 rad; gyro bias ≈ σ_g/√T ≈ 8e-5 rad/s
    assert np.abs(err[6:8]).max() < 2e-3
    assert abs(err[8]) < 1e-6  # yaw is given (body-z error is yaw only up to tilt-error products)
    np.testing.assert_allclose(x0.bg, bg0, atol=4e-4)


def test_static_init_tilted_and_its_P0():
    """Tilted still IMU (roll 0.2, pitch −0.1, yaw 1.0) with white noise only."""
    yaw, pitch, roll = 1.0, -0.1, 0.2
    q = so3.quat_mul(so3.quat_mul(so3.exp([0, 0, yaw]), so3.exp([0, pitch, 0])), so3.exp([roll, 0, 0]))
    t = sim.imu_timestamps(5.0, 200.0)
    rng = np.random.default_rng(0)
    noise = ImuNoise(sigma_a=NOISE.sigma_a, sigma_g=NOISE.sigma_g)
    acc_e, gyr_e, _, _ = sim.imu_noise(len(t), 0.005, noise, rng, bg0=[2e-3, 1e-3, -1e-3])
    imu = ImuStream(t, sim.specific_force(q, np.zeros(3)) + acc_e, gyr_e)
    x0 = eskf.static_init(imu, int(t[-1]) + 1, yaw=yaw, p0=np.ones(3))
    truth = NominalState(np.ones(3), np.zeros(3), q, np.zeros(3), np.array([2e-3, 1e-3, -1e-3]))
    err = boxminus(truth, x0)
    sig_tilt = noise.sigma_a / (9.81 * np.sqrt(5.0))
    assert np.abs(err[6:9]).max() < 5 * sig_tilt
    assert np.abs(err[12:15]).max() < 5 * noise.sigma_g / np.sqrt(5.0)
    P0 = eskf.static_init_P0(x0, 5.0, NOISE, sigma_ba0=0.05, sigma_yaw=0.05, sigma_p=1.0)
    np.testing.assert_allclose(P0, P0.T, atol=0)
    assert np.linalg.eigvalsh(P0).min() > 0
    # the tilt error is δθ ≈ (1/g)[û]× δb_a + averaging noise: removing the bias term leaves only that noise and yaw
    u = so3.quat_to_rot(x0.q).T @ np.array([0.0, 0.0, 1.0])
    A = np.zeros((3, 15))
    A[:, 6:9], A[:, 9:12] = np.eye(3), -so3.skew(u) / 9.81
    C = A @ P0 @ A.T
    Uperp = np.eye(3) - np.outer(u, u)
    np.testing.assert_allclose(Uperp @ C @ Uperp, sig_tilt**2 * Uperp, atol=1e-15)
    assert P0[6:9, 6:9].trace() > 100 * sig_tilt**2  # while the tilt variance itself is bias-dominated


# ---------------------------------------------------------------------- gate


def test_gate_rejects_a_20m_outlier_and_logs_it():
    r = sim.simulate(sim.SimConfig(duration_s=10.0, imu_noise=NOISE, seed=8))  # seed 8: no natural NIS > 11.345
    fixes = list(r.gps)
    m = next(i for i, f in enumerate(fixes) if f.t_ns == 8 * 10**9)
    fixes[m] = PositionFix(fixes[m].t_ns, fixes[m].p_W + np.array([20.0, 0.0, 0.0]), fixes[m].sigma)
    f = ESKF(r.truth.state(0), P0_DEFAULT, EskfConfig(noise=NOISE))
    ups = eskf.run(f, r.imu_consistent, fixes)
    bad = ups[m]
    assert not bool(bad.accepted) and float(bad.nis) > 100
    assert [(rj.t_ns, rj.run) for rj in f.rejections] == [(8 * 10**9, None)]
    assert all(bool(u.accepted) for i, u in enumerate(ups) if i != m)  # the next fixes are fine again
    L = f.log.arrays()
    row = np.flatnonzero((L["kind"] == 1) & (L["t_ns"] == 8 * 10**9))[0]
    np.testing.assert_array_equal(L["x"][row], L["x"][row - 1])  # rejected: state untouched
    np.testing.assert_array_equal(L["P_diag"][row], L["P_diag"][row - 1])


def test_gate_off_accepts_the_outlier():
    x = random_state()
    f = ESKF(x, P0_DEFAULT, EskfConfig(noise=NOISE, gate=None))
    u = f.update_position(PositionFix(np.int64(0), x.p + 20.0, 1.0))
    assert bool(u.accepted) and f.rejections == []


def test_update_uses_joseph_form_and_stays_symmetric():
    """One update by hand (inverse-based, standard form) vs the code (solve, Joseph); identity reset."""
    x = random_state()
    A = np.random.default_rng(3).standard_normal((15, 15))
    P = A @ A.T * 0.01 + np.eye(15) * 1e-3
    f = ESKF(x, P, EskfConfig(noise=NOISE, lever_arm=LEVER, reset="identity", gate=None))
    z = x.p + so3.quat_to_rot(x.q) @ LEVER + np.array([0.3, -0.1, 0.2])
    u = f.update_position(PositionFix(np.int64(0), z, 0.5), inject=False)
    _, H = eskf.position_model(x, LEVER)
    S = H @ P @ H.T + 0.25 * np.eye(3)
    K = P @ H.T @ np.linalg.inv(S)
    np.testing.assert_allclose(u.S, S, rtol=1e-12)
    np.testing.assert_allclose(f.dx, K @ u.y, rtol=1e-9, atol=1e-14)
    np.testing.assert_allclose(f.P, (np.eye(15) - K @ H) @ P, rtol=1e-8, atol=1e-14)
    np.testing.assert_array_equal(f.P, f.P.T)
    assert float(u.nis) == pytest.approx(u.y @ np.linalg.inv(S) @ u.y, rel=1e-12)


# ------------------------------------------------------------------ hygiene


def test_min_eigenvalue_check_raises_on_a_broken_P():
    x = random_state()
    P = P0_DEFAULT.copy()
    P[0, 0] = -1.0
    with pytest.raises(FloatingPointError):
        ESKF(x, P, EskfConfig(noise=NOISE, check_psd=True))
    f = ESKF(x, P0_DEFAULT, EskfConfig(noise=NOISE, check_psd=True))
    f.propagate(IMU, 0.005)
    assert f.min_eig > 0


def test_quaternion_stays_normalized_and_options_validated():
    x = random_state()
    f = ESKF(x, P0_DEFAULT, EskfConfig(noise=NOISE))
    for _ in range(50):
        f.propagate(IMU, 0.005)
        f.update_position(PositionFix(f.t_ns, f.x.p + 0.1, 1.0))
    assert abs(np.linalg.norm(f.x.q) - 1.0) < 1e-15
    with pytest.raises(ValueError):
        EskfConfig(noise=NOISE, discretization="rk4")
    with pytest.raises(ValueError):
        EskfConfig(noise=NOISE, reset="none")


def test_fix_off_the_imu_grid_raises():
    r = sim.simulate(sim.SimConfig(duration_s=1.0))
    f = ESKF(r.truth.state(0), P0_DEFAULT, EskfConfig(noise=NOISE))
    with pytest.raises(ValueError, match="not on an IMU timestamp"):
        eskf.run(f, r.imu_consistent, [PositionFix(np.int64(12_345), np.zeros(3), 1.0)])
