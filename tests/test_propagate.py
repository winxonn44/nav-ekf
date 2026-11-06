"""propagate.py: the §6 integrator, exactness on the consistent stream, O(dt) on the analytic one."""

import numpy as np
import pytest

from navekf_py import sim, so3
from navekf_py.propagate import dead_reckon, propagate
from navekf_py.types import ImuSample, ImuStream, NominalState


def rot_err(q_ref, q):
    return np.linalg.norm(so3.log(so3.quat_mul(so3.quat_conj(q_ref), q)), axis=-1)


def test_one_step_matches_conventions_section6():
    """A hand-written §6 step, with biases, using plain numpy matrix algebra."""
    rng = np.random.default_rng(5)
    x = NominalState(rng.standard_normal(3), rng.standard_normal(3), so3.quat_normalize(rng.standard_normal(4)),
                     0.05 * rng.standard_normal(3), 0.01 * rng.standard_normal(3))  # fmt: skip
    imu = ImuSample(np.int64(0), np.array([0.3, -0.2, 9.9]), np.array([0.4, -0.3, 0.8]))
    dt, g_W = 0.005, np.array([0, 0, -9.81])

    a_W = so3.quat_to_rot(x.q) @ (imu.acc - x.ba) + g_W
    p = x.p + x.v * dt + 0.5 * a_W * dt**2
    v = x.v + a_W * dt
    q = so3.quat_normalize(so3.quat_mul(x.q, so3.exp((imu.gyr - x.bg) * dt)))

    y = propagate(x, imu, dt)
    np.testing.assert_allclose(y.p, p, atol=1e-15)
    np.testing.assert_allclose(y.v, v, atol=1e-15)
    np.testing.assert_allclose(y.q, q, atol=1e-16)
    np.testing.assert_array_equal(y.ba, x.ba)
    np.testing.assert_array_equal(y.bg, x.bg)
    assert np.linalg.norm(y.q) == pytest.approx(1.0, abs=1e-15)


def test_consistent_stream_dead_reckoning_is_exact():
    """§7.1: noise-free DR on the consistent stream reproduces q, v, p_ref to ~1e-9 (60 s at 200 Hz)."""
    cfg = sim.SimConfig(duration_s=60.0, imu_rate_hz=200.0)
    r = sim.simulate(cfg)
    assert len(r.truth.t_ns) == 12_001 and r.imu_consistent.kind == "consistent"
    x = dead_reckon(r.truth.state(0), r.imu_consistent, cfg.g)
    assert rot_err(r.truth.q, x.q).max() < 1e-9
    assert np.abs(x.v - r.truth.v).max() < 1e-9
    assert np.abs(x.p - r.p_ref_consistent).max() < 1e-9
    # p_ref is the trapezoid integral of the true v: close to, but not exactly, the analytic p
    assert np.abs(r.p_ref_consistent - r.truth.p).max() < 1e-4


def test_analytic_stream_error_is_first_order():
    """§7.2: DR error against the analytic truth shrinks as O(dt) (log-log slope ≈ 1 over 3 rates)."""
    rates = np.array([100.0, 200.0, 400.0])
    errs = []
    for rate in rates:
        r = sim.simulate(sim.SimConfig(duration_s=30.0, imu_rate_hz=rate))
        x = dead_reckon(r.truth.state(0), r.imu_analytic)
        errs.append(np.linalg.norm(x.p - r.truth.p, axis=1).max())
    slope = np.polyfit(np.log(1 / rates), np.log(errs), 1)[0]
    assert slope == pytest.approx(1.0, abs=0.05)
    assert errs[-1] < 0.1  # 4.4 cm after 30 s at 400 Hz


def test_dead_reckon_bias_correction():
    """A known constant bias, removed by the nominal bias estimate, gives the exact result again."""
    cfg = sim.SimConfig(duration_s=10.0, ba0=[0.05, -0.02, 0.1], bg0=[0.01, 0.0, -0.02])
    r = sim.simulate(cfg)
    x = dead_reckon(r.truth.state(0), r.imu_consistent)  # truth.state carries the true biases
    assert np.abs(x.p - r.p_ref_consistent).max() < 1e-10
    x0 = r.truth.state(0)
    x0.ba[:] = 0.0  # ignoring the bias makes it drift
    assert np.abs(dead_reckon(x0, r.imu_consistent).p - r.p_ref_consistent).max() > 1.0


def test_batched_dead_reckoning_equals_single_runs():
    r = sim.simulate(sim.SimConfig(duration_s=3.0, imu_noise=sim.ImuNoise(1e-2, 1e-3, 0, 0)))
    rng = np.random.default_rng(0)
    N = 4
    acc = r.imu_analytic.acc[:, None, :] + 0.01 * rng.standard_normal((len(r.truth.t_ns), N, 3))
    gyr = np.repeat(r.imu_analytic.gyr[:, None, :], N, 1)
    x0 = r.truth.state(0)
    batch = dead_reckon(x0, ImuStream(r.truth.t_ns, acc, gyr))
    for i in range(N):
        single = dead_reckon(x0, ImuStream(r.truth.t_ns, acc[:, i], gyr[:, i]))
        np.testing.assert_array_equal(batch.p[:, i], single.p)
        np.testing.assert_array_equal(batch.q[:, i], single.q)


def test_output_layout_one_state_per_timestamp():
    imu = sim.static_imu(1.0, 100.0)
    x0 = NominalState(np.zeros(3), np.zeros(3), np.array([1.0, 0, 0, 0]), np.zeros(3), np.zeros(3))
    out = dead_reckon(x0, imu)
    assert out.p.shape == (101, 3) and out.q.shape == (101, 4)
    np.testing.assert_allclose(out.p, 0.0, atol=1e-13)  # still, level, perfect IMU stays put
    np.testing.assert_array_equal(out.q, np.tile([1.0, 0, 0, 0], (101, 1)))
