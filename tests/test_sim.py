"""sim.py: analytic derivatives, the two IMU streams, noise statistics, GPS."""

from pathlib import Path

import numpy as np
import pytest

from navekf_py import sim, so3
from navekf_py.types import ImuNoise

ROOT = Path(__file__).resolve().parents[1]


def test_yaml_config_loads():
    cfg = sim.load_config(ROOT / "configs" / "sim.yaml")
    assert cfg.imu_rate_hz == 200.0 and cfg.gps.rate_hz == 10.0
    assert cfg.imu_noise.sigma_g == pytest.approx(1.7e-4)
    assert cfg.gps.dropouts == [[30.0, 40.0]]


def test_analytic_derivatives_match_finite_differences():
    cfg = sim.TrajectoryConfig()
    t = np.linspace(0.0, 60.0, 241)  # crosses static, ramp and steady motion
    h = 1e-5
    f, fp, fm = sim.evaluate(cfg, t), sim.evaluate(cfg, t + h), sim.evaluate(cfg, t - h)
    np.testing.assert_allclose((fp["p"] - fm["p"]) / (2 * h), f["v"], atol=1e-7)
    np.testing.assert_allclose((fp["v"] - fm["v"]) / (2 * h), f["a"], atol=1e-7)
    # body rate: Log(q(t−h)* ⊗ q(t+h)) / 2h
    w_num = so3.log(so3.quat_mul(so3.quat_conj(fm["q"]), fp["q"])) / (2 * h)
    np.testing.assert_allclose(w_num, f["w_B"], atol=1e-8)


def test_trajectory_is_a_figure8_with_attitude_motion():
    cfg = sim.TrajectoryConfig()
    f = sim.evaluate(cfg, np.linspace(cfg.static_s + cfg.ramp_s, 60.0, 2000))
    assert np.ptp(f["p"][:, 0]) > 30 and np.ptp(f["p"][:, 1]) > 15 and np.ptp(f["p"][:, 2]) > 2
    peak_rate = np.abs(f["w_B"]).max(0)
    assert np.all(peak_rate > 0.03)  # roll, pitch and yaw rates all non-trivial


def test_static_segment_then_smooth_start():
    cfg = sim.TrajectoryConfig(static_s=5.0, ramp_s=5.0)
    f = sim.evaluate(cfg, np.linspace(0, 5.0, 50))
    for k in ("v", "a", "w_B"):
        np.testing.assert_array_equal(f[k], 0.0)
    np.testing.assert_allclose(f["p"], np.tile(f["p"][0], (50, 1)))
    # continuity of p, v, a across the ramp ends
    for t0 in (5.0, 10.0):
        a, b = sim.evaluate(cfg, t0 - 1e-9), sim.evaluate(cfg, t0 + 1e-9)
        for k in ("p", "v", "a"):
            np.testing.assert_allclose(a[k], b[k], atol=1e-7)


def test_timestamps_are_int64_ns():
    t = sim.imu_timestamps(1.0, 200.0)
    assert t.dtype == np.int64 and len(t) == 201 and t[1] == 5_000_000 and t[-1] == 1_000_000_000
    with pytest.raises(ValueError):
        sim.imu_timestamps(1.0, 300.0)  # 3.333... ms is not an integer number of ns


def test_streams_agree_to_first_order():
    """Consistent and analytic streams differ by O(dt) (one is a step average, the other a point value)."""
    errs = []
    for rate in (100.0, 200.0, 400.0):
        r = sim.simulate(sim.SimConfig(duration_s=20.0, imu_rate_hz=rate))
        errs.append(np.abs(r.imu_consistent.acc[:-1] - r.imu_analytic.acc[:-1]).max())
    assert errs[0] / errs[1] == pytest.approx(2.0, rel=0.1)
    assert errs[1] / errs[2] == pytest.approx(2.0, rel=0.1)


def test_noise_free_by_default_and_seeded():
    r = sim.simulate(sim.SimConfig(duration_s=2.0))
    np.testing.assert_array_equal(r.truth.ba, 0.0)
    noisy = sim.SimConfig(duration_s=2.0, imu_noise=ImuNoise(1e-2, 1e-3, 1e-3, 1e-4), seed=7)
    a, b = sim.simulate(noisy), sim.simulate(noisy)
    np.testing.assert_array_equal(a.imu_analytic.acc, b.imu_analytic.acc)
    np.testing.assert_array_equal(a.gps[3].p_W, b.gps[3].p_W)
    noisy.seed = 8
    assert not np.array_equal(sim.simulate(noisy).imu_analytic.acc, a.imu_analytic.acc)
    # the same noise realization is on both streams: their difference doesn't depend on the noise
    np.testing.assert_allclose(a.imu_analytic.acc - a.imu_consistent.acc, r.imu_analytic.acc - r.imu_consistent.acc, atol=1e-12)
    np.testing.assert_allclose(a.imu_analytic.gyr - a.imu_consistent.gyr, r.imu_analytic.gyr - r.imu_consistent.gyr, atol=1e-12)


def test_imu_noise_statistics():
    """White noise has per-sample std σ/√dt; a bias random walk has variance σ_b² t (conventions §4)."""
    n, dt = 20_000, 0.005
    noise = ImuNoise(sigma_a=2e-3, sigma_g=1.7e-4, sigma_ba=3e-3, sigma_bg=2e-5)
    acc_err, gyr_err, ba, bg = sim.imu_noise(n, dt, noise, np.random.default_rng(0))
    white_a, white_g = acc_err - ba, gyr_err - bg
    assert white_a.std() == pytest.approx(noise.sigma_a / np.sqrt(dt), rel=0.02)
    assert white_g.std() == pytest.approx(noise.sigma_g / np.sqrt(dt), rel=0.02)
    steps_a, steps_g = np.diff(ba, axis=0), np.diff(bg, axis=0)
    assert steps_a.std() == pytest.approx(noise.sigma_ba * np.sqrt(dt), rel=0.02)
    assert steps_g.std() == pytest.approx(noise.sigma_bg * np.sqrt(dt), rel=0.02)
    np.testing.assert_array_equal(ba[0], 0.0)


def test_gps_rate_noise_lever_arm_and_dropouts():
    lever = [0.5, -0.2, 0.3]
    cfg = sim.SimConfig(duration_s=60.0, gps=sim.GpsConfig(sigma=1.0, lever_arm=lever, dropouts=[[20.0, 25.0]]))
    r = sim.simulate(cfg)
    t = np.array([f.t_ns for f in r.gps])
    assert np.all(np.diff(t)[np.diff(t) != 5_100_000_000] == 100_000_000)  # 10 Hz except the gap
    assert len(r.gps) == 601 - 50
    assert not np.any((t >= 20e9) & (t < 25e9))
    # residual against the true antenna position is N(0, 1 m²)
    idx = np.searchsorted(r.truth.t_ns, t)
    np.testing.assert_array_equal(r.truth.t_ns[idx], t)  # fixes land on IMU timestamps
    antenna = r.truth.p[idx] + np.einsum("kij,j->ki", so3.quat_to_rot(r.truth.q[idx]), lever)
    res = np.array([f.p_W for f in r.gps]) - antenna
    assert abs(res.mean()) < 0.1 and res.std() == pytest.approx(1.0, rel=0.08)
    np.testing.assert_array_equal(r.gps[0].R_meas, np.eye(3))
