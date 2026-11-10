"""E4: EuRoC loader, static init, synthetic fixes, outage placement, on a synthetic EuRoC-layout fixture (no dataset)."""

import numpy as np
import pytest

from navekf_py import eskf, euroc, euroc_eval, so3
from navekf_py.euroc_fixture import R_BS_X_UP, FixtureConfig, write_fixture
from navekf_py.io import read_tum
from navekf_py.types import ImuNoise, ImuStream

INIT = euroc.InitConfig()
STRICT = euroc.InitConfig(
    acc_std_max=1e-6, gyro_std_max=1e-6
)  # noise-free fixture: the default thresholds pass the slow ramp out of rest
FIXES_LOW = euroc.FixConfig(rate_hz=10.0, sigma=1e-6, seed=7)
ZERO = ImuNoise()


@pytest.fixture
def root(tmp_path):
    return tmp_path


def load(root, **kw):
    cfg = FixtureConfig(**kw)
    return euroc.load_sequence(root, write_fixture(root, cfg))


def filter_cfg(noise=None):
    return eskf.EskfConfig(noise=noise or ImuNoise(sigma_a=2e-3, sigma_g=1.7e-4, sigma_ba=3e-3, sigma_bg=2e-5), g=9.81)


# ------------------------------------------------------------------------- loader


def test_timestamps_keep_all_19_digits(root):
    seq = load(root)
    assert seq.imu.t_ns.dtype == np.int64
    assert np.diff(seq.imu.t_ns).max() == np.diff(seq.imu.t_ns).min() == 5_000_000
    assert seq.imu.t_ns[0] % 1000 == 0 and seq.imu.t_ns[0] > 10**18  # a double would have lost this


def test_ground_truth_starts_after_imu(root):
    seq = load(root, gt_delay_s=0.5)
    assert int(seq.gt.t_ns[0]) - int(seq.imu.t_ns[0]) == 500_000_000
    assert euroc.data_report(seq, INIT)["gt_starts_after_imu_s"] == pytest.approx(0.5)


def test_sensor_yaml_noise_densities(root):
    seq = load(root, noise=ImuNoise(sigma_a=2e-3, sigma_g=1.7e-4, sigma_ba=3e-3, sigma_bg=2e-5))
    assert seq.noise == ImuNoise(sigma_a=2e-3, sigma_g=1.7e-4, sigma_ba=3e-3, sigma_bg=2e-5)
    assert seq.rate_hz == 200.0


def test_duplicate_timestamp_raises(root):
    name = write_fixture(root)
    path = root / name / "mav0" / "imu0" / "data.csv"
    lines = path.read_text().splitlines()
    lines[5] = lines[4]
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match="strictly increase"):
        euroc.load_sequence(root, name)


def test_imu_t_bs_rotation_is_applied(root):
    """Samples written in a sensor frame and rotated back by T_BS read the same as the identity fixture."""
    plain = load(root, name="plain")
    rotated = load(root, name="rotated", imu_R_BS=R_BS_X_UP)
    np.testing.assert_allclose(rotated.imu.acc, plain.imu.acc, atol=1e-12)
    np.testing.assert_allclose(rotated.imu.gyr, plain.imu.gyr, atol=1e-12)


def test_imu_t_bs_translation_raises(root):
    name = write_fixture(root)
    path = root / name / "mav0" / "imu0" / "sensor.yaml"
    text = path.read_text()
    assert "data: [1, 0, 0, 0," in text
    path.write_text(text.replace("data: [1, 0, 0, 0,", "data: [1, 0, 0, 0.1,"))
    with pytest.raises(ValueError, match="translation"):
        euroc.load_sequence(root, name)


def test_csv_header_comment_is_skipped_and_columns_map(root):
    seq = load(root, static_s=5.0)
    # still and level at the start: specific force up, no rotation
    m = seq.imu.t_ns < seq.imu.t_ns[0] + 1_000_000_000
    np.testing.assert_allclose(seq.imu.acc[m].mean(0), [0.0, 0.0, 9.81], atol=1e-9)
    np.testing.assert_allclose(seq.imu.gyr[m].mean(0), [0.0, 0.0, 0.0], atol=1e-12)


# ---------------------------------------------------------------------- stillness


def test_still_segment_found_at_start(root):
    seq = load(root, static_s=5.0)
    segments = euroc.find_still_segments(seq.imu, seq.rate_hz, INIT)
    assert segments[0].t_start_ns == seq.imu.t_ns[0]
    assert 4.5 <= segments[0].duration_s <= 5.5
    assert euroc.data_report(seq, INIT)["begins_still"]


def test_moving_start_is_not_still_and_later_stillness_is_used():
    fs, n = 200.0, 2000
    t = np.arange(n, dtype=np.int64) * 5_000_000
    acc = np.tile([0.0, 0.0, 9.81], (n, 1))
    gyr = np.zeros((n, 3))
    rng = np.random.default_rng(0)
    acc[:800] += rng.normal(0.0, 1.0, (800, 3))  # moving for the first 4 s
    gyr[:800] += rng.normal(0.0, 0.1, (800, 3))
    imu = ImuStream(t, acc, gyr, kind="euroc")
    segments = euroc.find_still_segments(imu, fs, INIT)
    assert len(segments) == 1
    assert segments[0].t_start_ns == 4 * euroc.NS_PER_S


def test_no_still_segment_raises():
    fs, n = 200.0, 1000
    t = np.arange(n, dtype=np.int64) * 5_000_000
    rng = np.random.default_rng(1)
    imu = ImuStream(t, np.array([0.0, 0.0, 9.81]) + rng.normal(0.0, 1.0, (n, 3)), rng.normal(0.0, 0.1, (n, 3)), kind="euroc")
    gt = euroc.GroundTruth(
        t, np.zeros((n, 3)), np.tile([1.0, 0, 0, 0], (n, 1)), np.zeros((n, 3)), np.zeros((n, 3)), np.zeros((n, 3))
    )
    seq = euroc.Sequence("x", imu, gt, ZERO, fs)
    with pytest.raises(ValueError, match="no still segment"):
        euroc.select_still_segment(seq, INIT)


def test_init_with_x_up_body_keeps_heading_and_is_tilt_correct(root):
    """The EuRoC body has x up, where ZYX Euler angles are singular; the tilt-corrected init must work anyway."""
    seq = load(root, x_up=True, gt_delay_s=0.0)
    seg = euroc.select_still_segment(seq, STRICT)
    k = int(np.searchsorted(seq.gt.t_ns, seg.t_end_ns))
    q_ref = seq.gt.q[k]
    x0, tilt = euroc.static_init_tilt_corrected(seq.imu, seg, q_ref, seq.gt.p[k])
    assert tilt == pytest.approx(0.0, abs=1e-9)  # the reference attitude is already consistent
    assert np.linalg.norm(so3.log(so3.quat_mul(so3.quat_conj(q_ref), x0.q))) < 1e-9

    # now tilt the reference by 3° about a horizontal axis: heading stays, gravity lands on +z
    dq = so3.exp(np.array([0.05, 0.0, 0.0]))
    q_bad = so3.quat_normalize(so3.quat_mul(dq, q_ref))
    x1, tilt1 = euroc.static_init_tilt_corrected(seq.imu, seg, q_bad, seq.gt.p[k])
    assert tilt1 == pytest.approx(0.05, abs=1e-6)
    m = (seq.imu.t_ns >= seg.t_start_ns) & (seq.imu.t_ns < seg.t_end_ns)
    a = seq.imu.acc[m].mean(0)
    up = so3.quat_to_rot(x1.q) @ (a / np.linalg.norm(a))
    np.testing.assert_allclose(up, [0.0, 0.0, 1.0], atol=1e-9)
    # world-frame gravity correction removes the deliberate tilt
    assert np.linalg.norm(so3.log(so3.quat_mul(so3.quat_conj(q_ref), x1.q))) < 1e-9


# ----------------------------------------------------------------- fixes and outages


def test_fixes_are_seeded_on_imu_timestamps_and_stop_at_ground_truth(root):
    seq = load(root)
    k0 = 100
    cfg = euroc.FixConfig(rate_hz=10.0, sigma=0.1, seed=3)
    f = euroc.synthesize_fixes(seq, k0, cfg, seq_index=0)
    t = np.array([int(x.t_ns) for x in f], dtype=np.int64)
    assert set(t.tolist()) <= set(seq.imu.t_ns.tolist())
    assert np.all(np.diff(t) == 100_000_000)
    assert t[0] == seq.imu.t_ns[k0]
    assert t[-1] <= seq.gt.t_ns[-1]

    again = euroc.synthesize_fixes(seq, k0, cfg, seq_index=0)
    np.testing.assert_array_equal([x.p_W for x in f], [x.p_W for x in again])
    other = euroc.synthesize_fixes(seq, k0, cfg, seq_index=1)
    assert not np.allclose([x.p_W for x in f], [x.p_W for x in other])
    assert not np.allclose(
        [x.p_W for x in f], [x.p_W for x in euroc.synthesize_fixes(seq, k0, euroc.FixConfig(10.0, 0.1, 4), seq_index=0)]
    )


def test_fix_noise_matches_sigma_and_truth(root):
    seq = load(root)
    cfg = euroc.FixConfig(rate_hz=10.0, sigma=0.1, seed=3)
    f = euroc.synthesize_fixes(seq, 100, cfg, seq_index=0)
    t = np.array([int(x.t_ns) for x in f], dtype=np.int64)
    p = np.array([x.p_W for x in f])
    e = p - euroc.gt_position_at(seq.gt, t)
    assert np.std(e) == pytest.approx(0.1, rel=0.15)
    assert abs(np.mean(e)) < 0.1 / np.sqrt(len(e)) * 4


def test_fix_rate_must_divide_imu_rate(root):
    seq = load(root)
    with pytest.raises(ValueError, match="divide"):
        euroc.synthesize_fixes(seq, 0, euroc.FixConfig(rate_hz=7.0), 0)


def test_outage_windows_are_placed_by_rule():
    t0, end = 1_000_000_000, 101_000_000_000
    w = euroc.outage_windows(t0, end, duration_s=10.0, warmup_s=20.0, recovery_s=5.0)
    assert w[0] == (21_000_000_000, 31_000_000_000)
    assert w[1] == (36_000_000_000, 46_000_000_000)
    assert all(b + 5_000_000_000 <= end for _, b in w)
    assert w == euroc.outage_windows(t0, end, 10.0, 20.0, 5.0)
    assert euroc.outage_windows(t0, end, 60.0, 20.0, 5.0) == [(21_000_000_000, 81_000_000_000)]
    assert euroc.outage_windows(t0, end, 90.0, 20.0, 5.0) == []


def test_drop_fixes_removes_only_inside_window(root):
    seq = load(root)
    f = euroc.synthesize_fixes(seq, 0, euroc.FixConfig(rate_hz=10.0, sigma=0.1, seed=0), 0)
    t = np.array([int(x.t_ns) for x in f])
    w = (t[20], t[30])
    kept = euroc.drop_fixes(f, w)
    kt = np.array([int(x.t_ns) for x in kept])
    assert len(kept) == len(f) - 10
    assert t[19] in kt and t[20] not in kt and t[29] not in kt and t[30] in kt


# ----------------------------------------------------------------- the whole chain


def test_whole_chain_tracks_truth_on_noise_free_fixture(root):
    """Loader, static init, seeded fixes and filter: noise-free data, fixes at 1 µm → millimetre-level tracking."""
    seq = load(root, x_up=True, static_s=5.0)
    setup = euroc.prepare_run(seq, 0, STRICT, FIXES_LOW, filter_cfg())
    rows = euroc_eval.run_python(setup, setup.fixes)
    post = euroc_eval.posterior(rows)
    t = post[:, euroc_eval.C_T].astype(np.int64)
    e = post[:, euroc_eval.C_P] - euroc.gt_position_at(seq.gt, t)
    assert np.max(np.linalg.norm(e, axis=1)) < 1e-3
    assert euroc_eval.nis_summary(rows)["rejected"] == 0


def test_outage_scoring_and_truth_replay(root):
    """On noise-free data the scored outage error is tiny and truth-start replay reproduces ground truth."""
    seq = load(root, x_up=True, static_s=5.0)
    setup = euroc.prepare_run(seq, 0, STRICT, FIXES_LOW, filter_cfg())
    t0 = int(setup.imu.t_ns[0])
    window = (t0 + 12 * euroc.NS_PER_S, t0 + 17 * euroc.NS_PER_S)
    fixes = euroc.drop_fixes(setup.fixes, window)
    rows = euroc_eval.run_python(setup, fixes)
    o = euroc_eval.score_outage(rows, seq.gt, 5.0, window, fixes)
    assert o.t_since_fix_s[-1] == pytest.approx(5.0, abs=0.1)
    assert o.err_norm.max() < 1e-2
    assert o.left_norm_s is None
    assert o.first_accept_after_s == pytest.approx(0.0, abs=1e-9)
    assert o.fixes_after == sum(int(f.t_ns) >= window[1] for f in setup.fixes)
    assert euroc_eval.truth_start_replay(setup, seq.gt, o.last_fix_ns, window[1]) < 1e-6


def test_tum_files_have_quaternion_order_and_common_timestamps(root, tmp_path):
    seq = load(root, x_up=True, static_s=5.0)
    setup = euroc.prepare_run(seq, 0, STRICT, FIXES_LOW, filter_cfg())
    rows = euroc_eval.run_python(setup, setup.fixes)
    ref, est = euroc_eval.write_tum_files(rows, seq.gt, tmp_path / "out")
    tr, pr, qr = read_tum(ref)
    te, pe, qe = read_tum(est)
    np.testing.assert_array_equal(tr, te)
    assert np.all(qe[:, 0] >= 0)
    np.testing.assert_allclose(pr, pe, atol=1e-3)
    d = so3.log(so3.quat_mul(so3.quat_conj(qr), qe))
    assert np.max(np.linalg.norm(d, axis=-1)) < 5e-3  # a wrong order would be order 1


def test_gimbal_x_up_fixture_starts_near_pitch_minus_90(root):
    """Guards the claim that this fixture matches the real data's attitude regime."""
    seq = load(root, x_up=True)
    R = so3.quat_to_rot(seq.gt.q[100])
    # S's x axis along world up for the (level) static start
    np.testing.assert_allclose(R @ np.array([1.0, 0.0, 0.0]), [0.0, 0.0, 1.0], atol=1e-6)


def test_evo_metrics_separate_alignment(tmp_path):
    pytest.importorskip("evo")
    t = np.arange(400, dtype=np.int64) * 5_000_000 + 10**18
    p = np.column_stack([np.linspace(0, 5, 400), np.sin(np.linspace(0, 6, 400)), 0.1 * np.cos(np.linspace(0, 3, 400))])
    q = np.tile([1.0, 0.0, 0.0, 0.0], (400, 1))
    from navekf_py.io import write_tum

    ref, est = tmp_path / "ref.tum", tmp_path / "est.tum"
    write_tum(ref, t, p, q)
    write_tum(est, t, p + np.array([1.0, 0.0, 0.0]), q)
    m = euroc_eval.evo_metrics(ref, est)
    assert m["ate_unaligned_rmse_m"] == pytest.approx(1.0, abs=1e-6)
    assert m["ate_se3_rmse_m"] == pytest.approx(0.0, abs=1e-6)
    assert m["rpe_trans_rmse_m"] == pytest.approx(0.0, abs=1e-6)


def test_seed_spread_keeps_seed_zero_and_ranges():
    def rec(seq, d, err, axis=False):
        return {"sequence": seq, "duration_s": d, "end_error_m": err, "left_norm": False, "left_axis": axis, "locked": False}

    per_seed = [
        [rec("A", 10.0, 1.0, True), rec("A", 10.0, 3.0), rec("B", 10.0, 5.0)],
        [rec("A", 10.0, 2.0), rec("A", 10.0, 4.0), rec("B", 10.0, 7.0, True)],
        [rec("A", 10.0, 6.0), rec("A", 10.0, 8.0, True), rec("B", 10.0, 9.0, True)],
    ]
    rows = {r["sequence"]: r for r in euroc_eval.seed_spread(per_seed)}
    assert rows["A"]["windows"] == 2 and rows["all"]["windows"] == 3 and rows["all"]["seeds"] == 3
    assert rows["A"]["end_error_m"] == {"seed0": 2.0, "median": 3.0, "min": 2.0, "max": 7.0}
    assert rows["all"]["left_axis"] == {"seed0": 1.0, "median": 1.0, "min": 1.0, "max": 2.0}

    with pytest.raises(ValueError, match="number of outage windows"):
        euroc_eval.seed_spread([per_seed[0], per_seed[1][:-1]])
