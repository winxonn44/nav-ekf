"""E3: the C++ filter matches the Python reference within 1e-9 at every step.

Calls the built ``run_dataset`` binary, so it needs a C++ build. The binary is taken from
``$NAVEKF_RUN_DATASET`` or ``build/run_dataset``. Without a binary the tests skip, unless
``NAVEKF_REQUIRE_BINARY`` is set (CI), when they fail. Golden files are generated here, in a
temporary directory, and never stored. ``pytest -s`` prints the per-scenario maxima.
"""

import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from navekf_py import eskf, euroc, euroc_eval, parity
from navekf_py.euroc_fixture import FixtureConfig, write_fixture
from navekf_py.io import read_tum
from navekf_py.types import ImuNoise

TOL = 1e-9
REPO = Path(__file__).resolve().parents[1]
IX_STATE = slice(2, 18)
IX_P = slice(18, 138)


@pytest.fixture(scope="module")
def binary() -> Path:
    path = Path(os.environ.get("NAVEKF_RUN_DATASET", REPO / "build" / "run_dataset"))
    if not path.exists():
        msg = f"C++ binary not found at {path}"
        if os.environ.get("NAVEKF_REQUIRE_BINARY"):
            pytest.fail(msg)
        pytest.skip(msg)
    return path


def run_cpp(binary: Path, d: Path) -> np.ndarray:
    cmd = [str(binary), "--imu", str(d / "imu.csv"), "--gps", str(d / "gps.csv"), "--config", str(d / "config.yaml"),
           "--tum", str(d / "cpp.tum"), "--log", str(d / "cpp_log.csv"), "--parity"]  # fmt: skip
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    assert (d / "cpp_log.csv").read_text().splitlines()[0] == parity.log_header()
    return np.loadtxt(d / "cpp_log.csv", delimiter=",", skiprows=1)


@pytest.mark.parametrize("name", list(parity.SCENARIOS))
def test_cpp_matches_python_at_every_step(binary, tmp_path, name):
    case, py = parity.export(name, tmp_path)
    cpp = run_cpp(binary, tmp_path)

    assert cpp.shape == py.shape
    np.testing.assert_array_equal(cpp[:, 0], py[:, 0], err_msg="timestamps")
    np.testing.assert_array_equal(cpp[:, 1], py[:, 1], err_msg="row kinds")
    np.testing.assert_array_equal(cpp[:, -1], py[:, -1], err_msg="gate decisions")

    d_state = np.abs(cpp[:, IX_STATE] - py[:, IX_STATE])
    d_P = np.abs(cpp[:, IX_P] - py[:, IX_P])
    updates = py[:, 1] == 1
    assert np.array_equal(np.isnan(cpp[:, -2]), np.isnan(py[:, -2]))
    assert np.array_equal(np.isnan(py[:, -2]), ~updates)
    d_nis = np.abs(cpp[updates, -2] - py[updates, -2])
    rejected = int(np.sum(py[updates, -1] == 0))

    print(f"\nPARITY {name}: rows={len(py)} updates={int(updates.sum())} rejected={rejected} "
          f"max|dstate|={d_state.max():.3e} max|dP|={d_P.max():.3e} max|dNIS|={d_nis.max():.3e}")  # fmt: skip
    assert d_state.max() <= TOL
    assert d_P.max() <= TOL
    assert d_nis.max() <= TOL
    if case.expect_rejections:
        assert rejected > 0
    else:
        assert rejected == 0


def test_dropout_and_lockout_are_really_exercised(binary, tmp_path):
    """The noisy scenario has a hole in its fixes; the lock-out scenario never recovers."""
    _, py = parity.export("noisy_dropout_outliers", tmp_path / "noisy")
    t_up = py[py[:, 1] == 1, 0] * 1e-9
    assert not np.any((t_up >= 15.0) & (t_up < 25.0))
    assert np.sum(py[py[:, 1] == 1, -1] == 0) >= 3

    _, lock = parity.export("lockout", tmp_path / "lockout")
    late = (lock[:, 1] == 1) & (lock[:, 0] >= 5e9)
    assert late.sum() == 51 and np.all(lock[late, -1] == 0)
    cpp = run_cpp(binary, tmp_path / "lockout")
    np.testing.assert_array_equal(cpp[late, -1], 0.0)


def test_tum_quaternion_order_and_default_log(binary, tmp_path):
    """TUM has one posterior per IMU time, ``qx qy qz qw``, ``w ≥ 0``; the default log has the §8 columns."""
    case, py = parity.export("first_jr", tmp_path)
    d = tmp_path
    subprocess.run([str(binary), "--imu", str(d / "imu.csv"), "--gps", str(d / "gps.csv"), "--config", str(d / "config.yaml"),
                    "--tum", str(d / "a.tum"), "--log", str(d / "a.csv")], check=True, capture_output=True)  # fmt: skip
    t, p, q = read_tum(d / "a.tum")
    np.testing.assert_array_equal(t, case.imu.t_ns)
    assert np.all(q[:, 0] >= 0)

    # the pose at t_k is the last Python row at that time (update if any, otherwise the IMU step)
    k = 500  # 2.5 s, no fix: IMU row only
    row = py[py[:, 0] == t[k]][-1]
    np.testing.assert_allclose(p[k], row[2:5], atol=1e-8)
    expected = row[8:12] if row[8] >= 0 else -row[8:12]
    np.testing.assert_allclose(q[k], expected, atol=1e-8)

    head = (d / "a.csv").read_text().splitlines()[0].split(",")
    assert len(head) == 2 + 16 + 15 + 2
    assert head[-2:] == ["nis", "accepted"]


def test_cpp_matches_python_on_euroc_layout_fixture(binary, tmp_path):
    """The E4 chain (loader, init, seeded fixes, a 10 s outage) runs identically through both filters."""
    noise = ImuNoise(sigma_a=2e-3, sigma_g=1.7e-4, sigma_ba=3e-3, sigma_bg=2e-5)
    seq = euroc.load_sequence(tmp_path, write_fixture(tmp_path, FixtureConfig(duration_s=30.0, noise=noise, x_up=True, seed=11)))
    cfg = eskf.EskfConfig(noise=noise, g=9.81)
    setup = euroc.prepare_run(seq, 0, euroc.InitConfig(), euroc.FixConfig(rate_hz=10.0, sigma=0.1, seed=3), cfg)
    t0 = int(setup.imu.t_ns[0])
    window = (t0 + 8 * euroc.NS_PER_S, t0 + 18 * euroc.NS_PER_S)
    fixes = euroc.drop_fixes(setup.fixes, window)
    case = euroc_eval.as_case(setup, fixes)
    py = parity.run_python(case)

    d = tmp_path / "parity"
    parity.write_case(case, d)
    parity.write_python_log(d / "python_log.csv", py)
    cpp = euroc_eval.run_cpp(binary, d / "imu.csv", d / "gps.csv", d / "config.yaml", d, parity_mode=True)

    assert cpp.shape == py.shape
    np.testing.assert_array_equal(cpp[:, 0:2], py[:, 0:2])
    np.testing.assert_array_equal(cpp[:, -1], py[:, -1])
    assert np.abs(cpp[:, IX_STATE] - py[:, IX_STATE]).max() <= TOL
    assert np.abs(cpp[:, IX_P] - py[:, IX_P]).max() <= TOL
