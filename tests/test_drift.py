"""Derivation §7: dead-reckoning drift laws by Monte Carlo, one noise source at a time.

100 runs per source, 60 s at 100 Hz, still and level IMU. The empirical
per-axis horizontal RMS position error must stay within ±20% of the §7 curve
for t ≥ 1 s (sampling error at 100 runs is ~7% per point, 1σ), and the
log–log slope must match 1.5 / 2 / 2.5 / 3. About 1.5 s in total.
"""

import numpy as np
import pytest

from navekf_py import drift

N_RUNS = 100


@pytest.fixture(scope="module", params=list(drift.SLOPES))
def result(request):
    return drift.run(request.param, n_runs=N_RUNS, duration_s=60.0, rate_hz=100.0, seed=0)


def test_rms_matches_theory(result):
    m = result.t >= 1.0
    ratio = result.rms[m] / result.theory[m]
    assert ratio.min() > 0.8 and ratio.max() < 1.2, f"{result.source}: ratio in [{ratio.min():.3f}, {ratio.max():.3f}]"


def test_loglog_slope(result):
    slope = drift.loglog_slope(result.t, result.rms)
    assert slope == pytest.approx(drift.SLOPES[result.source], abs=0.1), f"{result.source}: slope {slope:.3f}"


def test_theory_table_values():
    """Spot-check the §7 formulas at t = 10 s."""
    g = 9.81
    assert drift.theory("accel_noise", 10.0, 1.0) == pytest.approx(10**1.5 / np.sqrt(3))
    assert drift.theory("accel_bias", 10.0, 1.0) == pytest.approx(50.0)
    assert drift.theory("gyro_noise", 10.0, 1.0) == pytest.approx(g * 10**2.5 / np.sqrt(20))
    assert drift.theory("gyro_bias", 10.0, 1.0) == pytest.approx(g * 1000 / 6)
