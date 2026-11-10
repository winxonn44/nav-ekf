"""consistency.py: bands, NEES/NIS, and Monte Carlo consistency of the ESKF (derivation §8)."""

from pathlib import Path

import numpy as np
import pytest

from navekf_py import consistency as c
from navekf_py import eskf, sim

ROOT = Path(__file__).resolve().parents[1]


def test_bands_and_gate_match_the_derivation():
    np.testing.assert_allclose(c.chi2_band(15, 50), (13.52, 16.56), atol=5e-3)
    np.testing.assert_allclose(c.chi2_band(3, 50), (2.36, 3.72), atol=5e-3)
    np.testing.assert_allclose(c.chi2_band(15, 1), (6.26, 27.49), atol=5e-3)
    assert eskf.GATE_99_3DOF == pytest.approx(11.345, abs=5e-4)


def test_nees_uses_the_covariance():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((4, 15, 15))
    P = A @ np.swapaxes(A, -1, -2) + np.eye(15)
    e = rng.standard_normal((4, 15))
    ref = np.einsum("ni,nij,nj->n", e, np.linalg.inv(P), e)
    np.testing.assert_allclose(c.nees(e, P), ref, rtol=1e-10)


@pytest.fixture(scope="module")
def smoke():
    """10 runs × 60 s (configs/sim.yaml: 10 s dropout), consistent stream, default filter. ~1.5 s."""
    r = c.monte_carlo(c.McConfig(sim=sim.load_config(ROOT / "configs" / "sim.yaml"), n_runs=10, seed=0))
    return r, c.summarize(r)


def test_initial_error_is_a_draw_from_P0(smoke):
    r, _ = smoke
    assert r.t_tick_ns[0] == 0
    # posterior after the t = 0 fix; still χ²(15)-ish over 10 runs (band for N = 10)
    lo, hi = c.chi2_band(15, 10, 0.999)
    assert lo < r.nees[0].mean() < hi


def test_smoke_monte_carlo_is_consistent(smoke):
    _, s = smoke
    lo, hi = s["nees_band"]
    assert lo < s["nees_mean"] < hi and abs(s["nees_mean"] - 15) < 1.2
    assert s["nees_in_band"] >= 0.85
    lo, hi = s["nis_band"]
    assert lo < s["nis_mean"] < hi and abs(s["nis_mean"] - 3) < 0.25
    assert s["nis_in_band"] >= 0.85


def test_false_rejection_rate_is_about_one_percent(smoke):
    """No outliers, gate at the 99% quantile: ~1% of ~5000 fixes rejected (binomial 3σ ≈ ±0.4%)."""
    r, s = smoke
    n = r.accepted.size
    assert n > 4500
    assert 0.006 <= s["reject_rate"] <= 0.015
    assert r.n_rejected == int((~r.accepted).sum())


@pytest.mark.slow
def test_50_run_bands_hold_on_average_over_batches():
    """The E2 headline: NEES/NIS of 50-run batches × 120 s with a 30 s dropout, gate off, ~40 s.

    The band is per time step, so a consistent filter spends 95% of the time inside it on average,
    however correlated the errors are; correlation only widens the spread between batches (one batch
    has been seen as low as 76%). So the test averages over independent batches. The batch-to-batch
    std of the NEES share is about 3 points (12 batches, scripts/check_nees_coverage.py), so with 6
    batches 3 standard errors below 95% is 91%. The gate is off because it deliberately truncates
    the innovations; its effect is reported separately. Seeds are not used anywhere else.
    """
    cfg = sim.load_config(ROOT / "configs" / "e2_consistency.yaml")
    assert any(b - a >= 30 for a, b in cfg.gps.dropouts)
    batches = [c.summarize(c.monte_carlo(c.McConfig(sim=cfg, n_runs=50, seed=20000 + 50 * b, gate=None))) for b in range(6)]
    lo, hi = batches[0]["nees_band"]
    assert lo < np.mean([s["nees_mean"] for s in batches]) < hi
    lo, hi = batches[0]["nis_band"]
    assert lo < np.mean([s["nis_mean"] for s in batches]) < hi
    assert np.mean([s["nees_in_band"] for s in batches]) >= 0.91
    assert np.mean([s["nis_in_band"] for s in batches]) >= 0.93
