"""Tests for the vectorized exact Housworth-Stahl likelihood."""

import numpy as np
import pytest

from cointf.fastlik import (
    Meioses,
    log_fstar_fast,
    log_sstar_fast,
    loglik_fast,
    series_kernel,
)
from cointf.naive import log_fstar, log_sstar, loglik_meiosis_naive
from cointf.simintf import simulate_meiosis


@pytest.mark.parametrize("nu", [0.5, 1.0, 4.0, 20.0, 100.0])
def test_series_matches_naive(nu):
    x = np.logspace(-5, np.log10(3.0), 300)
    np.testing.assert_allclose(log_fstar_fast(x, nu), log_fstar(x, nu), atol=1e-10)
    np.testing.assert_allclose(log_sstar_fast(x, nu), log_sstar(x, nu), atol=1e-10)


@pytest.mark.parametrize(
    "nu,p", [(1.0, 0.05), (4.0, 0.05), (12.0, 0.3), (2.0, 0.0), (4.0, 1.0)]
)
def test_testbed_matches_naive(testbed, nu, p):
    data, prm = testbed
    L = prm["L"]
    ref = np.array([loglik_meiosis_naive(d, L, nu, p) for d in data])
    assert loglik_fast(data, L, nu, p) == pytest.approx(ref.sum(), abs=1e-9)
    np.testing.assert_allclose(
        loglik_fast(data, L, nu, p, per_meiosis=True), ref, atol=1e-9
    )


@pytest.mark.parametrize("nu,p", [(4.0, 0.1), (10.0, 0.4)])
def test_mixed_lengths_match_naive(nu, p):
    rng = np.random.default_rng(4)
    Ls = rng.choice([0.6, 1.5, 2.8], size=300)
    data = [simulate_meiosis(L, nu, p, rng) for L in Ls]
    ref = np.array([loglik_meiosis_naive(d, L, nu, p) for d, L in zip(data, Ls)])
    got = loglik_fast(data, Ls, nu, p, per_meiosis=True)
    np.testing.assert_allclose(got, ref, atol=1e-9)


@pytest.mark.parametrize("p", [0.0, 0.3, 1.0])
def test_nu1_is_poisson(testbed, p):
    data, prm = testbed
    rng = np.random.default_rng(1)
    Ls = rng.uniform(0.5, 3.0, size=len(data)) + prm["L"]
    assert loglik_fast(data, Ls, 1.0, p) == pytest.approx(-Ls.sum(), abs=1e-8)


@pytest.mark.parametrize("xs", [[0.0, 0.5], [0.5, 1.5], [0.2, 1.7], [-0.1]])
def test_out_of_range_positions_raise(xs):
    with pytest.raises(ValueError):
        loglik_fast([xs], 1.5, 4.0, 0.05)


@pytest.mark.parametrize(
    "kw",
    [
        dict(nu=0.0, p=0.1, L=1.0),
        dict(nu=2.0, p=1.2, L=1.0),
        dict(nu=2.0, p=0.1, L=0.0),
    ],
)
def test_invalid_parameters_raise(kw):
    with pytest.raises(ValueError):
        loglik_fast([[0.3]], kw["L"], kw["nu"], kw["p"])


def test_custom_kernel_is_used(testbed):
    data, prm = testbed
    calls = []
    base = series_kernel(4.0)

    def counting(u):
        calls.append(u.size)
        return base(u)

    got = loglik_fast(data, prm["L"], 4.0, 0.05, kernel=counting)
    assert calls and sum(calls) > 0
    assert got == loglik_fast(data, prm["L"], 4.0, 0.05)


def test_meioses_container_reuse(testbed):
    data, prm = testbed
    mz = Meioses(data, prm["L"])
    for nu, p in [(2.0, 0.1), (6.0, 0.02)]:
        assert loglik_fast(mz, None, nu, p) == pytest.approx(
            loglik_fast(data, prm["L"], nu, p), abs=1e-12
        )


def test_empty_meioses_and_no_crossovers():
    assert loglik_fast([], 1.0, 4.0, 0.1) == 0.0
    ref = loglik_meiosis_naive([], 1.2, 4.0, 0.1)
    assert loglik_fast([[], []], 1.2, 4.0, 0.1) == pytest.approx(2 * ref, abs=1e-12)
