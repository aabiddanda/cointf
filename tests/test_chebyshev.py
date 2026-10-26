"""Tests for the piecewise Chebyshev kernel (`cointf.chebyshev`).

The kernel is approximate inside ``[u_min, u_max]`` and exact outside, so it
is compared against the exact series of `cointf.fastlik`.
"""

import doctest

import numpy as np
import pytest

import cointf.chebyshev as chebyshev
from cointf.chebyshev import ChebyshevKernel, PiecewiseChebyshev
from cointf.fastlik import Meioses, log_fstar_fast, log_sstar_fast, loglik_fast
from cointf.simintf import simulate_meioses

dense_u = np.concatenate(
    [np.logspace(-6, np.log10(0.25), 2000), np.linspace(0.25, 5.0, 8000)]
)


@pytest.mark.parametrize("nu", [0.3, 0.7, 1.0, 1.5, 2.5, 4.3, 8.5, 20.3, 50.0, 100.0])
def test_kernel_matches_series(nu):
    kern = ChebyshevKernel()
    np.testing.assert_allclose(
        kern.log_fstar(dense_u, nu), log_fstar_fast(dense_u, nu), rtol=0, atol=1e-11
    )
    np.testing.assert_allclose(
        kern.log_sstar(dense_u, nu), log_sstar_fast(dense_u, nu), rtol=0, atol=1e-11
    )


def test_outside_range_is_exact():
    kern = ChebyshevKernel(u_min=1e-3, u_max=2.0)
    u = np.array([0.0, 1e-5, 5e-4, 2.5, 7.0])
    np.testing.assert_array_equal(kern.log_fstar(u, 4.0), log_fstar_fast(u, 4.0))
    np.testing.assert_array_equal(kern.log_sstar(u, 4.0), log_sstar_fast(u, 4.0))


def test_endpoints_of_range():
    kern = ChebyshevKernel()
    u = np.array([kern.u_min, kern.u_split, kern.u_max])
    np.testing.assert_allclose(
        kern.log_fstar(u, 4.0), log_fstar_fast(u, 4.0), atol=1e-12
    )


def test_piecewise_fit_of_polynomial_needs_one_linear_piece():
    fit = PiecewiseChebyshev(lambda u: 1.0 + u - 0.5 * u**2, 1e-6, 0.25, 5.0)
    assert np.sum(~fit.log_scale) == 1
    u = np.concatenate([np.logspace(-6, -1, 11), np.linspace(0.3, 5.0, 11)])
    np.testing.assert_allclose(fit(u), 1.0 + u - 0.5 * u**2, atol=1e-13)


@pytest.mark.parametrize("nu,p", [(1.5, 0.05), (4.0, 0.05), (4.3, 0.0), (20.3, 0.2)])
def test_loglik_matches_exact_kernel(testbed, nu, p):
    data, prm = testbed
    mz = Meioses(data, prm["L"])
    exact = loglik_fast(mz, None, nu, p, per_meiosis=True)
    approx = loglik_fast(mz, None, nu, p, kernel=ChebyshevKernel(), per_meiosis=True)
    np.testing.assert_allclose(approx, exact, rtol=0, atol=1e-12)


def test_fits_are_cached_and_reused(testbed):
    data, prm = testbed
    mz = Meioses(data, prm["L"])
    kern = ChebyshevKernel()
    for p in [0.0, 0.05, 0.3]:
        loglik_fast(mz, None, 4.0, p, kernel=kern)
    assert list(kern.cache) == [4.0]


def test_cache_is_bounded():
    kern = ChebyshevKernel(max_cached=3)
    for nu in [1.5, 2.0, 2.5, 3.0, 3.5]:
        kern.fits(nu)
    assert len(kern.cache) <= 3


def test_per_meiosis_nu_few_distinct_values():
    data = simulate_meioses(L=2.0, nu=4.0, p=0.05, n=300, seed=8)
    nu = np.where(np.arange(300) % 2 == 0, 3.0, 6.0)
    kern = ChebyshevKernel()
    exact = loglik_fast(data, 2.0, nu, 0.05, per_meiosis=True)
    approx = loglik_fast(data, 2.0, nu, 0.05, kernel=kern, per_meiosis=True)
    np.testing.assert_allclose(approx, exact, atol=1e-12)
    assert sorted(kern.cache) == [3.0, 6.0]


def test_per_meiosis_nu_many_distinct_values_falls_back():
    data = simulate_meioses(L=2.0, nu=4.0, p=0.05, n=100, seed=8)
    nu = np.linspace(2.0, 8.0, 100)
    kern = ChebyshevKernel(max_distinct=10)
    exact = loglik_fast(data, 2.0, nu, 0.05, per_meiosis=True)
    approx = loglik_fast(data, 2.0, nu, 0.05, kernel=kern, per_meiosis=True)
    np.testing.assert_array_equal(approx, exact)
    assert not kern.cache


def test_long_maps_use_series_beyond_u_max():
    data = simulate_meioses(L=8.0, nu=4.0, p=0.05, n=40, seed=10)
    exact = loglik_fast(data, 8.0, 4.0, 0.05, per_meiosis=True)
    approx = loglik_fast(
        data, 8.0, 4.0, 0.05, kernel=ChebyshevKernel(), per_meiosis=True
    )
    np.testing.assert_allclose(approx, exact, atol=1e-11)


def test_invalid_range_raises():
    with pytest.raises(ValueError):
        ChebyshevKernel(u_min=0.5, u_split=0.25)


def test_docstring_examples_run():
    assert doctest.testmod(chebyshev).failed == 0


def test_fit_raises_when_pieces_run_out():
    with pytest.raises(RuntimeError):
        PiecewiseChebyshev(np.sin, 1e-6, 0.25, 500.0, tol=1e-14, max_pieces=4)


def test_n_pieces_matches_coefficients():
    fit = ChebyshevKernel().fits(4.0)[0]
    assert fit.n_pieces == fit.coefs.shape[1] == fit.u_lo.size


def test_error_stays_small_as_nu_varies(testbed):
    # Piece layouts change with nu; the likelihood error must stay small
    # across those changes, not only at a few chosen values.
    data, prm = testbed
    mz = Meioses(data, prm["L"])
    kern = ChebyshevKernel(max_cached=8)
    for nu in np.linspace(3.0, 5.0, 41):
        exact = loglik_fast(mz, None, nu, 0.05)
        approx = loglik_fast(mz, None, nu, 0.05, kernel=kern)
        assert abs(approx - exact) < 1e-10


@pytest.mark.parametrize("nu,p", [(0.6, 0.1), (1.0, 0.4), (5.0, 1.0)])
def test_loglik_edge_parameters(testbed, nu, p):
    data, prm = testbed
    exact = loglik_fast(data, prm["L"], nu, p, per_meiosis=True)
    approx = loglik_fast(
        data, prm["L"], nu, p, kernel=ChebyshevKernel(), per_meiosis=True
    )
    np.testing.assert_allclose(approx, exact, atol=1e-12)


def test_crossovers_at_chromosome_ends():
    # Zero distances to the ends use the exact series (S*(0) = 1).
    data = [[0.0, 0.7, 1.5], [0.0], [1.5]]
    exact = loglik_fast(data, 1.5, 4.0, 0.05, per_meiosis=True)
    approx = loglik_fast(
        data, 1.5, 4.0, 0.05, kernel=ChebyshevKernel(), per_meiosis=True
    )
    np.testing.assert_allclose(approx, exact, atol=1e-12)
