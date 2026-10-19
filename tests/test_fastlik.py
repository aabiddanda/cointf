"""Tests for the fast exact Housworth-Stahl likelihood (`cointf.fastlik`).

Every likelihood test compares against the slow reference implementation in
`cointf.naive`, which is itself checked against brute-force enumeration,
the nu = 1 Poisson case and simulated count probabilities in test_naive.py.
"""

import doctest

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import cointf.fastlik as fastlik
from cointf.fastlik import (
    Meioses,
    log_fstar_fast,
    log_sstar_fast,
    log_tail_fast,
    loglik_fast,
    series_kernel,
)
from cointf.naive import (
    log_fstar,
    log_sstar,
    log_tail_integral,
    loglik_meiosis_naive,
)
from cointf.simintf import simulate_meiosis

atol_naive = 1e-9


def naive_per_meiosis(data, L, nu, p):
    """Reference per-meiosis log-likelihoods with broadcast parameters."""
    M = len(data)
    L, nu, p = (np.broadcast_to(np.asarray(v, float), (M,)) for v in (L, nu, p))
    return np.array(
        [loglik_meiosis_naive(d, a, b, c) for d, a, b, c in zip(data, L, nu, p)]
    )


def random_params(M, rng, p_edges=True):
    """Per-meiosis nu over [0.4, 60] (log-uniform) and p over [0, 1]."""
    nu = np.exp(rng.uniform(np.log(0.4), np.log(60.0), M))
    p = rng.uniform(0.0, 1.0, M)
    if p_edges:
        p[:3], p[3:6] = 0.0, 1.0
    return nu, p


# --------------------------------------------------------------------------
# series functions
# --------------------------------------------------------------------------
@pytest.mark.parametrize("nu", [0.3, 0.5, 1.0, 4.0, 20.0, 100.0])
def test_series_match_naive(nu):
    x = np.logspace(-5, np.log10(3.0), 300)
    np.testing.assert_allclose(log_fstar_fast(x, nu), log_fstar(x, nu), atol=1e-10)
    np.testing.assert_allclose(log_sstar_fast(x, nu), log_sstar(x, nu), atol=1e-10)


@pytest.mark.parametrize("nu", [0.3, 1.0, 4.0, 20.0, 100.0])
def test_log_tail_matches_naive(nu):
    x = np.array([0.0, 1e-4, 0.1, 0.7, 1.5, 3.0])
    ref = np.array([log_tail_integral(v, nu) for v in x])
    np.testing.assert_allclose(log_tail_fast(x, nu), ref, atol=1e-10)
    assert log_tail_fast(0.0, nu)[0] == pytest.approx(0.0, abs=1e-15)  # G(0) = 1


@pytest.mark.parametrize("fn", [log_fstar_fast, log_sstar_fast, log_tail_fast])
def test_series_accept_per_element_nu(fn):
    rng = np.random.default_rng(0)
    x = rng.uniform(1e-4, 2.5, 200)
    nu = np.exp(rng.uniform(np.log(0.3), np.log(100.0), 200))
    expected = np.array([fn(a, b)[0] for a, b in zip(x, nu)])
    np.testing.assert_allclose(fn(x, nu), expected, rtol=0, atol=1e-12)


@pytest.mark.parametrize("fn", [log_fstar_fast, log_sstar_fast, log_tail_fast])
def test_series_shapes(fn):
    assert fn(0.5, 4.0).shape == (1,)
    assert fn(np.ones((2, 3)), 4.0).shape == (6,)


def test_fstar_at_zero():
    assert log_fstar_fast(0.0, 4.0)[0] == -np.inf
    assert log_fstar_fast(0.0, 1.0)[0] == pytest.approx(0.0, abs=1e-15)  # f*(0)=1
    assert log_fstar_fast(0.0, 0.5)[0] == np.inf


def test_sstar_limits():
    assert log_sstar_fast(0.0, 4.0)[0] == pytest.approx(0.0, abs=1e-15)
    assert np.all(np.diff(log_sstar_fast(np.linspace(0, 3, 50), 4.0)) <= 0)


def test_nu1_series_is_exponential():
    # nu = 1: f* is Exp(1), so log f* = log S* = -x and log G = -x.
    x = np.linspace(0.01, 3.0, 40)
    np.testing.assert_allclose(log_fstar_fast(x, 1.0), -x, atol=1e-12)
    np.testing.assert_allclose(log_sstar_fast(x, 1.0), -x, atol=1e-12)
    np.testing.assert_allclose(log_tail_fast(x, 1.0), -x, atol=1e-12)


def test_series_kernel_returns_both():
    u = np.array([0.1, 0.5, 1.0])
    lf, lS = series_kernel(u, 4.0)
    np.testing.assert_array_equal(lf, log_fstar_fast(u, 4.0))
    np.testing.assert_array_equal(lS, log_sstar_fast(u, 4.0))


# --------------------------------------------------------------------------
# likelihood vs the naive reference
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "nu,p",
    [(1.0, 0.05), (4.0, 0.05), (12.0, 0.3), (2.0, 0.0), (4.0, 1.0), (0.4, 0.2)],
)
def test_pooled_testbed_matches_naive(testbed, nu, p):
    data, prm = testbed
    ref = naive_per_meiosis(data, prm["L"], nu, p)
    assert loglik_fast(data, prm["L"], nu, p) == pytest.approx(
        ref.sum(), abs=atol_naive
    )
    got = loglik_fast(data, prm["L"], nu, p, per_meiosis=True)
    np.testing.assert_allclose(got, ref, atol=atol_naive)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_per_meiosis_parameters_match_naive(testbed, seed):
    data, prm = testbed
    nu, p = random_params(len(data), np.random.default_rng(seed))
    ref = naive_per_meiosis(data, prm["L"], nu, p)
    got = loglik_fast(data, prm["L"], nu, p, per_meiosis=True)
    np.testing.assert_allclose(got, ref, atol=atol_naive)


def test_per_meiosis_nu_only_and_p_only(testbed):
    data, prm = testbed
    nu, p = random_params(len(data), np.random.default_rng(5), p_edges=False)
    for args in [(nu, 0.1), (4.0, p)]:
        ref = naive_per_meiosis(data, prm["L"], *args)
        got = loglik_fast(data, prm["L"], *args, per_meiosis=True)
        np.testing.assert_allclose(got, ref, atol=atol_naive)


def test_simulated_heterogeneous_meioses_match_naive():
    # Each meiosis simulated from (and evaluated at) its own nu, p and L.
    rng = np.random.default_rng(4)
    M = 300
    Ls = rng.choice([0.4, 1.5, 2.8], size=M)
    nu, p = random_params(M, rng, p_edges=False)
    data = [simulate_meiosis(L, a, b, rng) for L, a, b in zip(Ls, nu, p)]
    ref = naive_per_meiosis(data, Ls, nu, p)
    got = loglik_fast(data, Ls, nu, p, per_meiosis=True)
    np.testing.assert_allclose(got, ref, atol=atol_naive)


@pytest.mark.parametrize("nu,p", [(4.0, 0.1), (10.0, 0.4)])
def test_mixed_lengths_match_naive(nu, p):
    rng = np.random.default_rng(7)
    Ls = rng.choice([0.6, 1.5, 2.8], size=300)
    data = [simulate_meiosis(L, nu, p, rng) for L in Ls]
    ref = naive_per_meiosis(data, Ls, nu, p)
    np.testing.assert_allclose(
        loglik_fast(data, Ls, nu, p, per_meiosis=True), ref, atol=atol_naive
    )


def test_many_crossovers_in_one_meiosis():
    # One outlier meiosis with many crossovers among ordinary ones.
    rng = np.random.default_rng(9)
    data = [simulate_meiosis(1.5, 4.0, 0.05, rng) for _ in range(50)]
    data.append(np.sort(rng.uniform(0, 1.5, 25)))
    ref = naive_per_meiosis(data, 1.5, 3.0, 0.3)
    got = loglik_fast(data, 1.5, 3.0, 0.3, per_meiosis=True)
    np.testing.assert_allclose(got, ref, atol=atol_naive)
    assert np.isfinite(got).all()


@settings(max_examples=60, deadline=None)
@given(
    st.lists(
        st.lists(st.floats(0.0, 1.0, allow_nan=False), max_size=5, unique=True),
        min_size=1,
        max_size=6,
    ),
    st.floats(0.2, 3.0),
    st.lists(st.floats(0.3, 80.0), min_size=6, max_size=6),
    st.lists(st.floats(0.0, 1.0), min_size=6, max_size=6),
)
def test_property_random_inputs_match_naive(fracs, L, nus, ps):
    data = [np.unique(np.asarray(f) * L) for f in fracs]
    M = len(data)
    nu, p = np.array(nus[:M]), np.array(ps[:M])
    ref = naive_per_meiosis(data, L, nu, p)
    got = loglik_fast(data, L, nu, p, per_meiosis=True)
    np.testing.assert_allclose(got, ref, atol=1e-8, rtol=1e-10)


# --------------------------------------------------------------------------
# model properties
# --------------------------------------------------------------------------
def test_per_meiosis_sums_to_total(testbed):
    data, prm = testbed
    nu, p = random_params(len(data), np.random.default_rng(3))
    per = loglik_fast(data, prm["L"], nu, p, per_meiosis=True)
    assert loglik_fast(data, prm["L"], nu, p) == pytest.approx(per.sum(), abs=1e-12)


def test_constant_arrays_equal_scalars(testbed):
    data, prm = testbed
    M = len(data)
    scalar = loglik_fast(data, prm["L"], 4.0, 0.05, per_meiosis=True)
    arrays = loglik_fast(
        data,
        np.full(M, prm["L"]),
        np.full(M, 4.0),
        np.full(M, 0.05),
        per_meiosis=True,
    )
    np.testing.assert_array_equal(arrays, scalar)


@pytest.mark.parametrize("p", [0.0, 0.3, 1.0])
def test_nu1_is_rate_one_poisson(testbed, p):
    data, prm = testbed
    rng = np.random.default_rng(1)
    Ls = rng.uniform(0.5, 3.0, size=len(data)) + prm["L"]
    assert loglik_fast(data, Ls, 1.0, p) == pytest.approx(-Ls.sum(), abs=1e-8)


def test_nu1_with_per_meiosis_p_is_poisson(testbed):
    data, prm = testbed
    p = np.random.default_rng(2).uniform(0, 1, len(data))
    got = loglik_fast(data, prm["L"], 1.0, p, per_meiosis=True)
    np.testing.assert_allclose(got, -prm["L"], atol=1e-8)


def test_p1_ignores_nu(testbed):
    data, prm = testbed
    nu = np.random.default_rng(6).uniform(0.5, 50, len(data))
    got = loglik_fast(data, prm["L"], nu, 1.0, per_meiosis=True)
    np.testing.assert_allclose(got, -prm["L"], atol=1e-12)


def test_heterogeneous_nu_changes_likelihood(testbed):
    data, prm = testbed
    nu = np.where(np.arange(len(data)) % 2 == 0, 2.0, 8.0)
    het = loglik_fast(data, prm["L"], nu, 0.05)
    assert het != pytest.approx(loglik_fast(data, prm["L"], 2.0, 0.05))
    assert het != pytest.approx(loglik_fast(data, prm["L"], 8.0, 0.05))


def test_testbed_mle_near_truth(testbed):
    data, prm = testbed
    mz = Meioses(data, prm["L"])
    nus = np.array([1.0, 2.0, 3.0, 4.0, 6.0, 10.0])
    ll = [loglik_fast(mz, None, v, prm["p"]) for v in nus]
    assert nus[int(np.argmax(ll))] in (3.0, 4.0)


# --------------------------------------------------------------------------
# data handling
# --------------------------------------------------------------------------
def test_positions_at_chromosome_ends_match_naive():
    data = [[0.0, 0.7], [0.4, 1.5], [0.0, 1.5], [0.0], [1.5]]
    for nu, p in [(4.0, 0.05), (0.5, 0.3), (1.0, 0.0)]:
        ref = naive_per_meiosis(data, 1.5, nu, p)
        got = loglik_fast(data, 1.5, nu, p, per_meiosis=True)
        np.testing.assert_allclose(got, ref, atol=atol_naive)
        assert np.isfinite(got).all()


def test_unsorted_input_is_sorted():
    assert loglik_fast([[1.1, 0.2, 0.6]], 1.5, 4.0, 0.1) == pytest.approx(
        loglik_fast([[0.2, 0.6, 1.1]], 1.5, 4.0, 0.1), abs=1e-14
    )


@pytest.mark.parametrize(
    "xs", [[0.5, 1.6], [-0.1], [0.2, np.nan], [0.3, np.inf], [0.4, 0.4]]
)
def test_invalid_positions_raise(xs):
    with pytest.raises(ValueError):
        loglik_fast([xs], 1.5, 4.0, 0.05)


@pytest.mark.parametrize(
    "kw",
    [
        dict(nu=0.0, p=0.1, L=1.0),
        dict(nu=-1.0, p=0.1, L=1.0),
        dict(nu=np.nan, p=0.1, L=1.0),
        dict(nu=np.inf, p=0.1, L=1.0),
        dict(nu=2.0, p=1.2, L=1.0),
        dict(nu=2.0, p=-0.1, L=1.0),
        dict(nu=2.0, p=np.nan, L=1.0),
        dict(nu=2.0, p=0.1, L=0.0),
        dict(nu=2.0, p=0.1, L=np.nan),
        dict(nu=[2.0, 0.0], p=0.1, L=1.0),
        dict(nu=2.0, p=[0.1, 1.5], L=1.0),
    ],
)
def test_invalid_parameters_raise(kw):
    with pytest.raises(ValueError):
        loglik_fast([[0.3], [0.6]], kw["L"], kw["nu"], kw["p"])


@pytest.mark.parametrize(
    "kw",
    [
        dict(nu=[2.0, 3.0, 4.0], p=0.1, L=1.0),
        dict(nu=2.0, p=[0.1], L=1.0),
        dict(nu=2.0, p=0.1, L=[1.0, 1.0, 1.0]),
        dict(nu=np.ones((2, 1)), p=0.1, L=1.0),
    ],
)
def test_wrong_parameter_shapes_raise(kw):
    with pytest.raises(ValueError, match="length 2"):
        loglik_fast([[0.3], [0.6]], kw["L"], kw["nu"], kw["p"])


def test_L_and_meioses_arguments():
    mz = Meioses([[0.3]], 1.0)
    with pytest.raises(ValueError, match="must be None"):
        loglik_fast(mz, 1.0, 4.0, 0.1)
    with pytest.raises(ValueError, match="required"):
        loglik_fast([[0.3]], None, 4.0, 0.1)


def test_meioses_container_reuse(testbed):
    data, prm = testbed
    mz = Meioses(data, prm["L"])
    for nu, p in [(2.0, 0.1), (6.0, 0.02)]:
        assert loglik_fast(mz, None, nu, p) == loglik_fast(data, prm["L"], nu, p)


def test_meioses_attributes():
    mz = Meioses([[0.9, 0.2], [], np.array([0.5])], [1.2, 1.0, 2.0])
    assert mz.M == 3
    np.testing.assert_array_equal(mz.counts, [2, 0, 1])
    np.testing.assert_array_equal(mz.L, [1.2, 1.0, 2.0])
    assert repr(mz) == "Meioses(M=3, crossovers=3, max_count=2)"
    assert repr(Meioses([], 1.0)) == "Meioses(M=0, crossovers=0, max_count=0)"


def test_empty_data_and_no_crossovers():
    assert loglik_fast([], 1.0, 4.0, 0.1) == 0.0
    assert loglik_fast([], 1.0, 4.0, 0.1, per_meiosis=True).shape == (0,)
    ref = loglik_meiosis_naive([], 1.2, 4.0, 0.1)
    assert loglik_fast([[], []], 1.2, 4.0, 0.1) == pytest.approx(2 * ref, abs=1e-12)
    got = loglik_fast([[], []], 1.2, [4.0, 9.0], [0.1, 0.5], per_meiosis=True)
    ref2 = naive_per_meiosis([[], []], 1.2, [4.0, 9.0], [0.1, 0.5])
    np.testing.assert_allclose(got, ref2, atol=atol_naive)


# --------------------------------------------------------------------------
# kernels
# --------------------------------------------------------------------------
def test_kernel_called_once_with_scalar_nu_when_pooled(testbed):
    data, prm = testbed
    calls = []

    def kernel(u, nu):
        calls.append((u.shape, np.ndim(nu)))
        return series_kernel(u, nu)

    got = loglik_fast(data, prm["L"], 4.0, 0.05, kernel=kernel)
    assert len(calls) == 1 and calls[0][1] == 0
    assert got == loglik_fast(data, prm["L"], 4.0, 0.05)


def test_kernel_gets_per_distance_nu_when_heterogeneous(testbed):
    data, prm = testbed
    nu = np.linspace(2.0, 8.0, len(data))
    seen = {}

    def kernel(u, nu_u):
        seen["shapes"] = (u.shape, np.shape(nu_u))
        return series_kernel(u, nu_u)

    got = loglik_fast(data, prm["L"], nu, 0.05, kernel=kernel)
    assert seen["shapes"][0] == seen["shapes"][1]
    assert got == loglik_fast(data, prm["L"], nu, 0.05)


def test_kernel_with_bad_output_raises():
    def bad(u, nu):
        return np.zeros(u.size + 1), np.zeros(u.size)

    with pytest.raises(ValueError, match="kernel"):
        loglik_fast([[0.3, 0.8]], 1.5, 4.0, 0.1, kernel=bad)


def test_approximate_kernel_error_propagates_boundedly(testbed):
    # A kernel perturbed by |delta| <= eps per value moves each meiosis by at
    # most (number of kernel values for that meiosis) * eps.
    data, prm = testbed
    eps = 1e-6

    def perturbed(u, nu):
        lf, lS = series_kernel(u, nu)
        return lf + eps, lS - eps

    mz = Meioses(data, prm["L"])
    exact = loglik_fast(mz, None, 4.0, 0.05, per_meiosis=True)
    approx = loglik_fast(mz, None, 4.0, 0.05, kernel=perturbed, per_meiosis=True)
    n = mz.counts
    bound = (2 * n + n * (n - 1) / 2) * eps
    assert np.all(np.abs(approx - exact) <= bound + 1e-12)


# --------------------------------------------------------------------------
# documentation
# --------------------------------------------------------------------------
def test_docstring_examples_run():
    result = doctest.testmod(fastlik, raise_on_error=False)
    assert result.attempted > 0 and result.failed == 0
