"""Tests for the phase-unknown (quartet) likelihood (`cointf.quartet`).

The recursion is checked against two references: the brute-force 3^n
labeling sum in `cointf.naive`, and an independent identity that pools two
single-meiosis likelihoods over every split of the crossovers between the
two meioses.
"""

import doctest
import itertools

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.integrate import quad
from scipy.special import logsumexp

import cointf.quartet as quartet
from cointf.chebyshev import ChebyshevKernel
from cointf.fastlik import Meioses
from cointf.naive import loglik_meiosis_naive, loglik_quartet_naive
from cointf.quartet import loglik_quartet
from cointf.simintf import simulate_quartets


def split_loglik(xs, L, nu, p):
    """Sum single-meiosis densities over all 2^n splits into two meioses."""
    xs = list(xs)
    terms = []
    for in_a in itertools.product([False, True], repeat=len(xs)):
        a = [x for x, s in zip(xs, in_a) if s]
        b = [x for x, s in zip(xs, in_a) if not s]
        terms.append(
            loglik_meiosis_naive(a, L, nu, p) + loglik_meiosis_naive(b, L, nu, p)
        )
    return float(logsumexp(terms))


small_datasets = [
    [],
    [0.7],
    [0.2, 1.1],
    [0.05, 0.6, 1.3, 1.95],
    [0.1, 0.15, 0.9, 1.0, 1.7],
]


@pytest.mark.parametrize("nu,p", [(4.0, 0.1), (10.0, 0.5), (2.0, 0.0), (0.5, 0.2)])
def test_matches_brute_force(nu, p):
    for xs in small_datasets:
        got = loglik_quartet([xs], 2.0, nu, p)
        assert got == pytest.approx(loglik_quartet_naive(xs, 2.0, nu, p), abs=1e-10)


@pytest.mark.parametrize("nu,p", [(4.0, 0.1), (12.0, 0.03), (1.7, 0.0)])
def test_matches_pooled_single_meioses(nu, p):
    for xs in small_datasets:
        got = loglik_quartet([xs], 2.0, nu, p)
        assert got == pytest.approx(split_loglik(xs, 2.0, nu, p), abs=1e-10)


@pytest.mark.parametrize("p", [0.0, 0.3, 1.0])
def test_nu1_is_rate_two_poisson(p):
    data = simulate_quartets(L=1.8, nu=3.0, p=0.1, n=50, seed=4)
    got = loglik_quartet(data, 1.8, 1.0, p, per_meiosis=True)
    expected = np.array([len(d) for d in data]) * np.log(2.0) - 3.6
    np.testing.assert_allclose(got, expected, atol=1e-10)


def test_p1_ignores_nu():
    data = simulate_quartets(L=1.5, nu=4.0, p=0.05, n=30, seed=5)
    a = loglik_quartet(data, 1.5, 4.0, 1.0)
    b = loglik_quartet(data, 1.5, 20.0, 1.0)
    assert a == b


def test_simulated_quartets_match_brute_force():
    data = simulate_quartets(L=1.5, nu=6.0, p=0.08, n=40, seed=9)
    data = [d for d in data if len(d) <= 7]
    got = loglik_quartet(data, 1.5, 6.0, 0.08, per_meiosis=True)
    ref = [loglik_quartet_naive(d, 1.5, 6.0, 0.08) for d in data]
    np.testing.assert_allclose(got, ref, atol=1e-10)


def test_per_quartet_parameters():
    data = simulate_quartets(L=1.5, nu=4.0, p=0.05, n=20, seed=2)
    rng = np.random.default_rng(0)
    nu = rng.uniform(1.5, 12.0, len(data))
    p = rng.uniform(0.0, 0.3, len(data))
    got = loglik_quartet(data, 1.5, nu, p, per_meiosis=True)
    ref = [
        loglik_quartet(d[None, :] if d.size else [d], 1.5, a, b)
        for d, a, b in zip(data, nu, p)
    ]
    np.testing.assert_allclose(got, ref, atol=1e-12)


def test_density_integrates_over_one_crossover():
    """P(N = 0) and P(N = 1) of pooled data vs simulation."""
    L, nu, p, n = 1.0, 5.0, 0.05, 40000
    counts = np.bincount([len(d) for d in simulate_quartets(L, nu, p, n, seed=3)])
    p0 = np.exp(loglik_quartet([[]], L, nu, p))
    p1, _ = quad(lambda x: np.exp(loglik_quartet([[x]], L, nu, p)), 0, L)
    for prob, freq in zip([p0, p1], counts[:2] / n):
        se = np.sqrt(prob * (1 - prob) / n)
        assert abs(prob - freq) < 4 * se


def test_quartet_mle_near_truth():
    data = simulate_quartets(L=2.0, nu=6.0, p=0.05, n=1500, seed=12)
    mz = Meioses(data, 2.0)
    nus = np.array([3.0, 4.5, 6.0, 8.0, 11.0])
    ll = [loglik_quartet(mz, None, v, 0.05) for v in nus]
    assert nus[int(np.argmax(ll))] == 6.0


def test_chebyshev_kernel_matches_series():
    mz = Meioses(simulate_quartets(L=2.5, nu=4.3, p=0.05, n=500, seed=6), 2.5)
    exact = loglik_quartet(mz, None, 4.3, 0.05, per_meiosis=True)
    approx = loglik_quartet(
        mz, None, 4.3, 0.05, kernel=ChebyshevKernel(), per_meiosis=True
    )
    np.testing.assert_allclose(approx, exact, atol=1e-11)


def test_empty_and_invalid():
    assert loglik_quartet([], 1.0, 4.0, 0.1) == 0.0
    with pytest.raises(ValueError):
        loglik_quartet([[0.5]], 1.0, -1.0, 0.1)
    with pytest.raises(ValueError):
        loglik_quartet([[0.5]], 1.0, 4.0, 1.5)
    with pytest.raises(ValueError):
        loglik_quartet([[1.5]], 1.0, 4.0, 0.1)


def test_docstring_examples_run():
    assert doctest.testmod(quartet).failed == 0


@pytest.mark.parametrize("nu,p", [(4.0, 0.1), (9.0, 0.02)])
def test_mixed_lengths_match_brute_force(nu, p):
    rng = np.random.default_rng(11)
    Ls = rng.uniform(0.5, 3.0, 12)
    data = [np.sort(rng.uniform(0, L, rng.integers(0, 6))) for L in Ls]
    got = loglik_quartet(data, Ls, nu, p, per_meiosis=True)
    ref = [loglik_quartet_naive(d, L, nu, p) for d, L in zip(data, Ls)]
    np.testing.assert_allclose(got, ref, atol=1e-10)


@settings(max_examples=40, deadline=None)
@given(
    fracs=st.lists(st.floats(0.001, 0.999), min_size=0, max_size=6, unique=True),
    L=st.floats(0.3, 3.0),
    nu=st.floats(0.4, 30.0),
    p=st.floats(0.0, 1.0),
)
def test_property_random_inputs_match_brute_force(fracs, L, nu, p):
    xs = np.sort(np.array(fracs) * L)
    got = loglik_quartet([xs], L, nu, p)
    assert got == pytest.approx(loglik_quartet_naive(xs, L, nu, p), abs=1e-9)


def test_meioses_reuse_matches_lists():
    data = simulate_quartets(L=1.5, nu=4.0, p=0.05, n=50, seed=13)
    mz = Meioses(data, 1.5)
    for nu, p in [(2.0, 0.1), (6.0, 0.0), (1.0, 1.0)]:
        assert loglik_quartet(mz, None, nu, p) == pytest.approx(
            loglik_quartet(data, 1.5, nu, p), abs=1e-12
        )


def test_naive_quartet_rejects_bad_parameters():
    for kw in [dict(L=0.0), dict(nu=0.0), dict(p=1.5)]:
        args = dict(xs=[0.5], L=1.0, nu=4.0, p=0.1) | kw
        with pytest.raises(ValueError):
            loglik_quartet_naive(**args)
    assert loglik_quartet_naive([0.2, 0.7], 1.0, 4.0, 1.0) == pytest.approx(
        2 * np.log(2.0) - 2.0
    )
