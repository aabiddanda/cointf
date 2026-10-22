"""Tests for the exact (series-based) Housworth-Stahl likelihood."""

import itertools

import numpy as np
import pytest
from scipy.integrate import dblquad, quad
from scipy.stats import gamma

from cointf.naive import (
    log_fstar,
    log_sstar,
    log_tail_integral,
    loglik_meiosis_naive,
    loglik_naive,
)
from cointf.simintf import simulate_meioses


def fstar_value(x, nu):
    return np.exp(log_fstar(x, nu))[0]


def sstar_value(x, nu):
    return np.exp(log_sstar(x, nu))[0]


def brute_force_loglik(xs, L, nu, p):
    """Sum over all 2^n pathway labelings directly."""
    q = 1.0 - p
    total = 0.0
    for labels in itertools.product([0, 1], repeat=len(xs)):
        ys = [x for x, lab in zip(xs, labels) if lab == 1]  # interference
        m = len(xs) - len(ys)  # escape
        dens = p**m * np.exp(-p * L)
        if not ys:
            dens *= np.exp(log_tail_integral(q * L, nu))
        else:
            dens *= q * sstar_value(q * ys[0], nu)
            for a, b in zip(ys[:-1], ys[1:]):
                dens *= q * fstar_value(q * (b - a), nu)
            dens *= sstar_value(q * (L - ys[-1]), nu)
        total += dens
    return np.log(total)


def test_fstar_matches_series():
    for nu in [1.0, 2.5, 10.0]:
        for x in [0.05, 0.5, 1.0, 3.0]:
            k = np.arange(1, 81)
            ref = np.sum(0.5**k * gamma.pdf(x, a=k * nu, scale=0.5 / nu))
            assert fstar_value(x, nu) == pytest.approx(ref, rel=1e-10)


def test_fstar_nu1_is_exponential():
    xs = np.array([0.01, 0.5, 2.0, 5.0])
    np.testing.assert_allclose(log_fstar(xs, 1.0), -xs, atol=1e-10)
    np.testing.assert_allclose(log_sstar(xs, 1.0), -xs, atol=1e-10)


@pytest.mark.parametrize("nu", [1.0, 4.0, 12.0])
def test_tail_integral(nu):
    assert log_tail_integral(0.0, nu) == pytest.approx(0.0, abs=1e-10)
    for x in [0.3, 1.0, 2.5]:
        ref, _ = quad(lambda u: sstar_value(u, nu), x, np.inf)
        assert np.exp(log_tail_integral(x, nu)) == pytest.approx(ref, rel=1e-6)


@pytest.mark.parametrize("p", [0.0, 0.3, 1.0])
def test_no_interference_is_poisson(p):
    # nu = 1 makes both pathways Poisson, so the superposition is rate 1.
    L = 1.7
    for xs in [[], [0.4], [0.1, 0.9, 1.6]]:
        assert loglik_meiosis_naive(xs, L, 1.0, p) == pytest.approx(-L, abs=1e-9)


@pytest.mark.parametrize("nu,p", [(4.0, 0.1), (10.0, 0.5), (2.0, 0.0)])
def test_matches_brute_force(nu, p):
    L = 2.0
    for xs in [[], [0.7], [0.2, 1.1], [0.05, 0.6, 1.3, 1.95]]:
        got = loglik_meiosis_naive(xs, L, nu, p)
        ref = brute_force_loglik(xs, L, nu, p)
        assert got == pytest.approx(ref, rel=1e-9)


@pytest.mark.parametrize("nu,p", [(4.0, 0.05), (10.0, 0.2)])
def test_count_probabilities_match_simulation(nu, p):
    """P(N=0), P(N=1), P(N=2) from the likelihood vs the simulator."""
    L, n = 1.0, 40000
    data = simulate_meioses(L=L, nu=nu, p=p, n=n, seed=11)
    counts = np.bincount([len(d) for d in data], minlength=3) / n

    def dens(*xs):
        return np.exp(loglik_meiosis_naive(sorted(xs), L, nu, p))

    p0 = dens()
    p1, _ = quad(dens, 0, L)
    p2, _ = dblquad(lambda y, x: dens(x, y), 0, L, lambda x: x, L)
    for prob, freq in zip([p0, p1, p2], counts[:3]):
        se = np.sqrt(prob * (1 - prob) / n)
        assert abs(prob - freq) < 4 * se + 1e-3


def test_finite_near_chromosome_ends():
    # Crossovers near the ends.
    for xs in [[0.0062, 1.0723], [1.2486, 1.4952], [1.4946]]:
        for nu in [4.0, 8.0, 12.0]:
            assert np.isfinite(loglik_meiosis_naive(xs, 1.5, nu, 0.05))


# ── Fixed test bed (tests/data, true nu=4, p=0.05) ───────────────────────────


def test_testbed_matches_simulator(testbed):
    data, prm = testbed
    sim = simulate_meioses(
        L=prm["L"], nu=prm["nu"], p=prm["p"], n=prm["n"], seed=prm["seed"]
    )
    assert len(data) == prm["n"]
    for d, s in zip(data, sim):
        np.testing.assert_allclose(d, s, rtol=1e-12)


def test_testbed_loglik(testbed):
    data, prm = testbed
    L, p = prm["L"], prm["p"]
    nus = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 12.0]
    lls = np.array([loglik_naive(data, L, nu, p) for nu in nus])
    assert np.all(np.isfinite(lls))
    # nu = 1 is a rate-1 Poisson process: log-likelihood is exactly -n * L.
    assert lls[0] == pytest.approx(-prm["n"] * L, abs=1e-8)
    # Reference values; update deliberately if the model definition changes.
    assert lls[nus.index(4.0)] == pytest.approx(-269.79, abs=0.01)
    assert nus[int(np.argmax(lls))] in (3.0, 4.0)


@pytest.mark.parametrize(
    "L,nu,p", [(0.0, 4.0, 0.1), (1.0, 0.0, 0.1), (1.0, 4.0, -0.1), (1.0, 4.0, 1.1)]
)
def test_invalid_parameters_raise(L, nu, p):
    with pytest.raises(ValueError):
        loglik_meiosis_naive([0.3], L, nu, p)


def test_simulate_meioses_rejects_bad_arguments():
    with pytest.raises(ValueError):
        simulate_meioses(L=1.0, nu=4.0, p=0.05, n=0, seed=1)
    with pytest.raises(ValueError):
        simulate_meioses(L=1.0, nu=4.0, p=0.05, n=5, seed=0)
