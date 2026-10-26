"""Tests for the simulator (`cointf.simintf`).

The likelihood tests rely on simulated data following the model exactly, so
these check the simulator's own properties: stationarity, mean counts, the
pathway edge cases and argument validation.
"""

import doctest

import numpy as np
import pytest
from scipy.stats import kstest

import cointf.simintf as simintf
from cointf.simintf import (
    escape_crossovers,
    interference_crossovers,
    simulate_meioses,
    simulate_meiosis,
    simulate_quartets,
)


@pytest.mark.parametrize("nu,p", [(1.0, 0.0), (4.0, 0.05), (12.0, 0.3)])
def test_mean_count_is_L(nu, p):
    L, n = 1.7, 20000
    counts = np.array([len(d) for d in simulate_meioses(L, nu, p, n, seed=3)])
    assert abs(counts.mean() - L) < 4 * counts.std() / np.sqrt(n)


def test_interference_reduces_count_variance():
    # nu = 1 is Poisson (variance L). Interference makes chiasma counts more
    # regular, but each chromatid keeps each chiasma with probability 1/2,
    # so the variance cannot fall below the binomial floor L / 2.
    L, n = 2.0, 20000
    var = {
        nu: np.var([len(d) for d in simulate_meioses(L, nu, 0.0, n, seed=4)])
        for nu in (1.0, 8.0)
    }
    assert abs(var[1.0] - L) < 0.1
    assert 0.5 * L < var[8.0] < 0.65 * L


@pytest.mark.parametrize("nu,p", [(4.0, 0.05), (10.0, 0.0)])
def test_positions_are_stationary(nu, p):
    # A stationary process puts crossovers uniformly along the chromosome.
    L = 1.5
    pos = np.concatenate(simulate_meioses(L, nu, p, 5000, seed=5))
    assert kstest(pos / L, "uniform").pvalue > 1e-3


def test_quartets_pool_two_meioses():
    L, n = 1.5, 20000
    quartets = simulate_quartets(L, 4.0, 0.05, n, seed=6)
    counts = np.array([len(d) for d in quartets])
    assert len(quartets) == n
    assert abs(counts.mean() - 2 * L) < 4 * counts.std() / np.sqrt(n)
    assert all(np.all(np.diff(d) > 0) for d in quartets)
    assert all(d.size == 0 or (d[0] >= 0 and d[-1] <= L) for d in quartets)


def test_seeds_are_reproducible():
    a = simulate_quartets(1.5, 4.0, 0.05, 20, seed=9)
    b = simulate_quartets(1.5, 4.0, 0.05, 20, seed=9)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))


def test_pathway_edge_cases():
    rng = np.random.default_rng(0)
    assert interference_crossovers(2.0, 4.0, 1.0, rng).size == 0
    assert escape_crossovers(2.0, 0.0, rng).size == 0


def test_default_rng():
    xs = simulate_meiosis(1.5, 4.0, 0.05)
    assert np.all((xs >= 0) & (xs <= 1.5))


@pytest.mark.parametrize(
    "kw",
    [dict(L=0.0), dict(L=-1.0), dict(nu=0.0), dict(p=-0.1), dict(p=1.5)],
)
def test_simulate_meiosis_rejects_bad_parameters(kw):
    args = dict(L=1.0, nu=4.0, p=0.05) | kw
    with pytest.raises(ValueError):
        simulate_meiosis(**args)


def test_simulate_quartets_rejects_bad_arguments():
    with pytest.raises(ValueError):
        simulate_quartets(L=1.0, nu=4.0, p=0.05, n=0, seed=1)
    with pytest.raises(ValueError):
        simulate_quartets(L=1.0, nu=4.0, p=0.05, n=5, seed=0)


def test_docstring_examples_run():
    assert doctest.testmod(simintf).failed == 0
