"""Shared fixtures for cointf tests."""

import ast
import csv
from pathlib import Path

import pytest

data_dir = Path(__file__).parent / "data"

# Test bed: 200 meioses simulated with
#   simulate_meioses(L=1.5, nu=4.0, p=0.05, n=200, seed=7)
# stored as a TSV with columns ID, L, nu, p, X, where X is a quoted list of
# crossover positions in Morgans, e.g. '[0.41, 0.68]'.
testbed_path = data_dir / "testbed_L1.5_nu4_p0.05_n200_seed7.tsv"
testbed_params = dict(L=1.5, nu=4.0, p=0.05, n=200, seed=7)


def read_meioses(path):
    """Read crossover positions from a test bed TSV."""
    with open(path) as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    return [ast.literal_eval(r["X"].strip("'")) for r in rows]


@pytest.fixture(scope="session")
def testbed():
    """(data, params) for the fixed simulated test bed."""
    return read_meioses(testbed_path), testbed_params
