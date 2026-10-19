"""Shared fixtures for cointf tests."""

import ast
import csv
from pathlib import Path

import pytest

data_dir = Path(__file__).parent / "data"

# Test bed: 200 meioses simulated with
#   cointf simulate --nu 4.0 --p 0.05 --length 1.5 --n 200 --seed 7
testbed_path = data_dir / "testbed_L1.5_nu4_p0.05_n200_seed7.tsv"
testbed_params = dict(L=1.5, nu=4.0, p=0.05, n=200, seed=7)


def read_meioses(path):
    """Read crossover positions from a `cointf simulate` TSV."""
    with open(path) as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    return [ast.literal_eval(r["X"].strip("'")) for r in rows]


@pytest.fixture(scope="session")
def testbed():
    """(data, params) for the fixed simulated test bed."""
    return read_meioses(testbed_path), testbed_params
