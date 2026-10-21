"""Python package to implement fast likelihoods for the Houseworth-Stahl model."""

__version__ = "0.0.1b"

from .simintf import simulate_meiosis, simulate_meioses  # noqa
from .naive import loglik_meiosis_naive, loglik_naive  # noqa
from .fastlik import Meioses, loglik_fast  # noqa
