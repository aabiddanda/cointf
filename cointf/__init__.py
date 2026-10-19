"""Python package to implement fast likelihoods for the Houseworth-Stahl model."""

__version__ = "0.0.1b"

from .stahl import Stahl  # noqa
from .simintf import SimStahl  # noqa
from .naive import loglik_meiosis_naive, loglik_naive  # noqa
from .fastlik import Meioses, loglik_fast  # noqa
