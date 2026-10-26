"""Fast, exact likelihoods for the Housworth-Stahl model of crossover interference."""

__version__ = "0.1.0a"

from .simintf import simulate_meiosis, simulate_meioses, simulate_quartets  # noqa
from .naive import loglik_meiosis_naive, loglik_naive, loglik_quartet_naive  # noqa
from .fastlik import Meioses, loglik_fast  # noqa
from .quartet import loglik_quartet  # noqa
from .chebyshev import ChebyshevKernel  # noqa
