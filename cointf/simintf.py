import numpy as np
from scipy.stats import gamma, poisson


class SimStahl:
    def __init__(self, L, n=100):
        """Initialize a crossover simulator."""
        self.L = L
        self.n = n

    def sim_stahl(self, p=0.05, nu=1.0):
        """Simulate crossovers according to the stahl model."""
        assert (p > 0) and (p < 1.0)
        assert nu > 0
        nchrom = len(self.L)
        for c in nchrom:
            """Now simulate each individual..."""
            pass
