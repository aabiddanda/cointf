import numpy as np
from scipy.optimize import minimize
from stahl_saddlepoint import loglik_meiosis


class Stahl:
    def __init__(self):
        """Implementation of Stahl et al Model."""
        self.phat

    def loglik(self, data, p=0.1, nu=2):
        """Calculate the log-likelihood of the meioses under the Houseworth-Stahl model.

        Data is a list of lists.
        """
        assert p >= 0.0
        assert p <= 1.0
        assert nu >= 1.0
        assert len(data) > 1
        loglik = 0.0
        for d in data:
            loglik += loglik_meiosis(d, p=p, nu=nu)
        return loglik

    def fit_stahl(self, data, L, **kwargs):
        """Fit data from the Houseworth-Stahl model to numerically infer parameters."""
        opt_res = minimize(
            lambda x: self.loglik(data, p=x[0], nu=x[1]),
            x0=[0.1, 1],
            bounds=[(0, 1), (0.1, 30)],
            **kwargs,
        )
        return opt_res.x
