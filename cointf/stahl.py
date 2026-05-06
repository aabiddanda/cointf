import numpy as np
from scipy.optimize import minimize
from stahl_saddlepoint import loglik_meiosis


class Stahl:
    def __init__(self):
        """Implementation of the Houseworth-Stahl Model for interference."""
        pass

    def loglik(self, data, L, p=0.1, nu=2):
        """Calculate the log-likelihood of the meioses under the Houseworth-Stahl model.

        Data is a list of lists.
        """
        assert p >= 0.0
        assert p <= 1.0
        assert nu >= 1.0
        assert len(data) > 1
        loglik = 0.0
        for d in data:
            lp = loglik_meiosis(np.array(d, dtype=float), L=L, p=p, nu=nu)
            print(lp, d)
            loglik += lp
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

    def est_confint(self, data, L, **kwargs):
        """Estimate confidence intervals for parameters using profile-likelihoods"""
        pass

    def likelihood_ratio(self, data, L, p=0.1, nu=2.0):
        """Estimating the likelihood between this model and the model with no interference."""
        ll_tot = self.loglik(data, L, p=p, nu=nu)
        ll_null = self.loglik(data, L, p=0.0, nu=nu)
        return ll_tot, ll_null
