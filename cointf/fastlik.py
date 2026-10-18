"""Vectorized exact likelihood for the Housworth-Stahl model.

Same model and recursion as `cointf.naive`, but:

  - f* and S* = 1 - F* come from a 60-term gamma series evaluated directly
    with `gammaln` / `gammaincc` (no scipy.stats overhead), and
  - the forward recursion over pathway labels runs for all meioses at once on
    a padded array, so the cost of one evaluation is a handful of numpy calls
    per crossover index rather than a Python loop over meioses.

The kernel (u -> log f*(u), log S*(u)) is a pluggable callable, so a tabulated
kernel can replace the series without changing the recursion.
"""

import numpy as np
from scipy.special import gammaincc, gammaln, logsumexp

from .naive import log_tail_integral

_K = np.arange(1, 61, dtype=np.float64)
_LOGW = -_K * np.log(2.0)
_W = np.exp(_LOGW)


def log_fstar_fast(x, nu):
    """log f*(x; nu) from the 60-term gamma series (vectorized over x)."""
    x = np.atleast_1d(np.asarray(x, dtype=np.float64))[:, None]
    a, b = _K * nu, 2.0 * nu
    with np.errstate(divide="ignore", invalid="ignore"):
        logpdf = (a - 1.0) * np.log(x) + a * np.log(b) - b * x - gammaln(a)
    return logsumexp(_LOGW + logpdf, axis=1)


def log_sstar_fast(x, nu):
    """log(1 - F*(x; nu)) from the 60-term gamma series (vectorized over x).

    The truncated tail contributes less than 2^-60 in absolute terms.
    """
    x = np.atleast_1d(np.asarray(x, dtype=np.float64))[:, None]
    S = np.sum(_W * gammaincc(_K * nu, 2.0 * nu * x), axis=1)
    with np.errstate(divide="ignore"):
        return np.log(S)


def series_kernel(nu):
    """Default kernel: u -> (log f*(u; nu), log S*(u; nu)) from the series."""

    def kernel(u):
        return log_fstar_fast(u, nu), log_sstar_fast(u, nu)

    return kernel


class Meioses:
    """Crossover positions padded to an (M, nmax) array, plus pair indices.

    Parameters
    ----------
    data : Sequence of crossover-position arrays, one per meiosis (Morgans).
    L    : Chromosome length in Morgans: a scalar, or one value per meiosis.
    """

    def __init__(self, data, L):
        self.M = len(data)
        L = np.asarray(L, dtype=np.float64)
        self.L = np.broadcast_to(L, (self.M,)).copy() if L.ndim == 0 else L.copy()
        if self.L.shape != (self.M,):
            raise ValueError(f"L must be a scalar or have length {self.M}")
        if np.any(self.L <= 0):
            raise ValueError("L must be > 0")
        self.counts = np.array([len(d) for d in data], dtype=int)
        self.nmax = int(self.counts.max()) if self.M else 0
        X = np.full((self.M, self.nmax), np.nan)
        for m, d in enumerate(data):
            d = np.sort(np.asarray(d, dtype=np.float64))
            if d.size and (d[0] <= 0.0 or d[-1] >= self.L[m]):
                raise ValueError(
                    f"meiosis {m}: crossover positions must lie in (0, {self.L[m]})"
                )
            X[m, : d.size] = d
        self.X = X
        self.valid = ~np.isnan(X)
        self.iu, self.ju = np.triu_indices(self.nmax, k=1)
        self.pair_valid = self.valid[:, self.ju]  # j valid implies i valid


def loglik_fast(data_or_meioses, L, nu, p, kernel=None, per_meiosis=False):
    """Exact log-likelihood of crossover positions, vectorized across meioses.

    Parameters
    ----------
    data_or_meioses : Sequence of crossover-position arrays, or a `Meioses`.
    L      : Chromosome length(s) in Morgans; ignored if a `Meioses` is given.
    nu     : Interference shape parameter (> 0).
    p      : Escape fraction in [0, 1].
    kernel : Optional callable u -> (log f*(u), log S*(u)) for this nu.
             Defaults to the 60-term series.
    per_meiosis : If True, return the per-meiosis log-likelihoods.

    Returns
    -------
    float, or np.ndarray of shape (M,) if `per_meiosis`.
    """
    if nu <= 0:
        raise ValueError(f"nu must be > 0, got {nu!r}")
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"p must be in [0, 1], got {p!r}")
    mz = (
        data_or_meioses
        if isinstance(data_or_meioses, Meioses)
        else Meioses(data_or_meioses, L)
    )
    if mz.M == 0:
        return np.empty(0) if per_meiosis else 0.0
    q = 1.0 - p
    with np.errstate(divide="ignore", invalid="ignore"):
        log_p = np.log(p)
        if q == 0.0:
            # Pure Poisson process with rate 1.
            ll = -mz.L + mz.counts * log_p
            return ll if per_meiosis else float(ll.sum())
        ll = _recursion(mz, nu, p, q, log_p, kernel or series_kernel(nu))
    return ll if per_meiosis else float(ll.sum())


def _recursion(mz, nu, p, q, log_p, kernel):
    """Forward recursion over pathway labels for all meioses (log-space)."""
    log_q = np.log(q)
    X, V, pv = mz.X, mz.valid, mz.pair_valid
    Lcol = mz.L[:, None]

    # Every u at which the kernel is needed, evaluated in a single call.
    gaps = X[:, mz.ju] - X[:, mz.iu]
    u_first = q * X[V]
    u_last = q * (Lcol - X)[V]
    u_gap = q * gaps[pv]
    n1, n2 = u_first.size, u_last.size
    lf, lS = kernel(np.concatenate([u_first, u_last, u_gap]))
    lf, lS = np.asarray(lf, dtype=np.float64), np.asarray(lS, dtype=np.float64)

    # first[m, j]  : log density that the first interference XO is X[m, j]
    # last[m, j]   : log P(no further interference XO after X[m, j])
    # gap[m, i, j] : log density of an interference gap X[m, j] - X[m, i]
    first = np.full(X.shape, -np.inf)
    first[V] = log_q + lS[:n1]
    last = np.full(X.shape, -np.inf)
    last[V] = lS[n1 : n1 + n2]
    gm = np.full(gaps.shape, -np.inf)
    gm[pv] = log_q + lf[n1 + n2 :]
    gap = np.full((mz.M, mz.nmax, mz.nmax), -np.inf)
    gap[:, mz.iu, mz.ju] = gm

    # alpha[m, s]: s = 0 means no interference XO yet, s = i + 1 means the
    # latest one is X[m, i].
    alpha = np.full((mz.M, mz.nmax + 1), -np.inf)
    alpha[:, 0] = 0.0
    for j in range(mz.nmax):
        new = alpha + log_p  # X[:, j] from the escape pathway
        trans = np.concatenate([first[:, j : j + 1], gap[:, :j, j]], axis=1)
        new[:, j + 1] = logsumexp(alpha[:, : j + 1] + trans, axis=1)
        alpha = np.where((mz.counts > j)[:, None], new, alpha)

    # log G(qL), computed once per distinct chromosome length.
    uL, inv = np.unique(mz.L, return_inverse=True)
    logG = np.array([log_tail_integral(q * Lu, nu) for Lu in uL])[inv]
    end = np.concatenate([logG[:, None], last], axis=1)
    return logsumexp(alpha + end, axis=1) - p * mz.L
