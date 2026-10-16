"""Exact (series-based) likelihood for the Housworth-Stahl model.

This is a slow reference implementation to compare against the saddlepoint
likelihood in `stahl_saddlepoint`. It uses the same generative model as
`cointf.simintf`:

  - Interference pathway: a stationary renewal process with gap density
    q * f*(q x; nu), where q = 1 - p and
        f*(x; nu) = sum_{k>=1} (1/2)^k Gamma(x; k*nu, rate=2*nu).
  - Escape pathway: a homogeneous Poisson process with rate p.

Observed crossovers are the superposition of the two, with pathway labels
unobserved. The likelihood sums over all labelings with a forward recursion
on the position of the most recent interference crossover (O(n^2) per
meiosis). f*, its survival function and its integrated tail are evaluated
from the truncated gamma series in log-space.
"""

import numpy as np
from scipy.special import gammaincc, logsumexp
from scipy.stats import gamma

_KMAX = 200


def _log_weights(kmax):
    k = np.arange(1, kmax + 1, dtype=np.float64)
    return k, -k * np.log(2.0)


def log_fstar(x, nu, kmax=_KMAX):
    """log f*(x; nu) via the truncated gamma series (vectorized over x)."""
    x = np.atleast_1d(np.asarray(x, dtype=np.float64))
    k, logw = _log_weights(kmax)
    terms = logw[None, :] + gamma.logpdf(x[:, None], a=k[None, :] * nu, scale=0.5 / nu)
    return logsumexp(terms, axis=1)


def log_sstar(x, nu, kmax=_KMAX):
    """log(1 - F*(x; nu)) via the truncated gamma series (vectorized over x)."""
    x = np.atleast_1d(np.asarray(x, dtype=np.float64))
    k, logw = _log_weights(kmax)
    terms = logw[None, :] + gamma.logsf(x[:, None], a=k[None, :] * nu, scale=0.5 / nu)
    return logsumexp(terms, axis=1)


def log_tail_integral(x, nu, kmax=_KMAX):
    """log of G(x) = int_x^inf (1 - F*(u; nu)) du.

    For a Gamma(a, rate b) component, int_x^inf S(u) du = (a/b) Q(a+1, bx) - x Q(a, bx),
    with Q the regularized upper incomplete gamma function. G(0) = E[f*] = 1.
    """
    k, _ = _log_weights(kmax)
    a = k * nu
    b = 2.0 * nu
    comp = (a / b) * gammaincc(a + 1.0, b * x) - x * gammaincc(a, b * x)
    val = np.sum(0.5**k * comp)
    return np.log(val) if val > 0.0 else -np.inf


def loglik_meiosis_naive(xs, L, nu, p, kmax=_KMAX):
    """Exact log-likelihood of one meiosis' crossover positions on [0, L].

    Parameters
    ----------
    xs  : Crossover positions in Morgans, all in (0, L).
    L   : Chromosome length in Morgans (> 0).
    nu  : Interference shape parameter (> 0).
    p   : Escape fraction in [0, 1].

    Returns
    -------
    float
        Log of the joint density of the observed crossover positions.
    """
    if L <= 0:
        raise ValueError(f"L must be > 0, got {L!r}")
    if nu <= 0:
        raise ValueError(f"nu must be > 0, got {nu!r}")
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"p must be in [0, 1], got {p!r}")
    xs = np.sort(np.asarray(xs, dtype=np.float64))
    n = xs.size
    q = 1.0 - p
    with np.errstate(divide="ignore"):
        log_p = np.log(p)
        log_q = np.log(q)

    if q == 0.0:
        # Pure Poisson process with rate 1.
        return -L + n * log_p

    # Precompute renewal terms (on the rescaled axis u = q * x).
    # first[j]  : log density that the first interference XO is at xs[j]
    # last[j]   : log P(no further interference XO in (xs[j], L])
    # gap[i, j] : log density of an interference gap xs[j] - xs[i], i < j
    first = log_q + log_sstar(q * xs, nu, kmax) if n else np.empty(0)
    last = log_sstar(q * (L - xs), nu, kmax) if n else np.empty(0)
    gap = np.full((n, n), -np.inf)
    iu, ju = np.triu_indices(n, k=1)
    if iu.size:
        gap[iu, ju] = log_q + log_fstar(q * (xs[ju] - xs[iu]), nu, kmax)

    # alpha[s]: log joint density of labels so far, where s = 0 means no
    # interference XO yet and s = i + 1 means the latest one is xs[i].
    alpha = np.full(n + 1, -np.inf)
    alpha[0] = 0.0
    for j in range(n):
        new = alpha + log_p  # xs[j] from the escape pathway
        trans = np.concatenate(([first[j]], gap[:j, j]))
        new[j + 1] = logsumexp(alpha[: j + 1] + trans)
        alpha = new

    end = np.concatenate(([log_tail_integral(q * L, nu, kmax)], last))
    return float(logsumexp(alpha + end) - p * L)


def loglik_naive(data, L, nu, p, kmax=_KMAX):
    """Exact log-likelihood summed over independent meioses."""
    return sum(loglik_meiosis_naive(d, L, nu, p, kmax=kmax) for d in data)
