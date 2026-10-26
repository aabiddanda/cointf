r"""Reference (slow) exact likelihood for the Housworth-Stahl model.

A direct, one-meiosis-at-a-time implementation used to validate
:mod:`cointf.fastlik`. It uses the same generative model as
:mod:`cointf.simintf`:

- Interference pathway: a stationary renewal process with gap density
  :math:`q f^*(qx; \nu)`, where :math:`q = 1 - p` and

  .. math::

      f^*(x; \nu) = \sum_{k \ge 1} 2^{-k}\,
      \mathrm{Gamma}(x;\, k\nu,\ \mathrm{rate}=2\nu).

- Escape pathway: a homogeneous Poisson process with rate :math:`p`.

Observed crossovers are the superposition of the two, with pathway labels
unobserved. The likelihood sums over all labelings with a forward recursion
on the position of the most recent interference crossover
(:math:`O(n^2)` per meiosis). :math:`f^*`, its survival function and its
integrated tail are evaluated in log space from a 200-term gamma series.

:func:`loglik_quartet_naive` is the matching reference for phase-unknown
(quartet) data, the pooled crossovers of two meioses. It sums directly over
all :math:`3^n` labelings (interference in meiosis A, interference in
meiosis B, or escape) and is used to validate :mod:`cointf.quartet`.
"""

import itertools

import numpy as np
from scipy.special import gammaincc, logsumexp
from scipy.stats import gamma

__all__ = [
    "log_fstar",
    "log_sstar",
    "log_tail_integral",
    "loglik_meiosis_naive",
    "loglik_naive",
    "loglik_quartet_naive",
]

default_kmax = 200


def series_log_weights(kmax):
    """Return the gamma-series indices and their log weights.

    Parameters
    ----------
    kmax : int
        Number of series terms.

    Returns
    -------
    k : numpy.ndarray
        Indices ``1, ..., kmax`` as floats.
    log_weight : numpy.ndarray
        Log weights ``-k * log(2)``.
    """
    k = np.arange(1, kmax + 1, dtype=np.float64)
    return k, -k * np.log(2.0)


def log_fstar(x, nu, kmax=default_kmax):
    r"""Log density of the unit-rate inter-crossover distance.

    Parameters
    ----------
    x : array_like
        Distances in Morgans, ``> 0``.
    nu : float
        Interference parameter, ``> 0``.
    kmax : int, default 200
        Number of gamma-series terms.

    Returns
    -------
    numpy.ndarray
        :math:`\log f^*(x; \nu)`, shape ``(x.size,)``.
    """
    x = np.atleast_1d(np.asarray(x, dtype=np.float64))
    k, logw = series_log_weights(kmax)
    terms = logw[None, :] + gamma.logpdf(x[:, None], a=k[None, :] * nu, scale=0.5 / nu)
    return logsumexp(terms, axis=1)


def log_sstar(x, nu, kmax=default_kmax):
    r"""Log survival function of the inter-crossover distance.

    Parameters
    ----------
    x : array_like
        Distances in Morgans, ``>= 0``.
    nu : float
        Interference parameter, ``> 0``.
    kmax : int, default 200
        Number of gamma-series terms.

    Returns
    -------
    numpy.ndarray
        :math:`\log(1 - F^*(x; \nu))`, shape ``(x.size,)``.
    """
    x = np.atleast_1d(np.asarray(x, dtype=np.float64))
    k, logw = series_log_weights(kmax)
    terms = logw[None, :] + gamma.logsf(x[:, None], a=k[None, :] * nu, scale=0.5 / nu)
    return logsumexp(terms, axis=1)


def log_tail_integral(x, nu, kmax=default_kmax):
    r"""Log integrated survival function of the inter-crossover distance.

    Computes :math:`\log G(x)` with :math:`G(x) = \int_x^\infty (1 - F^*(u; \nu))\, du`.
    For a gamma component with shape :math:`a` and rate :math:`b`,
    :math:`\int_x^\infty S(u)\, du = (a/b) Q(a + 1, bx) - x Q(a, bx)`, with
    :math:`Q` the regularised upper incomplete gamma function;
    :math:`G(0) = 1`.

    Parameters
    ----------
    x : float
        Length in Morgans, ``>= 0``.
    nu : float
        Interference parameter, ``> 0``.
    kmax : int, default 200
        Number of gamma-series terms.

    Returns
    -------
    float
        :math:`\log G(x)`, or ``-inf`` if the series is not positive.
    """
    k = series_log_weights(kmax)[0]
    a = k * nu
    b = 2.0 * nu
    comp = (a / b) * gammaincc(a + 1.0, b * x) - x * gammaincc(a, b * x)
    val = np.sum(0.5**k * comp)
    return np.log(val) if val > 0.0 else -np.inf


def loglik_meiosis_naive(xs, L, nu, p, kmax=default_kmax):
    """Exact log-likelihood of one meiosis' crossover positions.

    Parameters
    ----------
    xs : array_like
        Crossover positions in Morgans, in ``[0, L]`` and distinct.
    L : float
        Chromosome length in Morgans, ``> 0``.
    nu : float
        Interference parameter, ``> 0``.
    p : float
        Fraction of crossovers escaping interference, in ``[0, 1]``.
    kmax : int, default 200
        Number of gamma-series terms.

    Returns
    -------
    float
        Log of the joint density of the observed crossover positions.

    Raises
    ------
    ValueError
        If `L`, `nu` or `p` is out of range.

    See Also
    --------
    cointf.fastlik.loglik_fast : Fast, vectorised equivalent.
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


def loglik_naive(data, L, nu, p, kmax=default_kmax):
    """Exact log-likelihood summed over independent meioses.

    Parameters
    ----------
    data : sequence of array_like
        Crossover positions in Morgans, one sequence per meiosis.
    L : float
        Chromosome length in Morgans, shared by all meioses.
    nu : float
        Interference parameter, ``> 0``.
    p : float
        Fraction of crossovers escaping interference, in ``[0, 1]``.
    kmax : int, default 200
        Number of gamma-series terms.

    Returns
    -------
    float
        Sum of :func:`loglik_meiosis_naive` over the meioses.
    """
    return sum(loglik_meiosis_naive(d, L, nu, p, kmax=kmax) for d in data)


def loglik_quartet_naive(xs, L, nu, p, kmax=default_kmax):
    r"""Brute-force log-likelihood of one phase-unknown (quartet) dataset.

    Sums directly over all :math:`3^n` labelings of the observed crossovers
    as interference crossovers of meiosis A, of meiosis B, or escape
    crossovers (a rate-:math:`2p` Poisson process). Only practical for small
    :math:`n`; used to validate :func:`cointf.quartet.loglik_quartet`.

    Parameters
    ----------
    xs : array_like
        Pooled crossover positions of two meioses, in ``[0, L]`` and
        distinct.
    L : float
        Chromosome length in Morgans, ``> 0``.
    nu : float
        Interference parameter, ``> 0``.
    p : float
        Escape fraction in each meiosis, in ``[0, 1]``.
    kmax : int, default 200
        Number of gamma-series terms.

    Returns
    -------
    float
        Log of the joint density of the pooled crossover positions.

    Raises
    ------
    ValueError
        If `L`, `nu` or `p` is out of range.
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
    if q == 0.0:
        return n * np.log(2.0) - 2.0 * L

    with np.errstate(divide="ignore"):
        log_2p = np.log(2.0 * p)
    log_q = np.log(q)
    first = log_q + log_sstar(q * xs, nu, kmax) if n else np.empty(0)
    last = log_sstar(q * (L - xs), nu, kmax) if n else np.empty(0)
    gap = np.full((n, n), -np.inf)
    iu, ju = np.triu_indices(n, k=1)
    if iu.size:
        gap[iu, ju] = log_q + log_fstar(q * (xs[ju] - xs[iu]), nu, kmax)
    log_empty = log_tail_integral(q * L, nu, kmax)

    def log_renewal(ix):
        """Log density of one meiosis' interference crossovers at xs[ix]."""
        if not ix:
            return log_empty
        inner = sum(gap[a, b] for a, b in zip(ix[:-1], ix[1:]))
        return first[ix[0]] + inner + last[ix[-1]]

    terms = []
    for labels in itertools.product((0, 1, 2), repeat=n):
        in_a = [i for i, lab in enumerate(labels) if lab == 1]
        in_b = [i for i, lab in enumerate(labels) if lab == 2]
        n_escape = n - len(in_a) - len(in_b)
        escape = n_escape * log_2p if n_escape else 0.0
        terms.append(escape + log_renewal(in_a) + log_renewal(in_b))
    return float(logsumexp(terms) - 2.0 * p * L)
