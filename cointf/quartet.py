r"""Phase-unknown (quartet) likelihood for the Housworth-Stahl model.

In a nuclear family with two offspring and no grandparents, a parent's
crossovers are seen as switches in the siblings' shared inheritance, but it
is unknown in which of the two meioses each one happened. The observed
positions are then the superposition of two independent meioses with the
same :math:`(\nu, p)` and length :math:`L`.

Each observed crossover comes from the interference pathway of meiosis A,
from that of meiosis B, or from the escape pathway of either meiosis. The
two escape pathways superpose to a Poisson process of rate :math:`2p`, so
escape crossovers need no meiosis label. The likelihood sums over all
:math:`3^n` labelings with a forward recursion whose state is the pair of
latest interference crossovers in A and B, costing :math:`O(n^3)` per
quartet instead of :math:`O(3^n)`.

The interference terms are the same as for a single meiosis (:math:`f^*`
at gaps, :math:`S^*` at end distances and :math:`G` for an empty pathway),
so this module reuses :class:`cointf.fastlik.Meioses` and the kernel
interface of :func:`cointf.fastlik.loglik_fast`.
"""

import numpy as np

from .fastlik import logsumexp_last, prepare_inputs, renewal_terms, series_kernel

__all__ = ["loglik_quartet"]


def loglik_quartet(data, L, nu, p, *, kernel=None, per_meiosis=False):
    r"""Log-likelihood of phase-unknown crossovers pooled from two meioses.

    Parameters
    ----------
    data : sequence of array_like or Meioses
        Crossover positions in Morgans, one sequence per quartet (the pooled
        crossovers of two meioses of one chromosome), or a prepared
        :class:`cointf.fastlik.Meioses`.
    L : float or array_like or None
        Chromosome length(s) in Morgans; ``None`` when `data` is a
        :class:`cointf.fastlik.Meioses`.
    nu : float or array_like
        Interference parameter, ``> 0``: a scalar or one value per quartet.
        Both meioses of a quartet share it.
    p : float or array_like
        Escape fraction in each meiosis, in ``[0, 1]``: a scalar or one value
        per quartet. Both meioses of a quartet share it.
    kernel : callable, optional
        Kernel ``kernel(u_gap, nu_gap, u_end, nu_end) -> (log_f, log_S)``;
        see :func:`cointf.fastlik.series_kernel` (the default) and
        :class:`cointf.chebyshev.ChebyshevKernel`.
    per_meiosis : bool, default False
        If True, return one log-likelihood per entry of `data`, i.e. per
        quartet (one pair of pooled meioses), instead of the sum. Named as in
        :func:`cointf.fastlik.loglik_fast` to keep the two interfaces the
        same; it never splits a quartet into its two meioses.

    Returns
    -------
    float or numpy.ndarray
        The total log-likelihood, or an array of shape ``(M,)`` with one
        value per quartet if `per_meiosis` is True, where ``M`` is the
        number of quartets.

    Raises
    ------
    ValueError
        As for :func:`cointf.fastlik.loglik_fast`.

    See Also
    --------
    cointf.naive.loglik_quartet_naive : Brute-force :math:`3^n` reference.
    cointf.simintf.simulate_quartets : Simulate quartet data.

    Notes
    -----
    With ``nu = 1`` or ``p = 1`` the data are a rate-2 Poisson process and
    the log-likelihood is :math:`n \log 2 - 2L`.

    Examples
    --------
    >>> from cointf.quartet import loglik_quartet
    >>> data = [[0.2, 0.31, 0.9, 1.12], [0.7], []]
    >>> ll = loglik_quartet(data, 1.5, nu=4.0, p=0.05)
    """
    mz, nu, p = prepare_inputs(data, L, nu, p)
    if mz.M == 0:
        return np.empty(0) if per_meiosis else 0.0

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        ll = quartet_arrays(mz, nu, p, kernel or series_kernel)
    return ll if per_meiosis else float(ll.sum())


def quartet_arrays(mz, nu, p, kernel):
    """Compute per-quartet log-likelihoods for validated inputs.

    Parameters
    ----------
    mz : Meioses
        Prepared crossover data, one entry per quartet.
    nu, p : float or numpy.ndarray
        Validated parameters: floats or length-``mz.M`` arrays.
    kernel : callable
        Kernel; see :func:`cointf.fastlik.series_kernel`.

    Returns
    -------
    numpy.ndarray
        Log-likelihood of each quartet, shape ``(mz.M,)``.
    """
    p_m, poisson, log_q, lf, lS, log_G = renewal_terms(mz, nu, p, kernel)
    log_2p = np.log(2.0 * p_m)
    ll = np.empty(mz.M)
    ll[mz.no_crossover] = 2.0 * log_G[mz.no_crossover]
    for n, idx, iu, ju, eb, gb in mz.groups:
        k = idx.size
        lq = log_q[idx, None]
        first = lq + lS[eb[0] : eb[1]].reshape(k, n)
        last = lS[eb[1] : eb[2]].reshape(k, n)
        gap = np.full((k, n, n), -np.inf)
        if iu.size:
            gap[:, iu, ju] = lq + lf[gb[0] : gb[1]].reshape(k, iu.size)
        lp = log_2p[idx, None, None]
        # alpha[:, a, b]: a (b) = 0 means no interference crossover yet in
        # meiosis A (B); a = i + 1 means the latest one is the i-th position.
        # A and B are exchangeable, so alpha is symmetric in (a, b).
        alpha = np.full((k, n + 1, n + 1), -np.inf)
        alpha[:, 0, 0] = 0.0
        for j in range(n):
            new = alpha + lp  # j-th crossover from an escape pathway
            trans = np.concatenate([first[:, j : j + 1], gap[:, :j, j]], axis=1)
            # Assign the j-th crossover to B: sum over B's previous state.
            to_b = logsumexp_last(alpha[:, : j + 1, : j + 1] + trans[:, None, :])
            new[:, : j + 1, j + 1] = to_b
            new[:, j + 1, : j + 1] = to_b  # assigned to A, by symmetry
            alpha = new
        end = np.concatenate([log_G[idx, None], last], axis=1)
        both = alpha + end[:, :, None] + end[:, None, :]
        ll[idx] = logsumexp_last(both.reshape(k, -1))
    ll -= 2.0 * p_m * mz.L
    if poisson.any():
        ll[poisson] = mz.counts[poisson] * np.log(2.0) - 2.0 * mz.L[poisson]
    return ll
