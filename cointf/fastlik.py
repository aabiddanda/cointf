r"""Fast exact likelihood for the Housworth-Stahl gamma-escape model.

Same model and recursion as :mod:`cointf.naive`, organised for speed:

- :math:`f^*` and :math:`S^* = 1 - F^*` come from a 60-term gamma series
  evaluated directly with ``gammaln`` / ``gammaincc``.
- Meioses are grouped by their number of crossovers, and the forward
  recursion over pathway labels runs once per group on a dense array. The
  cost of one evaluation is a few numpy calls per crossover index rather
  than a Python loop over meioses, and memory scales with the meioses
  actually present rather than with the largest one.
- All kernel evaluations needed for one likelihood call are made in a
  single vectorised call, so the kernel can be swapped (e.g. for a
  tabulated approximation) without touching the recursion.

The interference parameter :math:`\nu` and escape fraction :math:`p` may
be scalars (a pooled fit) or arrays with one value per meiosis (e.g. when
they depend on covariates).

Notes
-----
Interference crossovers form a stationary renewal process with gap
density :math:`q f^*(qx; \nu)`, :math:`q = 1 - p`, where

.. math::

    f^*(x; \nu) = \sum_{k \ge 1} 2^{-k}\,
    \mathrm{Gamma}(x;\, k\nu,\ \mathrm{rate}=2\nu).

Escape crossovers form an independent Poisson process of rate :math:`p`
(per Morgan). Observed crossovers are the superposition, with pathway
labels unobserved; the likelihood sums over all labelings with an
:math:`O(n^2)` forward recursion per meiosis with :math:`n` crossovers.
"""

import numpy as np
from scipy.special import gammaincc, gammaln, xlogy

__all__ = [
    "Meioses",
    "log_fstar_fast",
    "log_sstar_fast",
    "log_tail_fast",
    "loglik_fast",
    "series_kernel",
]

n_series_terms = 60
series_k = np.arange(1, n_series_terms + 1, dtype=np.float64)
series_log_weight = -series_k * np.log(2.0)
series_weight = np.exp(series_log_weight)


def logsumexp_last(a):
    """Compute log-sum-exp along the last axis.

    A lean replacement for :func:`scipy.special.logsumexp` on the small,
    repeated reductions in the recursion, where scipy's per-call overhead
    dominates.

    Parameters
    ----------
    a : numpy.ndarray
        Array of log values; ``-inf`` entries are allowed.

    Returns
    -------
    numpy.ndarray
        ``log(sum(exp(a), axis=-1))``, with ``-inf`` where every entry along
        the axis is ``-inf``.
    """
    m = np.max(a, axis=-1, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log(np.sum(np.exp(a - m), axis=-1)) + m[..., 0]


def series_shape_rate(x, nu):
    """Return distances as a column with the gamma shapes and rates of the series.

    Parameters
    ----------
    x : array_like
        Distances in Morgans; flattened to 1-D.
    nu : float or array_like
        Interference parameter: a scalar, or one value per element of `x`.

    Returns
    -------
    x : numpy.ndarray
        Distances, shape ``(n, 1)``.
    a : numpy.ndarray
        Gamma shapes ``k * nu``: shape ``(n_series_terms,)`` for scalar `nu`,
        otherwise ``(n, n_series_terms)``.
    b : float or numpy.ndarray
        Gamma rates ``2 * nu``, broadcastable against `a`.
    """
    x = np.atleast_1d(np.asarray(x, dtype=np.float64))
    nu = np.asarray(nu, dtype=np.float64)
    if nu.ndim != 0:
        nu = np.broadcast_to(nu, x.shape).reshape(-1, 1)
    return x.reshape(-1, 1), series_k * nu, 2.0 * nu


def log_fstar_fast(x, nu):
    r"""Log density of the inter-crossover distance on the chiasma scale.

    Evaluates :math:`\log f^*(x; \nu)` from the first 60 terms of the gamma
    series (the omitted terms carry total weight :math:`2^{-60}`).

    Parameters
    ----------
    x : array_like
        Distances in Morgans, ``>= 0``. Flattened to 1-D.
    nu : float or array_like
        Interference parameter, ``> 0``. A scalar, or an array with the same
        number of elements as `x` (one value per distance).

    Returns
    -------
    numpy.ndarray
        :math:`\log f^*(x; \nu)`, shape ``(x.size,)``. At ``x = 0`` this is
        ``-inf`` for ``nu > 1``, ``0`` for ``nu = 1`` and ``+inf`` for
        ``nu < 1``.

    See Also
    --------
    cointf.naive.log_fstar : Slower 200-term reference implementation.
    """
    x, a, b = series_shape_rate(x, nu)
    with np.errstate(divide="ignore", invalid="ignore"):
        logpdf = xlogy(a - 1.0, x) + a * np.log(b) - b * x - gammaln(a)
    return logsumexp_last(series_log_weight + logpdf)


def log_sstar_fast(x, nu):
    r"""Log survival function of the inter-crossover distance.

    Evaluates :math:`\log S^*(x; \nu) = \log(1 - F^*(x; \nu))` from the first
    60 terms of the gamma series.

    Parameters
    ----------
    x : array_like
        Distances in Morgans, ``>= 0``. Flattened to 1-D.
    nu : float or array_like
        Interference parameter, ``> 0``. A scalar, or an array with the same
        number of elements as `x`.

    Returns
    -------
    numpy.ndarray
        :math:`\log S^*(x; \nu)`, shape ``(x.size,)``.

    See Also
    --------
    cointf.naive.log_sstar : Slower 200-term reference implementation.
    """
    x, a, b = series_shape_rate(x, nu)
    S = np.sum(series_weight * gammaincc(a, b * x), axis=1)
    with np.errstate(divide="ignore"):
        return np.log(S)


def log_tail_fast(x, nu):
    r"""Log integrated survival function of the inter-crossover distance.

    Evaluates :math:`\log G(x; \nu)` with
    :math:`G(x) = \int_x^\infty S^*(u; \nu)\, du`, using the closed form
    :math:`\int_x^\infty S_k(u)\, du = (a/b) Q(a + 1, bx) - x Q(a, bx)` for each
    gamma component (shape :math:`a`, rate :math:`b`, :math:`Q` the
    regularised upper incomplete gamma function). :math:`G(0) = 1`.

    Parameters
    ----------
    x : array_like
        Lengths in Morgans, ``>= 0``. Flattened to 1-D.
    nu : float or array_like
        Interference parameter, ``> 0``. A scalar, or an array with the same
        number of elements as `x`.

    Returns
    -------
    numpy.ndarray
        :math:`\log G(x; \nu)`, shape ``(x.size,)``. ``-inf`` where the series
        underflows to a non-positive value.

    See Also
    --------
    cointf.naive.log_tail_integral : Scalar 200-term reference implementation.
    """
    x, a, b = series_shape_rate(x, nu)
    comp = (a / b) * gammaincc(a + 1.0, b * x) - x * gammaincc(a, b * x)
    G = np.sum(series_weight * comp, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(G > 0.0, np.log(np.maximum(G, 0.0)), -np.inf)


def series_kernel(u, nu):
    r"""Default likelihood kernel: exact :math:`\log f^*` and :math:`\log S^*`.

    Any callable with this signature can be passed as ``kernel`` to
    :func:`loglik_fast`.

    Parameters
    ----------
    u : numpy.ndarray
        1-D array of distances on the interference scale (Morgans times
        :math:`q = 1 - p`), ``>= 0``.
    nu : float or numpy.ndarray
        Interference parameter: a scalar, or an array of the same shape as
        `u` giving the value for each distance.

    Returns
    -------
    log_f : numpy.ndarray
        :math:`\log f^*(u; \nu)`, same shape as `u`.
    log_S : numpy.ndarray
        :math:`\log S^*(u; \nu)`, same shape as `u`.
    """
    return log_fstar_fast(u, nu), log_sstar_fast(u, nu)


class Meioses:
    """Crossover data prepared for repeated likelihood evaluation.

    Validates and sorts the positions, groups meioses by crossover count and
    precomputes every distance the likelihood needs. Build it once and pass
    it to :func:`loglik_fast` for each parameter value.

    Parameters
    ----------
    data : sequence of array_like
        Crossover positions in Morgans, one 1-D sequence per meiosis (a
        meiosis here is one chromosome in one gamete). Positions must be
        finite, lie in ``[0, L]`` and be distinct within a meiosis; order
        does not matter.
    L : float or array_like
        Chromosome length in Morgans, ``> 0``: a scalar shared by all meioses
        or one value per meiosis.

    Attributes
    ----------
    M : int
        Number of meioses.
    L : numpy.ndarray
        Chromosome length per meiosis, shape ``(M,)``.
    counts : numpy.ndarray
        Number of crossovers per meiosis, shape ``(M,)``.
    groups : list of tuple
        One ``(n, idx, iu, ju, bounds)`` entry per distinct crossover count
        ``n >= 1``: the meiosis indices, the upper-triangle pair indices and
        the slice bounds of its first, last and gap distances in `x_raw`.
    x_raw : numpy.ndarray
        Every distance the likelihood needs, before scaling by :math:`q`.
    row : numpy.ndarray
        Meiosis index of each entry of `x_raw`.
    no_crossover : numpy.ndarray
        Indices of meioses without crossovers.

    Raises
    ------
    ValueError
        If `L` has the wrong shape or is not finite and positive, or a
        meiosis has non-finite, out-of-range or duplicated positions.

    Examples
    --------
    >>> from cointf.fastlik import Meioses, loglik_fast
    >>> mz = Meioses([[0.2, 0.9], [], [0.5]], L=1.2)
    >>> mz.M, mz.counts.tolist()
    (3, [2, 0, 1])
    >>> ll = loglik_fast(mz, None, nu=4.0, p=0.05)
    """

    def __init__(self, data, L):
        self.M = len(data)
        L = np.asarray(L, dtype=np.float64)
        if L.ndim == 0:
            L = np.full(self.M, float(L))
        if L.shape != (self.M,):
            raise ValueError(f"L must be a scalar or have length {self.M}")
        if not np.all(np.isfinite(L) & (L > 0)):
            raise ValueError("L must be finite and > 0")
        self.L = L.copy()

        sorted_data = []
        for m, d in enumerate(data):
            d = np.sort(np.asarray(d, dtype=np.float64).ravel())
            if d.size:
                if not np.all(np.isfinite(d)):
                    raise ValueError(f"meiosis {m}: positions must be finite")
                if d[0] < 0.0 or d[-1] > self.L[m]:
                    raise ValueError(
                        f"meiosis {m}: positions must lie in [0, {self.L[m]}]"
                    )
                if np.any(np.diff(d) == 0.0):
                    raise ValueError(f"meiosis {m}: duplicated crossover positions")
            sorted_data.append(d)
        self.counts = np.array([d.size for d in sorted_data], dtype=np.intp)

        # Group meioses by crossover count; record every kernel argument
        # (before scaling by q) and the meiosis it belongs to.
        self.groups = []
        xs, rows = [], []
        offset = 0
        for n in np.unique(self.counts[self.counts > 0]):
            idx = np.flatnonzero(self.counts == n)
            X = np.vstack([sorted_data[m] for m in idx])
            iu, ju = np.triu_indices(n, k=1)
            parts = [X, self.L[idx, None] - X, X[:, ju] - X[:, iu]]
            sizes = [part.size for part in parts]
            xs += [part.ravel() for part in parts]
            rows += [np.repeat(idx, part.shape[1]) for part in parts]
            bounds = offset + np.cumsum([0, *sizes])
            self.groups.append((int(n), idx, iu, ju, bounds))
            offset = bounds[-1]
        self.x_raw = np.concatenate(xs) if xs else np.empty(0)
        self.row = np.concatenate(rows) if rows else np.empty(0, dtype=np.intp)
        self.no_crossover = np.flatnonzero(self.counts == 0)

    def __repr__(self):
        """Return a short summary of the dataset.

        Returns
        -------
        str
            Number of meioses, total crossovers and the largest count.
        """
        return (
            f"Meioses(M={self.M}, crossovers={int(self.counts.sum())}, "
            f"max_count={int(self.counts.max()) if self.M else 0})"
        )


def as_per_meiosis(value, name, M):
    """Coerce a parameter to a float or a length-`M` array.

    Parameters
    ----------
    value : float or array_like
        A scalar, or one value per meiosis.
    name : str
        Parameter name, used in the error message.
    M : int
        Number of meioses.

    Returns
    -------
    float or numpy.ndarray
        A float if `value` is a scalar or an array of identical entries,
        otherwise a float array of shape ``(M,)``.

    Raises
    ------
    ValueError
        If `value` is an array whose shape is not ``(M,)``.
    """
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 0:
        return float(arr)
    if arr.shape != (M,):
        raise ValueError(f"{name} must be a scalar or have length {M}")
    if M and np.all(arr == arr[0]):
        return float(arr[0])
    return arr


def loglik_fast(data, L, nu, p, *, kernel=None, per_meiosis=False):
    r"""Exact log-likelihood of crossover positions under the gamma-escape model.

    Parameters
    ----------
    data : sequence of array_like or Meioses
        Crossover positions in Morgans, one sequence per meiosis, or a
        prepared :class:`Meioses` object (faster for repeated evaluation).
    L : float or array_like or None
        Chromosome length(s) in Morgans: a scalar or one value per meiosis.
        Must be ``None`` when `data` is a :class:`Meioses`, which carries its
        own lengths.
    nu : float or array_like
        Interference parameter, ``> 0``: a scalar shared by all meioses or
        one value per meiosis. ``nu = 1`` is no interference.
    p : float or array_like
        Fraction of crossovers escaping interference, in ``[0, 1]``: a scalar
        or one value per meiosis.
    kernel : callable, optional
        Function ``kernel(u, nu) -> (log_f, log_S)`` returning
        :math:`\log f^*` and :math:`\log S^*` at distances `u` (see
        :func:`series_kernel`, the default). `nu` is passed as a scalar
        when it is shared by all meioses, otherwise as an array matching
        `u`. Use this to plug in a tabulated approximation.
    per_meiosis : bool, default False
        If True, return one log-likelihood per meiosis instead of the sum.

    Returns
    -------
    float or numpy.ndarray
        The total log-likelihood, or an array of shape ``(M,)`` if
        `per_meiosis` is True.

    Raises
    ------
    ValueError
        If parameters are out of range or have the wrong shape, if `L` is
        given together with a :class:`Meioses`, or if the data are invalid
        (see :class:`Meioses`).

    See Also
    --------
    Meioses : Prepare data once for repeated evaluation.
    cointf.naive.loglik_meiosis_naive : Slow single-meiosis reference.

    Notes
    -----
    Passing an array whose entries are all equal is treated as the
    corresponding scalar, so a pooled fit costs the same either way.

    Examples
    --------
    >>> from cointf.fastlik import loglik_fast
    >>> data = [[0.31, 1.12], [0.70], []]
    >>> total = loglik_fast(data, 1.5, nu=4.0, p=0.05)
    >>> per = loglik_fast(data, 1.5, nu=[4.0, 6.0, 2.0], p=0.05, per_meiosis=True)
    >>> per.shape
    (3,)
    """
    if isinstance(data, Meioses):
        if L is not None:
            raise ValueError("L must be None when data is a Meioses object")
        mz = data
    else:
        if L is None:
            raise ValueError("L is required unless data is a Meioses object")
        mz = Meioses(data, L)

    nu = as_per_meiosis(nu, "nu", mz.M)
    p = as_per_meiosis(p, "p", mz.M)
    if not np.all(np.isfinite(nu) & (np.asarray(nu) > 0)):
        raise ValueError("nu must be finite and > 0")
    if not np.all((np.asarray(p) >= 0.0) & (np.asarray(p) <= 1.0)):
        raise ValueError("p must lie in [0, 1]")
    if mz.M == 0:
        return np.empty(0) if per_meiosis else 0.0

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        ll = loglik_arrays(mz, nu, p, kernel or series_kernel)
    return ll if per_meiosis else float(ll.sum())


def loglik_arrays(mz, nu, p, kernel):
    """Compute per-meiosis log-likelihoods for validated inputs.

    The computational core of :func:`loglik_fast`, without argument checks.

    Parameters
    ----------
    mz : Meioses
        Prepared crossover data.
    nu : float or numpy.ndarray
        Interference parameter: a float or a length-``mz.M`` array.
    p : float or numpy.ndarray
        Escape fraction: a float or a length-``mz.M`` array.
    kernel : callable
        Kernel ``kernel(u, nu) -> (log_f, log_S)``; see :func:`series_kernel`.

    Returns
    -------
    numpy.ndarray
        Log-likelihood of each meiosis, shape ``(mz.M,)``.

    Raises
    ------
    ValueError
        If the kernel's outputs are not shaped like its input.
    """
    M = mz.M
    p_m = np.broadcast_to(p, (M,))
    q_m = 1.0 - p_m
    poisson = q_m == 0.0  # p = 1: pure Poisson, handled in closed form below
    q_eff = np.where(poisson, 1.0, q_m)
    log_q = np.log(q_eff)
    log_p = np.log(p_m)

    # Every kernel evaluation in one call.
    scalar_q = np.ndim(p) == 0
    u = (q_eff[0] if scalar_q else q_eff[mz.row]) * mz.x_raw
    nu_u = nu if np.ndim(nu) == 0 else nu[mz.row]
    lf, lS = kernel(u, nu_u)
    lf = np.asarray(lf, dtype=np.float64)
    lS = np.asarray(lS, dtype=np.float64)
    if lf.shape != u.shape or lS.shape != u.shape:
        raise ValueError("kernel must return two arrays shaped like u")

    log_G = log_tail_per_meiosis(q_eff * mz.L, nu)
    ll = np.empty(M)
    ll[mz.no_crossover] = log_G[mz.no_crossover]
    for n, idx, iu, ju, b in mz.groups:
        k = idx.size
        lq = log_q[idx, None]
        first = lq + lS[b[0] : b[1]].reshape(k, n)
        last = lS[b[1] : b[2]].reshape(k, n)
        gap = np.full((k, n, n), -np.inf)
        if iu.size:
            gap[:, iu, ju] = lq + lf[b[2] : b[3]].reshape(k, iu.size)
        lp = log_p[idx, None]
        # alpha[:, s]: s = 0 means no interference crossover yet; s = i + 1
        # means the latest interference crossover is the i-th position.
        alpha = np.full((k, n + 1), -np.inf)
        alpha[:, 0] = 0.0
        for j in range(n):
            new = alpha + lp  # j-th crossover from the escape pathway
            trans = np.concatenate([first[:, j : j + 1], gap[:, :j, j]], axis=1)
            new[:, j + 1] = logsumexp_last(alpha[:, : j + 1] + trans)
            alpha = new
        end = np.concatenate([log_G[idx, None], last], axis=1)
        ll[idx] = logsumexp_last(alpha + end)
    ll -= p_m * mz.L
    if poisson.any():
        ll[poisson] = -mz.L[poisson]  # rate-1 Poisson with log p = 0
    return ll


def log_tail_per_meiosis(x, nu):
    r"""Evaluate :math:`\log G(x_m; \nu_m)` once per distinct pair.

    Parameters
    ----------
    x : numpy.ndarray
        Rescaled chromosome lengths :math:`q_m L_m`, shape ``(M,)``.
    nu : float or numpy.ndarray
        Interference parameter: a float or a length-``M`` array.

    Returns
    -------
    numpy.ndarray
        :math:`\log G` for each meiosis, shape ``(M,)``.
    """
    if np.ndim(nu) == 0:
        ux, inv = np.unique(x, return_inverse=True)
        return log_tail_fast(ux, nu)[inv.reshape(-1)]
    pairs = np.column_stack([x, nu])
    up, inv = np.unique(pairs, axis=0, return_inverse=True)
    return log_tail_fast(up[:, 0], up[:, 1])[inv.reshape(-1)]
