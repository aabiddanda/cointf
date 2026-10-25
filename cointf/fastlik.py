r"""Fast exact likelihood for the Housworth-Stahl model.

Same model and recursion as :mod:`cointf.naive`, organised for speed:

- :math:`f^*`, :math:`S^* = 1 - F^*` and the integrated tail :math:`G` are
  gamma series. For each distance only the terms that matter are evaluated:
  an index window is found by bisection with explicit error bounds, and the
  terms above it are summed in closed form. Typically about 10 of the
  terms need an incomplete gamma or log-gamma evaluation.
- The recursion needs :math:`\log f^*` only at gaps between crossovers and
  :math:`\log S^*` only at the distances to the chromosome ends, so each is
  evaluated only where it is used.
- Meioses are grouped by crossover count and the forward recursion runs
  once per group on a dense array, so cost is a few numpy calls per
  crossover index and memory scales with the meioses present.
- All kernel evaluations needed for one likelihood call are made in a
  single vectorised call, so the kernel can be swapped without touching
  the recursion.

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

Truncation. With :math:`z = 2\nu x` and shape :math:`a = k\nu`, the series
terms are bounded with the Chernoff exponent
:math:`e(a, z) = z - a + a \log(a / z)`: :math:`Q(a, z) \le e^{-e}` for
:math:`a < z` and :math:`1 - Q(a, z) \le e^{-e}` for :math:`a > z`, where
:math:`Q` is the regularised upper incomplete gamma function. Terms are
dropped, or replaced by their limit :math:`Q = 1`, only where the total
error is below :math:`e^{-40}` relative to a lower bound on the result. The
log terms of :math:`f^*` are concave in :math:`k`, so its window is the set
of terms within a fixed margin of the largest one. Series indices are
capped at ``max_series_terms``.
"""

import numpy as np
from scipy.special import gammaincc, gammaln

__all__ = [
    "Meioses",
    "log_fstar_fast",
    "log_sstar_fast",
    "log_tail_fast",
    "loglik_fast",
    "series_kernel",
]

max_series_terms = 200
truncation_margin = 40.0
bisection_steps = int(np.ceil(np.log2(max_series_terms + 2)))
log2 = np.log(2.0)


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


def gamma_exponent(a, z):
    r"""Chernoff exponent of a unit-rate gamma distribution.

    Parameters
    ----------
    a : numpy.ndarray
        Shapes, ``> 0``.
    z : numpy.ndarray
        Points, ``> 0``.

    Returns
    -------
    numpy.ndarray
        :math:`e(a, z) = z - a + a \log(a / z) \ge 0`. The upper tail
        :math:`Q(a, z)` is at most :math:`e^{-e}` when :math:`a < z`, and the
        lower tail :math:`1 - Q(a, z)` is at most :math:`e^{-e}` when
        :math:`a > z`.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        return z - a + a * np.log(a / z)


def first_true(pred, lo, hi):
    """Find the smallest integer in ``[lo, hi]`` where a predicate holds.

    Element-wise bisection for a predicate that is monotone (False, then
    True) on each interval.

    Parameters
    ----------
    pred : callable
        Maps an array of integer-valued floats to a boolean array.
    lo, hi : numpy.ndarray
        Inclusive bounds, as floats with integer values.

    Returns
    -------
    numpy.ndarray
        The first index where `pred` is True, or `hi` where it is never True
        below `hi`.
    """
    lo = np.array(lo, dtype=np.float64)
    hi = np.array(hi, dtype=np.float64)
    for _ in range(bisection_steps):
        mid = np.floor(0.5 * (lo + hi))
        ok = pred(mid)
        hi = np.where(ok, mid, hi)
        lo = np.where(ok, lo, mid + 1.0)
    return hi


def ragged_indices(klo, khi):
    """Flatten inclusive index windows into (row, index) pairs.

    Parameters
    ----------
    klo, khi : numpy.ndarray
        First and last series index for each element; empty where
        ``khi < klo``.

    Returns
    -------
    row : numpy.ndarray
        Element each pair belongs to.
    k : numpy.ndarray
        Series index of each pair, as floats.
    """
    width = np.maximum(khi - klo + 1, 0).astype(np.intp)
    row = np.repeat(np.arange(klo.size), width)
    start = np.cumsum(width) - width
    k = np.arange(row.size) - (start - klo.astype(np.intp))[row]
    return row, k.astype(np.float64)


class SeriesArguments:
    r"""Distances and parameters prepared for windowed series evaluation.

    Parameters
    ----------
    x : array_like
        Distances in Morgans, ``>= 0``. Flattened to 1-D.
    nu : float or array_like
        Interference parameter, ``> 0``: a scalar or one value per element
        of `x`.

    Attributes
    ----------
    x : numpy.ndarray
        Distances, shape ``(n,)``.
    nu : numpy.ndarray
        Interference parameter for each distance, shape ``(n,)``.
    z : numpy.ndarray
        Distances on the unit-rate scale, :math:`2 \nu x`.
    positive : numpy.ndarray
        Boolean mask of distances ``> 0``; the series functions handle
        ``x = 0`` separately.
    log_gamma_table : numpy.ndarray or None
        :math:`\log\Gamma(k\nu)` for ``k = 1, ..., max_series_terms + 1``
        when `nu` is a scalar, otherwise None.
    """

    def __init__(self, x, nu):
        self.x = np.atleast_1d(np.asarray(x, dtype=np.float64)).ravel()
        scalar = np.ndim(nu) == 0
        nu = np.asarray(nu, dtype=np.float64)
        self.nu = np.broadcast_to(nu, self.x.shape).ravel()
        self.z = 2.0 * self.nu * self.x
        self.positive = self.x > 0.0
        self.log_gamma_table = None
        if scalar:
            k = np.arange(1, max_series_terms + 2)
            self.log_gamma_table = gammaln(k * float(nu))

    def subset(self, mask):
        """Return the arguments restricted to `mask`.

        Parameters
        ----------
        mask : numpy.ndarray
            Boolean mask over the distances.

        Returns
        -------
        SeriesArguments
            A view on the selected distances sharing the log-gamma table.
        """
        out = object.__new__(SeriesArguments)
        out.x, out.nu, out.z = self.x[mask], self.nu[mask], self.z[mask]
        out.positive = self.positive[mask]
        out.log_gamma_table = self.log_gamma_table
        return out

    def log_gamma_shape(self, k, nu):
        r"""Return :math:`\log\Gamma(k\nu)`, from the table when available.

        Parameters
        ----------
        k : numpy.ndarray
            Series indices (integer-valued floats, at most
            ``max_series_terms + 1``).
        nu : numpy.ndarray
            Interference parameter matching `k`.

        Returns
        -------
        numpy.ndarray
            :math:`\log\Gamma(k\nu)`.
        """
        if self.log_gamma_table is None:
            return gammaln(k * nu)
        return self.log_gamma_table[k.astype(np.intp) - 1]

    def first_median_index(self):
        r"""Return the first index whose term has :math:`Q(k\nu, z) \ge 1/2`.

        Uses that a gamma median exceeds ``a - 1/3`` for shapes ``a >= 1``.

        Returns
        -------
        numpy.ndarray
            Index :math:`k_0 = \lceil \max(z + 1/3, 1) / \nu \rceil`, capped at
            ``max_series_terms``.
        """
        k0 = np.ceil(np.maximum(self.z + 1.0 / 3.0, 1.0) / self.nu)
        return np.clip(k0, 1.0, max_series_terms)


def fstar_log_term(args, k):
    r"""Log of the k-th term of the :math:`f^*` series.

    Parameters
    ----------
    args : SeriesArguments
        Prepared distances; all must be positive.
    k : numpy.ndarray
        Series index for each distance.

    Returns
    -------
    numpy.ndarray
        :math:`-k \log 2 + \log \mathrm{Gamma}(x;\, k\nu,\ 2\nu)`.
    """
    a = k * args.nu
    return (
        -k * log2
        + np.log(2.0 * args.nu)
        + (a - 1.0) * np.log(args.z)
        - args.z
        - args.log_gamma_shape(k, args.nu)
    )


def log_fstar_windowed(args):
    r"""Evaluate log f* at positive distances over the window of significant terms.

    Parameters
    ----------
    args : SeriesArguments
        Prepared distances; all must be positive.

    Returns
    -------
    numpy.ndarray
        :math:`\log f^*` for each distance.
    """
    n = args.x.size
    ones = np.ones(n)
    kmax = np.full(n, float(max_series_terms))

    term = lambda k: fstar_log_term(args, k)  # noqa: E731

    # The log terms are concave in k: find the largest, then the indices
    # within a margin of it. Dropped terms total < exp(-margin) * f*.
    mode = first_true(lambda k: term(k + 1.0) < term(k), ones, kmax)
    peak = term(mode)
    floor = peak - truncation_margin - np.log(max_series_terms)
    klo = first_true(lambda k: term(k) >= floor, ones, mode)
    khi = first_true(lambda k: term(k) < floor, mode, kmax + 1.0) - 1.0
    row, k = ragged_indices(klo, khi)
    sub = args.subset(row)
    total = np.bincount(
        row, weights=np.exp(fstar_log_term(sub, k) - peak[row]), minlength=n
    )
    return np.log(total) + peak


def keep_up_to(j, nu, z, threshold, shape_offset):
    r"""Check whether the series terms with index up to `j` must be kept.

    Dropping every index ``<= j`` costs at most :math:`Q(j\nu + c, z)` with
    ``c = shape_offset``, by the Chernoff bound when :math:`j\nu + c < z`.

    Parameters
    ----------
    j : numpy.ndarray
        Candidate last dropped index (``0`` drops nothing).
    nu, z : numpy.ndarray
        Interference parameter and unit-rate distance for each element.
    threshold : numpy.ndarray
        Log of the largest allowed error.
    shape_offset : float
        ``0`` for :math:`S^*`, ``1`` for :math:`G`.

    Returns
    -------
    numpy.ndarray
        True where the terms up to `j` cannot all be dropped.
    """
    a = j * nu + shape_offset
    droppable = (j == 0) | ((a < z) & (-gamma_exponent(a, z) <= threshold))
    return ~droppable


def sstar_window(args):
    """Index window for the S* series with error below the truncation margin.

    Parameters
    ----------
    args : SeriesArguments
        Prepared distances; all must be positive.

    Returns
    -------
    klo : numpy.ndarray
        First evaluated index; lower indices are dropped.
    khi : numpy.ndarray
        Last evaluated index; higher indices are taken as ``Q = 1``.
    """
    nu, z = args.nu, args.z
    k0 = args.first_median_index()
    threshold = -(k0 + 1.0) * log2 - truncation_margin  # S* >= 2^-(k0+1)

    # Taking Q = 1 for every index >= k costs at most 2^(1-k) e^-e(k nu, z).
    upper_ok = lambda k: (1.0 - k) * log2 - gamma_exponent(k * nu, z) <= threshold  # noqa: E731
    keep = lambda j: keep_up_to(j, nu, z, threshold, 0.0)  # noqa: E731
    khi = first_true(upper_ok, k0, np.full_like(k0, max_series_terms + 1.0)) - 1.0
    klo = first_true(keep, np.zeros_like(k0), k0)
    return klo, khi


def log_sstar_windowed(args):
    r"""Evaluate log S* at positive distances over the window of significant terms.

    Parameters
    ----------
    args : SeriesArguments
        Prepared distances; all must be positive.

    Returns
    -------
    numpy.ndarray
        :math:`\log S^*` for each distance.
    """
    klo, khi = sstar_window(args)
    row, k = ragged_indices(klo, khi)
    q = gammaincc(k * args.nu[row], args.z[row])
    total = np.bincount(row, weights=np.exp2(-k) * q, minlength=args.x.size)
    total = total + np.exp2(-khi)  # sum over k > khi of 2^-k
    return np.log(total)


def tail_window(args):
    """Index window for the G series with error below the truncation margin.

    Parameters
    ----------
    args : SeriesArguments
        Prepared distances; all must be positive.

    Returns
    -------
    klo : numpy.ndarray
        First evaluated index; lower indices are dropped.
    khi : numpy.ndarray
        Last evaluated index; higher indices are taken as ``Q = 1``.
    """
    nu, z = args.nu, args.z
    k0 = args.first_median_index()
    # G >= 2^-k0 (k0 nu - z) / (2 nu) >= 2^-k0 / (6 nu).
    threshold = -k0 * log2 - np.log(6.0 * nu) - truncation_margin

    # Taking Q = 1 for every index >= k costs at most
    # (k + 1) 2^(1-k) e^-e(k nu, z).
    upper_ok = lambda k: (  # noqa: E731
        np.log(k + 1.0) + (1.0 - k) * log2 - gamma_exponent(k * nu, z) <= threshold
    )
    keep = lambda j: keep_up_to(j, nu, z, threshold, 1.0)  # noqa: E731
    khi = first_true(upper_ok, k0, np.full_like(k0, max_series_terms + 1.0)) - 1.0
    klo = first_true(keep, np.zeros_like(k0), k0)
    return klo, khi


def log_tail_windowed(args):
    r"""Evaluate log G at positive distances over the window of significant terms.

    Parameters
    ----------
    args : SeriesArguments
        Prepared distances; all must be positive.

    Returns
    -------
    numpy.ndarray
        :math:`\log G` for each distance; ``-inf`` where the sum is not
        positive.
    """
    klo, khi = tail_window(args)
    row, k = ragged_indices(klo, khi)
    nu, z, x = args.nu[row], args.z[row], args.x[row]
    a = k * nu
    comp = 0.5 * k * gammaincc(a + 1.0, z) - x * gammaincc(a, z)
    total = np.bincount(row, weights=np.exp2(-k) * comp, minlength=args.x.size)
    # sum over k > khi of 2^-k (k / 2 - x), the Q = 1 limit of each term.
    total = total + np.exp2(-khi) * (0.5 * (khi + 2.0) - args.x)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(total > 0.0, np.log(np.maximum(total, 0.0)), -np.inf)


def evaluate_series(x, nu, windowed, at_zero):
    """Apply a windowed series to positive distances and a limit at zero.

    Parameters
    ----------
    x : array_like
        Distances in Morgans, ``>= 0``.
    nu : float or array_like
        Interference parameter: a scalar or one value per element of `x`.
    windowed : callable
        Maps a :class:`SeriesArguments` of positive distances to log values.
    at_zero : callable
        Maps the interference parameter at zero distances to log values.

    Returns
    -------
    numpy.ndarray
        Log values, shape ``(x.size,)``.
    """
    args = SeriesArguments(x, nu)
    out = np.empty(args.x.size)
    pos = args.positive
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        if pos.all():
            out[:] = windowed(args)
        else:
            if pos.any():
                out[pos] = windowed(args.subset(pos))
            out[~pos] = at_zero(args.nu[~pos])
    return out


def log_fstar_zero(nu):
    r"""Return log f*(0): -inf for nu > 1, 0 for nu = 1, +inf for nu < 1.

    Parameters
    ----------
    nu : numpy.ndarray
        Interference parameter.

    Returns
    -------
    numpy.ndarray
        :math:`\log f^*(0; \nu)`.
    """
    return np.where(nu > 1.0, -np.inf, np.where(nu < 1.0, np.inf, 0.0))


def log_fstar_fast(x, nu):
    r"""Log density of the unit-rate inter-crossover distance.

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

    Examples
    --------
    >>> from cointf.fastlik import log_fstar_fast
    >>> log_fstar_fast([0.5, 1.0], nu=1.0)  # nu = 1 is Exp(1)
    array([-0.5, -1. ])
    """
    return evaluate_series(x, nu, log_fstar_windowed, log_fstar_zero)


def log_sstar_fast(x, nu):
    r"""Log survival function of the inter-crossover distance.

    Evaluates :math:`\log S^*(x; \nu) = \log(1 - F^*(x; \nu))`.

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
        :math:`\log S^*(x; \nu)`, shape ``(x.size,)``; ``0`` at ``x = 0``.

    See Also
    --------
    cointf.naive.log_sstar : Slower 200-term reference implementation.
    """
    return evaluate_series(x, nu, log_sstar_windowed, np.zeros_like)


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
    return evaluate_series(x, nu, log_tail_windowed, np.zeros_like)


def series_kernel(u_gap, nu_gap, u_end, nu_end):
    r"""Default likelihood kernel: exact :math:`\log f^*` and :math:`\log S^*`.

    The recursion needs :math:`f^*` only at gaps between interference
    crossovers and :math:`S^*` only at distances to the chromosome ends, so
    a kernel receives the two sets separately. Any callable with this
    signature can be passed as ``kernel`` to :func:`loglik_fast`.

    Parameters
    ----------
    u_gap : numpy.ndarray
        1-D array of gaps between crossovers on the interference scale
        (Morgans times :math:`q = 1 - p`), ``>= 0``.
    nu_gap : float or numpy.ndarray
        Interference parameter: a scalar, or an array shaped like `u_gap`.
    u_end : numpy.ndarray
        1-D array of distances from a crossover to a chromosome end, on the
        interference scale, ``>= 0``.
    nu_end : float or numpy.ndarray
        Interference parameter: a scalar, or an array shaped like `u_end`.

    Returns
    -------
    log_f : numpy.ndarray
        :math:`\log f^*(u; \nu)` at `u_gap`.
    log_S : numpy.ndarray
        :math:`\log S^*(u; \nu)` at `u_end`.
    """
    return log_fstar_fast(u_gap, nu_gap), log_sstar_fast(u_end, nu_end)


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
        One ``(n, idx, iu, ju, end_bounds, gap_bounds)`` per crossover count
        ``n >= 1``: meiosis indices, upper-triangle pair indices, and slice
        bounds into `x_end` and `x_gap`.
    x_end : numpy.ndarray
        Distances from each crossover to the chromosome start and end, where
        :math:`S^*` is needed, before scaling by :math:`q`.
    row_end : numpy.ndarray
        Meiosis index of each entry of `x_end`.
    x_gap : numpy.ndarray
        Distances between pairs of crossovers, where :math:`f^*` is needed,
        before scaling by :math:`q`.
    row_gap : numpy.ndarray
        Meiosis index of each entry of `x_gap`.
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
        # (before scaling by q) and the meiosis it belongs to, keeping the
        # end distances (for S*) apart from the gaps (for f*).
        self.groups = []
        ends, end_rows, gaps, gap_rows = [], [], [], []
        end_offset = gap_offset = 0
        for n in np.unique(self.counts[self.counts > 0]):
            idx = np.flatnonzero(self.counts == n)
            X = np.vstack([sorted_data[m] for m in idx])
            iu, ju = np.triu_indices(n, k=1)
            ends += [X.ravel(), (self.L[idx, None] - X).ravel()]
            end_rows += [np.repeat(idx, n)] * 2
            gaps.append((X[:, ju] - X[:, iu]).ravel())
            gap_rows.append(np.repeat(idx, iu.size))
            end_bounds = end_offset + np.array([0, X.size, 2 * X.size])
            gap_bounds = gap_offset + np.array([0, gaps[-1].size])
            self.groups.append((int(n), idx, iu, ju, end_bounds, gap_bounds))
            end_offset, gap_offset = end_bounds[-1], gap_bounds[-1]

        self.x_end = np.concatenate(ends) if ends else np.empty(0)
        self.row_end = np.concatenate(end_rows) if ends else np.empty(0, np.intp)
        self.x_gap = np.concatenate(gaps) if gaps else np.empty(0)
        self.row_gap = np.concatenate(gap_rows) if gaps else np.empty(0, np.intp)
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
    r"""Exact log-likelihood of crossover positions under the Housworth-Stahl model.

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
        Function ``kernel(u_gap, nu_gap, u_end, nu_end) -> (log_f, log_S)``
        returning :math:`\log f^*` at the gaps `u_gap` and :math:`\log S^*`
        at the end distances `u_end` (see :func:`series_kernel`, the
        default). The `nu` arguments are scalars when `nu` is shared by all
        meioses, otherwise arrays matching their distances.
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
    mz, nu, p = prepare_inputs(data, L, nu, p)
    if mz.M == 0:
        return np.empty(0) if per_meiosis else 0.0

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        ll = loglik_arrays(mz, nu, p, kernel or series_kernel)
    return ll if per_meiosis else float(ll.sum())


def prepare_inputs(data, L, nu, p):
    """Validate likelihood arguments and build the :class:`Meioses` if needed.

    Parameters
    ----------
    data, L, nu, p
        As for :func:`loglik_fast`.

    Returns
    -------
    mz : Meioses
        Prepared crossover data.
    nu, p : float or numpy.ndarray
        Parameters as floats or length-``mz.M`` arrays.

    Raises
    ------
    ValueError
        As for :func:`loglik_fast`.
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
    return mz, nu, p


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
        Kernel ``kernel(u_gap, nu_gap, u_end, nu_end) -> (log_f, log_S)``;
        see :func:`series_kernel`.

    Returns
    -------
    numpy.ndarray
        Log-likelihood of each meiosis, shape ``(mz.M,)``.

    Raises
    ------
    ValueError
        If the kernel's outputs are not shaped like its inputs.
    """
    p_m, poisson, log_q, lf, lS, log_G = renewal_terms(mz, nu, p, kernel)
    log_p = np.log(p_m)
    ll = np.empty(mz.M)
    ll[mz.no_crossover] = log_G[mz.no_crossover]
    for n, idx, iu, ju, eb, gb in mz.groups:
        k = idx.size
        lq = log_q[idx, None]
        first = lq + lS[eb[0] : eb[1]].reshape(k, n)
        last = lS[eb[1] : eb[2]].reshape(k, n)
        gap = np.full((k, n, n), -np.inf)
        if iu.size:
            gap[:, iu, ju] = lq + lf[gb[0] : gb[1]].reshape(k, iu.size)
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


def renewal_terms(mz, nu, p, kernel):
    """Evaluate every interference-pathway term the recursion needs.

    Shared by the single-meiosis and quartet recursions.

    Parameters
    ----------
    mz : Meioses
        Prepared crossover data.
    nu, p : float or numpy.ndarray
        Validated parameters: floats or length-``mz.M`` arrays.
    kernel : callable
        Kernel ``kernel(u_gap, nu_gap, u_end, nu_end) -> (log_f, log_S)``.

    Returns
    -------
    p_m : numpy.ndarray
        Escape fraction per meiosis, shape ``(M,)``.
    poisson : numpy.ndarray
        Meioses with ``p = 1``, which the caller handles in closed form; their
        other terms are computed with ``q = 1`` and are not meaningful.
    log_q : numpy.ndarray
        :math:`\\log q` per meiosis.
    lf : numpy.ndarray
        :math:`\\log f^*` at ``q * mz.x_gap``.
    lS : numpy.ndarray
        :math:`\\log S^*` at ``q * mz.x_end``.
    log_G : numpy.ndarray
        :math:`\\log G(q L)` per meiosis.

    Raises
    ------
    ValueError
        If the kernel's outputs are not shaped like its inputs.
    """
    p_m = np.broadcast_to(p, (mz.M,))
    q_m = 1.0 - p_m
    poisson = q_m == 0.0
    q_eff = np.where(poisson, 1.0, q_m)

    # Every kernel evaluation in one call: f* at gaps, S* at end distances.
    scalar_q, scalar_nu = np.ndim(p) == 0, np.ndim(nu) == 0
    u_gap = (q_eff[0] if scalar_q else q_eff[mz.row_gap]) * mz.x_gap
    u_end = (q_eff[0] if scalar_q else q_eff[mz.row_end]) * mz.x_end
    nu_gap = nu if scalar_nu else nu[mz.row_gap]
    nu_end = nu if scalar_nu else nu[mz.row_end]
    lf, lS = kernel(u_gap, nu_gap, u_end, nu_end)
    lf = np.asarray(lf, dtype=np.float64)
    lS = np.asarray(lS, dtype=np.float64)
    if lf.shape != u_gap.shape or lS.shape != u_end.shape:
        raise ValueError("kernel must return arrays shaped like u_gap and u_end")

    log_G = log_tail_per_meiosis(q_eff * mz.L, nu)
    return p_m, poisson, np.log(q_eff), lf, lS, log_G


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
