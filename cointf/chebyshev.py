r"""Piecewise Chebyshev kernel for the fast likelihood.

For a fixed :math:`\nu`, :math:`\log f^*(u; \nu)` and :math:`\log S^*(u; \nu)`
are smooth in :math:`u > 0`, so each can be replaced by a piecewise
Chebyshev interpolant built once from the exact series and then evaluated
with a short Clenshaw recurrence. :class:`ChebyshevKernel` has the kernel
signature of :func:`cointf.fastlik.series_kernel`, so it plugs into
:func:`cointf.fastlik.loglik_fast` unchanged.

Notes
-----
- Fits are in log space, so the interpolation error is relative in
  :math:`f^*` and :math:`S^*`. For :math:`f^*` the singular part
  :math:`(\nu - 1)\log u` is subtracted before fitting and added back after.
- Pieces below ``u_split`` use the variable :math:`\log u`, where the
  functions are nearly linear; pieces above use :math:`u`, where
  :math:`f^*` has bumps near :math:`u = 1/2, 1, \dots` for large
  :math:`\nu`. Pieces are halved until the trailing Chebyshev coefficients
  fall below the tolerance.
- Distances outside ``[u_min, u_max]`` fall back to the exact series, so the
  kernel is exact there and approximate inside. ``tol`` is an acceptance
  threshold on trailing coefficients, not a guaranteed error bound; see
  :class:`ChebyshevKernel`.
- One fit is built per distinct :math:`\nu` and cached. When :math:`\nu`
  differs between meioses and there are more than ``max_distinct`` distinct
  values in one call, the kernel falls back to the exact series.
"""

import numpy as np

from .fastlik import log_fstar_fast, log_sstar_fast

__all__ = ["ChebyshevKernel", "PiecewiseChebyshev"]


def chebyshev_nodes(n_nodes):
    r"""Chebyshev points of the first kind and the values-to-coefficients matrix.

    Parameters
    ----------
    n_nodes : int
        Number of nodes, i.e. polynomial degree plus one.

    Returns
    -------
    nodes : numpy.ndarray
        Points :math:`\cos(\pi (j + 1/2) / N)` on ``[-1, 1]``, shape ``(N,)``.
    to_coefs : numpy.ndarray
        Matrix mapping values at `nodes` to Chebyshev coefficients, shape
        ``(N, N)``.
    """
    j = np.arange(n_nodes) + 0.5
    nodes = np.cos(np.pi * j / n_nodes)
    to_coefs = (2.0 / n_nodes) * np.cos(
        np.pi * np.outer(np.arange(n_nodes), j) / n_nodes
    )
    to_coefs[0] *= 0.5
    return nodes, to_coefs


def clenshaw(coefs, x):
    """Evaluate Chebyshev series with one coefficient row per point.

    Parameters
    ----------
    coefs : numpy.ndarray
        Coefficients, shape ``(N, n)``: column ``i`` belongs to point ``i``.
    x : numpy.ndarray
        Points on ``[-1, 1]``, shape ``(n,)``.

    Returns
    -------
    numpy.ndarray
        :math:`\\sum_k c_{k i} T_k(x_i)`, shape ``(n,)``.
    """
    two_x = 2.0 * x
    b1 = np.zeros_like(x)
    b2 = np.zeros_like(x)
    for k in range(coefs.shape[0] - 1, 0, -1):
        b1, b2 = coefs[k] + two_x * b1 - b2, b1
    return coefs[0] + x * b1 - b2


class PiecewiseChebyshev:
    r"""Adaptive piecewise Chebyshev interpolant of a smooth function of u.

    Parameters
    ----------
    fn : callable
        Vectorised function of a 1-D array of ``u > 0``.
    u_min, u_split, u_max : float
        Domain ``[u_min, u_max]``; pieces below `u_split` are fitted in
        :math:`\log u`, pieces above in :math:`u`.
    n_nodes : int
        Nodes per piece (degree plus one).
    tol : float
        Acceptance threshold, not an error bound: a piece is accepted when
        its last three coefficients are below ``tol / 4`` times
        ``max(1, max |fn|)`` on the piece.
    max_pieces : int
        Upper bound on the number of pieces.

    Attributes
    ----------
    u_lo : numpy.ndarray
        Lower end of each piece in u, sorted, shape ``(P,)``.
    center, half : numpy.ndarray
        Midpoint and half-width of each piece in its own variable.
    log_scale : numpy.ndarray
        True for pieces fitted in :math:`\log u`.
    coefs : numpy.ndarray
        Coefficients, shape ``(n_nodes, P)``.
    n_evaluations : int
        Calls of `fn` made while building, counted per point.

    Raises
    ------
    RuntimeError
        If the tolerance is not reached within `max_pieces` pieces.
    """

    def __init__(
        self, fn, u_min, u_split, u_max, n_nodes=16, tol=1e-12, max_pieces=4096
    ):
        nodes, to_coefs = chebyshev_nodes(n_nodes)
        # Pending intervals as (lo, hi, log_scale) in the fitting variable.
        pending = [(np.log(u_min), np.log(u_split), True), (u_split, u_max, False)]
        done = []
        self.n_evaluations = 0
        while pending:
            lo = np.array([a for a, _, _ in pending])
            hi = np.array([b for _, b, _ in pending])
            logs = np.array([s for _, _, s in pending])
            mid, half = 0.5 * (lo + hi), 0.5 * (hi - lo)
            s = mid[:, None] + half[:, None] * nodes[None, :]
            u = np.where(logs[:, None], np.exp(s), s)
            values = np.asarray(fn(u.ravel()), dtype=np.float64).reshape(u.shape)
            self.n_evaluations += values.size
            c = values @ to_coefs.T
            scale = np.maximum(1.0, np.max(np.abs(values), axis=1))
            ok = np.max(np.abs(c[:, -3:]), axis=1) <= 0.25 * tol * scale
            ok &= np.all(np.isfinite(values), axis=1)
            nxt = []
            for i, item in enumerate(pending):
                if ok[i]:
                    done.append((item, c[i]))
                else:
                    nxt += [(lo[i], mid[i], logs[i]), (mid[i], hi[i], logs[i])]
            if len(done) + len(nxt) > max_pieces:
                raise RuntimeError("Chebyshev fit did not converge within max_pieces")
            pending = nxt

        lo_u = np.array([np.exp(a) if s else a for (a, _, s), _ in done])
        order = np.argsort(lo_u)
        self.u_min, self.u_max = float(u_min), float(u_max)
        self.u_lo = lo_u[order]
        bounds = np.array([(a, b) for (a, b, _), _ in done])[order]
        self.center = bounds.mean(axis=1)
        self.half = 0.5 * (bounds[:, 1] - bounds[:, 0])
        self.log_scale = np.array([s for (_, _, s), _ in done])[order]
        self.coefs = np.array([c for _, c in done])[order].T.copy()

    @property
    def n_pieces(self):
        """Number of pieces."""
        return self.u_lo.size

    def __call__(self, u):
        """Evaluate the interpolant at points inside ``[u_min, u_max]``.

        Parameters
        ----------
        u : numpy.ndarray
            1-D array of points in ``[u_min, u_max]``.

        Returns
        -------
        numpy.ndarray
            Interpolated values, shape ``(u.size,)``.
        """
        # exp(log(u_min)) can round above u_min, so clip the first piece.
        piece = np.maximum(np.searchsorted(self.u_lo, u, side="right") - 1, 0)
        s = np.where(self.log_scale[piece], np.log(u), u)
        x = np.clip((s - self.center[piece]) / self.half[piece], -1.0, 1.0)
        return clenshaw(self.coefs[:, piece], x)


class ChebyshevKernel:
    r"""Likelihood kernel with piecewise Chebyshev fits of log f* and log S*.

    Parameters
    ----------
    tol : float, default 1e-12
        Acceptance threshold for each piece: its last three Chebyshev
        coefficients must be below ``tol / 4`` times the largest absolute
        value being fitted on the piece (at least 1). This is a heuristic,
        not a bound. With the default, measured errors in :math:`\log f^*`
        and :math:`\log S^*` are about 1e-14 for :math:`\nu \le 4` and grow
        to about 3e-12 at :math:`\nu = 100`, where the fitted values are
        largest.
    u_min, u_max : float, default 1e-6 and 5.0
        Distances (on the interference scale) inside this range use the fit;
        the rest use the exact series.
    u_split : float, default 0.25
        Boundary between pieces fitted in :math:`\log u` and in :math:`u`.
    n_nodes : int, default 16
        Nodes per piece.
    max_distinct : int, default 16
        Largest number of distinct :math:`\nu` values in one call for which
        fits are built; above it the call uses the exact series.
    max_cached : int, default 256
        Fits kept in the cache before it is cleared.

    Examples
    --------
    >>> from cointf.chebyshev import ChebyshevKernel
    >>> from cointf.fastlik import loglik_fast
    >>> data = [[0.31, 1.12], [0.70], []]
    >>> kernel = ChebyshevKernel()
    >>> exact = loglik_fast(data, 1.5, nu=4.0, p=0.05)
    >>> approx = loglik_fast(data, 1.5, nu=4.0, p=0.05, kernel=kernel)
    >>> bool(abs(exact - approx) < 1e-10)
    True
    """

    def __init__(
        self,
        tol=1e-12,
        u_min=1e-6,
        u_max=5.0,
        u_split=0.25,
        n_nodes=16,
        max_distinct=16,
        max_cached=256,
    ):
        if not 0.0 < u_min < u_split < u_max:
            raise ValueError("need 0 < u_min < u_split < u_max")
        self.tol, self.n_nodes = tol, n_nodes
        self.u_min, self.u_split, self.u_max = u_min, u_split, u_max
        self.max_distinct, self.max_cached = max_distinct, max_cached
        self.cache = {}

    def fits(self, nu):
        """Return the cached fits for one value of nu, building them if needed.

        Parameters
        ----------
        nu : float
            Interference parameter, ``> 0``.

        Returns
        -------
        fit_f : PiecewiseChebyshev
            Fit of :math:`\\log f^*(u) - (\\nu - 1) \\log u`.
        fit_s : PiecewiseChebyshev
            Fit of :math:`\\log S^*(u)`.
        """
        nu = float(nu)
        if nu not in self.cache:
            if len(self.cache) >= self.max_cached:
                self.cache.clear()
            opts = dict(
                u_min=self.u_min,
                u_split=self.u_split,
                u_max=self.u_max,
                n_nodes=self.n_nodes,
                tol=self.tol,
            )
            regular_f = lambda u: log_fstar_fast(u, nu) - (nu - 1.0) * np.log(u)  # noqa: E731
            self.cache[nu] = (
                PiecewiseChebyshev(regular_f, **opts),
                PiecewiseChebyshev(lambda u: log_sstar_fast(u, nu), **opts),
            )
        return self.cache[nu]

    def evaluate(self, u, nu, which):
        r"""Evaluate log f* or log S* at distances u.

        Parameters
        ----------
        u : array_like
            Distances, ``>= 0``. Flattened to 1-D.
        nu : float or array_like
            A scalar, or one value per element of `u`.
        which : {"f", "S"}
            :math:`\log f^*` or :math:`\log S^*`.

        Returns
        -------
        numpy.ndarray
            Values, shape ``(u.size,)``.
        """
        u = np.atleast_1d(np.asarray(u, dtype=np.float64)).ravel()
        exact = log_fstar_fast if which == "f" else log_sstar_fast
        if np.ndim(nu) == 0:
            return self.evaluate_shared(u, float(nu), which, exact)
        nu = np.asarray(nu, dtype=np.float64).ravel()
        distinct, inverse = np.unique(nu, return_inverse=True)
        if distinct.size > self.max_distinct:
            return exact(u, nu)
        out = np.empty(u.size)
        for i, v in enumerate(distinct):
            sel = inverse == i
            out[sel] = self.evaluate_shared(u[sel], float(v), which, exact)
        return out

    def evaluate_shared(self, u, nu, which, exact):
        """Evaluate at distances `u` for one scalar `nu`.

        Parameters
        ----------
        u : numpy.ndarray
            1-D distances.
        nu : float
            Interference parameter.
        which : {"f", "S"}
            Function to evaluate.
        exact : callable
            Exact series used outside the fitted range.

        Returns
        -------
        numpy.ndarray
            Values, shape ``(u.size,)``.
        """
        inside = (u >= self.u_min) & (u <= self.u_max)
        if not inside.any():
            return exact(u, nu)
        fit_f, fit_s = self.fits(nu)
        out = np.empty(u.size)
        ui = u[inside]
        if which == "f":
            out[inside] = fit_f(ui) + (nu - 1.0) * np.log(ui)
        else:
            out[inside] = fit_s(ui)
        if not inside.all():
            out[~inside] = exact(u[~inside], nu)
        return out

    def log_fstar(self, u, nu):
        r"""Approximate :math:`\log f^*(u; \nu)`; see :func:`cointf.fastlik.log_fstar_fast`."""
        return self.evaluate(u, nu, "f")

    def log_sstar(self, u, nu):
        r"""Approximate :math:`\log S^*(u; \nu)`; see :func:`cointf.fastlik.log_sstar_fast`."""
        return self.evaluate(u, nu, "S")

    def __call__(self, u_gap, nu_gap, u_end, nu_end):
        """Kernel interface of :func:`cointf.fastlik.series_kernel`.

        Parameters
        ----------
        u_gap, nu_gap, u_end, nu_end
            As for :func:`cointf.fastlik.series_kernel`.

        Returns
        -------
        log_f : numpy.ndarray
            :math:`\\log f^*` at `u_gap`.
        log_S : numpy.ndarray
            :math:`\\log S^*` at `u_end`.
        """
        return self.log_fstar(u_gap, nu_gap), self.log_sstar(u_end, nu_end)
