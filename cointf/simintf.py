"""Simulation of crossover positions under the Housworth-Stahl model.

The interference pathway is a stationary renewal process with gap density
``f*`` rescaled to rate ``1 - p``; the escape pathway is a homogeneous
Poisson process with rate ``p``. Observed crossovers are their superposition.
"""

import numpy as np

__all__ = ["simulate_meioses", "simulate_meiosis", "simulate_quartets"]


size_biased_kmax = 200
size_biased_k = np.arange(1, size_biased_kmax + 1, dtype=np.float64)
# Cumulative distribution of the size-biased geometric P(K = k) = k 2^-(k+1).
size_biased_cmf = np.cumsum(size_biased_k * 0.5 ** (size_biased_k + 1))


def sample_size_biased_k(rng: np.random.Generator, size: int) -> np.ndarray:
    r"""Sample the size-biased geometric distribution by inverse CDF.

    Parameters
    ----------
    rng : numpy.random.Generator
        Random number generator.
    size : int
        Number of draws.

    Returns
    -------
    numpy.ndarray
        Integer draws :math:`K \ge 1` with :math:`P(K = k) = k\, 2^{-(k+1)}`,
        shape ``(size,)``.
    """
    return np.searchsorted(size_biased_cmf, rng.random(size)) + 1


def interference_crossovers(
    L: float,
    nu: float,
    p: float,
    rng: np.random.Generator,
) -> np.ndarray:
    r"""Simulate crossovers from the interference pathway.

    A stationary renewal process with gap density :math:`q f^*(qx; \nu)`,
    :math:`q = 1 - p`, so that the pathway has rate :math:`q` per Morgan.
    Each gap is drawn as :math:`\mathrm{Gamma}(K\nu,\ \mathrm{rate}=2\nu q)`
    with :math:`K` geometric on ``{1, 2, ...}`` with success probability 1/2.

    Parameters
    ----------
    L : float
        Chromosome length in Morgans, ``> 0``.
    nu : float
        Interference parameter, ``> 0``.
    p : float
        Escape fraction, in ``[0, 1]``.
    rng : numpy.random.Generator
        Random number generator.

    Returns
    -------
    numpy.ndarray
        Sorted crossover positions in ``[0, L]``; empty if ``p = 1``.

    Notes
    -----
    Stationarity requires the first position to follow the excess-life
    density :math:`q S^*(qx; \nu)`. It is drawn as :math:`WU` with
    :math:`W \sim \mathrm{Gamma}(K\nu + 1,\ \mathrm{rate}=2\nu q)`,
    :math:`K` size-biased geometric (see :func:`sample_size_biased_k`) and
    :math:`U` uniform on ``[0, 1]``.
    """
    if p >= 1.0:
        return np.empty(0, dtype=np.float64)

    q = 1.0 - p
    scale = 0.5 / (nu * q)  # gamma scale 1 / (2 nu q)

    # First crossover from the excess-life distribution.
    K = int(sample_size_biased_k(rng, 1)[0])
    W = rng.gamma(shape=K * nu + 1.0, scale=scale)
    x = W * rng.random()

    positions = []
    while x <= L:
        positions.append(x)
        k = rng.geometric(p=0.5)
        x += rng.gamma(shape=k * nu, scale=scale)

    return np.array(positions, dtype=np.float64) if positions else np.empty(0)


def escape_crossovers(L: float, p: float, rng: np.random.Generator) -> np.ndarray:
    """Simulate crossovers from the escape pathway.

    A homogeneous Poisson process of rate `p` per Morgan: ``Poisson(p L)``
    crossovers placed uniformly on ``[0, L]``.

    Parameters
    ----------
    L : float
        Chromosome length in Morgans, ``> 0``.
    p : float
        Escape fraction, in ``[0, 1]``.
    rng : numpy.random.Generator
        Random number generator.

    Returns
    -------
    numpy.ndarray
        Sorted crossover positions in ``[0, L]``; empty if ``p = 0``.
    """
    if p <= 0.0:
        return np.empty(0, dtype=np.float64)
    n = rng.poisson(lam=p * L)
    return np.sort(rng.uniform(0.0, L, size=n)) if n > 0 else np.empty(0)


def simulate_meiosis(
    L: float,
    nu: float,
    p: float,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Simulate one meiosis under the Housworth-Stahl model.

    Crossovers are the superposition of the interference pathway
    (rate ``1 - p``) and the escape pathway (rate ``p``).

    Parameters
    ----------
    L : float
        Chromosome length in Morgans, ``> 0``.
    nu : float
        Interference parameter, ``> 0``. ``nu = 1`` is no interference and
        ``nu > 1`` positive interference.
    p : float
        Fraction of crossovers escaping interference, in ``[0, 1]``.
        ``p = 0`` is the pure interference model and ``p = 1`` a Poisson
        process.
    rng : numpy.random.Generator, optional
        Random number generator; a fresh :func:`numpy.random.default_rng`
        if None.

    Returns
    -------
    numpy.ndarray
        Sorted crossover positions in Morgans, in ``[0, L]``; may be empty.

    Raises
    ------
    ValueError
        If `L`, `nu` or `p` is out of range.

    See Also
    --------
    simulate_meioses : Simulate many meioses from one seed.

    Notes
    -----
    The expected number of crossovers is `L` for every ``(nu, p)``. For
    ``nu > 1`` the interference pathway has sub-Poisson counts.

    Examples
    --------
    >>> import numpy as np
    >>> xs = simulate_meiosis(1.5, nu=4.0, p=0.05, rng=np.random.default_rng(1))
    >>> bool(np.all((xs >= 0) & (xs <= 1.5)))
    True
    """
    if L <= 0:
        raise ValueError(f"L must be > 0, got {L!r}")
    if nu <= 0:
        raise ValueError(f"nu must be > 0, got {nu!r}")
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"p must be in [0, 1], got {p!r}")

    if rng is None:
        rng = np.random.default_rng()

    parts = []

    xo_int = interference_crossovers(L, nu, p, rng)
    if len(xo_int):
        parts.append(xo_int)

    xo_esc = escape_crossovers(L, p, rng)
    if len(xo_esc):
        parts.append(xo_esc)

    if not parts:
        return np.empty(0, dtype=np.float64)

    result = np.concatenate(parts) if len(parts) > 1 else parts[0]
    return np.sort(result)


def simulate_meioses(L, nu, p, n, seed):
    """Simulate independent meioses under the Housworth-Stahl model.

    Parameters
    ----------
    L : float
        Chromosome length in Morgans, ``> 0``.
    nu : float
        Interference parameter, ``> 0``.
    p : float
        Fraction of crossovers escaping interference, in ``[0, 1]``.
    n : int
        Number of meioses, ``> 0``.
    seed : int
        Seed for :func:`numpy.random.default_rng`, ``> 0``.

    Returns
    -------
    list of numpy.ndarray
        Sorted crossover positions in Morgans, one array per meiosis.

    Raises
    ------
    ValueError
        If `n` or `seed` is not positive, or a model parameter is out of range.

    See Also
    --------
    simulate_meiosis : Simulate a single meiosis.

    Examples
    --------
    >>> data = simulate_meioses(L=1.5, nu=4.0, p=0.05, n=3, seed=7)
    >>> len(data)
    3
    """
    if n <= 0:
        raise ValueError(f"n must be > 0, got {n!r}")
    if seed <= 0:
        raise ValueError(f"seed must be > 0, got {seed!r}")
    rng = np.random.default_rng(seed=seed)
    return [simulate_meiosis(L=L, nu=nu, p=p, rng=rng) for _ in range(n)]


def simulate_quartets(L, nu, p, n, seed):
    """Simulate phase-unknown (quartet) crossover data.

    Each dataset pools the crossovers of two independent meioses with the
    same parameters, as observed for one parent in a family with two
    offspring when it is unknown which offspring inherited each crossover.

    Parameters
    ----------
    L : float
        Chromosome length in Morgans, ``> 0``.
    nu : float
        Interference parameter, ``> 0``.
    p : float
        Fraction of crossovers escaping interference, in ``[0, 1]``.
    n : int
        Number of quartets, ``> 0``.
    seed : int
        Seed for :func:`numpy.random.default_rng`, ``> 0``.

    Returns
    -------
    list of numpy.ndarray
        Sorted pooled crossover positions in Morgans, one array per quartet.

    Raises
    ------
    ValueError
        If `n` or `seed` is not positive, or a model parameter is out of range.

    See Also
    --------
    cointf.quartet.loglik_quartet : Likelihood of quartet data.

    Examples
    --------
    >>> data = simulate_quartets(L=1.5, nu=4.0, p=0.05, n=3, seed=7)
    >>> len(data)
    3
    """
    if n <= 0:
        raise ValueError(f"n must be > 0, got {n!r}")
    if seed <= 0:
        raise ValueError(f"seed must be > 0, got {seed!r}")
    rng = np.random.default_rng(seed=seed)
    return [
        np.sort(
            np.concatenate(
                [simulate_meiosis(L=L, nu=nu, p=p, rng=rng) for _ in range(2)]
            )
        )
        for _ in range(n)
    ]
