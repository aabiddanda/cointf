import numpy as np


_KMAX = 200
_ks = np.arange(1, _KMAX + 1, dtype=np.float64)
_CMF = np.cumsum(_ks * 0.5 ** (_ks + 1))  # cumulative mass function, sum -> 1


def _sample_K_batch(rng: np.random.Generator, size: int) -> np.ndarray:
    """Batch sample of K ~ P(K=k) = k*(1/2)^{k+1} by precomputed inverse CDF."""
    return np.searchsorted(_CMF, rng.random(size)) + 1  # k in {1, 2, ...}


def _sample_excess_life(nu: float, rng: np.random.Generator) -> float:
    """Sample the first crossover position from the stationary f*(x;nu) renewal.

    The excess-life density is h(x; nu) = 1 - F*(x; nu).

    Uses the inspection-paradox decomposition:
        K  ~ P(K=k) = k*(1/2)^{k+1}   (size-biased geometric)
        W  ~ Gamma(K*nu+1, rate=2*nu)  (size-biased f* component)
        X0 = W * U,  U ~ Uniform[0,1]
    """
    K = int(_sample_K_batch(rng, 1)[0])
    W = rng.gamma(shape=K * nu + 1.0, scale=0.5 / nu)
    return W * rng.random()


def _sample_fstar(nu: float, rng: np.random.Generator) -> float:
    """Sample one inter-crossover gap from f*(x; nu).

    f*(x; nu) = sum_{k>=1} (1/2)^k * Gamma(k*nu, rate=2*nu) density.
    Method: K ~ Geometric(1/2) on {1,2,...}, then X ~ Gamma(K*nu, rate=2*nu).
    Mean = E[K] * (1/2) = 2 * (1/2) = 1 Morgan.
    """
    k = rng.geometric(p=0.5)  # k in {1, 2, ...}
    return rng.gamma(shape=k * nu, scale=0.5 / nu)


def _interference_crossovers(
    L: float,
    nu: float,
    p: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate crossover positions from the gamma-interference pathway.

    Crossovers are placed directly from a stationary f*(x; nu) renewal
    process (no explicit Bernoulli thinning step -- the thinning is already
    encoded in f*). To achieve a crossover RATE of (1-p) per Morgan
    (so that the two pathways together give rate 1), we scale the chromosome:
    effectively sample a renewal on [0, L*(1-p)] then rescale back.

    In practice: place crossovers with f* inter-gaps on [0, L] but with
    f* scaled to have mean 1/(1-p) -- i.e. scale all gaps by 1/(1-p).
    Equivalently, place crossovers with Gamma(k*nu, rate=2*nu*(1-p)) gaps.
    """
    if p >= 1.0:
        return np.empty(0, dtype=np.float64)

    q = 1.0 - p  # interference-pathway crossover rate per Morgan
    scale = 0.5 / (
        nu * q
    )  # = 1 / (2*nu*q); gap mean = nu * scale = 1/(2q) chiasma mean
    # After geometric thinning: crossover mean = 1/q

    # First crossover: excess-life of the scaled renewal
    # Excess-life sampler: W ~ Gamma(K*nu+1, rate=2*nu*q), X0 = W*U
    K = int(_sample_K_batch(rng, 1)[0])
    W = rng.gamma(shape=K * nu + 1.0, scale=scale)
    x = W * rng.random()

    positions = []
    while x <= L:
        positions.append(x)
        k = rng.geometric(p=0.5)
        x += rng.gamma(shape=k * nu, scale=scale)

    return np.array(positions, dtype=np.float64) if positions else np.empty(0)


# ── Poisson-pathway crossovers on [0, L] ──────────────────────────────────────


def _poisson_crossovers(L: float, p: float, rng: np.random.Generator) -> np.ndarray:
    """Generate crossover positions from the Poisson escape pathway.

    N ~ Poisson(p * L) crossovers placed uniformly on [0, L].
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
    """Simulate crossover positions under the Housworth-Stahl model.

    Parameters
    ----------
    L   : Chromosome length in Morgans  (> 0).
    nu  : Interference shape parameter  (> 0).
          nu = 1  ->  no interference.
          nu > 1  ->  positive interference.
    p   : Escape fraction in [0, 1].
          p = 0  ->  pure interference model.
          p = 1  ->  pure Poisson model.
    rng : numpy.random.Generator (optional). Created with default_rng() if None.

    Returns
    -------
    np.ndarray, shape (n,)
        Sorted crossover positions in Morgan units, all in (0, L).
        May be empty (zero crossovers).

    Notes
    -----
    Expected number of crossovers = L regardless of (nu, p).
    The interference pathway gives sub-Poisson (Fano < 1) crossover counts
    for nu > 1; the Poisson pathway gives Fano = 1 for its own counts.
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

    xo_int = _interference_crossovers(L, nu, p, rng)
    if len(xo_int):
        parts.append(xo_int)

    xo_esc = _poisson_crossovers(L, p, rng)
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
