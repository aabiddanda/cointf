# stahl_saddlepoint.pyx
#
# Saddlepoint approximation for the renewal density f*(x; nu) and its CDF
# F*(x; nu) arising in the Housworth-Stahl crossover-interference model.
#
# DOMAIN NOTE
# -----------
# The MGF M_{f*}(t) = M_gamma(t)/(2-M_gamma(t)) requires M_gamma(t)<2, i.e.
# t < t* = 2*nu*(1-2^{-1/nu}).  This is STRICTLY less than 2*nu.
# K'(t) is strictly increasing on (-inf, t*), with K'(0)=1 and K'(t*-)=+inf,
# so the saddlepoint equation K'(t_hat)=x has a unique solution for all x>0.

# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False

from libc.math cimport (log, exp, sqrt, fabs, pow, M_PI,
                        INFINITY, NAN, isnan, copysign)

cdef double INV_SQRT_2PI = 0.3989422804014327
cdef int    MAX_ITER = 60
cdef double TOL = 1e-12

# ── Normal PDF/CDF (Abramowitz & Stegun 26.2.17, max err < 7.5e-8) ────────────

cdef inline double _norm_pdf(double z) nogil:
    return INV_SQRT_2PI * exp(-0.5 * z * z)

cdef inline double _norm_cdf(double z) nogil:
    cdef double t, poly, p
    cdef bint neg = (z < 0.0)
    if neg:
        z = -z
    t = 1.0 / (1.0 + 0.2316419 * z)
    poly = t * (0.319381530 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))))
    p = 1.0 - _norm_pdf(z) * poly
    return (1.0 - p) if neg else p

# ── True upper domain boundary ─────────────────────────────────────────────────

cdef inline double _t_upper(double nu) nogil:
    """t* = 2*nu*(1 - 2^{-1/nu}): where M_gamma(t)=2, CGF diverges."""
    return 2.0 * nu * (1.0 - pow(2.0, -1.0 / nu))


def t_upper(double nu):
    """Upper boundary t* of the CGF domain for given nu."""
    return _t_upper(nu)


cdef inline double _u(double t, double nu) nogil:
    return pow(1.0 - t / (2.0 * nu), -nu)


cdef inline double _u_d1(double t, double nu) nogil:
    return 0.5 * pow(1.0 - t / (2.0 * nu), -(nu + 1.0))


cdef inline double _u_d2(double t, double nu) nogil:
    return (nu + 1.0) / (4.0 * nu) * pow(1.0 - t / (2.0 * nu), -(nu + 2.0))


cdef double _cgf(double t, double nu) nogil:
    cdef double u = _u(t, nu)
    return log(u) - log(2.0 - u)


cdef double _cgf_d1(double t, double nu) nogil:
    cdef double u = _u(t, nu)
    cdef double ud = _u_d1(t, nu)
    return 2.0 * ud / (u * (2.0 - u))


cdef double _cgf_d2(double t, double nu) nogil:
    cdef double u = _u(t, nu)
    cdef double ud = _u_d1(t, nu)
    cdef double udd = _u_d2(t, nu)
    cdef double D = u * (2.0 - u)
    cdef double N = 2.0 * ud
    cdef double Nd = 2.0 * udd
    cdef double Dd = 2.0 * (1.0 - u) * ud
    return (Nd * D - N * Dd) / (D * D)


# Python-visible wrappers
def cgf(double t, double nu):
    """K(t): CGF of f*(·;nu). Domain: t < t*(nu)."""
    if t >= _t_upper(nu):
        raise ValueError(f"t must be < t*={_t_upper(nu):.6f}")
    return _cgf(t, nu)


def cgf_d1(double t, double nu):
    """K'(t). At t=0: K'(0)=1 = mean of f*."""
    if t >= _t_upper(nu):
        raise ValueError(f"t must be < t*={_t_upper(nu):.6f}")
    return _cgf_d1(t, nu)


def cgf_d2(double t, double nu):
    """K''(t). At t=0: K''(0)=(nu+1)/(2nu) = variance of f*."""
    if t >= _t_upper(nu):
        raise ValueError(f"t must be < t*={_t_upper(nu):.6f}")
    return _cgf_d2(t, nu)


# ── Newton-Raphson saddlepoint solver ─────────────────────────────────────────
# Solve K'(t_hat)=x on (-inf, t*).
# Warm start: linear approximation around t=0 using K'(0)=1, K''(0)=(nu+1)/(2nu).
# Clamp into (t_lo, 0.9999*t*) after each step.


cdef double _find_saddlepoint(double x, double nu) nogil:
    cdef double t_star, t, kd1, kd2, step
    cdef int i
    t_star = _t_upper(nu)
    # Warm start from K'(0)=1, K''(0)=(nu+1)/(2nu)
    t = (x - 1.0) * 2.0 * nu / (nu + 1.0)
    # Clamp: upper guard is strict, lower is loose
    if t >= 0.9 * t_star:
        t = 0.9 * t_star
    if t <= -500.0:
        t = -500.0

    for i in range(MAX_ITER):
        kd1 = _cgf_d1(t, nu)
        kd2 = _cgf_d2(t, nu)
        step = (kd1 - x) / kd2
        t -= step
        if t >= 0.99999 * t_star:
            t = 0.99999 * t_star
        if t <= -500.0:
            t = -500.0
        if fabs(step) < TOL * (1.0 + fabs(t)):
            return t
    return NAN


def find_saddlepoint(double x, double nu):
    """Solve K'(t_hat)=x. Returns t_hat in (-inf, t*(nu))."""
    if x <= 0.0:
        raise ValueError("x must be > 0.")
    cdef double t = _find_saddlepoint(x, nu)
    if isnan(t):
        raise RuntimeError(f"Newton-Raphson failed for x={x}, nu={nu}")
    return t


cdef double _fstar_sp(double x, double nu) nogil:
    cdef double t_hat, kval, kd2
    if x <= 0.0:
        return 0.0
    t_hat = _find_saddlepoint(x, nu)
    if isnan(t_hat):
        return NAN
    kval = _cgf(t_hat, nu)
    kd2 = _cgf_d2(t_hat, nu)
    return exp(kval - t_hat * x - 0.5 * log(2.0 * M_PI * kd2))


def fstar_sp(double x, double nu):
    """First-order saddlepoint approximation to f*(x; nu)."""
    if nu <= 0.0:
        raise ValueError("nu must be > 0.")
    if x <= 0.0:
        return 0.0
    cdef double v = _fstar_sp(x, nu)
    if isnan(v):
        raise RuntimeError(f"fstar_sp failed for x={x}, nu={nu}")
    return v


cdef double _Fstar_lr(double x, double nu) nogil:
    cdef double t_hat, kval, kd2, inner, r_hat, u_hat, v, kappa2, kappa3, sd, z, skew_c
    if x <= 0.0:
        return 0.0
    t_hat = _find_saddlepoint(x, nu)
    if isnan(t_hat):
        return NAN
    kval = _cgf(t_hat, nu)
    kd2 = _cgf_d2(t_hat, nu)
    inner = 2.0 * (t_hat * x - kval)
    if inner < 0.0:
        inner = 0.0
    r_hat = copysign(sqrt(inner), t_hat)
    u_hat = t_hat * sqrt(kd2)

    if fabs(r_hat) < 1.0e-5:
        # Near x=mu=1 (t_hat~0), L-R has a 0/0 form.
        # Use Edgeworth expansion with skewness correction:
        #   F*(mu+delta) ~ Phi(z) + phi(z)*[kappa3/(6*kappa2^{3/2})]*(1-z^2)
        # where z = delta/sqrt(kappa2),
        #   kappa2 = K''(0) = (nu+1)/(2*nu),
        #   kappa3 = K'''(0) = (nu+1)(nu+2)/(4*nu^2) + 1/2  [analytic]
        kappa2 = (nu + 1.0) / (2.0 * nu)
        kappa3 = (nu + 1.0) * (nu + 2.0) / (4.0 * nu * nu) + 0.5
        sd = sqrt(kappa2)
        z = (x - 1.0) / sd
        skew_c = kappa3 / (6.0 * kappa2 * sd)
        v = _norm_cdf(z) + _norm_pdf(z) * skew_c * (1.0 - z * z)
        if v < 0.0:
            v = 0.0
        if v > 1.0:
            v = 1.0
        return v

    v = _norm_cdf(r_hat) + _norm_pdf(r_hat) * (1.0 / r_hat - 1.0 / u_hat)
    if v < 0.0:
        v = 0.0
    if v > 1.0:
        v = 1.0
    return v


def Fstar_lr(double x, double nu):
    """Lugannani-Rice CDF approximation to F*(x; nu)."""
    if nu <= 0.0:
        raise ValueError("nu must be > 0.")
    if x <= 0.0:
        return 0.0
    cdef double v = _Fstar_lr(x, nu)
    if isnan(v):
        raise RuntimeError(f"Fstar_lr failed for x={x}, nu={nu}")
    return v


cdef double _h_mixture(double x, double nu, double p) nogil:
    cdef double fp, fu, Fp, Fu, q
    if x <= 0.0:
        return 0.0
    q = 1.0 - p
    fp = _fstar_sp(x, 1.0)
    fu = _fstar_sp(x, nu)
    Fp = _Fstar_lr(x, 1.0)
    Fu = _Fstar_lr(x, nu)
    return (p * p * fp
            + 2.0 * p * q * (1.0 - Fp) * (1.0 - Fu)
            + q * q * fu)


def h_mixture(double x, double nu, double p):
    """Two-pathway mixture density h(x; nu, p) — scalar."""
    if nu <= 0.0:
        raise ValueError("nu must be > 0.")
    if (p < 0.0) or (p > 1.0):
        raise ValueError("p must be in [0,1].")
    return _h_mixture(x, nu, p)


def loglik_meiosis(xs, double L, double nu, double p):
    """For a single-meiosis calculate the log-likeliood under the stahl model."""
    cdef int j
    cdef double ll, x0, xn, g_val, S_val, h_val
    cdef double Fp0, Fu0, Fpn, Fun
    if xs.size == 0:
        # The case of no observed crossovers.
        Fpn = _Fstar_lr(L, 1.0)
        Fun = _Fstar_lr(L, nu)
        S_val = p * (1.0 - Fpn) + (1.0 - p) * (1.0 - Fun)
        ll = 0.0
        if S_val > 0.0:
            ll += log(S_val)
    else:
        # At least a single-crossover available for analysis.
        x0 = xs[0]
        xn = L - xs[-1]
        # g(x0) = p*(1-F*(x0;1)) + (1-p)*(1-F*(x0;nu))
        Fp0 = _Fstar_lr(x0, 1.0)
        Fu0 = _Fstar_lr(x0, nu)
        g_val = p * (1.0 - Fp0) + (1.0 - p) * (1.0 - Fu0)

        # S(xn) = p*(1-F*(xn;1)) + (1-p)*(1-F*(xn;nu))
        Fpn = _Fstar_lr(xn, 1.0)
        Fun = _Fstar_lr(xn, nu)
        S_val = p * (1.0 - Fpn) + (1.0 - p) * (1.0 - Fun)

        ll = (log(g_val) if g_val > 0.0 else -INFINITY)
        if S_val > 0.0:
            ll += log(S_val)
        else:
            ll = -INFINITY
        for j in range(1, xs.size):
            h_val = _h_mixture(xs[j] - xs[j-1], nu, p)
            ll += (log(h_val) if h_val > 0.0 else -INFINITY)
    return ll
