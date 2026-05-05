# cython: boundscheck=False
# cython: cdivision=True
# cython: wraparound=False

from libc.math cimport exp, log, lgamma

cdef extern from "math.h":
    float INFINITY


cdef double logsumexp(double[:] x):
    """Cython implementation of the logsumexp trick."""
    cdef int i, n
    cdef double m = -1e32
    cdef double c = 0.0
    n = x.size
    for i in range(n):
        m = max(m, x[i])
    for i in range(n):
        c += exp(x[i] - m)
    return m + log(c)

cdef double gammaf(double x, double alpha, double scale):
    """Calculation of the renewal density..."""
    cdef double log_density
    # log(f(x)) = alpha*log(scale) - lgamma(alpha) + (alpha-1)*log(x) - x/scale
    log_density = alpha * log(scale) - lgamma(alpha) + (alpha - 1) * log(x) - x/scale
    return exp(log_density)

cdef double renewal_density(double x, double nu, int kmax=25):
    """Calculate the renewal density using a sum approximation."""
    cdef int k
    cdef double d
    for k in range(kmax):
        d += (0.5**k)*gammaf(x, nu, 2*nu)
    return d
