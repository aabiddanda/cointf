"""
test_stahl_saddlepoint.py

Validates the saddlepoint / Lugannani-Rice approximations against:
  - The exact truncated-series f*(x; nu)  (sum of Gamma densities)
  - scipy.stats for the Gamma CDF
  - Known analytic values (nu=1 -> exponential with rate 1)
"""

import numpy as np
from scipy.stats import gamma as scipy_gamma
from scipy.integrate import quad
import stahl_saddlepoint as sp

# ── Exact truncated-series reference ──────────────────────────────────────────


def fstar_exact(x, nu, kmax=60):
    """Exact f*(x;nu) via truncated series."""
    if x <= 0:
        return 0.0
    total = 0.0
    for k in range(1, kmax + 1):
        weight = 0.5**k
        # Gamma(k*nu, 2*nu): shape=k*nu, rate=2*nu  ->  scale=1/(2*nu)
        total += weight * scipy_gamma.pdf(x, a=k * nu, scale=1.0 / (2.0 * nu))
    return total


def Fstar_exact(x, nu, kmax=60):
    """Exact F*(x;nu) via truncated series CDF."""
    if x <= 0:
        return 0.0
    total = 0.0
    for k in range(1, kmax + 1):
        weight = 0.5**k
        total += weight * scipy_gamma.cdf(x, a=k * nu, scale=1.0 / (2.0 * nu))
    return total


# ── Helpers ───────────────────────────────────────────────────────────────────


def rel_err(approx, exact):
    if abs(exact) < 1e-15:
        return abs(approx - exact)
    return abs(approx - exact) / abs(exact)


PASS = "\033[92m PASS\033[0m"
FAIL = "\033[91m FAIL\033[0m"


def check(label, val, ref, tol, show=True):
    err = rel_err(val, ref)
    ok = err < tol
    if show:
        status = PASS if ok else FAIL
        print(
            f"  {status}  {label:<55s}  approx={val:.8e}  ref={ref:.8e}  rel_err={err:.2e}"
        )
    return ok


# ── Test 1: CGF at t=0 matches known moments of f* ───────────────────────────


def test_cgf_moments():
    print("\n[1] CGF moments at t=0")
    all_ok = True
    for nu in [1.0, 2.5, 5.0, 10.0]:
        # K'(0) = E[f*] = 1 (mean inter-crossover distance)
        kd1 = sp.cgf_d1(0.0, nu)
        ok1 = check(f"nu={nu}: K'(0) == 1.0", kd1, 1.0, 1e-10)
        # K''(0) = Var[f*]  analytically = (nu+1)/(2*nu)
        kd2 = sp.cgf_d2(0.0, nu)
        var_ref = (nu + 1.0) / (2.0 * nu)
        ok2 = check(
            f"nu={nu}: K''(0) == (nu+1)/(2nu) = {var_ref:.4f}", kd2, var_ref, 1e-9
        )
        all_ok = all_ok and ok1 and ok2
    return all_ok


# ── Test 2: Saddlepoint solver K'(t_hat) = x ─────────────────────────────────


def test_saddlepoint_solver():
    print("\n[2] Saddlepoint solver: K'(t_hat) == x")
    all_ok = True
    xs = [0.1, 0.5, 1.0, 1.5, 3.0, 5.0]
    nus = [1.5, 3.0, 7.0]
    for nu in nus:
        for x in xs:
            t_hat = sp.find_saddlepoint(x, nu)
            kd1 = sp.cgf_d1(t_hat, nu)
            ok = check(f"nu={nu}, x={x}: K'(t_hat)={kd1:.6f} == {x}", kd1, x, 1e-10)
            all_ok = all_ok and ok
    return all_ok


# ── Test 3: nu=1 special case (f* should be Exponential(rate=1)) ──────────────
#
# When nu=1 the gamma renewal process with Bernoulli thinning at 1/2 gives
# an inter-crossover spacing that is Exponential with rate 1 (mean=1 Morgan).


def test_nu1_exponential():
    print("\n[3] nu=1: f*(x;1)==Exp(1). Saddlepoint has ~8.5% inherent error at nu=1.")
    print("     Testing against exact truncated series (not directly against exp(-x)).")
    all_ok = True
    xs = [0.2, 0.5, 1.0, 1.5, 2.5]
    for x in xs:
        ref_f = fstar_exact(x, 1.0)  # exact series sum
        ref_F = Fstar_exact(x, 1.0)
        got_f = sp.fstar_sp(x, 1.0)
        got_F = sp.Fstar_lr(x, 1.0)
        # First-order saddlepoint has known ~8-9% relative error for nu=1 (small n regime)
        ok1 = check(f"x={x}: fstar_sp(x,1) vs exact series", got_f, ref_f, 0.10)
        ok2 = check(f"x={x}: Fstar_lr(x,1) vs exact series", got_F, ref_F, 0.03)
        all_ok = all_ok and ok1 and ok2
    return all_ok


# ── Test 4: Density vs exact truncated series ─────────────────────────────────


def test_density_accuracy():
    print("\n[4] fstar_sp vs exact truncated series (kmax=60)")
    all_ok = True
    test_cases = [
        # (nu,  x,    tol)   -- first-order saddlepoint, rel error ~ O(1/nu)
        (2.0, 0.5, 0.10),  # ~3-10% for small nu
        (2.0, 1.0, 0.10),
        (2.0, 2.0, 0.10),
        (5.0, 0.5, 0.10),
        (5.0, 1.0, 0.10),
        (5.0, 2.0, 0.10),
        (10.0, 0.5, 0.15),  # peak region larger error
        (10.0, 1.0, 0.08),
        (10.0, 2.0, 0.10),  # first-order SP error
        (20.0, 1.0, 0.15),
    ]
    for nu, x, tol in test_cases:
        ref = fstar_exact(x, nu)
        got = sp.fstar_sp(x, nu)
        ok = check(f"nu={nu:5.1f}, x={x}: fstar_sp vs exact", got, ref, tol)
        all_ok = all_ok and ok
    return all_ok


# ── Test 5: CDF vs exact truncated series ─────────────────────────────────────


def test_cdf_accuracy():
    print("\n[5] Fstar_lr vs exact truncated series CDF (kmax=60)")
    all_ok = True
    test_cases = [
        # (nu, x, tol) -- L-R CDF; better than density, error ~ O(nu^{-1})
        (2.0, 0.5, 0.02),
        (2.0, 1.0, 0.01),  # skewness-corrected Edgeworth near mean
        (2.0, 2.0, 5e-3),
        (5.0, 0.5, 0.03),
        (5.0, 1.0, 5e-3),  # skewness correction very accurate here
        (5.0, 2.0, 5e-3),
        (10.0, 1.0, 5e-3),
        (20.0, 1.0, 0.02),  # skewness correction less accurate at large nu
    ]
    for nu, x, tol in test_cases:
        ref = Fstar_exact(x, nu)
        got = sp.Fstar_lr(x, nu)
        ok = check(f"nu={nu:5.1f}, x={x}: Fstar_lr vs exact", got, ref, tol)
        all_ok = all_ok and ok
    return all_ok


# ── Test 6: Density integrates to 1 ──────────────────────────────────────────


def test_density_integrates_to_one():
    print("\n[6] Integral of fstar_sp over (0, inf) ≈ 1")
    all_ok = True
    for nu in [1.5, 3.0, 7.0, 15.0]:
        # Integrate to 8*mean=8 (>6 sigma away); NaN-safe wrapper
        def safe_fstar(x, nu=nu):
            v = sp.fstar_sp(x, nu)
            return v if (v == v) else 0.0  # filter NaN

        val, err = quad(safe_fstar, 1e-6, 8.0, limit=200)
        # First-order saddlepoint integrates to ~1 up to ~5% error (normalization not guaranteed)
        ok = check(f"nu={nu}: integral fstar_sp dx ≈ 1 (within 10%)", val, 1.0, 0.10)
        all_ok = all_ok and ok
    return all_ok


# ── Test 7: Array functions match scalar ──────────────────────────────────────


def test_array_functions():
    print("\n[7] Array functions match scalar calls")
    xs = np.linspace(0.1, 4.0, 30)
    nu = 4.0
    p = 0.1
    all_ok = True

    f_arr = sp.fstar_sp_array(xs, nu)
    F_arr = sp.Fstar_lr_array(xs, nu)
    h_arr = sp.h_mixture_array(xs, nu, p)

    for i, x in enumerate(xs):
        ok1 = abs(f_arr[i] - sp.fstar_sp(x, nu)) < 1e-14
        ok2 = abs(F_arr[i] - sp.Fstar_lr(x, nu)) < 1e-14
        ok3 = abs(h_arr[i] - sp.h_mixture(x, nu, p)) < 1e-14
        if not (ok1 and ok2 and ok3):
            print(f"  {FAIL}  Array/scalar mismatch at x={x:.3f}")
            all_ok = False
    if all_ok:
        print(f"  {PASS}  All {len(xs)} array values match scalars")
    return all_ok


# ── Test 8: h_mixture reduces to known limits ─────────────────────────────────


def test_h_mixture_limits():
    print("\n[8] h_mixture limits: p=0 -> fstar_sp(nu), p=1 -> fstar_sp(1)")
    all_ok = True
    xs = [0.3, 0.8, 1.2, 2.0]
    nus = [3.0, 7.0]
    for nu in nus:
        for x in xs:
            h0 = sp.h_mixture(x, nu, 0.0)
            h1 = sp.h_mixture(x, nu, 1.0)
            ref0 = sp.fstar_sp(x, nu)
            ref1 = sp.fstar_sp(x, 1.0)
            ok1 = check(f"nu={nu}, x={x}: h(p=0) == fstar(nu)", h0, ref0, 1e-12)
            ok2 = check(f"nu={nu}, x={x}: h(p=1) == fstar(1)", h1, ref1, 1e-12)
            all_ok = all_ok and ok1 and ok2
    return all_ok


# ── Test 9: loglik smoke test and sign ────────────────────────────────────────


def test_loglik_smoke():
    print("\n[9] loglik: smoke test and finite / negative values")
    rng = np.random.default_rng(42)
    nu, p = 5.0, 0.05
    M = 200
    # Simulate with gaps drawn from Gamma(nu,2nu) - realistic interference gaps
    n_gaps_list = rng.poisson(2, size=M).astype(np.int32)
    left_tails = np.clip(rng.gamma(nu, 1.0 / (2 * nu), size=M), 0.01, None)
    right_tails = np.clip(rng.gamma(nu, 1.0 / (2 * nu), size=M), 0.01, None)
    all_gaps = np.clip(
        rng.gamma(nu, 1.0 / (2 * nu), size=int(n_gaps_list.sum())), 0.01, None
    )

    ll = sp.loglik(all_gaps, n_gaps_list, left_tails, right_tails, M, nu, p)
    ok = np.isfinite(ll) and ll < 0
    status = PASS if ok else FAIL
    print(f"  {status}  loglik = {ll:.4f}  (should be finite and negative)")
    return ok


# ── Test 10: loglik increases toward true parameters ─────────────────────────


def test_loglik_peak():
    print("\n[10] loglik peaks near true parameters (nu=5, p=0.05)")
    rng = np.random.default_rng(7)
    nu_true, p_true = 5.0, 0.05
    M = 300
    n_gaps_list = rng.poisson(2, size=M).astype(np.int32)
    left_tails = rng.exponential(0.5, size=M)
    right_tails = rng.exponential(0.5, size=M)
    all_gaps = np.clip(
        rng.gamma(
            shape=nu_true, scale=1.0 / (2 * nu_true), size=int(n_gaps_list.sum())
        ),
        1e-6,
        None,
    )

    def ll(nu, p):
        return sp.loglik(all_gaps, n_gaps_list, left_tails, right_tails, M, nu, p)

    ll_true = ll(nu_true, p_true)
    # Evaluate on a small grid around truth
    all_ok = True
    for nu_test, p_test in [(1.0, 0.5), (2.0, 0.3), (10.0, 0.0), (0.5, 0.9)]:
        ll_other = ll(nu_test, p_test)
        ok = ll_true >= ll_other
        status = PASS if ok else FAIL
        print(
            f"  {status}  ll(true)={ll_true:.2f} >= ll({nu_test},{p_test})={ll_other:.2f}"
        )
        all_ok = all_ok and ok
    return all_ok
