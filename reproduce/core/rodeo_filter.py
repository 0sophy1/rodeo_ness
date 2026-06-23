"""
Rodeo zero-sector filtering (draft: Yeo, Jeong, Shin, Kim).

For an M-eigenmode with eigenvalue phi_j, n successful Rodeo cycles with times {t_l} give
    beta_j = prod_l cos(phi_j t_l / 2)          (draft Eq. 21, 31)
Zero modes (phi_0 = phi_1 = 0) -> beta = 1 (preserved).

Two schedules:
  (A) Gaussian baseline (draft Appendix A):  t_l ~ N(0, t_rms^2), t_rms = c/g.
      Averaged residual weight  E[|beta_j|^2] = prod_l (1 + exp(-phi_j^2 t_rms^2/2))/2
                                              <= q(c)^n,  q(c) = (1+exp(-c^2/2))/2 < 1.
  (B) Geometric deterministic schedule (optimized Rodeo, refs [26,27]):
      t_l = tau0 * r^(l-1), removing the random-time fluctuation.

Readout reuses the QPE machinery with the replacement alpha0^(j) -> beta_j (draft Eq. 22).
"""
import numpy as np
from model import (vec, liouvillian, steady_state, embedding_M, sx, sy, sz, I2)
from qpe_filter import Q_operator, O_super, build_input_state


# ----------------------------------------------------------------------
# Rodeo filtering factors
# ----------------------------------------------------------------------
def rodeo_betas(phis, times):
    """beta_j = prod_l cos(phi_j t_l / 2) for each eigenvalue phi_j."""
    phis = np.asarray(phis)
    times = np.asarray(times)
    # shape (modes, cycles)
    M = np.cos(np.outer(phis, times) / 2.0)
    return np.prod(M, axis=1)


def gaussian_schedule(n, t_rms, rng):
    """n Gaussian-distributed evolution times with rms width t_rms."""
    return rng.normal(0.0, t_rms, size=n)


def geometric_schedule(n, tau0, r):
    """Deterministic geometric schedule t_l = tau0 * r^(l-1)."""
    return tau0 * r ** np.arange(n)


# ----------------------------------------------------------------------
# Averaged-bound validation (draft Eqs. 32-34)
# ----------------------------------------------------------------------
def expected_residual_weight(phi, t_rms, n):
    """E[|beta_j|^2] for Gaussian schedule, closed form (draft Eq. 33)."""
    return ((1.0 + np.exp(-phi**2 * t_rms**2 / 2.0)) / 2.0) ** n


# ----------------------------------------------------------------------
# Rodeo-filtered readout (eigenmode level, same structure as QPE)
# ----------------------------------------------------------------------
def rodeo_filtered_readout(h, times, O):
    """R_O = <psi| Q |psi> for a single Rodeo schedule {times}."""
    M = embedding_M(h)
    phis, V = np.linalg.eigh(M)
    xi = build_input_state(h)
    c = V.conj().T @ xi
    beta = rodeo_betas(phis, times)
    psi = V @ (beta * c)
    Q = Q_operator(O)
    return np.vdot(psi, Q @ psi)


def rodeo_estimate(h, times, O):
    """Observable estimate <O> = R_O / R_I for one schedule realization."""
    R_O = rodeo_filtered_readout(h, times, O)
    R_I = rodeo_filtered_readout(h, times, I2)
    return (R_O / R_I).real


def rodeo_estimate_averaged(h, n, t_rms, O, n_real, rng):
    """Average observable estimate over n_real Gaussian-schedule realizations.
    Returns mean and std of the estimate, and mean total evolution time."""
    ests = []
    depths = []
    for _ in range(n_real):
        times = gaussian_schedule(n, t_rms, rng)
        ests.append(rodeo_estimate(h, times, O))
        depths.append(np.sum(np.abs(times)))
    return np.mean(ests), np.std(ests), np.mean(depths)


if __name__ == "__main__":
    np.set_printoptions(precision=6, suppress=True)
    g = 0.5
    # --- validate the averaged residual-weight bound, Eq. (33)-(34) ---
    print("Validation of averaged residual-weight bound (Gaussian schedule)")
    print("  closed-form Eq.(33) vs Monte-Carlo average of |beta_j|^2")
    rng = np.random.default_rng(0)
    c = 2.0
    t_rms = c / g
    q = (1 + np.exp(-c**2/2))/2
    print(f"  c={c}, t_rms={t_rms}, q(c)={q:.5f}")
    for phi in [0.5, 1.0, 1.77]:          # representative nonzero |phi| of M (>= g)
        for n in [1, 4, 8]:
            mc = np.mean([np.prod(np.cos(phi*gaussian_schedule(n,t_rms,rng)/2))**2
                          for _ in range(20000)])
            cf = expected_residual_weight(phi, t_rms, n)
            bound = q**n
            print(f"  phi={phi:>4}  n={n}:  MC={mc:.5f}  closed-form={cf:.5f}  q^n bound={bound:.5f}")

    print()
    print("Rodeo observable estimates (Gaussian schedule, averaged over realizations)")
    t0_unused = None
    for h in [0.5, 1.0, 1.5]:
        rho_ss, _, _ = steady_state(h)
        sy_exact = np.trace(sy @ rho_ss).real
        sz_exact = np.trace(sz @ rho_ss).real
        print("=" * 70)
        print(f"h = {h}:  exact <sy>={sy_exact:.6f}  <sz>={sz_exact:.6f}")
        print(f"{'n':>3} {'<sy>_avg':>12} {'<sz>_avg':>12} {'err_y':>12} {'err_z':>12} {'depth':>10}")
        for n in range(1, 13):
            my, sdy, depth = rodeo_estimate_averaged(h, n, t_rms, sy, 400, rng)
            mz, sdz, _ = rodeo_estimate_averaged(h, n, t_rms, sz, 400, rng)
            print(f"{n:>3} {my:>12.6f} {mz:>12.6f} "
                  f"{abs(my-sy_exact):>12.2e} {abs(mz-sz_exact):>12.2e} {depth:>10.3f}")
