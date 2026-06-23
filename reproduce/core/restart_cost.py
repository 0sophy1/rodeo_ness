"""
Restart-on-failure cost model (the draft's second claimed advantage).

We compare the EXPECTED TOTAL controlled-evolution depth to obtain one successful
filtered output, INCLUDING failed attempts and restarts.

Setup: input |xi> expanded in eigenmodes of M with amplitudes c_j (weights w_j=|c_j|^2,
normalized). Zero modes have phi=0 and survive every filter.

QPE (single terminal measurement):
  - one attempt runs the full circuit (depth D_QPE = t0(2^t-1)) and then measures.
  - success prob p0(t) = sum_j w_j |alpha0^(j)|^2   (zero modes contribute 1).
  - a FAILURE still costs the full D_QPE (no early termination).
  => E[total depth] = D_QPE / p0.

Rodeo (per-cycle measurement, abort on first failure):
  - cumulative surviving weight after k cycles: W_k = sum_j w_j (f_j^(k))^2,
    where f_j^(k) = prod_{m<=k} cos(phi_j t_m/2). W_0 = 1.
  - P(fail exactly at cycle k) = W_{k-1} - W_k.
  - depth consumed if fail at k: D_k = sum_{l<=k} |t_l|.
  - full-run success prob P = W_n.
  => E[total depth] = (1/P) * sum_k (W_{k-1}-W_k) D_k  +  D_n.
  A failure that happens early costs only the first few cycles -- this is the saving.
"""
import numpy as np
from model import embedding_M, steady_state, sx, sy, sz, I2
from qpe_filter import build_input_state, qpe_alpha0
from cost_compare import (G, T0, qpe_epsilon, qpe_depth, nonzero_phis,
                          rodeo_epsilon_deterministic, rodeo_depth_deterministic)
from cost_compare import deterministic_schedule
from rodeo_filter import rodeo_betas, gaussian_schedule


def eigen_weights(h):
    """Return (phis, weights w_j) for |xi> expanded in eigenmodes of M (normalized)."""
    M = embedding_M(h)
    phis, V = np.linalg.eigh(M)
    xi = build_input_state(h)
    c = V.conj().T @ xi
    w = np.abs(c)**2
    w = w / w.sum()
    return phis, w


# ---------------- QPE ----------------
def qpe_p0(h, t):
    phis, w = eigen_weights(h)
    a2 = np.array([qpe_alpha0(p, t, T0)**2 for p in phis])
    return float(np.sum(w * a2))


def qpe_expected_total_depth(h, t):
    return qpe_depth(t) / qpe_p0(h, t)


# ---------------- Rodeo (deterministic, exact) ----------------
def rodeo_expected_total_depth_det(h, n, a_det):
    phis, w = eigen_weights(h)
    times = deterministic_schedule(n, a_det)
    abs_t = np.abs(times)
    Dk = np.cumsum(abs_t)                       # depth through cycle k (1-indexed: Dk[k-1])
    # cumulative cos products f_j^(k)
    cos_mat = np.cos(np.outer(phis, times) / 2.0)   # (modes, cycles)
    f = np.cumprod(cos_mat, axis=1)                  # f[:,k-1] = f_j^(k)
    W = np.empty(n + 1)
    W[0] = 1.0
    for k in range(1, n + 1):
        W[k] = float(np.sum(w * f[:, k - 1]**2))
    P = W[n]
    fail_term = sum((W[k - 1] - W[k]) * Dk[k - 1] for k in range(1, n + 1))
    return (fail_term / P) + Dk[-1], P


# ---------------- Rodeo (Gaussian, Monte-Carlo the whole restart process) ----------------
def rodeo_expected_total_depth_gauss_mc(h, n, t_rms, n_trials, rng):
    """Directly simulate restart-until-success; return mean total depth and mean #attempts."""
    phis, w = eigen_weights(h)
    totals = []
    attempts_list = []
    for _ in range(n_trials):
        total = 0.0
        attempts = 0
        while True:
            attempts += 1
            times = gaussian_schedule(n, t_rms, rng)
            abs_t = np.abs(times)
            a = w.copy()                          # current (normalized) weights
            consumed = 0.0
            succeeded = True
            for ell in range(n):
                consumed += abs_t[ell]
                cosf = np.cos(phis * times[ell] / 2.0)**2
                p_succ = float(np.sum(a * cosf))   # conditional success prob this cycle
                # update weights (unnormalized then renormalize)
                a = a * cosf
                if a.sum() <= 0:
                    succeeded = False; break
                a = a / a.sum()
                if rng.random() > p_succ:
                    succeeded = False; break
            total += consumed
            if succeeded:
                break
        totals.append(total)
        attempts_list.append(attempts)
    return np.mean(totals), np.mean(attempts_list)


if __name__ == "__main__":
    h = 0.5
    A_DET = 3.0
    phis, w = eigen_weights(h)
    zero_w = w[np.abs(phis) < 1e-9].sum()
    print(f"weight on zero sector |c0|^2+|c1|^2 = {zero_w:.4f}  (asymptotic success prob)")
    print()

    # ---- VALIDATION: Rodeo P from formula == ||beta o c||^2 / ||c||^2 ----
    print("Validation: full-run success prob P (formula vs direct norm):")
    for n in [3, 6, 10]:
        _, P = rodeo_expected_total_depth_det(h, n, A_DET)
        times = deterministic_schedule(n, A_DET)
        beta = rodeo_betas(phis, times)
        P_direct = np.sum(w * beta**2)
        print(f"  n={n:>2}: P_formula={P:.5f}  P_norm={P_direct:.5f}  match={np.isclose(P,P_direct)}")
    print()

    # ---- VALIDATION: Gaussian MC expected total depth vs analytic restart (using E per-cycle) ----
    print("Validation: Gaussian MC #attempts vs 1/E[P]  (sanity, not exact equal):")
    rng = np.random.default_rng(3)
    for n in [4, 8]:
        mc_depth, mc_att = rodeo_expected_total_depth_gauss_mc(h, n, 4.0, 3000, rng)
        print(f"  n={n}: MC mean total depth={mc_depth:.2f}, MC mean attempts={mc_att:.2f}")
    print()

    # ---- The comparison: expected TOTAL depth (with restarts) vs naive single-run depth ----
    print("Expected TOTAL controlled-evolution depth (incl. restarts) to one success:")
    print(f"{'eps':>9} | {'QPE single':>11} {'QPE total':>10} | {'Rodeo single':>13} {'Rodeo total':>12} {'P_run':>7}")
    for target in [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]:
        # QPE: find smallest t with eps<=target
        t = next((tt for tt in range(1, 30) if qpe_epsilon(h, tt) <= target), None)
        qd = qpe_depth(t); qtot = qpe_expected_total_depth(h, t)
        # Rodeo det: smallest n
        n = next((nn for nn in range(1, 80) if rodeo_epsilon_deterministic(h, nn, A_DET) <= target), None)
        rd = rodeo_depth_deterministic(n, A_DET)
        rtot, P = rodeo_expected_total_depth_det(h, n, A_DET)
        print(f"{target:>9.0e} | {qd:>11.1f} {qtot:>10.1f} | {rd:>13.1f} {rtot:>12.1f} {P:>7.3f}")
