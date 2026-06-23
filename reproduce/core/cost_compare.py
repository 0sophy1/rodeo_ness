"""
Resource cost comparison: QPE filtering vs Rodeo filtering on the R&S single-spin model.

Figure of merit for the filtering error epsilon = worst-case residual weight of a
nonzero mode in the filtered state:
    QPE:    epsilon_QPE(t) = max_{j!=0,1} |alpha0^(j)|^2          (R&S Eq. 12)
    Rodeo:  epsilon_Rodeo(n) = max_{j!=0,1} E[|beta_j|^2]         (draft Eq. 33)

Controlled-evolution depth (total simulated time under M):
    QPE:    T = t0 * sum_{k=0}^{t-1} 2^k = t0 (2^t - 1)           (geometric QPE ladder)
    Rodeo:  T = n * E[|t_l|] = n * t_rms * sqrt(2/pi)             (draft Eq. 36)

Gate cost = (2N+1) * k * T   (same Suzuki-Trotter prefactor for both; draft App. C).
For the single spin: 2N+1 = 3 qubits, M is 3-local => (k+1)=3 => k=2.

This script produces the numbers; plotting is done separately.
"""
import numpy as np
from model import embedding_M, steady_state, sx, sy, sz, I2
from rodeo_filter import expected_residual_weight, geometric_schedule, rodeo_betas
from qpe_filter import qpe_alpha0

G = 0.5            # Liouvillian gap / spectral separation of M for this model
T0 = 1.0 / 5.0     # QPE time scaling (R&S)
N_SPIN = 1
K_LOCAL = 2        # M is 3-local = (k+1)-local => k = 2
PREFACTOR = (2 * N_SPIN + 1) * K_LOCAL   # = 6


def nonzero_phis(h):
    """The nonzero eigenvalues of M (the 6 modes with |phi| >= g)."""
    M = embedding_M(h)
    phis = np.linalg.eigvalsh(M)
    return phis[np.abs(phis) > 1e-9]


# ---------------- QPE branch ----------------
def qpe_epsilon(h, t):
    """Worst-case residual leakage to the zero bin over nonzero modes."""
    phis = nonzero_phis(h)
    leak = np.array([qpe_alpha0(phi, t, T0)**2 for phi in phis])
    return np.max(leak)


def qpe_depth(t):
    """Total controlled-evolution time on the geometric QPE ladder."""
    return T0 * (2**t - 1)


# ---------------- Rodeo branch ----------------
def rodeo_epsilon_gaussian(h, n, t_rms):
    """Worst-case expected residual weight over nonzero modes (closed form)."""
    phis = nonzero_phis(h)
    w = np.array([expected_residual_weight(phi, t_rms, n) for phi in phis])
    return np.max(w)


def rodeo_depth_gaussian(n, t_rms):
    """Expected total evolution time: n * E[|t_l|], E[|t|] = t_rms sqrt(2/pi)."""
    return n * t_rms * np.sqrt(2.0 / np.pi)


def rodeo_epsilon_geometric(h, n, tau0, r):
    """Worst-case residual weight for a deterministic geometric schedule."""
    phis = nonzero_phis(h)
    times = geometric_schedule(n, tau0, r)
    beta = rodeo_betas(phis, times)
    return np.max(beta**2)


def rodeo_depth_geometric(n, tau0, r):
    times = geometric_schedule(n, tau0, r)
    return np.sum(np.abs(times))


if __name__ == "__main__":
    h = 0.5
    c = 2.0
    t_rms = c / G

    print(f"Model: single spin, h={h}, g={G}, prefactor (2N+1)k = {PREFACTOR}")
    print(f"Nonzero |phi| of M: {np.unique(np.round(np.abs(nonzero_phis(h)),4))}")
    print()

    print("QPE filtering:")
    print(f"{'t':>3} {'epsilon':>12} {'depth T':>12} {'gates':>12}")
    qpe_pts = []
    for t in range(1, 14):
        eps = qpe_epsilon(h, t)
        T = qpe_depth(t)
        gates = PREFACTOR * T
        qpe_pts.append((eps, T, gates))
        print(f"{t:>3} {eps:>12.3e} {T:>12.3f} {gates:>12.1f}")

    print()
    print(f"Rodeo filtering (Gaussian, c={c}, t_rms={t_rms}):")
    print(f"{'n':>3} {'epsilon':>12} {'depth T':>12} {'gates':>12}")
    rod_pts = []
    for n in range(1, 26):
        eps = rodeo_epsilon_gaussian(h, n, t_rms)
        T = rodeo_depth_gaussian(n, t_rms)
        gates = PREFACTOR * T
        rod_pts.append((eps, T, gates))
        if n <= 13 or n % 3 == 0:
            print(f"{n:>3} {eps:>12.3e} {T:>12.3f} {gates:>12.1f}")

    print()
    # Demonstrate the scaling laws by fitting
    qpe_pts = np.array(qpe_pts)
    rod_pts = np.array(rod_pts)
    # QPE: depth vs eps should be ~ eps^(-1/2). Fit log T = a + b log eps.
    mask = qpe_pts[:, 0] > 1e-12
    bq = np.polyfit(np.log(qpe_pts[mask, 0]), np.log(qpe_pts[mask, 1]), 1)
    print(f"QPE   fit:  log(depth) = {bq[1]:.3f} + ({bq[0]:.3f}) log(eps)   "
          f"=> depth ~ eps^({bq[0]:.3f})   [expect -0.5]")
    # Rodeo: depth vs log(1/eps) should be linear. Fit T = a + b * log(1/eps).
    mask = rod_pts[:, 0] > 1e-12
    br = np.polyfit(np.log(1.0 / rod_pts[mask, 0]), rod_pts[mask, 1], 1)
    print(f"Rodeo fit:  depth = {br[1]:.3f} + ({br[0]:.3f}) log(1/eps)   "
          f"=> depth ~ log(1/eps)  [linear]")


# ---- Deterministic (low-discrepancy) schedule helpers ----
def van_der_corput(n, base=2):
    """Deterministic low-discrepancy sequence in [0,1)."""
    seq = []
    for i in range(1, n + 1):
        x, denom, k = 0.0, 1.0, i
        while k > 0:
            denom *= base
            x += (k % base) / denom
            k //= base
        seq.append(x)
    return np.array(seq)


def deterministic_schedule(n, a, g=G):
    """Deterministic optimized Rodeo schedule: times bounded at scale 1/g,
    spread by a low-discrepancy sequence => no random fluctuation, depth
    linear in n (hence logarithmic in 1/eps)."""
    return (a / g) * van_der_corput(n)


def rodeo_epsilon_deterministic(h, n, a):
    """Worst-case residual weight for the deterministic schedule (exact)."""
    phis = nonzero_phis(h)
    times = deterministic_schedule(n, a)
    beta = rodeo_betas(phis, times)
    return np.max(beta**2)


def rodeo_depth_deterministic(n, a):
    return np.sum(np.abs(deterministic_schedule(n, a)))
