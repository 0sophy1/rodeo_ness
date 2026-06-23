"""
Single-spin driven-dissipative model from Ramusat & Savona (Quantum 2021), Sec. 5.

  H = h * sigma_x
  Jump operator A = sigma^-  (decay along z)
  rho_dot = -i[H, rho] - (1/2){ sigma^+ sigma^-, rho } + sigma^- rho sigma^+

Liouvillian gap g = 1/2, independent of h. They set t0 = 1/5.

We build:
  - Liouvillian superoperator L  (4x4, vectorized, column-stacking convention)
  - Hermitian embedding M = [[0, L], [L^dag, 0]]  (8x8)
  - the two zero modes |eta0> = |0>|I>, |eta1> = |1>|rho_ss>
  - spectral separation g_M = min nonzero |phi_j| of M

Everything is validated numerically against the known steady state and gap.
"""
import numpy as np

# ---- Pauli matrices ----
I2 = np.eye(2, dtype=complex)
sx = np.array([[0, 1], [1, 0]], dtype=complex)
sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
sz = np.array([[1, 0], [0, -1]], dtype=complex)
sp = np.array([[0, 1], [0, 0]], dtype=complex)   # sigma^+ = |0><1|
sm = np.array([[0, 0], [1, 0]], dtype=complex)   # sigma^- = |1><0|

# ---- Vectorization: COLUMN stacking ----
# vec(A X B) = (B^T kron A) vec(X)
def vec(rho):
    return rho.reshape(-1, order='F')

def unvec(v):
    d = int(round(np.sqrt(v.size)))
    return v.reshape(d, d, order='F')

def liouvillian(h):
    H = h * sx
    A = sm
    Adag = A.conj().T
    d = 2
    Id = np.eye(d, dtype=complex)
    # -i[H, rho]  ->  -i ( I kron H - H^T kron I )
    L = -1j * (np.kron(Id, H) - np.kron(H.T, Id))
    # dissipator: A rho A^dag - 1/2 {A^dag A, rho}
    AdA = Adag @ A
    L += np.kron(A.conj(), A)                       # A rho A^dag -> (A^* kron A)
    L += -0.5 * np.kron(Id, AdA)                    # -1/2 A^dag A rho
    L += -0.5 * np.kron(AdA.T, Id)                  # -1/2 rho A^dag A
    return L

def steady_state(h):
    """Return normalized steady-state density matrix from null space of L."""
    L = liouvillian(h)
    w, v = np.linalg.eig(L)
    idx = np.argmin(np.abs(w))
    rho_v = v[:, idx]
    rho = unvec(rho_v)
    rho = 0.5 * (rho + rho.conj().T)               # Hermitize
    rho = rho / np.trace(rho)                       # normalize
    return rho, L, w

def liouvillian_gap(h):
    """Liouvillian gap g = min |Re(lambda)| over nonzero eigenvalues."""
    L = liouvillian(h)
    w = np.linalg.eigvals(L)
    re = np.abs(w.real)
    # nonzero eigenvalues = exclude the one closest to 0
    order = np.argsort(np.abs(w))
    nonzero = w[order[1:]]
    return np.min(np.abs(nonzero.real)), w

def embedding_M(h):
    """Hermitian embedding M = [[0, L],[L^dag, 0]] (8x8) and its zero modes."""
    L = liouvillian(h)
    dL = L.shape[0]
    M = np.zeros((2 * dL, 2 * dL), dtype=complex)
    M[:dL, dL:] = L
    M[dL:, :dL] = L.conj().T
    return M

def zero_modes(h):
    """Return |eta0> = |0>|I>, |eta1> = |1>|rho_ss> as 8-vectors, plus spectral data of M."""
    rho_ss, L, _ = steady_state(h)
    dL = L.shape[0]
    Iden = np.eye(2, dtype=complex)
    vI = vec(Iden) / np.sqrt(2)                     # |I> normalized: <I|I> = Tr(I^dag I)=2
    vrho = vec(rho_ss)
    vrho = vrho / np.linalg.norm(vrho)
    eta0 = np.concatenate([vI, np.zeros(dL, dtype=complex)])           # |0> block
    eta1 = np.concatenate([np.zeros(dL, dtype=complex), vrho])         # |1> block
    return eta0, eta1, rho_ss


if __name__ == "__main__":
    np.set_printoptions(precision=5, suppress=True)
    for h in [0.5, 1.0, 1.5]:
        print("=" * 60)
        print(f"h = {h}")
        rho_ss, L, wL = steady_state(h)
        print("steady state rho_ss =")
        print(rho_ss)
        print("  Tr(rho_ss) =", np.trace(rho_ss).real)
        print("  Hermitian? ", np.allclose(rho_ss, rho_ss.conj().T))
        evals = np.linalg.eigvalsh(rho_ss)
        print("  eigenvalues (should be >=0):", evals.real)
        print("  ||L rho_ss|| =", np.linalg.norm(L @ vec(rho_ss)))

        # observables
        sy_ev = np.trace(sy @ rho_ss).real
        sz_ev = np.trace(sz @ rho_ss).real
        sx_ev = np.trace(sx @ rho_ss).real
        print(f"  <sx>={sx_ev:.5f}  <sy>={sy_ev:.5f}  <sz>={sz_ev:.5f}")

        g, _ = liouvillian_gap(h)
        print(f"  Liouvillian gap g = {g:.5f}  (expected 0.5)")

        M = embedding_M(h)
        print("  M Hermitian? ", np.allclose(M, M.conj().T))
        phi = np.linalg.eigvalsh(M)
        print("  M eigenvalues:", np.sort(phi))
        nz = np.sort(np.abs(phi))
        g_M = nz[nz > 1e-9][0]
        print(f"  spectral separation g_M = min nonzero |phi| = {g_M:.5f}")

        # verify zero modes
        eta0, eta1, _ = zero_modes(h)
        print("  ||M eta0|| =", np.linalg.norm(M @ eta0))
        print("  ||M eta1|| =", np.linalg.norm(M @ eta1))
        print("  <eta0|eta1> =", np.vdot(eta0, eta1))
