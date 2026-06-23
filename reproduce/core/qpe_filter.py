"""
QPE zero-sector filtering (Ramusat & Savona, Quantum 2021).

We work in the eigenbasis of M. The QPE projection onto the zero bin acts on each
eigenmode |eta_j> (eigenvalue phi_j) by an amplitude alpha0(phi_j) given by R&S Eq.(9)/(12):

    |alpha0^(j)|^2 = (1/2^(2t)) * sin^2(pi 2^t varphi_j) / sin^2(pi varphi_j),
    varphi_j = M_eigenvalue * t0   (dimensionless phase, in [0,1) by choice of t0)

For the zero modes phi_0 = phi_1 = 0, alpha0 = 1 (preserved).

Observable readout (R&S Eq. 17): with input |xi> = (|0>|I> + |1>|0>)/sqrt2,
the measured quantity <psi3|I (x) Q|psi3> = c1 Tr(O rho_ss) + spurious terms.
Setting O = I gives c1 Tr(rho_ss) = c1, so the ratio yields <O> = Tr(O rho_ss).

We simulate this exactly at the eigenmode level: build |xi>, expand in eigenmodes of M,
apply the QPE leakage factors, and evaluate the readout for O and O=I.
"""
import numpy as np
from model import (vec, unvec, liouvillian, steady_state, embedding_M, zero_modes,
                   sx, sy, sz, I2)


def build_input_state(h):
    """|xi> = (|0>|I> + |1>|0>)/sqrt2  as an 8-vector (R&S Eq. 6).
    Note: |I> here is the *unnormalized* vectorized identity (R&S keep <I|I>=Tr(I)=2).
    Then |0> in the |1> block is the vectorized |0><0| projector? 
    R&S Eq.(6): |xi> = (|0>|I> + |1>|0>)/sqrt2 where |0> means |0><0| of the system.
    """
    Iden = np.eye(2, dtype=complex)
    vI = vec(Iden)                       # vectorized identity, norm sqrt(2)
    rho0 = np.zeros((2, 2), dtype=complex)
    rho0[0, 0] = 1.0                     # |0><0|
    v0 = vec(rho0)
    dL = 4
    xi = np.concatenate([vI, v0]) / np.sqrt(2)
    return xi


def Q_operator(O):
    """Q = X (x) O acting on the (2N+1)-qubit second register: [[0,O],[O,0]] (R&S Eq.15)."""
    dL = O_super(O).shape[0]
    Q = np.zeros((2 * dL, 2 * dL), dtype=complex)
    Osup = O_super(O)
    Q[:dL, dL:] = Osup
    Q[dL:, :dL] = Osup
    return Q


def O_super(O):
    """Superoperator O = I (x) O_hat acting on vectorized space (R&S Sec.4).
    Tr(O rho) = <I| O_super |rho>. With column stacking, vec(O rho) = (I kron O) vec(rho)."""
    return np.kron(I2, O)


def qpe_alpha0(phi, t, t0):
    """Leakage amplitude to the zero bin for a mode with M-eigenvalue phi (R&S Eq.12).
    Returns alpha0 (real, >=0 magnitude; sign/phase irrelevant for the |.|^2 analysis,
    but we keep the signed sinc-like value for coherent readout)."""
    varphi = phi * t0                     # dimensionless phase
    if abs(varphi) < 1e-14:
        return 1.0
    num = np.sin(np.pi * (2**t) * varphi)
    den = (2**t) * np.sin(np.pi * varphi)
    return num / den


def qpe_filtered_readout(h, t, t0, O):
    """
    Exact eigenmode-level simulation of QPE filtering + readout.
    Returns R_O = <psi3| I(x)Q |psi3>  (unnormalized, before dividing by R_I).
    """
    M = embedding_M(h)
    phis, V = np.linalg.eigh(M)           # eigenvalues phi_j, eigenvectors columns
    xi = build_input_state(h)             # 8-vector on the M-register

    # expand xi in eigenmodes: xi = sum_j c_j |eta_j>
    c = V.conj().T @ xi                   # coefficients

    # apply QPE leakage: zero modes -> factor 1, nonzero modes -> alpha0
    alpha = np.array([qpe_alpha0(phi, t, t0) for phi in phis])
    psi3 = V @ (alpha * c)                # filtered (unnormalized) state on M-register

    # readout: R_O = <psi3| Q |psi3> with Q = X (x) O_super
    Q = Q_operator(O)
    R_O = np.vdot(psi3, Q @ psi3)
    return R_O, psi3


def qpe_estimate(h, t, t0, O):
    """Observable estimate <O> = R_O / R_I (R&S Eq.17)."""
    R_O, _ = qpe_filtered_readout(h, t, t0, O)
    R_I, _ = qpe_filtered_readout(h, t, t0, I2)
    return (R_O / R_I).real


if __name__ == "__main__":
    np.set_printoptions(precision=6, suppress=True)
    t0 = 1.0 / 5.0
    for h in [0.5, 1.0, 1.5]:
        rho_ss, _, _ = steady_state(h)
        sy_exact = np.trace(sy @ rho_ss).real
        sz_exact = np.trace(sz @ rho_ss).real
        print("=" * 70)
        print(f"h = {h}:  exact <sy>={sy_exact:.6f}  <sz>={sz_exact:.6f}")
        print(f"{'t':>3} {'<sy>_est':>12} {'<sz>_est':>12} {'err_y':>12} {'err_z':>12}")
        for t in range(1, 11):
            ey = qpe_estimate(h, t, t0, sy)
            ez = qpe_estimate(h, t, t0, sz)
            print(f"{t:>3} {ey:>12.6f} {ez:>12.6f} "
                  f"{abs(ey-sy_exact):>12.2e} {abs(ez-sz_exact):>12.2e}")
