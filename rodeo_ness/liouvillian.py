"""Liouvillian construction and Hermitian embedding for open-system NESS.

This module builds the vectorized Liouvillian ``L`` of a Markovian open
quantum system and the Hermitian embedding ``M = [[0, L], [L^dag, 0]]``
whose zero sector encodes the non-equilibrium steady state (NESS).

The conventions follow Ramusat & Savona, Quantum 5, 399 (2021):
column-stacking vectorization ``|X> = sum_jk X_jk |j> |k>`` and the
embedding of Eq. (4) there.  The dissipative transverse-field Ising chain
used throughout the paper is provided by :func:`tfim_liouvillian`.
"""

from __future__ import annotations

import numpy as np

# Single-qubit operators (computational Z basis)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
I2 = np.eye(2, dtype=complex)
SIGMA_MINUS = np.array([[0, 0], [1, 0]], dtype=complex)  # |down><up|


def _op_on_site(op: np.ndarray, site: int, n_sites: int) -> np.ndarray:
    """Embed a single-site operator into an ``n_sites``-qubit Hilbert space."""
    factors = [I2] * n_sites
    factors[site] = op
    out = factors[0]
    for f in factors[1:]:
        out = np.kron(out, f)
    return out


def vectorized_liouvillian(
    H: np.ndarray, jumps: list[np.ndarray]
) -> np.ndarray:
    """Return the vectorized Liouvillian matrix ``L`` (column-stacking).

    Implements Ramusat & Savona Eq. (2):

        L = -i (I (x) H - H^T (x) I)
            - 1/2 sum_j ( I (x) A_j^dag A_j + (A_j^dag A_j)^T (x) I
                          - 2 A_j^* (x) A_j ).

    Parameters
    ----------
    H : ndarray
        System Hamiltonian (Hermitian).
    jumps : list of ndarray
        Lindblad jump operators ``A_j``.

    Returns
    -------
    ndarray
        The ``d^2 x d^2`` Liouvillian matrix acting on the vectorized
        density matrix, where ``d`` is the Hilbert-space dimension.
    """
    dim = H.shape[0]
    Id = np.eye(dim, dtype=complex)
    L = -1j * (np.kron(Id, H) - np.kron(H.T, Id))
    for A in jumps:
        AdA = A.conj().T @ A
        L += -0.5 * (
            np.kron(Id, AdA)
            + np.kron(AdA.T, Id)
            - 2.0 * np.kron(A.conj(), A)
        )
    return L


def hermitian_embedding(L: np.ndarray) -> np.ndarray:
    """Return the Hermitian embedding ``M = [[0, L], [L^dag, 0]]``.

    The nonzero eigenvalues of ``M`` are ``+/- sigma_i(L)`` (the singular
    values of ``L``); its kernel is spanned by the two zero modes
    ``|0>|I>`` and ``|1>|rho_ss>``.
    """
    dim = L.shape[0]
    M = np.zeros((2 * dim, 2 * dim), dtype=complex)
    M[:dim, dim:] = L
    M[dim:, :dim] = L.conj().T
    return M


def tfim_liouvillian(
    N: int, J: float = 0.0, h: float = 1.0, gamma: float = 1.0
) -> np.ndarray:
    """Vectorized Liouvillian of the dissipative transverse-field Ising chain.

    System Hamiltonian (Ramusat & Savona Appendix A form, open chain):

        H = (J/4) sum_<j,k> Z_j Z_k + (h/2) sum_j X_j,

    with local relaxation jump operators ``A_j = sqrt(gamma) sigma_j^-``.
    The single-spin benchmark of the paper is ``N=1, J=0`` (any ``h``),
    for which the Liouvillian gap is ``gamma/2``.

    Parameters
    ----------
    N : int
        Number of spins.
    J : float
        Ising coupling strength.
    h : float
        Transverse-field strength.
    gamma : float
        Dissipation (relaxation) rate.

    Returns
    -------
    ndarray
        The vectorized Liouvillian ``L`` of dimension ``4^N x 4^N``.
    """
    H = np.zeros((2 ** N, 2 ** N), dtype=complex)
    for j in range(N):
        H += (h / 2.0) * _op_on_site(X, j, N)
    for j in range(N - 1):
        H += (J / 4.0) * (_op_on_site(Z, j, N) @ _op_on_site(Z, j + 1, N))
    jumps = [np.sqrt(gamma) * _op_on_site(SIGMA_MINUS, j, N) for j in range(N)]
    return vectorized_liouvillian(H, jumps)


def single_spin_liouvillian(h: float = 0.5) -> np.ndarray:
    """Vectorized Liouvillian of the single-spin benchmark (R&S model).

    ``H = h X``, jump operator ``sigma^-``.  Liouvillian gap is ``1/2``
    for all ``h``.  Exact NESS observables at ``h=0.5``:
    ``<sigma_y> = 2/3``, ``<sigma_z> = -1/3``, ``<sigma_x> = 0``.
    """
    return vectorized_liouvillian(h * X, [SIGMA_MINUS])


def spectral_separation(L: np.ndarray, tol: float = 1e-9) -> float:
    """Spectral separation ``g = min_{j != 0,1} |phi_j|`` of the embedding.

    Equivalently the smallest nonzero singular value of ``L``.
    """
    M = hermitian_embedding(L)
    evals = np.linalg.eigvalsh(M)
    nz = np.abs(evals[np.abs(evals) > tol])
    return float(nz.min())


def decay_rate(L: np.ndarray, tol: float = 1e-9) -> float:
    """Asymptotic decay rate ``g_decay = min_{lambda != 0} |Re lambda|``.

    This is the usual Liouvillian gap; it coincides with
    :func:`spectral_separation` for normal generators (and for the
    single-spin model) but differs in general.
    """
    evals = np.linalg.eigvals(L)
    re = np.abs(evals.real)
    nz = re[re > tol]
    return float(nz.min())


def steady_state(L: np.ndarray) -> np.ndarray:
    """Return the normalized NESS density matrix (right zero mode of ``L``)."""
    evals, evecs = np.linalg.eig(L)
    idx = int(np.argmin(np.abs(evals)))
    dim = int(round(np.sqrt(L.shape[0])))
    rho = evecs[:, idx].reshape(dim, dim)
    rho = rho / np.trace(rho)
    return rho
