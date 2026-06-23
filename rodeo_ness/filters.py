"""Rodeo and phase-estimation zero-sector filters.

This module implements the two filtering primitives compared in the paper,
at the level of the residual amplitudes that enter the cost analysis:

* Rodeo filter: residual factor ``beta_j = prod_l cos(phi_j t_l / 2)``
  for nonzero modes ``phi_j``, with a deterministic low-discrepancy
  (van der Corput) schedule or a Gaussian schedule.
* Phase-estimation filter: zero-bin leakage amplitude
  ``alpha_0^(j)`` for an ``m``-qubit phase register.

The worst-case residual *weight* (max over nonzero modes) is the quantity
bounded in the complexity table and used for the cost comparison.
"""

from __future__ import annotations

import numpy as np

from .liouvillian import hermitian_embedding


# --------------------------------------------------------------------------
# Schedules
# --------------------------------------------------------------------------
def van_der_corput(k: int, base: int = 2) -> float:
    """The ``k``-th term of the van der Corput low-discrepancy sequence."""
    x, f = 0.0, 1.0 / base
    while k > 0:
        x += (k % base) * f
        k //= base
        f /= base
    return x


def vdc_schedule(n_cycles: int, total_depth: float, base: int = 2) -> np.ndarray:
    """Deterministic schedule of ``n_cycles`` times summing to ``total_depth``.

    Times are spread by the van der Corput sequence (offset by 1/2 so that
    no time is zero) and rescaled so that ``sum_l t_l = total_depth``.
    """
    pat = np.array([0.5 + van_der_corput(k) for k in range(1, n_cycles + 1)])
    pat *= total_depth / pat.sum()
    return pat


# --------------------------------------------------------------------------
# Embedding spectrum helper
# --------------------------------------------------------------------------
def nonzero_modes(L: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    """Return the nonzero eigenvalues ``phi_j`` of the embedding ``M``."""
    M = hermitian_embedding(L)
    evals = np.linalg.eigvalsh(M)
    return evals[np.abs(evals) > tol]


# --------------------------------------------------------------------------
# Rodeo filter
# --------------------------------------------------------------------------
def rodeo_residual(phi: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Rodeo residual factors ``beta_j`` for modes ``phi`` and schedule ``times``.

    ``beta_j = prod_l cos(phi_j t_l / 2)``.
    """
    cos = np.cos(np.outer(phi, times) / 2.0)
    return np.prod(cos, axis=1)


def rodeo_worst_weight(
    phi: np.ndarray, total_depth: float, n_range=range(2, 50)
) -> float:
    """Best (over cycle count) worst-case Rodeo residual weight at fixed depth.

    For each number of cycles ``n`` in ``n_range`` a van der Corput schedule
    filling ``total_depth`` is built; the worst-case mode weight
    ``max_j |beta_j|^2`` is computed, and the minimum over ``n`` is returned.
    This is the fair, schedule-optimized Rodeo cost used in the paper.
    """
    best = 1.0
    for n in n_range:
        times = vdc_schedule(n, total_depth)
        weights = rodeo_residual(phi, times) ** 2
        best = min(best, float(weights.max()))
    return best


def rodeo_gaussian_expected_weight(
    phi: np.ndarray, n_cycles: int, t_rms: float
) -> np.ndarray:
    """Analytic expected weight ``E[|beta_j|^2]`` for a Gaussian schedule.

    ``E[|beta_j|^2] = ((1 + exp(-phi_j^2 t_rms^2 / 2)) / 2)^n`` (Appendix A).
    """
    single = (1.0 + np.exp(-(phi ** 2) * t_rms ** 2 / 2.0)) / 2.0
    return single ** n_cycles


# --------------------------------------------------------------------------
# Phase-estimation filter
# --------------------------------------------------------------------------
def qpe_leakage(phi: np.ndarray, t0: float, m: int) -> np.ndarray:
    """Zero-bin leakage weight ``|alpha_0^(j)|^2`` for an ``m``-qubit register.

    ``|alpha_0^(j)|^2 = sin^2(pi 2^m varphi_j) / (2^{2m} sin^2(pi varphi_j))``
    with the dimensionless phase ``varphi_j = phi_j t0 / (2 pi)``.
    """
    pj = phi * t0 / (2.0 * np.pi)
    num = np.sin(np.pi * 2 ** m * pj) ** 2
    den = np.clip(np.sin(np.pi * pj) ** 2, 1e-300, None)
    return num / den / (2 ** m) ** 2


def qpe_worst_weight(phi: np.ndarray, t0: float, total_depth: float) -> float:
    """Best (over register size) worst-case QPE leakage weight at fixed depth.

    Only registers whose largest controlled-evolution time
    ``t0 (2^m - 1)`` does not exceed ``total_depth`` are admissible.
    """
    best = 1.0
    for m in range(1, 22):
        if t0 * (2 ** m - 1) > total_depth:
            break
        weights = qpe_leakage(phi, t0, m)
        best = min(best, float(weights.max()))
    return best


# --------------------------------------------------------------------------
# Cost ratio
# --------------------------------------------------------------------------
def cost_ratio(
    L: np.ndarray, total_depth: float, t0: float = 0.2
) -> float:
    """Gate-cost ratio ``G_QPE / G_Rodeo`` at fixed controlled-evolution depth.

    Both filters share the same ``(2N+1)^k`` Suzuki-Trotter prefactor, so the
    gate-cost ratio equals the ratio of worst-case residual weights at the
    same depth.  A ratio greater than one means Rodeo is cheaper.
    """
    phi = nonzero_modes(L)
    return qpe_worst_weight(phi, t0, total_depth) / rodeo_worst_weight(
        phi, total_depth
    )
