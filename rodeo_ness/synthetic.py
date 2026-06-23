"""Synthetic embedding spectra for the spectral-density control experiment.

To separate the spectral separation ``g`` from the spectral *density*
(number of modes near zero) -- which the physical models do not allow,
since there both change with system size -- we construct spectra by hand:
fix the smallest mode at ``g`` and place additional modes above it.

The filters depend only on the embedding eigenvalues, so specifying the
list of nonzero modes is sufficient to evaluate the cost ratio.
"""

from __future__ import annotations

import numpy as np

from .filters import rodeo_worst_weight, qpe_worst_weight


def synthetic_spectrum(
    g: float, n_extra: int, top: float = 3.0
) -> np.ndarray:
    """Nonzero embedding modes: ``g`` plus ``n_extra`` modes up to ``top``.

    The smallest mode is exactly ``g``; the remaining modes are spread
    uniformly on ``[1.3 g, top]``.  Returns the array of positive modes
    (the embedding spectrum is symmetric, so the negatives are implied).
    """
    extra = np.linspace(1.3 * g, top, n_extra)
    return np.concatenate([[g], extra])


def synthetic_cost_ratio(
    g: float, n_extra: int, total_depth: float, t0: float = 0.2, top: float = 3.0
) -> float:
    """Cost ratio ``G_QPE / G_Rodeo`` for a synthetic spectrum at fixed depth."""
    phi = synthetic_spectrum(g, n_extra, top=top)
    return qpe_worst_weight(phi, t0, total_depth) / rodeo_worst_weight(
        phi, total_depth
    )
