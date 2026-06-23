"""rodeo_ness: Rodeo-algorithm steady-state preparation for open quantum systems.

Reference implementation accompanying
"Complexity Analysis and Efficient Implementation of the Rodeo Algorithm
for Steady-State Preparation".

The package provides:

* :mod:`rodeo_ness.liouvillian` -- Liouvillian construction, Hermitian
  embedding, and spectral quantities (separation ``g``, decay rate).
* :mod:`rodeo_ness.filters` -- Rodeo and phase-estimation zero-sector
  filters, schedules, and the gate-cost ratio.
* :mod:`rodeo_ness.synthetic` -- synthetic embedding spectra for the
  spectral-density control experiment.
"""

from .liouvillian import (
    tfim_liouvillian,
    single_spin_liouvillian,
    hermitian_embedding,
    vectorized_liouvillian,
    spectral_separation,
    decay_rate,
    steady_state,
)
from .filters import (
    van_der_corput,
    vdc_schedule,
    nonzero_modes,
    rodeo_residual,
    rodeo_worst_weight,
    rodeo_gaussian_expected_weight,
    qpe_leakage,
    qpe_worst_weight,
    cost_ratio,
)
from .synthetic import synthetic_spectrum, synthetic_cost_ratio

__version__ = "0.1.0"

__all__ = [
    "tfim_liouvillian",
    "single_spin_liouvillian",
    "hermitian_embedding",
    "vectorized_liouvillian",
    "spectral_separation",
    "decay_rate",
    "steady_state",
    "van_der_corput",
    "vdc_schedule",
    "nonzero_modes",
    "rodeo_residual",
    "rodeo_worst_weight",
    "rodeo_gaussian_expected_weight",
    "qpe_leakage",
    "qpe_worst_weight",
    "cost_ratio",
    "synthetic_spectrum",
    "synthetic_cost_ratio",
]
