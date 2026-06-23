# Documentation

This directory holds the figure-reproduction notebooks and the figures they generate.

## `notebooks/`

There is **one notebook per paper figure**, plus a quick-start tutorial. Each notebook
imports the relevant engine, recomputes the underlying quantities, explains the physics
in more depth than the paper caption, and writes its figure as a PDF next to itself.

Start with **`00_quickstart.ipynb`** — it introduces the model and the two filters, tours
the `rodeo_ness` API, and maps every figure to the model it uses and the comparison it
makes. Then open whichever figure you want.

Run a single notebook interactively with Jupyter, or execute them all non-interactively
from the repo root:

```bash
jupyter nbconvert --to notebook --execute --inplace docs/notebooks/*.ipynb
```

### Group A — single-spin benchmark (engine: `../reproduce/core/`)

| Notebook | Fig | What it produces |
|---|---|---|
| `Figure01_cost.ipynb`        | 1 | Power-law (QPE) vs logarithmic (Rodeo) depth vs target precision; fitted QPE exponent ~ 0.50. |
| `Figure02_restart.ipynb`     | 2 | Expected total depth with early-abort restarts; overhead (Rodeo ~ 1, QPE = 1/P). |
| `Figure03_obs_error.ipynb`   | 3 | Single-shot sigma_y error vs depth (QPE, Rodeo det., Rodeo Gauss. median + band). |
| `Figure05_observables.ipynb` | 5 | NESS recovery, 2x2 grid (QPE/Rodeo x sigma_y/sigma_z) over h in {0.5, 1.0, 1.5}. |

### Group B — separation sweep (engine: `rodeo_ness`)

| Notebook | Fig | What it produces |
|---|---|---|
| `Figure04_cost_vs_g.ipynb`   | 4  | Cost ratio collapses onto g and grows with g — the central result. |
| `Figure10_collapse.ipynb`    | 10 | Decay-rate scatter vs g-collapse. |
| `Figure11_interacting.ipynb` | 11 | Collapse survives interactions. |
| `Figure12_density.ipynb`     | 12 | Spectral-density saturation (synthetic spectra). |
| `Figure13_two_gaps.ipynb`    | 13 | g vs g_decay correlation; encoding origin of the quantum advantage. |

### Group C — circuit-level validation (engine: `rodeo_ness` + Qiskit)

| Notebook | Fig | What it produces |
|---|---|---|
| `Figure06_dynamic_circuit.ipynb`  | 6 | Compiled dynamic Rodeo circuit (measurement-conditioned cycles). |
| `Figure07_trotter.ipynb`          | 7 | Trotter decomposition into nine Pauli-term evolutions. |
| `Figure08_aer_convergence.ipynb`  | 8 | AerSimulator convergence to exact sigma_z = -1/3. |
| `Figure09_aer_cycle_saving.ipynb` | 9 | Gate-level early-abort saving (~35%). |

Group C requires the `circuit` extra (`pip install "rodeo_ness[circuit]"`); the
AerSimulator notebooks fall back to an exact-filter reference if Qiskit is absent.

## `figures/`

The PDFs produced by the notebooks. These are the same figures used in the paper; they
are fully reproducible from the notebooks. Improved versions with matching caption blocks
for the manuscript live in `../paper_figure_updates/`.
