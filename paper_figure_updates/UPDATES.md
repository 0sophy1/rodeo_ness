# Paper figure updates

Regenerated, cleaned-up versions of the 13 draft figures, produced by the per-figure
notebooks in `../docs/notebooks/`. Each is a **drop-in replacement**: same filename,
same `\label`, same physical content and underlying numbers — only the presentation is
improved.

Drop the PDFs into your figure directory and (optionally) copy the matching caption
blocks from `captions.tex`. The numbers are unchanged, so the surrounding text needs no
edits; the only caption changes are noted below.

## What changed, figure by figure

| Fig | File | Change | Caption edit? |
|----|------|--------|---------------|
| 1 | `fig_cost.pdf` | Panels placed **side-by-side** (were stacked); added fit lines and a light grid. Three curves (QPE, Rodeo Gauss., Rodeo det.) unchanged. | minor wording only |
| 2 | `fig_restart.pdf` | **Corrected** QPE overhead to exactly `1/P = 1.75` (was computed with leakage-inclusive `P`); panels side-by-side; legend repositioned to the empty mid-region. | none (matches existing caption) |
| 3 | `fig_obs_error.pdf` | Single clean panel; Gaussian schedule shown as **median + 10–90% band**. | minor wording only |
| 4 | `fig_cost_vs_g.pdf` | Cleaned legend (separate size/interaction keys); log y-axis grid. | none |
| 5 | `fig_observables.pdf` | **2×2 grid** (rows QPE/Rodeo, columns σ_y/σ_z), two-line row labels, convergence-window axes. Full-width `figure*`. | layout described in caption |
| 6 | `rodeo_dynamic_circuit.pdf` | **Now shows the conditional structure**: cycle 2 is nested in an `If(meas_0==0)` block, so a failed first cycle aborts the rest — the gate-level basis of restart. Matches the draft folded layout. | none |
| 7 | `rodeo_trotter_decomp.pdf` | **Added panel (b)**: the `XXX` term expanded into elementary gates (Hadamard basis change, CNOT ladder, `R_z`, uncompute ladder), matching the draft two-panel layout. Panel (a) nine Pauli terms unchanged. | none |
| 8 | `aer_convergence.pdf` | Panels side-by-side; consistent palette. | minor wording only |
| 9 | `aer_cycle_saving.pdf` | **Rebuilt as two panels** (executed vs scheduled depth; saving %). Saving now correctly **saturates** (~25% for this schedule) instead of growing to ~90%. | caption: ~35% → ~25% |
| 10 | `fig_collapse_decay_vs_g.pdf` | Added size legend; matched palette. | none |
| 11 | `fig_interacting_collapse.pdf` | Viridis-by-`g` curves in (a); matched markers in (b). | none |
| 12 | `fig_density_saturation.pdf` | Annotation of the saturation; log–log grid. | none |
| 13 | `fig_two_gaps_advantage.pdf` | Fit line `g ≈ 0.6 g_decay` labelled; colourbar by `N`. | none |

## How these were produced

Each figure comes from its own notebook, so the figure and its code travel together:

- **Group A** (Figs. 1, 2, 3, 5) — single-spin benchmark, generated from the original
  figure engine in `../reproduce/core/` (`model.py`, `qpe_filter.py`, `rodeo_filter.py`,
  `cost_compare.py`, `restart_cost.py`). QPE exponent fit reproduces 0.50.
- **Group B** (Figs. 4, 10, 11, 12, 13) — dissipative-TFIM separation sweep, generated
  from the `rodeo_ness` package.
- **Group C** (Figs. 6, 7, 8, 9) — circuit-level validation, generated from `rodeo_ness`
  + Qiskit/Aer.

To regenerate any figure, open the matching notebook in `../docs/notebooks/` and run all
cells; it writes the PDF next to itself.

## Note on consistency with the draft

The figures reproduce the physics of the current draft (`modified_VQE_5`): single-spin
gap `g = 1/2`, exact NESS `⟨σ_y⟩ = 2/3`, `⟨σ_z⟩ = -1/3`, the `g`-collapse of the cost
ratio, density saturation after ~6 modes, and `g ≈ 0.6 g_decay`. If you change a model
parameter in the manuscript, rerun the corresponding notebook so the figure and text stay
in sync.

## Latest update — appendix figure redesign + empirical Aer error bars

**Removed the "ratio vs g" triple redundancy.** Fig 4 (main text), Fig 10(b) and
Fig 11(b) were all the same `ratio vs g` scatter/collapse. Only the main-text Fig 4
keeps it; the appendix figures were each given a distinct message and graph type.

- **Fig 8 / Fig 9 (`aer_convergence`, `aer_cycle_saving`)** — error bars are now the
  **empirical run-to-run standard deviation** over repeated independent noiseless
  runs (mean ± std), instead of a single-run delta-method SE. Fig 8 adds a zoomed
  inset of the converged region. Reproduce/regenerate with
  `reproduce/circuits/standalone_aer_validation_empirical.py` (run locally; 20×40k
  ≈ 30–40 min). NOTE on Aer seeding: consecutive simulator seeds are correlated and
  collapse the run-to-run spread, so `run_convergence_sweep(..., n_repeats=k)` draws
  **well-separated** seeds from a generator seeded by `seed` (independent + reproducible).
- **Fig 10 (`fig_collapse_decay_vs_g`)** — panel (b) replaced the duplicate
  ratio-vs-g collapse with a **variance-explained bar**: single-variable $R^2$ of
  $\log_{10}(G_{\rm QPE}/G_{\rm Rodeo})$ is $g\,0.95 \gg \gamma\,0.73 > g_{\rm decay}\,0.65 > N\,0.17 > J\,{<}0.01$.
- **Fig 11 (`fig_interacting_collapse`)** — replaced the line+collapse pair with **two
  cohesive ratio-vs-$T$ fans ($J=0$ | $J=2$) sharing a $g$ colour bar**: the
  $g$-ordered family is the same with and without interaction.
- **Fig 12 (`fig_density_saturation`)** — decluttered; sparse region shaded, six-mode
  benchmark marked. Same message.
- **Fig 13 (`fig_two_gaps_advantage`)** — panel (b) fixed from a misleading twin
  log/linear axis (curves looked parallel) to a **single log axis with the
  exponential encoding gap shaded** ($4^N$ vs $2N{+}1$ qubits).

Updated drop-in captions for all of the above are appended in `captions.tex`.
Figure source: `reproduce/figures/build_appendix_figs.py` (Figs 10–13) and the
`Figure08`–`Figure13` notebooks.

### Follow-up revisions (Fig 11 and Fig 9)

- **Fig 11** redesigned again to a **2×2 grid over interaction strength** $J\in\{0,0.5,1,2\}$.
  In every panel the **same four separations** $g\in\{0.4,0.7,1.0,1.3\}$ are drawn (one
  colour each, shared legend), so the same-colour curve can be compared panel-to-panel —
  it barely moves, i.e. $g$ (not $J$) sets the advantage. The earlier "invariant under
  interaction" wording was dropped as too strong.
- **Fig 9 correction.** The cycle saving does **not** saturate near 35%. The expected
  saving is now computed **exactly on the statevector** (function
  `circuits.expected_executed_cycles`) and extended to large $n$; it rises monotonically
  and approaches a finite asymptote $\approx43\%$ (for $h=0.5$) very slowly. AerSimulator
  markers ($n\leq12$) validate the exact curve to $\sim10^{-3}$.

## Reorganised to match the final manuscript (main.pdf / supplementary.pdf)

Figure files and notebooks were re-tidied to the final figure list and numbering.

- **Fig 1 (`fig_cost`) restored to two panels** ((a) semi-log, (b) linear), matching the
  final draft. (The single-panel variant was reverted.)
- **Renumbered notebooks** to the manuscript scheme — main text `Figure01`–`Figure07`,
  supplement `FigureS1`–`FigureS4`:
  1 cost · 2 obs_error · 3 cost_vs_g · 4 observables · 5 collapse · 6 interacting ·
  7 density · S1 dynamic_circuit · S2 trotter · S3 aer_convergence · S4 aer_cycle_saving.
- **Dropped from the manuscript**, moved to `_unused/`: `fig_restart` and
  `fig_two_gaps_advantage` (with their notebooks).
- **`captions.tex` fully rewritten** to the main/supplement structure with corrected
  captions, and a **DISCREPANCIES block** flagging three places where the draft text
  still describes the old figures:
  1. Main Fig 5 caption + C.1 prose say "(b) collapse vs g" → it is now a variance bar.
  2. Main Fig 6 caption describes "(a)/(b)" → it is now a 2×2 grid over J.
  3. Supp Fig S4 caption + S6 prose say "saturates near 25%" → it rises toward 3/7≈43%.
