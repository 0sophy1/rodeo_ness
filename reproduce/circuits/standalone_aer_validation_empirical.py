#!/usr/bin/env python3
r"""
Standalone reproduction of the Aer validation figures (Fig 8 + Fig 9) with
*EMPIRICAL* error bars from repeated noiseless runs.

Why repeated runs?  A single seeded run only gives an analytic (delta-method)
estimate of the shot-noise error.  Here we instead run the whole dynamic-Rodeo
sweep ``N_REPEATS`` times with independent, well-separated simulator seeds and
report, for each cycle count ``n``:

    * the mean estimate over the repeats          (markers / line)
    * the empirical run-to-run standard deviation (error bars)

i.e. the shot-noise spread is *measured*, not approximated.  One sweep produces
both the convergence data (Fig 8) and the early-abort / cycle-saving data
(Fig 9), so this single script regenerates both figures -- no need to run the
expensive sweep twice.

IMPORTANT (Aer seeding): consecutive simulator seeds (1, 2, 3, ...) produce
strongly *correlated* samples on AerSimulator, which collapses the run-to-run
spread.  ``run_convergence_sweep`` therefore draws well-separated seeds from a
generator seeded by ``SEED`` -- independent realisations, still reproducible.

Run:
    python standalone_aer_validation_empirical.py

Timing: 20 repeats x 40000 shots over n = 1..12 is ~30-40 min on one CPU core.
To go faster while testing, lower N_REPEATS and/or SHOTS at the top of the file.
The empirical std stabilises by ~10-15 repeats.
"""
import os
import sys
import csv

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# --- locate the rodeo_ness package (repo root is two levels above this file) ---
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))
from rodeo_ness import circuits as C  # noqa: E402

# ================================ CONFIG ================================
H         = 0.5                 # single-spin field; exact NESS <sigma_z> = -1/3
N_VALUES  = list(range(1, 13))  # number of Rodeo cycles
SHOTS     = 40000               # noiseless shots per run
N_REPEATS = 20                  # independent runs -> empirical error bars
SEED      = 0                   # seeds the generator that draws the sim seeds
OUT_DIR   = HERE
# =======================================================================

rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"],
                 "mathtext.fontset": "dejavuserif", "axes.linewidth": 0.8})

BLUE, RED = "#0072B2", "#d62728"


def run():
    exact = C.exact_sz(H)
    print(f"[standalone] {N_REPEATS} repeats x {SHOTS} shots, n={N_VALUES[0]}..{N_VALUES[-1]}, "
          f"exact <sz> = {exact:+.4f}")
    print("[standalone] this is the slow part; progress is per repeat inside the sweep...")
    recs = C.run_convergence_sweep(h=H, n_values=N_VALUES, shots=SHOTS,
                                   seed=SEED, n_repeats=N_REPEATS)
    return exact, recs


def write_csv(recs, exact):
    path = os.path.join(OUT_DIR, "aer_validation_empirical.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n", "sz_mean", "sz_std_emp", "abs_error_sz",
                    "avg_executed_cycles_mean", "avg_executed_cycles_std",
                    "cycle_saving_mean", "cycle_saving_std", "n_repeats", "shots"])
        for r in recs:
            w.writerow([r["n"], r["sz_estimate"], r["sz_err"], r["abs_error_sz"],
                        r["avg_executed_cycles"], r["avg_executed_err"],
                        r["cycle_saving"], r["cycle_saving_err"], r["n_repeats"], SHOTS])
    print("[standalone] wrote", path)


def plot_fig8(recs, exact):
    n  = np.array([r["n"] for r in recs], float)
    sz = np.array([r["sz_estimate"] for r in recs], float)
    sd = np.array([r["sz_err"] for r in recs], float)         # empirical std
    ae = np.array([r["abs_error_sz"] for r in recs], float)

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4.5))
    axA.errorbar(n, sz, yerr=sd, fmt="o-", color=BLUE, ms=6, lw=1.5, capsize=3,
                 mfc="white", mec=BLUE, ecolor=BLUE, label="Aer (mean $\\pm$ std)")
    axA.axhline(exact, ls="--", color=RED, lw=1.5, label=r"exact NESS $-1/3$")
    axA.set_xlabel("number of Rodeo cycles $n$")
    axA.set_ylabel(r"$\langle\hat\sigma_z\rangle$")
    axA.set_title(f"(a) convergence to the NESS ({N_REPEATS} reps, {SHOTS // 1000}k shots)")
    axA.legend(fontsize=9, loc="lower right")
    axA.grid(alpha=0.2)
    # inset: zoom n>=6 around -1/3
    axin = inset_axes(axA, width="52%", height="42%", loc="upper right", borderpad=1.1)
    msk = n >= 6
    axin.errorbar(n[msk], sz[msk], yerr=sd[msk], fmt="o-", color=BLUE, ms=4, lw=1.2,
                  capsize=2, mfc="white", mec=BLUE, ecolor=BLUE)
    axin.axhline(exact, ls="--", color=RED, lw=1.2)
    axin.set_ylim(exact - 0.06, exact + 0.06)
    axin.set_title(r"zoom: $n\geq6$", fontsize=8)
    axin.tick_params(labelsize=7)
    axin.grid(alpha=0.2)

    axB.semilogy(n, ae, "o-", color=BLUE, ms=6, lw=1.5)
    floor = float(np.nanmedian(sd[n >= 6]))
    axB.axhspan(1e-4, floor, color="0.85", zorder=0)
    axB.axhline(floor, ls=":", color="0.4", lw=1.2,
                label=f"empirical shot-noise floor $\\approx${floor:.3f}")
    axB.set_xlabel("number of Rodeo cycles $n$")
    axB.set_ylabel(r"$|\langle\hat\sigma_z\rangle-(-1/3)|$")
    axB.set_title("(b) absolute error falls to the shot-noise floor")
    axB.legend(fontsize=9, loc="upper right")
    axB.grid(alpha=0.2, which="both")
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "aer_convergence_empirical.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("[standalone] wrote", path)


def plot_fig9(recs):
    n     = np.array([r["n"] for r in recs], float)
    avg   = np.array([r["avg_executed_cycles"] for r in recs], float)
    sav   = np.array([r["cycle_saving"] for r in recs], float) * 100.0

    # exact expected curve (no sampling), extended past the Aer range, + asymptote
    NN = np.arange(1, max(int(n.max()) + 1, 51))
    ex = np.array([C.expected_executed_cycles(H, int(k)) for k in NN])
    ex_sav = (1 - ex / NN) * 100
    slope = (C.expected_executed_cycles(H, 400) - C.expected_executed_cycles(H, 200)) / 200.0
    asymp = (1 - slope) * 100

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.5, 4.6))
    axA.plot(NN, NN, "--", color="0.6", lw=1.3, label="static (all $n$ cycles)")
    axA.plot(NN, ex, "-", color=BLUE, lw=2, label="expected (exact)")
    axA.plot(n, avg, "o", color=BLUE, ms=6, mfc="white", mec=BLUE, label=f"AerSimulator ({N_REPEATS} reps)")
    axA.set_xlabel("number of scheduled cycles $n$")
    axA.set_ylabel("executed cycle bodies / shot")
    axA.set_title("(a) early abort: executed vs scheduled")
    axA.legend(fontsize=9, loc="upper left"); axA.grid(alpha=0.2)

    axB.plot(NN, ex_sav, "-", color="#009E73", lw=2, label="expected (exact)")
    axB.plot(n, sav, "o", color="#009E73", ms=6, mfc="white", mec="#009E73", label=f"AerSimulator ({N_REPEATS} reps)")
    axB.axhline(asymp, ls="--", color="0.4", lw=1.3, label=f"$n\\to\\infty$ asymptote $\\approx{asymp:.0f}\\%$")
    axB.set_xlabel("number of scheduled cycles $n$")
    axB.set_ylabel("mean cycle saving (%)")
    axB.set_title("(b) saving keeps rising, slowly approaching its asymptote")
    axB.set_ylim(0, asymp + 4); axB.legend(fontsize=9, loc="lower right"); axB.grid(alpha=0.2)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "aer_cycle_saving_empirical.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("[standalone] wrote", path)


if __name__ == "__main__":
    exact, recs = run()
    write_csv(recs, exact)
    plot_fig8(recs, exact)
    plot_fig9(recs)
    print("[standalone] done -- Fig 8 (convergence) and Fig 9 (cycle saving) regenerated "
          "with empirical error bars.")
