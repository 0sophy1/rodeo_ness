"""Reproduce the circuit-level validation figures (Appendix E) without a notebook.

Runs the compiled dynamic Rodeo filter on a noiseless ``AerSimulator`` and writes:
  * aer_convergence.pdf   (Fig. 8) -- post-selected <sigma_z> -> exact NESS, with
                                       empirical (run-to-run) error bars and a zoomed inset, + |error| panel
  * aer_cycle_saving.pdf  (Fig. 9) -- executed vs scheduled cycles, + mean saving
  * aer_validation_data.csv        -- the underlying per-n table (incl. SEs)

Usage:
    pip install "rodeo_ness[circuit]"
    python -m reproduce.circuits.run_aer_validation        # from the repo root
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from rodeo_ness import circuits as circ

# N_REPEATS > 1 -> the sweep is repeated with independent, well-separated seeds and
# the error bars become the *empirical* run-to-run std (publication-grade). Lower it
# (or SHOTS) to run faster; 10 x 40k is ~18 min on one core.
H, SHOTS, SEED, N_REPEATS = 0.5, 40000, 0, 10
N_VALUES = list(range(1, 13))
OUT = Path(__file__).resolve().parent


def main() -> None:
    records = circ.run_convergence_sweep(h=H, n_values=N_VALUES, shots=SHOTS,
                                         seed=SEED, n_repeats=N_REPEATS)
    exact = circ.exact_sz(H)

    with (OUT / "aer_validation_data.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        w.writeheader(); w.writerows(records)

    N   = np.array([r["n"] for r in records])
    sz  = np.array([r["sz_estimate"] for r in records])
    sze = np.array([r["sz_err"] for r in records])
    err = np.array([r["abs_error_sz"] for r in records])
    m   = N >= 2

    # Fig. 8 -- convergence with error bars + zoomed inset
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4.3))
    axA.axhline(exact, ls="--", color="#D55E00", lw=1.3, zorder=1)
    axA.plot(N[m], sz[m], "-", color="#9ecae1", lw=1.0, zorder=2)
    axA.errorbar(N[m], sz[m], yerr=sze[m], fmt="o", color="#0072B2", ms=5, capsize=3,
                 elinewidth=1.0, zorder=3)
    axA.annotate(r"exact $-1/3$", xy=(N[m].max(), exact), xytext=(-3, 6),
                 textcoords="offset points", fontsize=8.5, color="#D55E00", ha="right", va="bottom")
    axA.set_xlabel("number of cycles $n$"); axA.set_ylabel(r"$\langle\sigma_z\rangle$ estimate")
    axA.set_title("(a) convergence to exact NESS")
    axA.set_ylim(-1.32, 0.12); axA.set_xticks(N[m]); axA.grid(alpha=0.25)
    axin = axA.inset_axes([0.36, 0.13, 0.60, 0.40])
    mi = N >= 6
    axin.axhline(exact, ls="--", color="#D55E00", lw=1.1)
    axin.errorbar(N[mi], sz[mi], yerr=sze[mi], fmt="o-", color="#0072B2", ms=4, lw=1.0,
                  capsize=2.5, elinewidth=0.9)
    axin.set_ylim(-0.40, -0.26); axin.set_xticks(N[mi]); axin.tick_params(labelsize=7)
    axin.set_title("zoom: converged", fontsize=7.5); axin.grid(alpha=0.25)
    axA.indicate_inset_zoom(axin, edgecolor="0.6")
    floor = float(np.nanmedian(sze[m])); axB.set_ylim(5e-4, 2)
    axB.axhspan(5e-4, floor, color="0.86", lw=0, label=r"shot-noise $1\sigma$" + f" ($\\approx${floor:.1e})")
    axB.semilogy(N[m], np.clip(err[m], 5e-4, None), "s-", color="#0072B2", lw=1.3, ms=5)
    axB.set_xlabel("number of cycles $n$"); axB.set_ylabel(r"$|\langle\sigma_z\rangle+1/3|$")
    axB.set_title("(b) absolute error vs shot-noise floor")
    axB.set_xticks(N[m]); axB.grid(alpha=0.25, which="both"); axB.legend(loc="upper right", fontsize=8.5)
    fig.suptitle("Circuit-level validation on AerSimulator (single-spin, $h=0.5$, $g=1/2$)")
    fig.tight_layout(); fig.savefig(OUT / "aer_convergence.pdf", bbox_inches="tight"); plt.close(fig)

    # Fig. 9 -- early-abort saving: exact expected curve (extended) + Aer markers
    static = np.array([r["static_cycles"] for r in records], float)
    avgc   = np.array([r["avg_executed_cycles"] for r in records])
    saving = np.array([100 * r["cycle_saving"] for r in records])
    NN = np.arange(1, max(int(N.max()) + 1, 51))
    exq = np.array([circ.expected_executed_cycles(H, int(k)) for k in NN])
    exq_sav = (1 - exq / NN) * 100
    slope = (circ.expected_executed_cycles(H, 400) - circ.expected_executed_cycles(H, 200)) / 200.0
    asymp = (1 - slope) * 100
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(11.5, 4.4))
    axa.plot(NN, NN, "--", color="0.6", lw=1.3, label="static (all $n$ cycles)")
    axa.plot(NN, exq, "-", color="#0072B2", lw=2, label="expected (exact)")
    axa.plot(N, avgc, "o", color="#0072B2", ms=6, mfc="white", mec="#0072B2", label=f"AerSimulator ({N_REPEATS} reps)")
    axa.set_xlabel("scheduled cycles $n$"); axa.set_ylabel("executed cycle bodies / shot")
    axa.set_title("(a) executed vs scheduled cycles"); axa.legend(fontsize=9); axa.grid(alpha=0.25)
    axb.plot(NN, exq_sav, "-", color="#009E73", lw=2, label="expected (exact)")
    axb.plot(N, saving, "o", color="#009E73", ms=6, mfc="white", mec="#009E73", label=f"AerSimulator ({N_REPEATS} reps)")
    axb.axhline(asymp, ls="--", color="0.4", lw=1.3, label=f"$n\\to\\infty$ asymptote $\\approx{asymp:.0f}\\%$")
    axb.set_xlabel("scheduled cycles $n$"); axb.set_ylabel("mean cycle saving (%)")
    axb.set_title("(b) saving keeps rising toward its asymptote")
    axb.set_ylim(0, asymp + 4); axb.legend(fontsize=9, loc="lower right"); axb.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(OUT / "aer_cycle_saving.pdf", bbox_inches="tight"); plt.close(fig)

    print(f"wrote Figs 8, 9 + CSV to {OUT}")
    print(f"  n=8 <sz>={sz[N==8][0]:+.4f} +/- {sze[N==8][0]:.4f} (exact {exact:+.4f})")


if __name__ == "__main__":
    main()
