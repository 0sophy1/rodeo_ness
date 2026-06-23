#!/usr/bin/env python3
"""Plot Fig 8 (convergence) and Fig 9 (cycle saving) from the accumulated Aer
checkpoint (aer_acc.json), using the empirical run-to-run std as error bars.
Writes the canonical-named PDFs into docs/figures, paper_figure_updates and
reproduce/circuits."""
import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

CKPT = "/home/claude/work/aer_acc.json"
ROOT = "/home/claude/work/repo/rodeo_ness"
sys.path.insert(0, ROOT)
from rodeo_ness import circuits as C  # noqa: E402
OUT_DIRS = [os.path.join(ROOT, "docs", "figures"),
            os.path.join(ROOT, "paper_figure_updates"),
            os.path.join(ROOT, "reproduce", "circuits")]
rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"],
                 "mathtext.fontset": "dejavuserif", "axes.linewidth": 0.8})
BLUE, RED, GREEN = "#0072B2", "#d62728", "#009E73"


def msd(a):
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    return (float(np.mean(a)) if a.size else np.nan,
            float(np.std(a, ddof=1)) if a.size > 1 else np.nan)


def save(fig, name):
    for d in OUT_DIRS:
        os.makedirs(d, exist_ok=True)
        fig.savefig(os.path.join(d, name), bbox_inches="tight")
    plt.close(fig)
    print("wrote", name, "to", len(OUT_DIRS), "dirs")


def load():
    d = json.load(open(CKPT))
    NV = d["NV"]; exact = d["exact"]; reps = d["completed"]
    n = np.array(NV, float)
    sz = np.array([msd(d["raw"][str(k)]["sz"])[0] for k in NV])
    szs = np.array([msd(d["raw"][str(k)]["sz"])[1] for k in NV])
    avg = np.array([msd(d["raw"][str(k)]["avg"])[0] for k in NV])
    avgs = np.array([msd(d["raw"][str(k)]["avg"])[1] for k in NV])
    sav = np.array([msd(d["raw"][str(k)]["sav"])[0] for k in NV])
    savs = np.array([msd(d["raw"][str(k)]["sav"])[1] for k in NV])
    ae = np.abs(sz - exact)
    return n, sz, szs, avg, avgs, sav, savs, ae, exact, reps, d["shots"]


def fig8(n, sz, szs, ae, exact, reps, shots):
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4.5))
    axA.errorbar(n, sz, yerr=szs, fmt="o-", color=BLUE, ms=6, lw=1.5, capsize=3,
                 mfc="white", mec=BLUE, ecolor=BLUE, label="Aer (mean $\\pm$ std)")
    axA.axhline(exact, ls="--", color=RED, lw=1.5, label=r"exact NESS $-1/3$")
    axA.set_xlabel("number of Rodeo cycles $n$"); axA.set_ylabel(r"$\langle\hat\sigma_z\rangle$")
    axA.set_title(f"(a) convergence to the NESS ({reps} reps, {shots // 1000}k shots)")
    axA.legend(fontsize=9, loc="lower right"); axA.grid(alpha=0.2)
    axin = inset_axes(axA, width="52%", height="42%", loc="upper right", borderpad=1.1)
    m = n >= 6
    axin.errorbar(n[m], sz[m], yerr=szs[m], fmt="o-", color=BLUE, ms=4, lw=1.2, capsize=2,
                  mfc="white", mec=BLUE, ecolor=BLUE)
    axin.axhline(exact, ls="--", color=RED, lw=1.2); axin.set_ylim(exact - 0.06, exact + 0.06)
    axin.set_title(r"zoom: $n\geq6$", fontsize=8); axin.tick_params(labelsize=7); axin.grid(alpha=0.2)
    axB.semilogy(n, ae, "o-", color=BLUE, ms=6, lw=1.5)
    floor = float(np.nanmedian(szs[n >= 6]))
    axB.axhspan(1e-4, floor, color="0.85", zorder=0)
    axB.axhline(floor, ls=":", color="0.4", lw=1.2, label=f"empirical shot-noise floor $\\approx${floor:.3f}")
    axB.set_xlabel("number of Rodeo cycles $n$"); axB.set_ylabel(r"$|\langle\hat\sigma_z\rangle-(-1/3)|$")
    axB.set_title("(b) absolute error falls to the shot-noise floor")
    axB.legend(fontsize=9, loc="upper right"); axB.grid(alpha=0.2, which="both")
    fig.tight_layout(); save(fig, "aer_convergence.pdf")


def fig9(n, avg, avgs, sav, savs, reps, h=0.5, n_max=50):
    # Exact expected curve (cheap, no sampling) extended to large n; the Aer means
    # (n<=12) validate it. The saving does NOT saturate at 35% -- it rises slowly
    # toward a finite asymptote (~43% for h=0.5).
    NN = np.arange(1, n_max + 1)
    ex = np.array([C.expected_executed_cycles(h, int(k)) for k in NN])
    ex_sav = (1 - ex / NN) * 100
    slope = (C.expected_executed_cycles(h, 400) - C.expected_executed_cycles(h, 200)) / 200.0
    asymp = (1 - slope) * 100

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.5, 4.6))
    axA.plot(NN, NN, "--", color="0.6", lw=1.3, label="static (all $n$ cycles)")
    axA.plot(NN, ex, "-", color=BLUE, lw=2, label="expected (exact)")
    axA.plot(n, avg, "o", color=BLUE, ms=6, mfc="white", mec=BLUE, label=f"AerSimulator ({reps} reps)")
    axA.set_xlabel("number of scheduled cycles $n$"); axA.set_ylabel("executed cycle bodies / shot")
    axA.set_title("(a) early abort: executed vs scheduled")
    axA.legend(fontsize=9, loc="upper left"); axA.grid(alpha=0.2)

    axB.plot(NN, ex_sav, "-", color=GREEN, lw=2, label="expected (exact)")
    axB.plot(n, sav * 100, "o", color=GREEN, ms=6, mfc="white", mec=GREEN, label=f"AerSimulator ({reps} reps)")
    axB.axhline(asymp, ls="--", color="0.4", lw=1.3, label=f"$n\\to\\infty$ asymptote $\\approx{asymp:.0f}\\%$")
    axB.set_xlabel("number of scheduled cycles $n$"); axB.set_ylabel("mean cycle saving (%)")
    axB.set_title("(b) saving keeps rising, slowly approaching its asymptote")
    axB.set_ylim(0, asymp + 4); axB.legend(fontsize=9, loc="lower right"); axB.grid(alpha=0.2)
    fig.tight_layout(); save(fig, "aer_cycle_saving.pdf")


if __name__ == "__main__":
    n, sz, szs, avg, avgs, sav, savs, ae, exact, reps, shots = load()
    print(f"plotting from {reps} reps @ {shots} shots; sz floor std ~ {np.nanmedian(szs[n>=6]):.4f}")
    fig8(n, sz, szs, ae, exact, reps, shots)
    fig9(n, avg, avgs, sav, savs, reps)
