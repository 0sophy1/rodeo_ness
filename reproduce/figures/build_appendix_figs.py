#!/usr/bin/env python3
"""Regenerate the separation figures of Appendix C (main-text Fig 5, 6, 7) as PDFs.

Writes each figure (canonical filename) into every directory in OUT_DIRS so the
paper's \\includegraphics paths pick up the new versions. Pure exact-diagonalization
+ matplotlib (no Aer), so it runs in a few seconds.

Final-draft figure map (function name kept for git history; see number below):
  fig10() -> main-text Fig 5  fig_collapse_decay_vs_g : (a) ratio vs decay rate g_decay
             (scatters, N-branches); (b) variance of log-ratio explained by each variable
             -> g (0.95) is the controlling variable
  fig11() -> main-text Fig 6  fig_interacting_collapse : 2x2 grid over J, four fixed
             separations g per panel (shared legend) -> g, not J, sets the advantage
  fig12() -> main-text Fig 7  fig_density_saturation   : density effect is subleading,
             saturates after a few modes
(The old fig13 / fig_two_gaps_advantage is dropped: it is not in the final manuscript.)
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import rcParams
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
import rodeo_ness as rn  # noqa: E402

OUT_DIRS = [os.path.join(ROOT, "docs", "figures"),
            os.path.join(ROOT, "paper_figure_updates")]
T, T0 = 15, 0.2
markers = {1: "o", 2: "s", 3: "^", 4: "D"}
Jcol = {0.0: "#0072B2", 1.0: "#D55E00", 2.0: "#009E73"}

rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"],
                 "mathtext.fontset": "dejavuserif", "axes.linewidth": 0.8})


def save(fig, name):
    for d in OUT_DIRS:
        os.makedirs(d, exist_ok=True)
        fig.savefig(os.path.join(d, name), bbox_inches="tight")
    plt.close(fig)
    print("wrote", name, "to", len(OUT_DIRS), "dirs")


def sweep_grid():
    g, gd, gam, Nv, Jv, lr = [], [], [], [], [], []
    for N in [1, 2, 3]:
        for J in [0.0, 1.0, 2.0]:
            for ga in np.linspace(0.6, 4.0, 9):
                L = rn.tfim_liouvillian(N, J=J, h=1.0, gamma=ga)
                g.append(rn.spectral_separation(L)); gd.append(rn.decay_rate(L)); gam.append(ga)
                Nv.append(N); Jv.append(J)
                lr.append(np.log10(rn.cost_ratio(L, total_depth=T, t0=T0)))
    return list(map(np.array, (g, gd, gam, Nv, Jv, lr)))


def variance_explained(gd, gam, Nv, Jv, g, lr):
    SStot = ((lr - lr.mean()) ** 2).sum()
    r2c = lambda x, d=4: 1 - (((lr - np.polyval(np.polyfit(x, lr, d), x)) ** 2).sum()) / SStot
    def r2g(x):
        p = np.zeros_like(lr)
        for v in np.unique(x):
            p[x == v] = lr[x == v].mean()
        return 1 - (((lr - p) ** 2).sum()) / SStot
    return {"spectral gap $g$": r2c(g), "dissipation $\\gamma$": r2c(gam),
            r"decay rate $g_{\rm decay}$": r2c(gd), "system size $N$": r2g(Nv),
            "interaction $J$": r2g(Jv)}


def fig10(g, gd, gam, Nv, Jv, lr):
    ratio = 10 ** lr
    R2 = variance_explained(gd, gam, Nv, Jv, g, lr)
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.5, 4.6))
    for N in [1, 2, 3]:
        for J in [0.0, 1.0, 2.0]:
            m = (Nv == N) & (Jv == J)
            axA.scatter(gd[m], ratio[m], marker=markers[N], color=Jcol[J], s=46,
                        alpha=0.78, edgecolor="k", linewidth=0.3)
    axA.axhline(1.0, ls=":", color="gray", lw=1.5); axA.set_yscale("log")
    axA.set_xlabel(r"decay rate $g_{\rm decay}=\min|\mathrm{Re}\,\lambda|$")
    axA.set_ylabel(r"$G_{\rm QPE}/G_{\rm Rodeo}$")
    axA.set_title("(a) vs decay rate: scatters, $N$-branches"); axA.grid(alpha=0.2, which="both")
    mh = [Line2D([], [], marker=markers[N], color="gray", ls="none", ms=7, label=f"$N={N}$") for N in [1, 2, 3]]
    ch = [Line2D([], [], marker="o", color=Jcol[J], ls="none", ms=7, label=f"$J={J:.0f}$") for J in [0.0, 1.0, 2.0]]
    la = axA.legend(handles=mh, loc="upper left", title="size", fontsize=8); axA.add_artist(la)
    axA.legend(handles=ch, loc="lower right", title="interaction", fontsize=8)
    labels = list(R2.keys()); vals = list(R2.values()); o = np.argsort(vals)
    labels = [labels[i] for i in o]; vals = [vals[i] for i in o]
    cols = ["#0072B2" if "gap $g$" in l else "#b0b0b0" for l in labels]
    axB.barh(range(len(labels)), vals, color=cols, edgecolor="k", linewidth=0.4, height=0.62)
    axB.set_yticks(range(len(labels))); axB.set_yticklabels(labels)
    for i, v in enumerate(vals):
        axB.text(v + 0.012, i, f"{v:.2f}", va="center", fontsize=9,
                 color="#0072B2" if "gap $g$" in labels[i] else "0.3")
    axB.set_xlim(0, 1.08); axB.axvline(1, ls=":", color="0.6", lw=0.8)
    axB.set_xlabel(r"variance of $\log_{10}(G_{\rm QPE}/G_{\rm Rodeo})$ explained ($R^2$)")
    axB.set_title("(b) the separation is the controlling variable"); axB.grid(alpha=0.25, axis="x")
    fig.tight_layout(); save(fig, "fig_collapse_decay_vs_g.pdf")
    return R2


def _gamma_for_g(J, g_target, N=2, h=1.0, grid=None):
    """Dissipation strength gamma that gives spectral separation g_target at this J."""
    if grid is None:
        grid = np.linspace(0.12, 7.0, 300)
    gs = np.array([rn.spectral_separation(rn.tfim_liouvillian(N, J=J, h=h, gamma=ga)) for ga in grid])
    if not np.all(np.diff(gs) >= -1e-9):
        order = np.argsort(gs); gs, grid = gs[order], grid[order]
    if g_target < gs.min() or g_target > gs.max():
        return None
    return float(np.interp(g_target, gs, grid))


def fig11():
    # 2x2 over interaction strength J; in every panel the SAME four separations g
    # are drawn (one colour each, fixed across panels), so the same-colour curve can
    # be compared panel-to-panel: it barely moves -> g, not J, sets the advantage.
    Js = [0.0, 0.5, 1.0, 2.0]
    g_targets = [0.4, 0.7, 1.0, 1.3]
    gcolors = ["#0072B2", "#009E73", "#E69F00", "#D55E00"]
    Tg = np.linspace(6, 30, 50)
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8), sharex=True, sharey=True)
    for ax, J in zip(axes.ravel(), Js):
        for gt, c in zip(g_targets, gcolors):
            ga = _gamma_for_g(J, gt)
            if ga is None:
                continue
            L = rn.tfim_liouvillian(2, J=J, h=1.0, gamma=ga)
            ax.semilogy(Tg, [rn.cost_ratio(L, total_depth=t, t0=T0) for t in Tg],
                        "-", color=c, lw=2.2, label=f"$g={gt:.1f}$")
        ax.axhline(1.0, ls=":", color="gray", lw=1.4)
        ax.set_title(f"$J={J:.1f}$", fontsize=11); ax.grid(alpha=0.2, which="both")
    for ax in axes[-1, :]:
        ax.set_xlabel("controlled-evolution depth $T$")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$G_{\rm QPE}/G_{\rm Rodeo}$")
    axes[0, 0].text(0.04, 0.95, "Rodeo cheaper", transform=axes[0, 0].transAxes,
                    fontsize=8.5, color="#555", va="top")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4,
               title="line colour = spectral separation $g$ (fixed across panels)",
               bbox_to_anchor=(0.5, -0.04), fontsize=10)
    fig.suptitle("Larger separation $g$ deepens the Rodeo advantage at each interaction strength $J$ "
                 "(TFIM, $N=2$)", y=0.995, fontsize=12)
    fig.tight_layout(rect=[0, 0.02, 1, 1])
    save(fig, "fig_interacting_collapse.pdf")


def fig12():
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    gcols = {0.5: "#0072B2", 0.7: "#D55E00", 0.9: "#009E73"}; modes = [1, 2, 3, 4, 6, 9, 14, 20, 40, 80, 150]
    for gg, c in gcols.items():
        nm = [ne + 1 for ne in modes]
        rt = [rn.synthetic_cost_ratio(gg, ne, total_depth=T, t0=T0) for ne in modes]
        ax.loglog(nm, rt, "o-", color=c, lw=2, ms=6.5, label=f"$g={gg}$")
    ax.axhline(1.0, ls=":", color="gray", lw=1.5)
    ax.axvspan(1, 7, color="0.93", zorder=0)
    ax.text(2.6, 1.25, "sparse spectra", fontsize=8.5, color="0.45", ha="center")
    ax.axvline(6, ls="--", color="0.5", lw=1.0)
    ax.text(6, 1.4e4, "single-spin\nbenchmark", fontsize=8, color="0.4", ha="center", va="top")
    ax.set_xlabel("number of embedding modes (at fixed $g$)")
    ax.set_ylabel(r"cost ratio $G_{\rm QPE}/G_{\rm Rodeo}$")
    ax.set_title("Spectral density is a subleading, saturating effect")
    ax.legend(title="separation $g$", loc="lower left"); ax.grid(alpha=0.2, which="both")
    ax.annotate("steep drop over the first\nfew modes, then saturates",
                xy=(9, rn.synthetic_cost_ratio(0.9, 8, total_depth=T, t0=T0)),
                xytext=(22, 700), fontsize=9, color="0.35", ha="center",
                arrowprops=dict(arrowstyle="->", color="0.5"))
    fig.tight_layout(); save(fig, "fig_density_saturation.pdf")


if __name__ == "__main__":
    g, gd, gam, Nv, Jv, lr = sweep_grid()
    R2 = fig10(g, gd, gam, Nv, Jv, lr)
    fig11(); fig12()
    print("\nvariance explained (main-text Fig 5b):",
          {k: round(v, 3) for k, v in sorted(R2.items(), key=lambda kv: -kv[1])})
