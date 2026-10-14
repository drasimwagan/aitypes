"""Generate all figures and the numerical-check table for the paper."""
import json
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.optimize import brentq
from model import (analytic_curves, f_excess, P_ss, lineage_prior_vars,
                   crossover_approx, monte_carlo)

OUT = "../paper/figures"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
C = {"lineage": "#2a78d6", "individual": "#eb6834", "hive": "#1baf7a", "darwinian": "#eda100"}
LS = {"lineage": "-", "individual": "--", "hive": "-.", "darwinian": ":"}
MK = {"lineage": "o", "individual": "s", "hive": "^", "darwinian": "D"}
NAME = {"lineage": "Amnesic lineage (LLM-like)", "individual": "Consolidating individual (human-like)",
        "hive": "Hive learner", "darwinian": "Darwinian lineage"}

mpl.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "pdf.fonttype": 42,
})

P0 = dict(N=100, L=20, s2=1.0, r=1.0, rho=0.5, rhoL=0.5, tau=20, Lam=10)


def regret(curves, p):
    f0 = f_excess(0.0, p["s2"], p["r"], p["L"])
    return {k: v - f0 for k, v in curves.items()}


def exact_crossing(p, lo=1e-9, hi=1e3):
    def d(logQ):
        a = analytic_curves(np.array([10**logQ]), p)
        return a["lineage"][0] - a["individual"][0]
    return 10 ** brentq(d, np.log10(lo), np.log10(hi))


# ------------------------------------------------------------------ Fig 2
def fig_regret():
    p = P0
    sig2 = p["s2"] + p["r"] / p["L"]
    Qa = np.logspace(-6, 1, 300)
    an = regret(analytic_curves(Qa, p), p)
    Qm = np.logspace(-5, 1, 13)
    _, mc_reg = monte_carlo(Qm, p, E=30000, burn=10000, seed=7)
    Qx, Qx_apx = exact_crossing(p), crossover_approx(p)

    fig, ax = plt.subplots(figsize=(4.6, 3.1))
    ax.grid(True, which="major", color=GRID, lw=0.5)
    order = ["darwinian", "lineage", "individual", "hive"]
    for k in order:
        ax.plot(Qa / sig2, an[k], LS[k], color=C[k], lw=1.6, label=NAME[k])
        ax.plot(Qm / sig2, np.clip(mc_reg[k], 1e-7, None), MK[k], ms=4.2, mfc="white",
                mec=C[k], mew=1.1, ls="none")
    ax.axvline(Qx / sig2, color=INK2, lw=0.7, ls=(0, (2, 2)))
    ax.text(Qx / sig2 * 1.15, 2.2e-6, r"$Q^\star$", color=INK2, fontsize=8)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(1e-6, 12); ax.set_ylim(1e-6, 400)
    ax.set_xlabel(r"world drift per episode, $Q/\sigma^2$")
    ax.set_ylabel(r"regret vs. oracle, $\varepsilon_S(Q)-f(0)$")
    # direct labels at right
    ylab = {"darwinian": 200, "lineage": 11, "individual": 1.1, "hive": 0.32}
    short = {"darwinian": "Darwinian", "lineage": "LLM-like", "individual": "human-like", "hive": "hive"}
    for k in order:
        ax.text(14, ylab[k], short[k], color=INK, fontsize=7, va="center", ha="left")
    # identity is carried by direct labels at the right edge + line style; no legend box
    ax.text(1.4e-6, 2e2, "lines: closed form\nmarkers: Monte Carlo", fontsize=6.5, color=INK2, va="top")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_regret.pdf", bbox_inches="tight")
    plt.close(fig)

    # numerical check table (analytic vs MC, raw excess loss)
    mc_raw, _ = monte_carlo(Qm, p, E=30000, burn=10000, seed=11)
    an_m = analytic_curves(Qm, p)
    rows = []
    for i, Q in enumerate(Qm):
        rows.append({"Q": Q, **{f"{k}_an": float(an_m[k][i]) for k in an_m},
                     **{f"{k}_mc": float(mc_raw[k][i]) for k in an_m}})
    maxrel = max(abs(r[f"{k}_mc"] - r[f"{k}_an"]) / r[f"{k}_an"] for r in rows for k in an_m)
    return {"Q_star_exact": Qx, "Q_star_approx": Qx_apx, "max_rel_err_mc": maxrel, "rows": rows}


# ------------------------------------------------------------------ Fig 3
def div_cmap():
    blue = ["#104281", "#256abf", "#5598e7", "#9ec5f4"]
    red = ["#f4b6b5", "#ec7c7b", "#e34948", "#a8302f"]
    cols = blue + ["#f0efec"] + red
    return LinearSegmentedColormap.from_list("bluered", cols)


def lineage_vs_individual(Q, p):
    a = regret(analytic_curves(Q, p), p)
    return np.log10(a["lineage"] / a["individual"])


def fig_phase():
    base = dict(P0, N=10_000)
    sig2 = base["s2"] + base["r"] / base["L"]
    Q = np.logspace(-8, 0, 220)
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.8), sharex=True, layout="constrained")
    cmap, vmax = div_cmap(), 2.0

    # (a) staleness abar vs drift
    Lams = np.unique(np.round(np.logspace(0, 3.3, 120)).astype(int))
    Z, abars = [], []
    for Lam in Lams:
        p = dict(base, Lam=int(Lam), tau=10)
        Z.append(lineage_vs_individual(Q, p)); abars.append(Lam + 5.5)
    Z = np.array(Z); abars = np.array(abars)
    ax = axes[0]
    m = ax.pcolormesh(Q / sig2, abars, Z, cmap=cmap, vmin=-vmax, vmax=vmax, shading="auto", rasterized=True)
    ax.contour(Q / sig2, abars, Z, levels=[0], colors=INK, linewidths=1.0)
    apx = [crossover_approx(dict(base, Lam=int(L), tau=10)) / sig2 for L in Lams]
    ax.plot(apx, abars, color="white", lw=1.0, ls=(0, (3, 2)))
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$Q/\sigma^2$"); ax.set_ylabel(r"mean staleness $\bar a$ (episodes)")
    ax.set_title(r"(a) staleness vs. drift  ($N=10^4$)", loc="left")
    ax.text(2e-8, 9, "LLM-like better", color="white", fontsize=7)
    ax.text(4e-4, 900, "human-like\nbetter", color="white", fontsize=7)

    # (b) instances N vs drift at fixed staleness
    Ns = np.unique(np.round(np.logspace(0, 7, 120)).astype(int))
    Z2 = np.array([lineage_vs_individual(Q, dict(base, N=int(n), Lam=45, tau=10)) for n in Ns])
    ax = axes[1]
    ax.pcolormesh(Q / sig2, Ns, Z2, cmap=cmap, vmin=-vmax, vmax=vmax, shading="auto", rasterized=True)
    ax.contour(Q / sig2, Ns, Z2, levels=[0], colors=INK, linewidths=1.0)
    apx2 = np.array([crossover_approx(dict(base, N=int(n), Lam=45, tau=10)) / sig2 for n in Ns])
    ok = Ns * base["rhoL"] > base["rho"]
    ax.plot(apx2[ok], Ns[ok], color="white", lw=1.0, ls=(0, (3, 2)))
    lim = base["s2"] / 1.0  # asymptote sigma^2/(rho (abar-1)^2) in units of sigma^2
    ax.axvline(1 / (base["rho"] * (50.5 - 1) ** 2), color=INK2, lw=0.6, ls=":")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$Q/\sigma^2$"); ax.set_ylabel(r"instances pooled $N$")
    ax.set_title(r"(b) experience vs. drift  ($\bar a=50.5$)", loc="left")
    cb = fig.colorbar(m, ax=axes, shrink=0.95, pad=0.01)
    cb.set_label(r"$\log_{10}$ regret ratio  LLM-like / human-like", fontsize=7)
    cb.outline.set_visible(False)
    fig.savefig(f"{OUT}/fig_phase.pdf", bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ Fig 1
def fig_spectra():
    yr = 3.15e7
    rows = [
        ("Bacterium\n(with CRISPR)", [("signalling", 1e0, 1e2), ("gene reg.", 1e2, 1e4),
                                      ("genome / population", 1e5, 1e9)],
         ["solid", "solid"]),
        ("Ant colony", [("individual ant", 1e-1, 1e2), ("pheromone trails", 3e2, 3e4),
                        ("colony", 1e6, 15 * yr), ("evolution", 1e9, 1e12)],
         ["solid", "solid", "solid"]),
        ("Human", [("neural", 1e-3, 1e0), ("working mem.", 1e0, 3e1), ("sleep consol.", 1e4, 3e5),
                   ("skills & knowledge", 1e6, 2e9), ("culture", 3e9, 1e11), ("genes", 2e11, 1e13)],
         ["solid", "solid", "solid", "solid", "weak"]),
        ("Deployed LLM", [("forward pass", 1e-2, 1e0), ("context", 1e0, 1e4),
                          ("weights (training runs)", 3e6, 3e8)],
         ["solid", "lagged"]),
        ("Hive learner\n(hypothetical)", [("forward pass", 1e-2, 1e0), ("context", 1e0, 1e4),
                                          ("shared consol.", 1e4, 1e6), ("weights", 1e6, 3e8)],
         ["solid", "solid", "solid"]),
    ]
    fig, ax = plt.subplots(figsize=(6.5, 3.3))
    bar_c = "#2a78d6"
    g = 1.25  # log-space gap at bar ends
    for i, (name, levels, links) in enumerate(rows):
        y = len(rows) - 1 - i
        for (lab, a, b) in levels:
            ax.plot([a * g, b / g], [y, y], color=bar_c, lw=5, solid_capstyle="round")
            ax.text(np.sqrt(a * b), y + 0.2, lab, ha="center", va="bottom", fontsize=6.3, color=INK)
        for j, style in enumerate(links):
            yy = y - 0.18
            mid = np.sqrt(levels[j][1] * levels[j][2]), np.sqrt(levels[j + 1][1] * levels[j + 1][2])
            ls = {"solid": "-", "weak": ":", "lagged": "--"}[style]
            ax.annotate("", xy=(mid[1], yy), xytext=(mid[0], yy),
                        arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.8, ls=ls,
                                        connectionstyle="arc3,rad=0.25", shrinkA=2, shrinkB=2,
                                        mutation_scale=7))
        if name.startswith("Deployed"):
            ax.text(2e9, y, "context is erased after each episode;\nreaches weights only via logged data, months later",
                    ha="left", va="center", fontsize=6, color=INK2, style="italic")
        if name.startswith("Hive"):
            ax.text(2e9, y, r"$\times N$ instances write into one slow memory", fontsize=6.3,
                    color=INK2, va="center", style="italic")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows][::-1], fontsize=7.5)
    ax.set_xscale("log"); ax.set_xlim(3e-4, 3e14); ax.set_ylim(-0.6, len(rows) - 0.35)
    ticks = [1e-3, 1, 60, 3600, 86400, yr, 100 * yr, 1e4 * yr, 1e6 * yr]
    ax.set_xticks(ticks)
    ax.set_xticklabels(["1 ms", "1 s", "1 min", "1 h", "1 day", "1 yr", "100 yr", "10 kyr", "1 Myr"])
    ax.minorticks_off()
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.grid(True, axis="x", color=GRID, lw=0.5)
    ax.set_xlabel("timescale of adaptation (order of magnitude, illustrative)")
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=INK2, lw=0.9, ls="-"), Line2D([], [], color=INK2, lw=0.9, ls="--"),
         Line2D([], [], color=INK2, lw=0.9, ls=":")]
    ax.legend(h, [r"consolidates ($\rho>0$)", "lagged, lineage-level only", "weak / indirect"],
              loc="upper right", frameon=False, fontsize=6.5, bbox_to_anchor=(1.04, 1.07))
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_spectra.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    import os
    os.makedirs(OUT, exist_ok=True)
    fig_spectra()
    fig_phase()
    res = fig_regret()
    json.dump(res, open(f"{OUT}/numerical_check.json", "w"), indent=1)
    print("Q* exact", res["Q_star_exact"], "approx", res["Q_star_approx"],
          "max rel err MC", res["max_rel_err_mc"])
