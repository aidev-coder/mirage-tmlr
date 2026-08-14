"""Layer saturation: the benchmark returns one number for layers worth very different things.

Main panel is the model with the widest saturated band; the three other saturating models
are drawn faint beneath it. The saturated set and its off-diagonal span are computed here
from the sweep, not taken from the prose, so the caption cannot drift from the figure.

Both series are AUROC.

    python -m src.figures.fig07_layers
"""

from __future__ import annotations

import numpy as np

from ._common import CORPUS, DOUBLE, OI, Sources, chance, need, plt, save, short, style

PROBE = "lr"          # gradient-trained logistic regression, the saturating case
TOL = 0.01            # "within 0.01 of its maximum in-distribution"


def curve(summary: dict, model: str):
    rows = need(summary, "curves", model, PROBE, where="extlayers_summary")
    rows = sorted(rows, key=lambda r: need(r, "layer"))
    return (np.array([need(r, "layer") for r in rows]),
            np.array([need(r, "in_dist") for r in rows]),   # in-distribution AUROC
            np.array([need(r, "off") for r in rows]))       # off-diagonal AUROC


def saturated(ind: np.ndarray, off: np.ndarray, tol: float = TOL):
    m = ind >= ind.max() - tol
    return m, int(m.sum()), float(off[m].min()), float(off[m].max())


def main() -> int:
    style()
    src = Sources()
    summary = src.load_glob(f"extlayers_summary_{CORPUS}.json")[0]
    models = [m for m in need(summary, "curves", where="extlayers_summary")
              if PROBE in summary["curves"][m]]

    scored = []
    for m in models:
        lay, ind, off = curve(summary, m)
        _, n, lo, hi = saturated(ind, off)
        scored.append((n, hi - lo, m))
    scored.sort(reverse=True)
    main_model = scored[0][2]
    others = [m for _, _, m in scored[1:4]]

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(DOUBLE, 4.6), height_ratios=[3, 1.15],
                                  sharex=False, layout="constrained")

    lay, ind, off = curve(summary, main_model)
    mask, n_sat, lo, hi = saturated(ind, off)

    ax.axhspan(ind.max() - TOL, min(1.0, ind.max()), color=OI["blue"], alpha=0.12, lw=0,
               label=f"within {TOL:.2f} of maximum in-distribution AUROC")
    for x in lay[mask]:
        ax.axvline(x, color=OI["blue"], alpha=0.07, lw=2.2, zorder=0)

    ax.plot(lay, ind, lw=1.6, color=OI["blue"], marker="o", ms=2.6,
            label="in-distribution AUROC")
    ax.plot(lay, off, lw=1.6, color=OI["vermillion"], marker="o", ms=2.6,
            label="off-diagonal AUROC")
    chance(ax, "y")

    # A vertical span at one x, not a line joining the extremes: joining them would read
    # as a trend, and the point is that these layers are indistinguishable to the benchmark.
    xb = float(lay.max()) + 1.6
    ax.annotate("", xy=(xb, lo), xytext=(xb, hi),
                arrowprops=dict(arrowstyle="<->", color="#3A4048", lw=1.0))
    for yv in (lo, hi):
        ax.plot([xb - 0.7, xb + 0.7], [yv, yv], color="#3A4048", lw=1.0)
    ax.text(xb + 1.2, (lo + hi) / 2, f"{hi - lo:.3f}", fontsize=7, color="#3A4048",
            ha="left", va="center", rotation=90)
    ax.set_xlim(-1.5, float(lay.max()) + 5.0)
    ax.text(0.985, 0.06,
            f"{n_sat} layers tie on the reported number;\n"
            f"off-diagonal across them spans {lo:.3f} to {hi:.3f}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5, color="#3A4048",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#D5D9DE", lw=0.6))

    ax.set_ylabel("AUROC")
    ax.set_ylim(0, 1.03)
    ax.set_xlabel("layer index")
    ax.set_title(f"{short(main_model)}: the benchmark cannot choose a layer", loc="left")
    ax.legend(loc="center left", bbox_to_anchor=(0.005, 0.42), frameon=False, fontsize=7.5)

    for m in others:
        l2, i2, o2 = curve(summary, m)
        msk, n2, lo2, hi2 = saturated(i2, o2)
        x = np.linspace(0, 1, len(l2))
        ax2.plot(x, i2, lw=1.0, color=OI["blue"], alpha=0.55)
        ax2.plot(x, o2, lw=1.0, color=OI["vermillion"], alpha=0.55)
        ax2.text(1.005, o2[-1], f"{short(m)}  ({n2} tied, {lo2:.2f}-{hi2:.2f})",
                 fontsize=6.2, color="#5A6068", va="center", transform=ax2.get_yaxis_transform()
                 if False else ax2.transData)
    chance(ax2, "y", label=False)
    ax2.set_ylim(0, 1.03)
    ax2.set_xlim(0, 1.55)
    ax2.set_xticks([])
    ax2.set_ylabel("AUROC")
    ax2.set_title("the other saturating models, depth normalised", loc="left", fontsize=8.5)

    save(fig, "fig07_layers", src, "Layer saturation")
    print(f"  main panel {short(main_model)}: {n_sat} tied layers, off-diagonal "
          f"{lo:.3f} to {hi:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
