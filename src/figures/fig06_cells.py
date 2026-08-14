"""Per-cell mean probe scores, 18 models, authored corpus.

The colourbar is a mean probe score on a zero-to-one scale, not AUROC. Diverging map
centred at 0.5 so "reads true" and "reads false" separate visually.

    python -m src.figures.fig06_cells
"""

from __future__ import annotations

import numpy as np

from ._common import CORPUS, DOUBLE, RESULTS, Sources, need, plt, save, short, style

CELLS = ("TT", "TA", "FT", "FA")


def table(src: Sources):
    rows = []
    for a in src.load_glob(f"stage3_saplma_*{CORPUS}*.json"):
        w = short(need(a, "model"))
        e = need(a, "per_layer", need(a, "headline_layer"), where=w)
        cs = need(e, "fielded_cell_scores", where=w)
        rows.append({
            "model": w,
            # mean probe score per cell, 0-1 — fielded_cell_scores.<cell>.mean
            "vals": [need(cs, c, "mean", where=w) for c in CELLS],
            # recoverability, used only to order the rows
            "recov": need(e, "mediation_allcell", "truth_beta_partialled", where=w),
        })
    return sorted(rows, key=lambda r: r["recov"])


def main() -> int:
    style()
    src = Sources()
    rows = table(src)
    M = np.array([r["vals"] for r in rows])

    fig, ax = plt.subplots(figsize=(DOUBLE * 0.62, 0.30 * len(rows) + 1.6))
    im = ax.imshow(M, cmap="RdBu", vmin=0.0, vmax=1.0, aspect="auto")

    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=7,
                    color="#FFFFFF" if abs(v - 0.5) > 0.34 else "#1A1C21")

    ax.set_xticks(range(len(CELLS)))
    ax.set_xticklabels(["TT\ntrue · common", "TA\ntrue · rare",
                        "FT\nfalse · common", "FA\nfalse · odd"], fontsize=8)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r["model"] for r in rows])
    ax.set_title("Mean probe score by cell, models ordered by recoverability", loc="left")
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)

    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("mean probe score (0-1), not AUROC")
    cb.outline.set_visible(False)

    save(fig, "fig06_cells", src, "Mean probe score by cell")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
