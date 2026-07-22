"""
Figures, regenerated from results/stage3_saplma_*.json only (never hand drawn).

fig_layer_sweep: off diagonal AUROC vs layer for each model, with the in
distribution curve for reference. The persistent gap across depth is the point.

fig_cell_scores: distribution of the fielded probe's P(true) per cell at each
model's headline layer. FT (fluent false) piling up near 1.0 is the mechanism.
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS = _ROOT / "results"
CELL_ORDER = ["TT", "TA", "FT", "FA"]
CELL_LABEL = {"TT": "true common", "TA": "true rare",
              "FT": "false fluent", "FA": "false odd"}


def _saplma_artifacts() -> list[dict]:
    arts = [json.loads(Path(f).read_text(encoding="utf-8"))
            for f in glob.glob(str(RESULTS / "stage3_saplma_*.json"))]
    return sorted(arts, key=lambda a: a["model"])


def fig_layer_sweep(out: str | Path | None = None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arts = _saplma_artifacts()
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for art in arts:
        L = [e["layer"] for e in art["per_layer"]]
        off = [e["adversarial"]["off_diagonal"]["auroc"] for e in art["per_layer"]]
        ind = [e["adversarial"]["headline_heldout_diagonal"]["auroc"] for e in art["per_layer"]]
        depth = [x / (len(L) - 1) for x in L]
        name = art["model"].split("/")[-1]
        line, = ax.plot(depth, off, marker="o", markersize=3, label=name)
        ax.plot(depth, ind, lw=0.8, ls=":", color=line.get_color(), alpha=0.7)
    ax.axhline(0.5, ls="--", lw=0.8, color="gray")
    ax.set_ylim(0.0, 1.03)
    ax.set_xlabel("relative depth")
    ax.set_ylabel("AUROC")
    ax.set_title("off diagonal (solid) vs in distribution (dotted)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = Path(out or RESULTS / "fig_layer_sweep.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def fig_cell_scores(out: str | Path | None = None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    arts = _saplma_artifacts()
    fig, axes = plt.subplots(1, len(arts), figsize=(4.2 * len(arts), 3.4), sharey=True)
    if len(arts) == 1:
        axes = [axes]
    for ax, art in zip(axes, arts):
        fh = art.get("fielded_scores_headline")
        if not fh:
            ax.set_title(art["model"].split("/")[-1] + " (no per item scores)")
            continue
        score = np.array(fh["score"])
        cell = np.array(fh["cell"])
        data = [score[cell == c] for c in CELL_ORDER]
        parts = ax.violinplot(data, showmedians=True, showextrema=False)
        for body in parts["bodies"]:
            body.set_alpha(0.6)
        ax.set_xticks(range(1, len(CELL_ORDER) + 1))
        ax.set_xticklabels([CELL_LABEL[c] for c in CELL_ORDER], rotation=20, fontsize=8)
        ax.axhline(0.5, ls="--", lw=0.8, color="gray")
        ax.set_title(art["model"].split("/")[-1] + f" (L{fh['layer']})", fontsize=9)
    axes[0].set_ylabel("fielded probe P(true)")
    fig.suptitle("the field's probe rates fluent falsehoods true", fontsize=10)
    fig.tight_layout()
    out = Path(out or RESULTS / "fig_cell_scores.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


if __name__ == "__main__":
    print(fig_layer_sweep())
    print(fig_cell_scores())
