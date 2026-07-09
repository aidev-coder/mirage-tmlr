"""
Figures, regenerated from results/*.json only (never hand-authored).

Figure 1 (planned, D-004): layer sweep — truth-probe AUROC and typicality-probe
AUROC per layer on the same axis. Co-peaking curves are the visual form of the
confound.
"""

from __future__ import annotations

import json
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent


def fig_layer_sweep(sweep_artifact: str | Path, out: str | Path | None = None):
    """sweep_artifact: results JSON with {'truth_sweep': [...], 'typicality_sweep': [...]}
    each a list of {layer, auroc, ci}."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    art = json.loads(Path(sweep_artifact).read_text())
    fig, ax = plt.subplots(figsize=(7, 4))
    for key, label in (("truth_sweep", "truth probe"), ("typicality_sweep", "typicality probe")):
        pts = art.get(key, [])
        if not pts:
            continue
        layers = [p["layer"] for p in pts]
        auc = [p["auroc"] for p in pts]
        lo = [p["ci"][0] for p in pts]
        hi = [p["ci"][1] for p in pts]
        ax.plot(layers, auc, marker="o", markersize=3, label=label)
        ax.fill_between(layers, lo, hi, alpha=0.2)
    ax.axhline(0.5, ls="--", lw=0.8, color="gray")
    ax.set_xlabel("layer")
    ax.set_ylabel("AUROC")
    ax.set_title(art.get("model", ""))
    ax.legend()
    fig.tight_layout()
    out = Path(out or _ROOT / "results" / "fig_layer_sweep.png")
    fig.savefig(out, dpi=200)
    return out
