"""Probes against the training-free baseline, one panel per corpus.

Positions carry the claim; no annotation declares a winner.

    python -m src.figures.fig05_bracket
"""

from __future__ import annotations

from ._common import (CORPUS, DOUBLE, HARVEST, OI, Sources, chance, need, plt, save,
                      short, style)


def probes(src: Sources, corpus_hash: str):
    out = []
    for a in src.load_glob(f"stage3_saplma_*{corpus_hash}*.json"):
        w = short(need(a, "model"))
        e = need(a, "per_layer", need(a, "headline_layer"), "adversarial", where=w)
        out.append((w, need(e, "off_diagonal", "auroc", where=w)))   # off-diagonal AUROC
    return sorted(out, key=lambda r: r[1])


def baselines(src: Sources, split: str):
    """Off-diagonal AUROC of reference perplexity, one value per reference model."""
    vals = {}
    for a in src.load_glob("pplbase_*_*.json"):
        ref = short(need(a, "reference_model"))
        vals[ref] = need(a, "splits", split, "off_diagonal", "auroc", where=ref)
    if not vals:
        raise KeyError(f"missing artifact key 'splits.{split}.off_diagonal.auroc'")
    return vals


def main() -> int:
    style()
    src = Sources()
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 3.6), sharey=True,
                             layout="constrained")

    for ax, (ch, split, name) in zip(axes, [(CORPUS, "authored", "authored corpus, n = 608"),
                                            (HARVEST, "harvested", "harvested corpus, n = 140")]):
        pr = probes(src, ch)
        base = baselines(src, split)
        lo, hi = min(base.values()), max(base.values())

        ax.axhspan(lo, hi, color=OI["orange"], alpha=0.22, lw=0, zorder=0,
                   label="perplexity baseline, two reference LMs")
        for v in base.values():
            ax.axhline(v, lw=1.0, color=OI["orange"], zorder=1)
        chance(ax, "y", label=(ax is axes[0]))

        ax.scatter(range(len(pr)), [p[1] for p in pr], s=26, color=OI["blue"], zorder=3,
                   label="probes (trained, hidden states)")
        ax.set_xticks([])
        ax.set_xlabel(name, fontsize=8)
        ax.set_ylim(0, 1.02)

    axes[0].set_ylabel("off-diagonal AUROC")
    axes[0].legend(loc="upper left", frameon=False, fontsize=7.5)
    for ax in axes:
        ax.margins(x=0.06)
    fig.suptitle("Probes against a detector that reads no hidden states\n"
                 "18 models per panel, ordered by off-diagonal AUROC",
                 x=0.01, ha="left", fontsize=9.5)
    save(fig, "fig05_bracket", src, "Probes against the training-free baseline")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
