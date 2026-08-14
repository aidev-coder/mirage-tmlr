"""The alignment spectrum, and what the reported number does across it.

x is the frequency read among true statements only, so truth is constant along it: whatever
separates the items is familiarity. Three training regimes sit at three places on that axis.
y is the number the field reports. It stays high everywhere.

Both axes are AUROC.

    python -m src.figures.fig08_spectrum
"""

from __future__ import annotations

import numpy as np

from ._common import (CORPUS, HARVEST, OI, SINGLE, Sources, need, plt, save, short, style)


def baseline(src: Sources):
    x, y = [], []
    for a in src.load_glob("pplbase_*_*.json"):
        w = short(need(a, "reference_model"))
        for split in ("authored", "harvested"):
            x.append(need(a, "splits", split, "frequency_read_true_only", "auroc", where=w))
            y.append(need(a, "splits", split, "diagonal", "auroc", where=w))
    return np.array(x), np.array(y)


def released(src: Sources):
    x, y = [], []
    for a in src.load_glob(f"transfer_*_{CORPUS}_20260814.json"):
        if need(a, "am_dir") != "azaria_mitchell_official":
            continue
        if a.get("topics") not in (None, "all"):
            continue
        w = short(need(a, "model"))
        x.append(need(a, "frequency_read_among_true_only", "auroc", where=w))
        y.append(need(a, "in_distribution_on_am", "auroc", where=w))
    return np.array(x), np.array(y)


def ours(src: Sources):
    fr = src.load_glob(f"freqread_true_only_{CORPUS}_*.json")[0]
    reads = {short(need(r, "model")): need(r, "frequency_read_true_only")
             for r in need(fr, "rows", where="freqread_true_only")}
    x, y = [], []
    for a in src.load_glob(f"stage3_saplma_*{CORPUS}*.json"):
        w = short(need(a, "model"))
        if w not in reads:
            continue
        e = need(a, "per_layer", need(a, "headline_layer"), "adversarial", where=w)
        x.append(reads[w])
        y.append(need(e, "headline_heldout_diagonal", "auroc", where=w))
    return np.array(x), np.array(y)


def main() -> int:
    style()
    src = Sources()
    groups = [
        ("reads no hidden states", *baseline(src), OI["orange"], "s"),
        ("trained on the released dataset", *released(src), OI["skyblue"], "^"),
        ("trained on our diagonal", *ours(src), OI["vermillion"], "o"),
    ]

    fig, ax = plt.subplots(figsize=(SINGLE, 3.3), layout="constrained")
    ax.axvline(0.5, ls=(0, (4, 3)), lw=0.8, color=OI["grey"], zorder=0,
               label="no frequency read")

    for label, x, y, colour, marker in groups:
        ax.scatter(x, y, s=26, color=colour, marker=marker, zorder=3,
                   label=f"{label}  ({x.min():.3f}-{x.max():.3f})")
        ax.plot([x.min(), x.max()], [np.median(y)] * 2, color=colour, lw=1.0, alpha=0.45,
                zorder=2)

    ax.set_xlabel("frequency read among true statements only (AUROC)")
    ax.set_ylabel("reported score, in-distribution AUROC")
    ax.set_xlim(0.45, 1.01)
    ax.set_ylim(0.6, 1.03)
    ax.set_title("The reported number barely moves across the spectrum", loc="left")
    ax.legend(loc="lower left", frameon=False, fontsize=7)
    save(fig, "fig08_spectrum", src, "Alignment spectrum")

    for label, x, y, *_ in groups:
        print(f"  {label}: frequency read {x.min():.3f}-{x.max():.3f}, "
              f"reported {y.min():.3f}-{y.max():.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
