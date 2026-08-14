"""In-distribution against off-diagonal AUROC, 18 models.

Both markers are AUROC; no mean probe score appears on this figure.

    python -m src.figures.fig02_dumbbell
"""

from __future__ import annotations

from ._common import CORPUS, DOUBLE, OI, Sources, chance, need, plt, save, short, style


def rows(src: Sources):
    out = []
    for a in src.load_glob(f"stage3_saplma_*{CORPUS}*.json"):
        w = short(need(a, "model"))
        e = need(a, "per_layer", need(a, "headline_layer"), "adversarial", where=w)
        out.append({
            "model": w,
            "in_dist": need(e, "headline_heldout_diagonal", "auroc", where=w),  # in-distribution AUROC
            "off": need(e, "off_diagonal", "auroc", where=w),                   # off-diagonal AUROC
            "ci": need(e, "off_diagonal", "ci", where=w),                       # its 95% bootstrap CI
        })
    return sorted(out, key=lambda r: r["off"], reverse=True)


def main() -> int:
    style()
    src = Sources()
    data = rows(src)
    y = list(range(len(data)))[::-1]

    fig, ax = plt.subplots(figsize=(DOUBLE, 0.30 * len(data) + 1.9),
                           layout="constrained")
    chance(ax, "x")

    for yi, r in zip(y, data):
        ax.plot([r["off"], r["in_dist"]], [yi, yi], lw=0.8, color="#D3D7DC", zorder=1)
        ax.plot(r["ci"], [yi, yi], lw=3.0, color=OI["vermillion"], alpha=0.30,
                solid_capstyle="round", zorder=2)

    ax.scatter([r["in_dist"] for r in data], y, marker="D", s=26, zorder=3,
               color=OI["blue"], label="in-distribution AUROC")
    ax.scatter([r["off"] for r in data], y, marker="o", s=30, zorder=3,
               color=OI["vermillion"], label="off-diagonal AUROC (95% CI)")

    ax.set_yticks(y)
    ax.set_yticklabels([r["model"] for r in data])
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("AUROC")
    ax.set_title("The benchmark reports 0.76 to 1.00 for every probe;\n"
                 "off-diagonal AUROC ranges from 0.08 to 0.89", loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=3,
              frameon=False, fontsize=7.5)
    ax.margins(y=0.02)
    save(fig, "fig02_dumbbell", src, "In-distribution against off-diagonal AUROC")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
