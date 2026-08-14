"""Recoverability against off-diagonal AUROC, with the stored hinge overlaid.

The hinge is not refitted here; its parameters come from the artifact.

Two estimators exist and they are different fits, so the choice is explicit:

  same_sample (default)  x from mechanism_saplma rows, hinge from mechanism_saplma
                         shape.hinge. This is the fit the section 3.2 equation quotes,
                         knee 0.765.
  canonical              x = recoverability_crossfitted (D-024), y = off_diagonal_crossfitted,
                         hinge taken from the 608 rung of knee_ladder_fit, knee 0.659.
                         No artifact stores a hinge fitted on the cross-fitted pair at full
                         corpus, so the ladder's top rung supplies it — same estimator,
                         same items.

    python -m src.figures.fig03_hinge [--estimator same_sample|canonical]
"""

from __future__ import annotations

import argparse

import numpy as np

from ._common import (CORPUS, OI, RESULTS, SINGLE, Sources, chance, need, plt, save,
                      short, style)


def same_sample(src: Sources):
    m = src.load(RESULTS / f"mechanism_saplma_{CORPUS}.json")
    pts = [(need(r, "recoverability", where="mechanism.rows"),   # same-sample estimator
            need(r, "off", where="mechanism.rows"),
            short(need(r, "model", where="mechanism.rows")))
           for r in need(m, "rows", where="mechanism")]
    h = need(m, "shape", "hinge", where="mechanism")
    return pts, {
        "knee": need(h, "knee"), "knee_ci": need(h, "knee_ci"),
        "floor": need(h, "floor"), "slope": need(h, "slope"),
        "aic_hinge": need(h, "aic"),
        "aic_linear": need(m, "shape", "linear", "aic", where="mechanism"),
        "r": need(m, "correlation", "r", where="mechanism"),
        "r_ci": need(m, "correlation", "ci", where="mechanism"),
        "label": "same-sample estimator, as in the section 3.2 equation",
    }


def canonical(src: Sources):
    pts = []
    for a in src.load_glob(f"crossfit_*_{CORPUS}_*.json"):
        w = short(need(a, "model"))
        pts.append((need(a, "recoverability_crossfitted", where=w),   # canonical, D-024
                    need(a, "off_diagonal_crossfitted", where=w), w))
    lad = src.load_glob(f"knee_ladder_fit_{CORPUS}_*.json")[-1]
    top = next((r for r in need(lad, "rungs", where="ladder") if need(r, "rung") == 608), None)
    if top is None:
        raise KeyError("missing artifact key 'rungs[rung=608]' in knee_ladder_fit")
    x = np.array([p[0] for p in pts]); y = np.array([p[1] for p in pts])
    return pts, {
        "knee": need(top, "knee"), "knee_ci": need(top, "knee_ci"),
        "floor": None, "slope": need(top, "slope"),
        "aic_hinge": None, "aic_linear": None,
        "r": float(np.corrcoef(x, y)[0, 1]), "r_ci": None,
        "label": "cross-fitted estimator (canonical, D-024)",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--estimator", choices=["same_sample", "canonical"], default="same_sample")
    a = ap.parse_args()

    style()
    src = Sources()
    pts, fit = (same_sample if a.estimator == "same_sample" else canonical)(src)

    fig, ax = plt.subplots(figsize=(SINGLE, 4.0))
    chance(ax, "y")

    lo, hi = fit["knee_ci"]
    ax.axvspan(lo, hi, color=OI["skyblue"], alpha=0.16, lw=0, zorder=0, label="knee 95% CI")
    ax.axvline(fit["knee"], ls=(0, (4, 3)), lw=1.0, color=OI["blue"], zorder=1,
               label=f"knee = {fit['knee']:.3f}")

    xs = np.array([p[0] for p in pts]); ys = np.array([p[1] for p in pts])
    ax.scatter(xs, ys, s=26, color=OI["black"], zorder=3, label="model")

    if fit["floor"] is not None:
        g = np.linspace(float(xs.min()), float(xs.max()), 200)
        ax.plot(g, fit["floor"] + fit["slope"] * np.maximum(0.0, g - fit["knee"]),
                lw=1.2, color=OI["vermillion"], zorder=2, label="hinge fit (from artifact)")

    rng = np.random.default_rng(0)                    # fixed seed: reruns are identical
    for (x, yv, name) in pts:
        ax.annotate(name, (x, yv), (x + 0.012 + rng.uniform(0, 0.004),
                                    yv + rng.uniform(-0.013, 0.013)),
                    fontsize=6.0, color="#5A6068", annotation_clip=False)

    note = f"r = {fit['r']:.3f}"
    if fit["r_ci"]:
        note += f" [{fit['r_ci'][0]:.3f}, {fit['r_ci'][1]:.3f}]"
    if fit["aic_hinge"] is not None:
        note += f"\nAIC: hinge {fit['aic_hinge']:.2f}, linear {fit['aic_linear']:.2f}"
    ax.text(0.02, 0.97, note, transform=ax.transAxes, va="top", fontsize=7.5, color="#3A4048")

    ax.set_xlabel("truth recoverability (partial regression coefficient)")
    ax.set_ylabel("off-diagonal AUROC")
    ax.set_title(fit["label"], loc="left")
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right", frameon=False)
    save(fig, f"fig03_hinge_{a.estimator}", src, "Recoverability against off-diagonal AUROC")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
