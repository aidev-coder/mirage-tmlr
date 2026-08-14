"""The knee ladder, both registered designs.

Points and intervals only. No trend line and no verdict annotation: the registered test
(decisions.md D-023) returned indeterminate, and a line through these points would assert
the reading the registration refused.

    python -m src.figures.fig04_ladder
"""

from __future__ import annotations

import numpy as np

from ._common import CORPUS, OI, RESULTS, SINGLE, Sources, need, plt, save, short, style


def primary(src: Sources):
    """Registered design: both quantities measured at n."""
    lad = src.load_glob(f"knee_ladder_fit_{CORPUS}_*.json")[-1]
    rg = need(lad, "rungs", where="knee_ladder_fit")
    return ([need(r, "rung") for r in rg],
            [need(r, "knee") for r in rg],          # fitted knee at this rung
            [need(r, "knee_ci") for r in rg],       # its bootstrap interval
            need(lad, "W_registered", where="knee_ladder_fit"))


def secondary(src: Sources):
    """Recoverability held at the full-corpus estimate, off-diagonal measured at n.

    The stored ladder artifact carries the primary design only, so this series is fitted
    here from committed per-model rows using the same hinge search. It is the one fit this
    package performs, and it runs on artifact values.
    """
    runs = [src.load(p) for p in sorted(RESULTS.glob(f"nladdercf_*_{CORPUS}_*.json"))]
    if not runs:
        raise FileNotFoundError(f"no artifact matches results/nladdercf_*_{CORPUS}_*.json")
    rungs = need(runs[0], "rungs", where="nladdercf")
    x = np.array([need(r, "recoverability_full_corpus", where=short(need(r, "model")))
                  for r in runs])

    def fit(xv, yv):
        best = None
        for k in np.linspace(float(xv.min()), float(xv.max()), 120):
            h = np.maximum(0.0, xv - k)
            if h.std() < 1e-9:
                continue
            A = np.column_stack([np.ones_like(h), h])
            coef, *_ = np.linalg.lstsq(A, yv, rcond=None)
            res = yv - A @ coef
            sse = float(res @ res)
            if best is None or sse < best[0]:
                best = (sse, float(k))
        return best[1]

    knees, cis = [], []
    for rg in rungs:
        y = np.array([np.mean([need(q, "off_diagonal_at_n")
                               for q in need(r, "rows", where="nladdercf")
                               if need(q, "rung") == rg]) for r in runs])
        knees.append(fit(x, y))
        # The artifact stores no interval for this design, so one is bootstrapped over
        # models with a fixed seed. Both series then carry an interval; showing a bar on
        # one series and not the other would read as selective reporting.
        rng = np.random.default_rng(20260814)
        draws = []
        for _ in range(600):
            i = rng.integers(0, len(x), len(x))
            if len(set(x[i].tolist())) > 3:
                draws.append(fit(x[i], y[i]))
        cis.append((float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))))
    return rungs, knees, cis


def main() -> int:
    style()
    src = Sources()
    rungs, knees, cis, W = primary(src)
    rungs2, knees2, cis2 = secondary(src)

    fig, ax = plt.subplots(figsize=(SINGLE, 3.6), layout="constrained")
    top = knees[-1]
    ax.axhspan(top - W / 2, top + W / 2, color=OI["grey"], alpha=0.13, lw=0, zorder=0,
               label=f"registered bound W = {W:.3f}, centred on the 608 knee")

    ax.errorbar(rungs, knees,
                yerr=[[k - c[0] for k, c in zip(knees, cis)],
                      [c[1] - k for k, c in zip(knees, cis)]],
                fmt="o", ms=5, lw=0, elinewidth=1.1, capsize=2.5, color=OI["blue"],
                zorder=3, label="both quantities measured at n")
    ax.errorbar([r + 8 for r in rungs2], knees2,
                yerr=[[k - c[0] for k, c in zip(knees2, cis2)],
                      [c[1] - k for k, c in zip(knees2, cis2)]],
                fmt="s", ms=4.5, mfc="none", mew=1.1, lw=0, elinewidth=1.0, capsize=2.5,
                color=OI["vermillion"], zorder=3,
                label="recoverability held at full corpus (bootstrapped CI)")

    ax.set_xlabel("probe training set size n (items)")
    ax.set_ylabel("fitted knee (recoverability units)")
    ax.set_xticks(rungs)
    ax.set_xlim(min(rungs) - 40, max(rungs) + 40)
    ax.set_title("Fitted knee against training size", loc="left")
    ax.legend(loc="lower right", frameon=False, fontsize=7)
    save(fig, "fig04_ladder", src, "Fitted knee against training size")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
