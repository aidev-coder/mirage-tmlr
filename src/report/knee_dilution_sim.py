"""Does measurement attenuation reproduce the knee drift the ladder showed?

EXPLORATORY by registration (decisions.md D-025). This cannot convert D-023's
INDETERMINATE verdict either way. Its only job is to say whether an identified
artifactual mechanism accounts for the observed drift.

Errors-in-variables makes a two-target prediction: noise in x drags the fitted knee
toward the data's centre AND flattens the slope. The ladder shows both moving together
(knee 0.545 -> 0.683, slope 2.08 -> 3.37 as n grows, i.e. both shrinking as n falls).
So take the full-corpus recoverability values, inject noise matched to each rung's
observed measurement spread, refit, and compare BOTH trajectories. Matching one and not
the other is not support.

    python -m src.report.knee_dilution_sim
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent


def fit_hinge(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Least-squares hinge y = a + b*max(0, x-k); returns (knee, slope)."""
    best = None
    for k in np.linspace(float(x.min()), float(x.max()), 120):
        h = np.maximum(0.0, x - k)
        if h.std() < 1e-9:
            continue
        A = np.column_stack([np.ones_like(h), h])
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        r = y - A @ coef
        sse = float(r @ r)
        if best is None or sse < best[0]:
            best = (sse, float(k), float(coef[1]))
    return best[1], best[2]


def main() -> int:
    runs = [json.loads(Path(f).read_text(encoding="utf-8"))
            for f in sorted(glob.glob(str(_ROOT / "results" / "nladder_*_44b4126cba1c_*.json")))]
    rungs = runs[0]["rungs"]
    x_full = np.array([r["recoverability_full_corpus"] for r in runs])

    print(f"models {len(runs)}   rungs {rungs}\n")
    print(f"{'rung':>6} {'noise SD':>9} | {'obs knee':>9} {'sim knee':>18} | "
          f"{'obs slope':>9} {'sim slope':>18}")

    rows = []
    for rg in rungs:
        # observed, primary design (both axes at n)
        xo = np.array([np.mean([q["recoverability_at_n"] for q in r["rows"] if q["rung"] == rg])
                       for r in runs])
        yo = np.array([np.mean([q["off_diagonal_at_n"] for q in r["rows"] if q["rung"] == rg])
                       for r in runs])
        k_obs, s_obs = fit_hinge(xo, yo)

        # measurement spread of recoverability at this rung, from the replicate subsamples
        sd = float(np.mean([np.std([q["recoverability_at_n"] for q in r["rows"] if q["rung"] == rg],
                                   ddof=1) for r in runs]))

        # simulate: full-precision x plus rung-matched noise, same y as observed at n
        rng = np.random.default_rng(20260814 + rg)
        ks, ss = [], []
        for _ in range(400):
            k, s = fit_hinge(x_full + rng.normal(0.0, sd, size=len(x_full)), yo)
            ks.append(k)
            ss.append(s)
        kq = np.percentile(ks, [2.5, 50, 97.5])
        sq = np.percentile(ss, [2.5, 50, 97.5])
        rows.append((rg, sd, k_obs, kq, s_obs, sq))
        print(f"{rg:>6} {sd:>9.3f} | {k_obs:>9.3f} {kq[1]:>7.3f} [{kq[0]:.3f},{kq[2]:.3f}] | "
              f"{s_obs:>9.3f} {sq[1]:>7.3f} [{sq[0]:.3f},{sq[2]:.3f}]")

    k_in = all(r[3][0] <= r[2] <= r[3][2] for r in rows)
    s_in = all(r[5][0] <= r[4] <= r[5][2] for r in rows)
    k_bad = [r[0] for r in rows if not (r[3][0] <= r[2] <= r[3][2])]
    s_bad = [r[0] for r in rows if not (r[5][0] <= r[4] <= r[5][2])]

    print("\n=== EXPLORATORY VERDICT (D-025: cannot move the registered result) ===")
    print(f"observed knee inside simulated interval at every rung : {k_in}"
          + ("" if k_in else f"  (misses at {k_bad})"))
    print(f"observed slope inside simulated interval at every rung: {s_in}"
          + ("" if s_in else f"  (misses at {s_bad})"))
    if k_in and s_in:
        print("BOTH targets reproduced -> attenuation accounts for the observed drift")
    elif k_in or s_in:
        print("ONE target reproduced -> not support; attenuation is not a sufficient account")
    else:
        print("NEITHER reproduced -> drift is not explained by x-measurement noise")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


def correlated_branch(seed: int = 20260814, draws: int = 500) -> int:
    """Does shared-item x-y noise hold the small-n slope up?

    The injected-noise model perturbs x alone. But recoverability and off-diagonal AUROC
    at a rung come from the SAME subsample, so their errors are correlated, and correlated
    x-y error biases a fitted slope upward — the counterweight the first simulation
    lacked. No correlation needs to be guessed: the replicate subsamples already carry it.
    Fit with each model's x and y taken from the SAME replicate (correlation intact), then
    from DIFFERENT replicates (correlation broken), and compare.

    EXPLORATORY (decisions.md D-025). Cannot move D-023's verdict.
    """
    runs = [json.loads(Path(f).read_text(encoding="utf-8"))
            for f in sorted(glob.glob(str(_ROOT / "results" / "nladder_*_44b4126cba1c_*.json")))]
    rungs = runs[0]["rungs"]
    rng = np.random.default_rng(seed)

    print(f"\n{'rung':>6} {'r(dx,dy)':>9} | {'paired slope':>22} {'broken slope':>22} {'delta':>7}")
    out = []
    for rg in rungs:
        X = np.array([[q["recoverability_at_n"] for q in r["rows"] if q["rung"] == rg] for r in runs])
        Y = np.array([[q["off_diagonal_at_n"] for q in r["rows"] if q["rung"] == rg] for r in runs])
        dx = (X - X.mean(axis=1, keepdims=True)).ravel()
        dy = (Y - Y.mean(axis=1, keepdims=True)).ravel()
        r_xy = float(np.corrcoef(dx, dy)[0, 1])

        nrep = X.shape[1]
        pa, br = [], []
        for _ in range(draws):
            i = rng.integers(0, nrep, len(runs))
            pa.append(fit_hinge(X[np.arange(len(runs)), i], Y[np.arange(len(runs)), i])[1])
            j = (i + rng.integers(1, nrep, len(runs))) % nrep      # guaranteed different rep
            br.append(fit_hinge(X[np.arange(len(runs)), i], Y[np.arange(len(runs)), j])[1])
        pq = np.percentile(pa, [2.5, 50, 97.5])
        bq = np.percentile(br, [2.5, 50, 97.5])
        out.append((rg, r_xy, pq, bq))
        print(f"{rg:>6} {r_xy:>9.3f} | {pq[1]:>7.3f} [{pq[0]:.3f},{pq[2]:.3f}] "
              f"{bq[1]:>7.3f} [{bq[0]:.3f},{bq[2]:.3f}] {pq[1]-bq[1]:>+7.3f}")

    infl = [o for o in out if o[2][1] > o[3][1]]
    print(f"\nrungs where intact pairing gives the higher slope: {len(infl)} of {len(out)}")
    small = [o for o in out if o[0] <= 300]
    print(f"mean delta at n<=300: {np.mean([o[2][1]-o[3][1] for o in small]):+.3f}   "
          f"at n>=450: {np.mean([o[2][1]-o[3][1] for o in out if o[0] >= 450]):+.3f}")
    return 0
