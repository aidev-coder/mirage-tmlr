"""Apply decisions.md D-023 (040b283, amended acab56a) to the ladder artifacts."""
import glob
import json
import os

import numpy as np

os.chdir(r".")

runs = [json.load(open(f, encoding="utf-8")) for f in glob.glob("results/nladder_*_44b4126cba1c_*.json")]
print(f"models: {len(runs)}")
rungs = runs[0]["rungs"]


def fit_hinge(x, y):
    """Least-squares hinge: y = a + b*max(0, x-k), k scanned."""
    best = None
    for k in np.linspace(min(x), max(x), 120):
        h = np.maximum(0.0, x - k)
        if h.std() < 1e-9:
            continue
        A = np.column_stack([np.ones_like(h), h])
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        resid = y - A @ coef
        sse = float(resid @ resid)
        if best is None or sse < best[0]:
            best = (sse, float(k), float(coef[0]), float(coef[1]))
    return best[1], best[2], best[3]


def knee_ci(x, y, seed=0, B=2000):
    rng = np.random.default_rng(seed)
    ks = []
    for _ in range(B):
        i = rng.integers(0, len(x), len(x))
        if len(set(x[i])) < 4:
            continue
        ks.append(fit_hinge(x[i], y[i])[0])
    return float(np.percentile(ks, 2.5)), float(np.percentile(ks, 97.5))


print(f"\n{'rung':>6} {'n':>5} {'knee k(n)':>10} {'95% CI':>20} {'slope':>8}")
primary = {}
for rg in rungs:
    x, y = [], []
    for r in runs:
        rows = [q for q in r["rows"] if q["rung"] == rg]
        x.append(np.mean([q["recoverability_at_n"] for q in rows]))
        y.append(np.mean([q["off_diagonal_at_n"] for q in rows]))
    x, y = np.array(x), np.array(y)
    k, a, b = fit_hinge(x, y)
    lo, hi = knee_ci(x, y)
    primary[rg] = (k, lo, hi)
    print(f"{rg:>6} {rows[0]['n']:>5} {k:>10.3f}   [{lo:.3f}, {hi:.3f}]   {b:>8.3f}")

print(f"\n{'rung':>6} {'knee (SECONDARY: rec held at full 608)':>44}")
secondary = {}
for rg in rungs:
    x = np.array([r["recoverability_full_corpus"] for r in runs])
    y = np.array([np.mean([q["off_diagonal_at_n"] for q in r["rows"] if q["rung"] == rg])
                  for r in runs])
    k, a, b = fit_hinge(x, y)
    secondary[rg] = k
    print(f"{rg:>6} {k:>44.3f}")

k_lo_rung, k_hi_rung = primary[rungs[0]][0], primary[rungs[-1]][0]
shift = abs(k_hi_rung - k_lo_rung)
W = 0.80 - 0.57   # registered reference interval width
ks = [primary[r][0] for r in rungs]
diffs = np.diff(ks)
mono = max(int((diffs > 0).sum()), int((diffs < 0).sum()))

print("\n=== D-023 CRITERION ===")
print(f"k(100) = {k_lo_rung:.3f}   k(608) = {k_hi_rung:.3f}   |shift| = {shift:.3f}")
print(f"registered W (reference bootstrap width) = {W:.3f}")
print(f"monotone rungs = {mono} of 4 steps (need >=3)")
if shift > W and mono >= 3:
    v = "MIGRATED -> section 3.2 is REWRITTEN"
elif shift > 0.5 * W or mono < 3:
    v = "INDETERMINATE -> publish curve, claim nothing"
else:
    v = "NO MIGRATION -> representational reading survives, scope stated"
print(f"VERDICT: {v}")
print(f"\nsecondary design knees: {[round(secondary[r], 3) for r in rungs]}")
print(f"secondary shift = {abs(secondary[rungs[-1]] - secondary[rungs[0]]):.3f}")
