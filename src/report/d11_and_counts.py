"""The conditions attached to D11 and to the "0 of 18" claim.

Three things the review left outstanding, all computable from committed artifacts:

  D11-clustered  the 0.453 correlation among endorsed probes is currently significant
                 under an item-level bootstrap, but the 64 pairs share 18 models and
                 clustering widened the pooled interval by half. Re-test it clustered.
  counts-with-CI every below-chance tally (48/98, 33/64, 25/55) is a point estimate.
                 The claim needs the count whose interval also excludes 0.5.
  per-model      "0 of 18 models fail under every architecture" replaced a fabricated
                 "10 of 18". It needs its quantitative form: per model, the best
                 off-diagonal across architectures with its interval.

    python -m src.report.d11_and_counts
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

_ROOT = Path(__file__).resolve().parent.parent.parent
CH = "44b4126cba1c"


def pairs_with_ci() -> list[dict]:
    """(in-dist, off, ci, probe, model). CI is present only where the artifact stores one."""
    out = []
    for f in glob.glob(str(_ROOT / "results" / f"stage3_saplma_*{CH}*.json")):
        a = json.loads(Path(f).read_text(encoding="utf-8"))
        e = a["per_layer"][a["headline_layer"]]["adversarial"]
        out.append({"in_dist": e["headline_heldout_diagonal"]["auroc"],
                    "off": e["off_diagonal"]["auroc"], "ci": e["off_diagonal"].get("ci"),
                    "probe": "saplma", "model": a["model"].split("/")[-1]})
    summ = _ROOT / "results" / f"extlayers_summary_{CH}.json"
    if summ.exists():
        for r in json.loads(summ.read_text(encoding="utf-8"))["at_two_layers"]:
            a = r.get("at_our_layer")
            if a:
                out.append({"in_dist": a["in_dist"], "off": a["off"], "ci": a.get("off_ci"),
                            "probe": r["probe"], "model": r["model"]})
    for f in glob.glob(str(_ROOT / "results" / f"drift_*{CH}*.json")):
        a = json.loads(Path(f).read_text(encoding="utf-8"))
        d = a["probes"]["drift"]
        out.append({"in_dist": d["in_dist"], "off": d["off"], "ci": d.get("off_ci"),
                    "probe": "drift", "model": a["model"].split("/")[-1]})
    return out


def main() -> int:
    P = pairs_with_ci()
    models = sorted({p["model"] for p in P})
    print(f"pairs {len(P)}  models {len(models)}  with stored CI: "
          f"{sum(1 for p in P if p['ci'])}\n")

    print("== D11 clustered ==")
    for floor in (0.0, 0.85, 0.90):
        sub = [p for p in P if p["in_dist"] >= floor]
        x = np.array([p["in_dist"] for p in sub]); y = np.array([p["off"] for p in sub])
        mm = np.array([p["model"] for p in sub])
        uniq = sorted(set(mm))
        rho = spearmanr(x, y)
        rng = np.random.default_rng(0)
        cl = []
        for _ in range(4000):
            pick = rng.choice(len(uniq), len(uniq), replace=True)
            idx = np.concatenate([np.flatnonzero(mm == uniq[j]) for j in pick])
            if len(set(x[idx])) > 2:
                cl.append(spearmanr(x[idx], y[idx]).statistic)
        lo, hi = np.percentile(cl, [2.5, 97.5])
        print(f"  floor {floor:.2f}: n={len(sub):3d}  rho {rho.statistic:+.3f} "
              f"(item-level p {rho.pvalue:.3f})  CLUSTERED 95% CI [{lo:+.3f}, {hi:+.3f}]  "
              f"{'excludes 0' if lo > 0 else 'CONTAINS 0'}")

    print("\n== below-chance counts, point estimate vs interval-excluding ==")
    for floor, label in ((0.0, "all"), (0.85, ">=0.85"), (0.90, ">=0.90")):
        sub = [p for p in P if p["in_dist"] >= floor]
        pt = sum(1 for p in sub if p["off"] < 0.5)
        haveci = [p for p in sub if p["ci"]]
        strict = sum(1 for p in haveci if p["ci"][1] < 0.5)
        print(f"  {label:7s} n={len(sub):3d}  point-estimate below chance {pt:3d}  |  "
              f"upper CI below 0.5: {strict:3d} of {len(haveci)} pairs carrying a CI")

    print("\n== per model: best off-diagonal across architectures ==")
    print(f"  {'model':24s} {'n arch':>6} {'best off':>9} {'CI':>18}  all-below-chance?")
    nfail = 0
    for m in sorted(models, key=lambda m: max(p["off"] for p in P if p["model"] == m)):
        ps = [p for p in P if p["model"] == m]
        best = max(ps, key=lambda p: p["off"])
        ci = best["ci"]
        allbad = all(p["off"] < 0.5 for p in ps)
        nfail += allbad
        cis = f"[{ci[0]:.3f}, {ci[1]:.3f}]" if ci else "not stored"
        print(f"  {m:24s} {len(ps):6d} {best['off']:9.3f} {cis:>18}  {'YES' if allbad else 'no'}")
    print(f"\n  models below chance under EVERY architecture: {nfail} of {len(models)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
