"""The D-batch statistics that need no GPU.

Four things the review asked for, all computed from committed artifacts:

  D9   model-clustered bootstrap for the 98-pair correlation. The pairs share 18 models,
       so an item-level bootstrap treats one phenomenon counted six times as six
       independent observations and reports an interval that is too narrow.
  D11  the same correlation restricted to pairs the benchmark itself would endorse,
       since nobody deploys a probe scoring 0.305 in-distribution.
  C8p  paired per-item bootstrap on probe-minus-perplexity. The unpaired comparison
       cannot say whether a +0.058 margin survives resampling the same items.
  GD   gendis discordance: per item, does the probe cross 0.5 upward while the model's
       own behaviour crosses downward? Scale-free, unlike comparing two means on two
       calibrations.

    python -m src.report.d_batch_local
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, spearmanr

from ..stats import auroc_with_ci  # noqa: F401  (kept for parity of provenance)

_ROOT = Path(__file__).resolve().parent.parent.parent
CH = "44b4126cba1c"


def _auroc(y: np.ndarray, s: np.ndarray) -> float:
    y = np.asarray(y, dtype=bool)
    if y.all() or (~y).all():
        return float("nan")
    r = np.argsort(np.argsort(s)) + 1.0
    n1, n0 = int(y.sum()), int((~y).sum())
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _pairs() -> list[tuple[float, float, str, str]]:
    out = []
    for f in glob.glob(str(_ROOT / "results" / f"stage3_saplma_*{CH}*.json")):
        a = json.loads(Path(f).read_text(encoding="utf-8"))
        e = a["per_layer"][a["headline_layer"]]["adversarial"]
        out.append((e["headline_heldout_diagonal"]["auroc"], e["off_diagonal"]["auroc"],
                    "saplma", a["model"].split("/")[-1]))
    summ = _ROOT / "results" / f"extlayers_summary_{CH}.json"
    if summ.exists():
        for r in json.loads(summ.read_text(encoding="utf-8"))["at_two_layers"]:
            if r.get("at_our_layer"):
                out.append((r["at_our_layer"]["in_dist"], r["at_our_layer"]["off"],
                            r["probe"], r["model"]))
    for f in glob.glob(str(_ROOT / "results" / f"drift_*{CH}*.json")):
        a = json.loads(Path(f).read_text(encoding="utf-8"))
        d = a["probes"]["drift"]
        out.append((d["in_dist"], d["off"], "drift", a["model"].split("/")[-1]))
    return out


def d9_clustered(pairs, seed=0, B=4000):
    x = np.array([p[0] for p in pairs])
    y = np.array([p[1] for p in pairs])
    models = np.array([p[3] for p in pairs])
    uniq = sorted(set(models))
    rho = spearmanr(x, y).statistic

    rng = np.random.default_rng(seed)
    item, clus = [], []
    for _ in range(B):
        i = rng.integers(0, len(x), len(x))
        if len(set(x[i])) > 2:
            item.append(spearmanr(x[i], y[i]).statistic)
        pick = rng.choice(len(uniq), len(uniq), replace=True)
        idx = np.concatenate([np.flatnonzero(models == uniq[j]) for j in pick])
        if len(set(x[idx])) > 2:
            clus.append(spearmanr(x[idx], y[idx]).statistic)

    print(f"\n== D9  clustered bootstrap ({len(pairs)} pairs, {len(uniq)} models) ==")
    print(f"  Spearman                      {rho:+.3f}")
    print(f"  item-level 95% CI (too narrow) [{np.percentile(item, 2.5):+.3f}, "
          f"{np.percentile(item, 97.5):+.3f}]  width {np.percentile(item,97.5)-np.percentile(item,2.5):.3f}")
    print(f"  MODEL-CLUSTERED 95% CI         [{np.percentile(clus, 2.5):+.3f}, "
          f"{np.percentile(clus, 97.5):+.3f}]  width {np.percentile(clus,97.5)-np.percentile(clus,2.5):.3f}")

    below = [p for p in pairs if p[1] < 0.5]
    per_model = {m: sum(1 for p in pairs if p[3] == m and p[1] < 0.5) for m in uniq}
    n_arch = {m: sum(1 for p in pairs if p[3] == m) for m in uniq}
    allfail = [m for m in uniq if per_model[m] == n_arch[m]]
    print(f"  pairs below chance {len(below)}/{len(pairs)}; "
          f"models failing under EVERY architecture {len(allfail)}/{len(uniq)}")


def d11_floor(pairs):
    print("\n== D11  restricted to probes the benchmark would endorse ==")
    for floor in (0.0, 0.85, 0.90):
        sub = [p for p in pairs if p[0] >= floor]
        x = np.array([p[0] for p in sub])
        y = np.array([p[1] for p in sub])
        r = spearmanr(x, y)
        print(f"  in-dist >= {floor:.2f}: n={len(sub):3d}  rho {r.statistic:+.3f}  "
              f"p {r.pvalue:.3f}  off-diag range [{y.min():.3f}, {y.max():.3f}]  "
              f"below chance {int((y < 0.5).sum())}")


def c8_paired(seed=0, B=4000):
    print("\n== C8p  paired per-item bootstrap, probe minus perplexity (off-diagonal) ==")
    for tag, ch in (("authored", CH), ("harvested", "d279c2cae5f4")):
        items = [json.loads(l) for l in
                 (_ROOT / "data" / "corpus" / f"mirage_2x2_v{ch}.jsonl").read_text(
                     encoding="utf-8").splitlines() if l.strip()]
        ppl = np.array([-it["typicality"]["reference_ppl"] for it in items])
        print(f"  -- {tag} (n={len(items)}) --")
        rows = []
        for f in sorted(glob.glob(str(_ROOT / "results" / f"stage3_saplma_*{ch}*.json"))):
            a = json.loads(Path(f).read_text(encoding="utf-8"))
            fs = a.get("fielded_scores_headline")
            if not fs or len(fs["score"]) != len(items):
                continue
            s = np.asarray(fs["score"], dtype=float)
            cell = np.asarray(fs["cell"])
            truth = np.asarray(fs["truth"], dtype=bool)
            off = np.flatnonzero(np.isin(cell, ["TA", "FT"]))
            d = _auroc(truth[off], s[off]) - _auroc(truth[off], ppl[off])
            rng = np.random.default_rng(seed)
            bs = []
            for _ in range(B):
                i = rng.choice(off, len(off), replace=True)
                if truth[i].all() or (~truth[i]).all():
                    continue
                bs.append(_auroc(truth[i], s[i]) - _auroc(truth[i], ppl[i]))
            lo, hi = np.percentile(bs, [2.5, 97.5])
            rows.append((a["model"].split("/")[-1], d, lo, hi))
        for m, d, lo, hi in sorted(rows, key=lambda r: -r[1]):
            verdict = "BEATS" if lo > 0 else ("loses" if hi < 0 else "tie")
            print(f"    {m:24s} delta {d:+.3f} [{lo:+.3f}, {hi:+.3f}]  {verdict}")
        beats = sum(1 for _, _, lo, _ in rows if lo > 0)
        print(f"    -> beats baseline with interval excluding zero: {beats} of {len(rows)}")


def gendis_discordance():
    print("\n== GD  per-item discordance on the models' own errors ==")
    print("     probe_p_true > 0.5 while behaviour_p_true < 0.5, on items the model got WRONG")
    for f in sorted(glob.glob(str(_ROOT / "results" / "gendis_*_cities_20260807.json"))):
        a = json.loads(Path(f).read_text(encoding="utf-8"))
        rec = [r for r in a["records"] if not r["correct"]]
        disc = [r for r in rec if r["probe_p_true"] > 0.5 and r["behavior_p_true"] < 0.5]
        n, k = len(rec), len(disc)
        ci = binomtest(k, n).proportion_ci() if n else None
        fr = [r["freq_log10"] for r in rec]
        print(f"  {a['model'].split('/')[-1]:24s} {k}/{n} = {k/n:.3f} "
              f"[{ci.low:.3f}, {ci.high:.3f}]   freq_log10 {min(fr):.2f}-{max(fr):.2f}")
    print("     composition: post-cache-fix items only, so every entity is RESOLVED and")
    print("     hence common; the contrast is measured on the common stratum alone.")
    print("     behaviour_p_true = the model's own affirmation probability for the claim.")


def main() -> int:
    pairs = _pairs()
    d9_clustered(pairs)
    d11_floor(pairs)
    c8_paired()
    gendis_discordance()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
