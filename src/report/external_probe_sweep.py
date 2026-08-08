"""
Run an external probe across every model and test whether the recoverability
mechanism holds for an architecture that is not ours.

The paper's mechanism claim is that a probe falls back on typicality exactly to the
extent that truth is not linearly available in the representation it reads. Measured
on our own SAPLMA probe that gives r = 0.964, but recoverability and the honest score
both came from the same apparatus, so the correlation could in principle be a
property of our probe rather than of the representation.

This script breaks that circularity two ways:

  1. It correlates an EXTERNAL probe's honest off-diagonal AUROC against the
     recoverability measured by OUR all-cell probe. Recoverability is a property of
     the representation; if the mechanism is real it should predict the confounding
     of a probe that had no part in measuring it.
  2. It also computes the external probe's OWN recoverability — the same probe fit
     on all four cells instead of the diagonal — so the relation can be stated
     entirely within the external architecture.

Usage:
    python -m src.report.external_probe_sweep \
        --probe mirage_hardness.probes_external.geometry_of_truth_mmprobe:GeometryOfTruthMMProbe \
        --corpus data/corpus/mirage_2x2_v6206fe484650.jsonl \
        --acts-dir data/activations/local --tag mmprobe
"""
from __future__ import annotations

import argparse
import importlib
import json
import re
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))

from src.eval.adversarial_split import DIAGONAL, OFF_DIAGONAL, adversarial_split  # noqa: E402
from src.stats import auroc_with_ci  # noqa: E402


def _load_probe(spec: str):
    mod, _, attr = spec.partition(":")
    return getattr(importlib.import_module(mod), attr)


def _corpus(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _our_recoverability(model: str, corpus_name: str) -> float | None:
    """Recoverability as measured by OUR all-cell probe — a property of the
    representation, computed without the external probe's involvement."""
    for f in (_ROOT / "results").glob("stage3_saplma_*.json"):
        a = json.loads(f.read_text(encoding="utf-8"))
        if a.get("corpus") == corpus_name and a["model"].split("/")[-1].lower() == model.lower():
            e = a["per_layer"][a["headline_layer"]]
            return e["mediation_allcell"].get("truth_beta_partialled")
    return None


def run(probe_spec: str, corpus_path: Path, acts_dir: Path, seed: int = 0) -> list[dict]:
    factory = _load_probe(probe_spec)
    items = _corpus(corpus_path)
    truth = np.array([bool(it["truth"]) for it in items])
    cells = np.array([it["cell"] for it in items])
    chash = corpus_path.stem.split("_v")[-1]
    off_idx = np.flatnonzero(np.isin(cells, OFF_DIAGONAL))
    diag_idx = np.flatnonzero(np.isin(cells, DIAGONAL))

    rows = []
    for npy in sorted(acts_dir.glob(f"*_{chash}_L*.npy")):
        m = re.match(rf"^(.+)_{chash}_L(\d+)\.npy$", npy.name)
        if not m:
            continue
        model, layer = m.group(1), int(m.group(2))
        X = np.load(npy).astype(np.float64)
        if len(X) != len(items):
            print(f"  skip {npy.name}: {len(X)} rows vs corpus {len(items)}")
            continue

        # the field's recipe: train on the diagonal, evaluate honestly off-diagonal
        adv = adversarial_split(X, truth, cells, lambda: factory(seed=seed), seed=seed)

        # the same external probe trained fairly on all four cells — its OWN
        # recoverability, stated entirely within this architecture
        fair = factory(seed=seed).fit(X, truth)
        fair_off = auroc_with_ci(truth[off_idx], np.asarray(fair.score(X))[off_idx], seed=seed)

        # per-cell readout of the fielded probe
        fielded = factory(seed=seed).fit(X[diag_idx], truth[diag_idx])
        s = np.asarray(fielded.score(X))
        per_cell = {c: round(float((s[cells == c] > 0.5).mean()), 4)
                    for c in ("TT", "TA", "FT", "FA")}

        rows.append({
            "model": model, "layer": layer,
            "in_dist": adv["headline_heldout_diagonal"]["auroc"],
            "off": adv["off_diagonal"]["auroc"], "off_ci": adv["off_diagonal"]["ci"],
            "gap": adv["gap"]["gap"], "gap_ci": adv["gap"]["ci"],
            "gap_excludes_zero": adv["gap"].get("excludes_zero"),
            "external_own_recoverability_off_auroc": fair_off["auroc"],
            "our_recoverability_truth_beta": _our_recoverability(model, corpus_path.name),
            "per_cell_frac_called_true": per_cell,
        })
        print(f"  {model:24s} L{layer:<3d} in-dist {adv['headline_heldout_diagonal']['auroc']:.3f} "
              f"off {adv['off_diagonal']['auroc']:.3f} gap {adv['gap']['gap']:+.3f} "
              f"| fair-trained off {fair_off['auroc']:.3f}", flush=True)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--acts-dir", default="data/activations/local")
    ap.add_argument("--tag", default="external")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = run(args.probe, Path(args.corpus), Path(args.acts_dir), args.seed)
    if not rows:
        raise SystemExit("no activation files matched the corpus")

    out = {"probe": args.probe, "corpus": Path(args.corpus).name,
           "n_models": len(rows), "rows": rows, "provenance": "measured",
           "seed": args.seed}

    for key, label in (("our_recoverability_truth_beta", "our all-cell probe (representation property)"),
                       ("external_own_recoverability_off_auroc", "the external probe's own fair training")):
        pairs = [(r[key], r["off"]) for r in rows if r.get(key) is not None]
        if len(pairs) > 2:
            x = np.array([p[0] for p in pairs]); y = np.array([p[1] for p in pairs])
            r = float(np.corrcoef(x, y)[0, 1])
            out[f"correlation_{key}"] = {"r": round(r, 4), "n": len(pairs), "measured_by": label}
            print(f"\ncorrelation(recoverability [{label}], honest off-diagonal) "
                  f"= {r:.3f}  (n={len(pairs)} models)")

    dest = _ROOT / "results" / f"external_sweep_{args.tag}_{Path(args.corpus).stem.split('_v')[-1]}.json"
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"-> {dest}")


if __name__ == "__main__":
    main()
