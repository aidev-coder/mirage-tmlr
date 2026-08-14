"""Persist the true-statements-only frequency read, per model.

This is the paper's opening statistic: strip every false statement, then ask the fielded
probe to separate common entities from rare ones. Truth is constant, so whatever separates
them is not truth.

It had been computed ad hoc and never written to an artifact, which meant the abstract's
range passed the checker only because those values happen to occur in unrelated runs. This
writes it once, from the per-item scores the Stage-3 artifacts already carry.

    python -m src.report.freqread_true_only
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np

from ..stats import auroc_with_ci

_ROOT = Path(__file__).resolve().parent.parent.parent
CORPUS = "44b4126cba1c"


def main() -> int:
    rows = []
    for f in sorted(glob.glob(str(_ROOT / "results" / f"stage3_saplma_*{CORPUS}*.json"))):
        a = json.loads(Path(f).read_text(encoding="utf-8"))
        fs = a.get("fielded_scores_headline")
        if not fs:
            continue
        s = np.asarray(fs["score"], dtype=float)
        cell = np.asarray(fs["cell"])
        keep = np.isin(cell, ["TT", "TA"])          # true statements only
        r = auroc_with_ci((cell[keep] == "TT"), s[keep], seed=0)
        rows.append({"model": a["model"], "layer": fs.get("layer"),
                     "n_true_items": int(keep.sum()),
                     "frequency_read_true_only": r["auroc"], "ci": r["ci"]})
    if not rows:
        raise RuntimeError("no stage3 artifact carried fielded_scores_headline")

    vals = [r["frequency_read_true_only"] for r in rows]
    out = {"experiment": "frequency_read_true_only",
           "note": "AUROC separating common from rare entities among statements that are "
                   "all true, using the diagonal-trained (fielded) probe. Truth is constant "
                   "across this contrast.",
           "corpus": f"mirage_2x2_v{CORPUS}.jsonl", "n_models": len(rows),
           "min": round(float(min(vals)), 4), "max": round(float(max(vals)), 4),
           "mean": round(float(np.mean(vals)), 4),
           "rows": sorted(rows, key=lambda r: -r["frequency_read_true_only"]),
           "provenance": "measured"}
    p = _ROOT / "results" / f"freqread_true_only_{CORPUS}_20260814.json"
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"{p.name}: {out['n_models']} models, {out['min']:.3f} to {out['max']:.3f}, "
          f"mean {out['mean']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
