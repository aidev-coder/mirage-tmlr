"""
MIRAGE hardness probe — the released constructive artifact (paper §5/§6).

Given a candidate truth x typicality corpus (and optionally a substrate for the
hidden-state canaries), reports whether it can support an UN-CONFOUNDED truth
claim, or whether it fails the way every attempt in this project failed. Any
future internal-state-probe paper that builds a decorrelation corpus should run
this and publish the verdict.

Four checks (thresholds in ../configs/corpus.yaml):
  1. crossing         typical vs atypical are genuinely separated on the axis.
  2. FT yield         the false-typical cell is actually populated (the cell that
                      exposes the confound). An empty/near-empty FT => the corpus
                      cannot test truth-vs-typicality; everything else is moot.
  2b. composition     no categorical metadata field (domain, source, template, ...)
                      that is VISIBLE IN THE TEXT can predict truth off-diagonal
                      after learning the association on the diagonal. MIRAGE's own
                      corpus failed this and we did not notice for three weeks: it
                      put 32% non-cities in diagonal-TRUE and 0% in diagonal-FALSE
                      while the off-diagonal reversed that, so a topic-reading probe
                      scored BELOW chance off-diagonal and we read it as a typicality
                      confound. This check is data-only and takes seconds. Run it.
  3. edit canary      (needs a substrate) a "was-edited" probe on hidden states is
                      near chance; else the false cell carries a generation
                      signature the truth probe can shortcut on.
  4. fragmentation    (needs a substrate) truth is not predictable from tokenization
     canary           features alone.

Corpus format: one JSON object per line with at least {truth, cell, edited,
entity, text, typicality:{entity_freq_log10, reference_ppl}}.

Usage:
  python mirage_hardness/run_check.py --corpus path.jsonl              # data-only checks
  # hidden-state canaries require a GPU substrate; run via Modal (see modal_app).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from src import corpus_build  # noqa: E402

FT_MIN_FRACTION = 0.15  # FT must be >= this share of a balanced 4-cell corpus (~0.25 ideal)


def ft_yield_check(items: list[dict]) -> dict:
    from collections import Counter
    c = Counter(it["cell"] for it in items)
    n = len(items)
    ft_frac = c.get("FT", 0) / max(n, 1)
    return {"gate": "ft_yield", "cell_counts": dict(c),
            "ft_fraction": round(ft_frac, 4), "threshold": FT_MIN_FRACTION,
            "pass": bool(ft_frac >= FT_MIN_FRACTION and c.get("FT", 0) >= 2)}


def run(items: list[dict], substrate=None, layer: int | None = None) -> dict:
    report = {"n_items": len(items), "checks": {}}
    report["checks"]["crossing"] = corpus_build.verify_crossing(items)
    report["checks"]["ft_yield"] = ft_yield_check(items)
    report["checks"]["composition"] = corpus_build.composition_canary(items)

    if substrate is not None:
        import numpy as np
        L = layer if layer is not None else int(
            substrate.model.config.num_hidden_layers * 0.5)
        H = substrate.hidden_states_matrix(
            [it["text"] for it in items], batch_size=32)[:, L, :].astype(np.float32)
        report["checks"]["edit_canary"] = corpus_build.edit_canary(
            H, np.array([it["edited"] for it in items]))
        feats = corpus_build.fragmentation_features(
            [it["text"] for it in items], [it["entity"] for it in items],
            substrate.tokenizer)
        report["checks"]["fragmentation_canary"] = corpus_build.fragmentation_canary(
            feats, np.array([it["truth"] for it in items]))
    else:
        report["checks"]["edit_canary"] = {"skipped": "needs substrate (GPU)"}
        report["checks"]["fragmentation_canary"] = {"skipped": "needs substrate (GPU)"}

    ran = [v for v in report["checks"].values() if "pass" in v]
    report["verdict"] = ("USABLE" if all(v["pass"] for v in ran) and substrate is not None
                         else "USABLE (data-only checks)" if all(v["pass"] for v in ran)
                         else "CONFOUNDED / UN-BUILDABLE")
    report["failed_checks"] = [k for k, v in report["checks"].items()
                               if "pass" in v and not v["pass"]]
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="MIRAGE hardness probe")
    ap.add_argument("--corpus", required=True, help="corpus .jsonl")
    args = ap.parse_args()
    items = [json.loads(line) for line in Path(args.corpus).read_text(
        encoding="utf-8").splitlines() if line.strip()]
    rep = run(items)
    print(json.dumps(rep, indent=2))
    print(f"\nVERDICT: {rep['verdict']}"
          + (f" | failed: {rep['failed_checks']}" if rep["failed_checks"] else ""))


if __name__ == "__main__":
    main()
