"""
Stage 1 — SAPLMA headline reproduction (the number the field reports).

Protocol, faithful to Azaria & Mitchell (2023):
  - data: the true/false statement CSVs (data/raw/fetch.py must have run)
  - probe: feedforward (256,128,64) on last-token hidden state, per layer
  - split: leave-one-topic-out — train on all topics but one, test on the
    held-out topic; repeat over topics; report per-topic and mean AUROC
  - layers: full sweep, never a cherry-picked best layer (D-004)

Gate (the project's standing directive §3 Stage 1): mean held-out-topic AUROC at the literature's
reported layers lands roughly in 0.7-0.9. Below that, the reproduction is
broken and the audit cannot proceed.

Output: results/stage1_saplma_<model>_<date>.json
  {model, per_topic: {topic: [per-layer {layer, auroc, ci}]}, mean_by_layer,
   n_statements, provenance: "measured"}

The negation sets (neg_*) are excluded from the headline by design — they are
the known generalization-failure axis and are evaluated separately.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import date
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
AM_DIR = _ROOT / "data" / "raw" / "azaria_mitchell"

HEADLINE_TOPICS = [
    "cities", "companies", "animals", "elements", "inventions", "facts",
    "generated", "capitals",
]


def load_statements(topics: list[str] | None = None) -> tuple[list[str], np.ndarray, np.ndarray]:
    """-> texts, labels (1=true), topic index array (aligned)."""
    topics = topics or HEADLINE_TOPICS
    texts, labels, topic_idx = [], [], []
    for ti, topic in enumerate(topics):
        path = AM_DIR / f"{topic}_true_false.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} missing — run `python data/raw/fetch.py` first")
        with open(path, encoding="utf-8-sig") as f:  # -sig: generated_*.csv has a BOM
            for row in csv.DictReader(f):
                texts.append(row["statement"])
                labels.append(int(row["label"]))
                topic_idx.append(ti)
    return texts, np.array(labels), np.array(topic_idx)


def run(substrate, topics: list[str] | None = None, seed: int = 20260709,
        max_statements_per_topic: int | None = None) -> dict:
    """Leave-one-topic-out SAPLMA sweep on `substrate` (src.substrate.Substrate)."""
    from .probes.saplma import layer_sweep

    topics = topics or HEADLINE_TOPICS
    texts, y, topic_idx = load_statements(topics)

    if max_statements_per_topic:  # smoke-run subsampling, recorded in output
        rng = np.random.default_rng(seed)
        keep = np.concatenate([
            rng.permutation(np.flatnonzero(topic_idx == t))[:max_statements_per_topic]
            for t in range(len(topics))])
        keep.sort()
        texts = [texts[i] for i in keep]
        y, topic_idx = y[keep], topic_idx[keep]

    H = substrate.batch_hidden_states(texts)   # [n, L+1, d], disk-cached

    per_topic: dict = {}
    for ti, topic in enumerate(topics):
        test_idx = np.flatnonzero(topic_idx == ti)
        train_idx = np.flatnonzero(topic_idx != ti)
        per_topic[topic] = layer_sweep(H, y, train_idx, test_idx, seed=seed)

    n_layers = H.shape[1]
    mean_by_layer = [
        {"layer": L,
         "mean_auroc": round(float(np.mean(
             [per_topic[t][L]["auroc"] for t in topics
              if per_topic[t][L]["auroc"] is not None])), 4)}
        for L in range(n_layers)]

    best_mean = max(m["mean_auroc"] for m in mean_by_layer)
    return {
        "detector": "saplma",
        "model": substrate.model_id,
        "protocol": "leave-one-topic-out",
        "topics": topics,
        "n_statements": int(len(texts)),
        "subsampled_per_topic": max_statements_per_topic,
        "per_topic": per_topic,
        "mean_by_layer": mean_by_layer,
        "gate_note": (
            f"best layer-mean held-out AUROC = {best_mean}; Stage-1 gate expects "
            "~0.7-0.9 (full curve reported; no layer cherry-picking downstream)"),
        "provenance": "measured",
        "seed": seed,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 1 SAPLMA headline reproduction")
    ap.add_argument("--model", required=True, help="substrate model id")
    ap.add_argument("--max-per-topic", type=int, default=None,
                    help="subsample for smoke runs (recorded in the artifact)")
    args = ap.parse_args()

    from .substrate import Substrate
    sub = Substrate(args.model)
    result = run(sub, max_statements_per_topic=args.max_per_topic)

    short = args.model.split("/")[-1].lower()
    out = _ROOT / "results" / f"stage1_saplma_{short}_{date.today():%Y%m%d}.json"
    out.write_text(json.dumps(result, indent=2))
    print(result["gate_note"])
    print(f"artifact: {out}")


if __name__ == "__main__":
    main()
