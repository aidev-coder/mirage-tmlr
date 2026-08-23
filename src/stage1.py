"""
Stage 1 — SAPLMA headline reproduction (the number the field reports).

Protocol, faithful to Azaria & Mitchell (2023):
  - data: the true/false statement CSVs (data/raw/fetch.py must have run)
  - probe: feedforward (256,128,64) on last-token hidden state, per layer
  - split: leave-one-topic-out — train on all topics but one, test on the
    held-out topic; repeat over topics; report per-topic and mean AUROC
  - layers: full sweep, never a cherry-picked best layer (D-004)

Gate ( Stage 1): mean held-out-topic AUROC at the literature's
reported layers lands roughly in 0.7-0.9. Below that, the reproduction is
broken and the audit cannot proceed.

Compute: extraction is batched on GPU and CACHED per topic to the activations
volume; the probe sweep runs on GPU (torch) and CHECKPOINTS per layer. A
cut-off resumes from the last cached shard / completed layer — it never
restarts (owner authorized a larger GPU + speed, overriding the §5 A10/L4
default; the torch probe is an accelerator with the same architecture as the
sklearn reference, cross-checked in the plumbing test). Set fast=False to fall
back to the reference sklearn probe.

Output: results/stage1_saplma_<model>_<date>.json
  {model, per_topic: {topic: [per-layer {layer, auroc, ci}]}, mean_by_layer,
   n_statements, provenance: "measured"}

The negation sets (neg_*) are excluded from the headline by design — they are
the known generalization-failure axis and are evaluated separately.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
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


# ── checkpoint helpers (all live on the activations volume) ────────────────

def _work_dir(substrate, sub_tag: str) -> Path:
    h = hashlib.sha256(substrate.model_id.encode()).hexdigest()[:12]
    w = Path(substrate.cache_dir) / "stage1" / f"{h}_{sub_tag}"
    w.mkdir(parents=True, exist_ok=True)
    return w


def _load_ckpt(path: Path) -> dict:
    if path.exists():
        return {int(k): v for k, v in json.loads(path.read_text()).items()}
    return {}


def _save_ckpt(path: Path, done: dict) -> None:
    path.write_text(json.dumps({str(k): v for k, v in done.items()}))


def _extract_or_load(substrate, texts, topic_idx, topics, work, batch_size,
                     commit_fn=None) -> np.ndarray:
    """Per-topic sharded extraction cache. Resumes: cached topics are loaded,
    only missing topics are (batched) extracted. Reassembled in original order."""
    parts = []
    for ti, topic in enumerate(topics):
        idx = np.flatnonzero(topic_idx == ti)
        shard = work / f"acts_{topic}.npy"
        if shard.exists():
            a = np.load(shard)
            if a.shape[0] == len(idx):
                print(f"  [cache] {topic}: loaded {a.shape}", flush=True)
                parts.append((idx, a))
                continue
            print(f"  [cache] {topic}: stale ({a.shape[0]} != {len(idx)}), re-extracting", flush=True)
        a = substrate.hidden_states_matrix(
            [texts[i] for i in idx], batch_size=batch_size, tag=topic)
        np.save(shard, a)
        if commit_fn:
            commit_fn()  # durable checkpoint: survives a container cut-off
        print(f"  [extract] {topic}: saved {a.shape}", flush=True)
        parts.append((idx, a))
    n = len(texts)
    L, d = parts[0][1].shape[1], parts[0][1].shape[2]
    H = np.zeros((n, L, d), dtype=parts[0][1].dtype)
    for idx, a in parts:
        H[idx] = a
    return H


def run(substrate, topics: list[str] | None = None, seed: int = 20260709,
        max_statements_per_topic: int | None = None, fast: bool = True,
        device: str | None = None, batch_size: int = 32,
        commit_fn=None) -> dict:
    """Leave-one-topic-out SAPLMA sweep on `substrate`, resumable + GPU-fast."""
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

    sub_tag = f"k{max_statements_per_topic}" if max_statements_per_topic else "full"
    work = _work_dir(substrate, sub_tag)

    H = _extract_or_load(substrate, texts, topic_idx, topics, work,
                         batch_size, commit_fn)

    if fast:
        from .probes.torch_mlp import layer_sweep_fast
    else:
        from .probes.saplma import layer_sweep

    per_topic: dict = {}
    for ti, topic in enumerate(topics):
        test_idx = np.flatnonzero(topic_idx == ti)
        train_idx = np.flatnonzero(topic_idx != ti)
        ckpt = work / f"probe_{topic}.json"
        if fast:
            done = _load_ckpt(ckpt)
            if done:
                print(f"  [resume] {topic}: {len(done)} layers already done", flush=True)

            def _persist(res, _ckpt=ckpt, _done=done):
                _done[res["layer"]] = res
                _save_ckpt(_ckpt, _done)
                if commit_fn:
                    commit_fn()

            curve = layer_sweep_fast(H, y, train_idx, test_idx, seed=seed,
                                     device=device, done_layers=done,
                                     on_layer=_persist)
        else:
            curve = layer_sweep(H, y, train_idx, test_idx, seed=seed)
        per_topic[topic] = curve
        best = max((c["auroc"] for c in curve if c["auroc"] is not None), default=None)
        print(f"  [probe] {topic}: best layer AUROC = {best}", flush=True)

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
        "probe_impl": "torch_gpu" if fast else "sklearn_cpu",
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
    ap.add_argument("--slow", action="store_true", help="use sklearn CPU probe")
    args = ap.parse_args()

    from .substrate import Substrate
    sub = Substrate(args.model)
    result = run(sub, max_statements_per_topic=args.max_per_topic, fast=not args.slow)

    short = args.model.split("/")[-1].lower()
    tag = f"_k{args.max_per_topic}" if args.max_per_topic else ""
    out = _ROOT / "results" / f"stage1_saplma_{short}{tag}_{date.today():%Y%m%d}.json"
    out.write_text(json.dumps(result, indent=2))
    print(result["gate_note"])
    print(f"artifact: {out}")


if __name__ == "__main__":
    main()
