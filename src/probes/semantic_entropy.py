"""
Semantic entropy (Farquhar et al. 2024) — the sampling-based BLACK-BOX reference
point in the probe bank. Sample K responses, cluster by bidirectional entailment,
compute entropy over cluster mass. High entropy = the model's answer distribution
is semantically dispersed = likely hallucination.

Clustering backends:
  nli    bidirectional-entailment via an NLI cross-encoder (faithful; GPU box)
  exact  normalized string match — a WEAK FALLBACK for plumbing tests only.
         Any result produced with it is tagged so it can never silently reach
         a table (the project's standing directive §1.2/§1.3).
"""

from __future__ import annotations

import re

import numpy as np


def _normalize(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()


def _cluster_exact(samples: list[str]) -> list[int]:
    seen: dict[str, int] = {}
    return [seen.setdefault(_normalize(s), len(seen)) for s in samples]


class NliClusterer:
    """Bidirectional-entailment clustering with a cross-encoder NLI model."""

    def __init__(self, model_id: str = "microsoft/deberta-large-mnli", device=None):
        from transformers import pipeline
        self.nli = pipeline("text-classification", model=model_id, device=device)

    def _entails(self, a: str, b: str) -> bool:
        r = self.nli({"text": a, "text_pair": b}, top_k=1)[0]
        return r["label"].upper().startswith("ENTAIL")

    def __call__(self, samples: list[str]) -> list[int]:
        labels = [-1] * len(samples)
        reps: list[tuple[int, str]] = []          # (cluster id, representative)
        for i, s in enumerate(samples):
            for cid, rep in reps:
                if self._entails(s, rep) and self._entails(rep, s):
                    labels[i] = cid
                    break
            if labels[i] == -1:
                cid = len(reps)
                reps.append((cid, s))
                labels[i] = cid
        return labels


def semantic_entropy(samples: list[str], clusterer=None) -> dict:
    """Entropy over semantic-cluster mass of K sampled responses."""
    fallback = clusterer is None
    labels = _cluster_exact(samples) if fallback else clusterer(samples)
    _, counts = np.unique(labels, return_counts=True)
    p = counts / counts.sum()
    return {
        "semantic_entropy": float(-(p * np.log(p)).sum()),
        "n_clusters": int(len(counts)),
        "n_samples": int(len(samples)),
        "clusterer": "exact_FALLBACK_not_for_reporting" if fallback else "nli",
    }
