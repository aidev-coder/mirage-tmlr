"""
Stage 2 orchestrator: candidate items -> typicality -> 2x2 cells -> 3 gates.

Runs the GPU-dependent scoring (reference perplexity, canary hidden states) and
returns the scored items + gate reports. finalize() (pure-python, enforces the
gates + D-006 signoff and writes the hash-versioned corpus) is called by the
caller ON THE LOCAL machine so the artifact lands in the repo, not the ephemeral
Modal container.

Cell assignment (D-002): typicality = perplexity under a canonical *cross-family*
reference LM; typical = bottom tercile, atypical = top tercile, MIDDLE TERCILE
DISCARDED a priori (§1.1). Canary hidden states + tokenizer come from a
representative probed substrate (per-substrate re-check happens at Stage 3).
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from . import corpus_build, corpus_gen
from .typicality import reference_perplexity

CANARY_LAYER_FRAC = 0.5  # mid-depth layer for the edit/fragmentation canaries


def _assign_cells(items: list[dict], ppl: np.ndarray) -> list[dict]:
    lo, hi = np.quantile(ppl, [1 / 3, 2 / 3])
    kept = []
    for it, p in zip(items, ppl):
        it["typicality"] = {"reference_ppl": float(p)}
        if p <= lo:
            band = "typical"
        elif p >= hi:
            band = "atypical"
        else:
            continue  # discard middle tercile (a priori — §1.1)
        it["cell"] = ("TT" if (it["truth"] and band == "typical") else
                      "TA" if it["truth"] else
                      "FT" if band == "typical" else "FA")
        kept.append(it)
    return kept


def _balance(items: list[dict], seed: int):
    by = defaultdict(list)
    for it in items:
        by[it["cell"]].append(it)
    n = min(len(v) for v in by.values()) if by else 0
    rng = np.random.default_rng(seed)
    out = []
    for v in by.values():
        for i in rng.choice(len(v), n, replace=False):
            out.append(v[int(i)])
    return out, {c: len(v) for c, v in by.items()}, n


def score_and_gate(reference_substrate, canary_substrate, edit_rate: float = 0.5,
                   seed: int = 20260712) -> dict:
    """GPU stage. Returns scored items + all three gate reports (no write)."""
    items = corpus_gen.build_candidate_items(edit_rate=edit_rate, seed=seed)
    texts = [it["text"] for it in items]
    print(f"[stage2] scoring reference ppl on {len(texts)} items "
          f"({reference_substrate.model_id})", flush=True)
    ppl = reference_perplexity(texts, reference_substrate)

    kept = _assign_cells(items, ppl)
    kept, raw_counts, per_cell_n = _balance(kept, seed)
    print(f"[stage2] cells (balanced to {per_cell_n}/cell): {raw_counts}", flush=True)

    crossing = corpus_build.verify_crossing(kept)

    L = int(canary_substrate.model.config.num_hidden_layers * CANARY_LAYER_FRAC)
    print(f"[stage2] canary hidden states at layer {L} "
          f"({canary_substrate.model_id})", flush=True)
    H = canary_substrate.hidden_states_matrix(
        [it["text"] for it in kept], batch_size=32)[:, L, :].astype(np.float32)
    edit_c = corpus_build.edit_canary(H, np.array([it["edited"] for it in kept]))
    feats = corpus_build.fragmentation_features(
        [it["text"] for it in kept], [it["entity"] for it in kept],
        canary_substrate.tokenizer)
    frag_c = corpus_build.fragmentation_canary(
        feats, np.array([it["truth"] for it in kept]))

    return {
        "items": kept,
        "meta": {"reference_model": reference_substrate.model_id,
                 "canary_model": canary_substrate.model_id, "canary_layer": L,
                 "per_cell_n": per_cell_n, "raw_counts": raw_counts,
                 "edit_rate": edit_rate, "seed": seed},
        "crossing": crossing, "edit_canary": edit_c, "fragmentation_canary": frag_c,
    }
