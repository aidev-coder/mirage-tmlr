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

from pathlib import Path

from . import corpus_build, corpus_gen
from .typicality import reference_perplexity

CANARY_LAYER_FRAC = 0.5  # mid-depth layer for the edit/fragmentation canaries
_ROOT = Path(__file__).resolve().parent.parent
FREQ_CACHE = _ROOT / "data" / "corpus" / "entity_freq.json"


def load_entity_freq() -> dict:
    import json
    return json.loads(FREQ_CACHE.read_text(encoding="utf-8"))


def _assign_cells(items: list[dict], ppl: np.ndarray, freq_map: dict) -> list[dict]:
    """D-007: typicality axis = entity frequency (common=typical, rare=atypical),
    which is separable from truth; perplexity is recorded as the cross-check only
    (ppl encodes truth — project notebook 2026-07-14). Middle tercile discarded a
    priori (§1.1)."""
    def item_logfreq(it):
        # mean log10-frequency over the statement's entities (subject + object).
        # object (e.g. country) spreads the axis where obscure subjects collapse
        # to zero; both together resolve typical vs atypical (D-007).
        vals = []
        for e in it.get("entities") or [it["entity"]]:
            v = (freq_map.get(e) or {}).get("log10")
            if v is not None:
                vals.append(float(v))
        return float(np.mean(vals)) if vals else np.nan

    logf = np.array([item_logfreq(it) for it in items])
    # RANK terciles (argsort), not value-quantiles: the frequency axis is skewed
    # with a zero-mass (obscure entities -> count 0), so value-quantiles collapse
    # (lo==hi). Ranking splits typical/atypical into balanced thirds regardless.
    idx_finite = np.flatnonzero(np.isfinite(logf))
    ranked = idx_finite[np.argsort(logf[idx_finite], kind="stable")]  # ascending freq
    n = len(ranked)
    t = n // 3
    atypical_idx = set(ranked[:t].tolist())        # lowest frequency = atypical
    typical_idx = set(ranked[n - t:].tolist())     # highest frequency = typical
    kept = []
    for i, (it, p, f) in enumerate(zip(items, ppl, logf)):
        if i in typical_idx:
            band = "typical"
        elif i in atypical_idx:
            band = "atypical"
        else:
            continue  # middle tercile discarded a priori (§1.1)
        it["typicality"] = {"entity_freq_log10": float(f), "reference_ppl": float(p),
                            "primary_axis": "entity_frequency_rank",
                            "ppl_is_crosscheck_only": True}
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


def _match_on_subword(items: list[dict], tokenizer, seed: int) -> list[dict]:
    """C3 control (stage2_self_review.md): match the entity sub-word-count
    histogram across cells so a probe can't read fragmentation as a proxy for
    truth/cell. Subsample each (cell, sub-word-bucket) to the per-bucket minimum
    across cells."""
    for it in items:
        eids = tokenizer(it["entity"])["input_ids"] if it["entity"] else []
        it["_sw"] = min(len(eids), 6)  # cap tail buckets
    cells = sorted({it["cell"] for it in items})
    by = defaultdict(list)
    for it in items:
        by[(it["cell"], it["_sw"])].append(it)
    buckets = sorted({it["_sw"] for it in items})
    rng = np.random.default_rng(seed)
    out = []
    for b in buckets:
        m = min(len(by[(c, b)]) for c in cells)
        if m == 0:
            continue
        for c in cells:
            pool = by[(c, b)]
            for i in rng.choice(len(pool), m, replace=False):
                out.append(pool[int(i)])
    for it in out:
        it.pop("_sw", None)
    return out


def _match_truth_subword(items: list[dict], tokenizer, seed: int) -> list[dict]:
    from collections import defaultdict
    for it in items:
        eids = tokenizer(it["entity"])["input_ids"] if it["entity"] else []
        sids = tokenizer(it["text"])["input_ids"]
        it["_sw"] = (min(len(eids), 6), min(len(sids) // 2, 8))
    by = defaultdict(lambda: {True: [], False: []})
    for it in items:
        by[it["_sw"]][it["truth"]].append(it)
    rng = np.random.default_rng(seed)
    out = []
    for d in by.values():
        m = min(len(d[True]), len(d[False]))
        for grp in (True, False):
            for i in rng.choice(len(d[grp]), m, replace=False):
                out.append(d[grp][int(i)])
    for it in out:
        it.pop("_sw", None)
    return out


def score_and_gate(reference_substrate, canary_substrate, edit_rate: float = 0.5,
                   seed: int = 20260712) -> dict:
    """GPU stage. Returns scored items + all three gate reports (no write)."""
    items = corpus_gen.build_candidate_items(edit_rate=edit_rate, seed=seed)
    texts = [it["text"] for it in items]
    print(f"[stage2] scoring reference ppl on {len(texts)} items "
          f"({reference_substrate.model_id})", flush=True)
    ppl = reference_perplexity(texts, reference_substrate)

    freq_map = load_entity_freq()
    full = _assign_cells(items, ppl, freq_map)
    raw_counts = {c: sum(1 for it in full if it["cell"] == c)
                  for c in ("TT", "TA", "FT", "FA")}
    L = int(canary_substrate.model.config.num_hidden_layers * CANARY_LAYER_FRAC)

    def canaries(pop):
        H = canary_substrate.hidden_states_matrix(
            [it["text"] for it in pop], batch_size=32)[:, L, :].astype(np.float32)
        e = corpus_build.edit_canary(H, np.array([it["edited"] for it in pop]))
        f = corpus_build.fragmentation_canary(
            corpus_build.fragmentation_features(
                [it["text"] for it in pop], [it["entity"] for it in pop],
                canary_substrate.tokenizer),
            np.array([it["truth"] for it in pop]))
        return e, f

    edit_c, frag_c = canaries(full)         # D-011: full corpus is the released corpus
    crossing = corpus_build.verify_crossing(full)
    kept = _match_truth_subword([dict(it) for it in full], canary_substrate.tokenizer, seed)
    kept_counts = {c: sum(1 for it in kept if it["cell"] == c) for c in ("TT", "TA", "FT", "FA")}
    edit_matched, frag_matched = canaries(kept)   # cross-check: truth-matched subset
    print(f"[stage2] raw={raw_counts} n={len(full)} (RELEASED, D-011) | "
          f"edit {edit_c.get('auroc')} {edit_c.get('ci')} pass={edit_c.get('pass')}"
          f" | frag(covariate) {frag_c.get('auroc')} {frag_c.get('ci')} pass={frag_c.get('pass')} | "
          f"truth-matched subset n={len(kept)} {kept_counts} "
          f"edit={edit_matched.get('auroc')} frag={frag_matched.get('auroc')}", flush=True)

    return {
        "items": full,
        "meta": {"reference_model": reference_substrate.model_id,
                 "canary_model": canary_substrate.model_id, "canary_layer": L,
                 "n_full": len(full), "n_released": len(full),
                 "raw_counts": raw_counts, "released_counts": raw_counts,
                 "edit_rate": edit_rate, "seed": seed},
        "crossing": crossing, "edit_canary": edit_c, "fragmentation_canary": frag_c,
        "evidence_matched": {"n": len(kept), "counts": kept_counts,
                             "edit_canary": edit_matched, "fragmentation_canary": frag_matched},
    }
