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

import json
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


def _assign_cells_v2(items: list[dict], ppl: np.ndarray, freq_map: dict,
                     min_per_cell: int = 20) -> tuple[list[dict], dict]:
    """v2 (2026-08-06). Two changes, both forced by defects found in v1.

    1. Terciles are ranked WITHIN each domain, not across the pooled corpus. v1
       ranked globally, so a domain whose entities were systematically rarer than
       cities landed wholesale in one typicality band. Combined with (2) below
       that made `domain` predict truth across the diagonal/off-diagonal split
       (AUROC 0.24) and manufactured the "collapse" we reported for three weeks.
    2. Entities whose frequency is UNRESOLVED (None) are dropped, not treated as
       zero. v1's cache recorded failed API calls as count=0, so 91% of entities
       carried a fake frequency and the typicality axis was really "did the fetch
       succeed". Anything without a measured count cannot be placed on this axis.

    A domain must populate all four cells with at least `min_per_cell` items or it
    is excluded entirely — a domain present in only some cells is exactly the leak
    v1 shipped.
    """
    def item_logfreq(it):
        vals = []
        for e in it.get("entities") or [it["entity"]]:
            v = (freq_map.get(e) or {}).get("log10")
            if v is not None:                      # None = unresolved, never 0
                vals.append(float(v))
        return float(np.mean(vals)) if vals else np.nan

    logf = np.array([item_logfreq(it) for it in items])
    domains = sorted({it.get("domain", "?") for it in items})
    kept, report = [], {}

    for dom in domains:
        idx = np.array([i for i, it in enumerate(items)
                        if it.get("domain", "?") == dom and np.isfinite(logf[i])])
        n_unresolved = sum(1 for i, it in enumerate(items)
                           if it.get("domain", "?") == dom and not np.isfinite(logf[i]))
        if len(idx) < 4 * min_per_cell:
            report[dom] = {"kept": 0, "reason": "too few entities with measured frequency",
                           "n_with_freq": int(len(idx)), "n_unresolved": n_unresolved}
            continue
        ranked = idx[np.argsort(logf[idx], kind="stable")]
        t = len(ranked) // 3
        atypical = set(ranked[:t].tolist())
        typical = set(ranked[len(ranked) - t:].tolist())

        staged = []
        for i in list(atypical | typical):
            it = items[i]
            band = "typical" if i in typical else "atypical"
            it["typicality"] = {"entity_freq_log10": float(logf[i]),
                                "reference_ppl": float(ppl[i]),
                                "primary_axis": "entity_frequency_rank_within_domain",
                                "ppl_is_crosscheck_only": True}
            it["cell"] = ("TT" if (it["truth"] and band == "typical") else
                          "TA" if it["truth"] else
                          "FT" if band == "typical" else "FA")
            staged.append(it)

        by_cell = defaultdict(list)
        for it in staged:
            by_cell[it["cell"]].append(it)
        counts = {c: len(by_cell.get(c, [])) for c in ("TT", "TA", "FT", "FA")}
        if min(counts.values()) < min_per_cell:
            report[dom] = {"kept": 0, "reason": "does not populate all four cells",
                           "cell_counts": counts, "n_unresolved": n_unresolved}
            continue
        report[dom] = {"kept": None, "cell_counts": counts, "n_unresolved": n_unresolved}
        kept.extend(staged)

    return kept, report


def _balance_by_domain(items: list[dict], seed: int) -> tuple[list[dict], dict]:
    """Equal items across a domain's OWN four cells, so domain is orthogonal to
    both truth and typicality by construction rather than by hope.

    v3 took one global minimum across every (domain, cell) pair, which let the
    smallest domain cap every other one — companies has 133 unique true statements,
    so every domain got 58. Orthogonality only needs P(true|domain) = P(typical|
    domain) = 0.5, which within-domain balancing already gives; domains do not have
    to match each other. Uncapped that would make cities ~76% of the corpus, so no
    domain is allowed a majority: a project whose headline died once to domain
    composition should not ship a pooled number that is mostly one topic.
    """
    rng = np.random.default_rng(seed)
    by = defaultdict(list)
    for it in items:
        by[(it.get("domain", "?"), it["cell"])].append(it)
    if not by:
        return [], {"per_domain_cell": 0}

    cells4 = ("TT", "TA", "FT", "FA")
    doms = sorted({d for d, _ in by})

    # `edited` is drawn 50/50 INSIDE every cell, so edit provenance is orthogonal
    # to truth AND to typicality by construction. Without this the swap generator
    # leaves FA edit-heavy (0.578 vs ~0.42 elsewhere), which puts an edit shortcut
    # on the training diagonal that vanishes off it — the same signature as the
    # confound being measured, worth up to +0.075 of gap on its own (§4.2, D-012).
    # Both constraints below are vacuous in some corpora and must not zero the
    # build out when they are. A harvested-error corpus has ONE domain (so "no
    # domain may exceed half" is unsatisfiable by definition and also meaningless,
    # since a constant cannot confound anything) and NO edited items (so an edit
    # split is undefined, and edit provenance is likewise constant). Applying
    # either blindly produced per-cell counts of zero and a division by zero.
    split_on_edit = len({bool(it["edited"]) for it in items}) > 1

    def halves(d, c):
        v = by[(d, c)]
        if not split_on_edit:
            return (v,)
        return ([x for x in v if x["edited"]], [x for x in v if not x["edited"]])

    n_parts = 2 if split_on_edit else 1
    cap = {d: min(min(len(p) for p in halves(d, c)) for c in cells4) for d in doms}
    if len(doms) > 1:
        others = {d: sum(v for k, v in cap.items() if k != d) for d in doms}
        per = {d: min(cap[d], others[d]) for d in doms}
    else:
        per = dict(cap)          # single domain: majority rule cannot apply

    out = []
    for d in doms:
        for c in cells4:
            for part in halves(d, c):
                for i in rng.choice(len(part), per[d], replace=False):
                    out.append(part[int(i)])
    total = 4 * n_parts * sum(per.values())
    if total == 0:
        raise RuntimeError(
            f"balancing produced an empty corpus; per-domain capacity {cap}. "
            "Some cell has no items to draw from.")
    return out, {"per_domain_part": per, "capacity_per_domain_part": cap,
                 "domains": doms, "total": total, "cells_per_domain": 4,
                 "split_on_edit": split_on_edit,
                 "per_domain_cell": {d: n_parts * per[d] for d in doms},
                 "max_domain_share": round(
                     max(4 * n_parts * per[d] / total for d in doms), 4),
                 "rule": ("within-domain balance"
                          + (", edited 50/50 within every cell" if split_on_edit
                             else ", no edit split (no item was edited)")
                          + ("; no domain may exceed 50% of items" if len(doms) > 1
                             else "; single domain"))}


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


def score_and_gate_v2(reference_substrate, canary_substrate, edit_rate: float = 0.5,
                      seed: int = 20260806, min_per_cell: int = 20) -> dict:
    """v2 GPU stage (2026-08-06). Per-domain terciles, per-(domain,cell) balancing,
    and the composition canary promoted to a HARD GATE.

    v1 shipped a corpus where `domain` predicted truth off-diagonal at AUROC 0.24
    and where 91% of frequencies were unmeasured fetch failures recorded as zero.
    Both were visible in committed artifacts and neither was gated. Everything here
    is arranged so those two failures cannot recur silently.
    """
    items = corpus_gen.build_candidate_items(edit_rate=edit_rate, seed=seed)
    texts = [it["text"] for it in items]
    print(f"[stage2-v2] scoring reference ppl on {len(texts)} items", flush=True)
    ppl = reference_perplexity(texts, reference_substrate)

    freq_map = load_entity_freq()
    n_freq = max(len(freq_map), 1)
    unresolved = sum(1 for v in freq_map.values()
                     if not isinstance(v.get("count"), int) or v["count"] < 0)
    frac_unresolved = unresolved / n_freq
    # A handful of entities the API never returns is tolerable — those items are
    # dropped from the axis rather than guessed at. A large fraction is not: v1's
    # cache was 91% unresolved-written-as-zero, which turned the typicality axis
    # into "did the fetch succeed". Refuse above 5% so that failure mode cannot
    # recur quietly, and report the number either way.
    print(f"[stage2-v2] frequency cache: {n_freq} entities, {unresolved} unresolved "
          f"({frac_unresolved:.1%}) — unresolved items are DROPPED, never zeroed",
          flush=True)
    if frac_unresolved > 0.05:
        raise RuntimeError(
            f"entity_freq.json is {frac_unresolved:.1%} unresolved ({unresolved}/"
            f"{n_freq}). Re-run data/raw/fetch_entity_freq.py to retry them; the "
            "typicality axis is not trustworthy at this coverage.")

    staged, domain_report = _assign_cells_v2(items, ppl, freq_map, min_per_cell)
    full, balance_report = _balance_by_domain(staged, seed)
    raw_counts = {c: sum(1 for it in full if it["cell"] == c)
                  for c in ("TT", "TA", "FT", "FA")}
    L = int(canary_substrate.model.config.num_hidden_layers * CANARY_LAYER_FRAC)

    H = canary_substrate.hidden_states_matrix(
        [it["text"] for it in full], batch_size=32)[:, L, :].astype(np.float32)
    edit_c = corpus_build.edit_canary(H, np.array([it["edited"] for it in full]))
    frag_c = corpus_build.fragmentation_canary(
        corpus_build.fragmentation_features(
            [it["text"] for it in full], [it["entity"] for it in full],
            canary_substrate.tokenizer),
        np.array([it["truth"] for it in full]))
    crossing = corpus_build.verify_crossing(full)
    composition = corpus_build.composition_canary(full)

    print(f"[stage2-v2] n={len(full)} {raw_counts} | balance={balance_report} | "
          f"crossing pass={crossing.get('pass')} | composition pass={composition.get('pass')} "
          f"failed={composition.get('failed_fields')} | edit {edit_c.get('auroc')} "
          f"pass={edit_c.get('pass')} | frag {frag_c.get('auroc')}", flush=True)
    for dom, rep in domain_report.items():
        print(f"    domain {dom}: {rep}", flush=True)

    return {
        "items": full,
        "meta": {"version": 2, "reference_model": reference_substrate.model_id,
                 "canary_model": canary_substrate.model_id, "canary_layer": L,
                 "n_full": len(full), "n_released": len(full),
                 "raw_counts": raw_counts, "released_counts": raw_counts,
                 "domain_report": domain_report, "balance": balance_report,
                 "edit_rate": edit_rate, "seed": seed},
        "crossing": crossing, "edit_canary": edit_c,
        "fragmentation_canary": frag_c, "composition_canary": composition,
    }


def score_and_gate_harvested(reference_substrate, canary_substrate, topic: str = "wikidata",
                             seed: int = 20260812, min_per_cell: int = 20) -> dict:
    """Build the 2x2 from HARVESTED MODEL ERRORS instead of authored entity swaps.

    Every corpus before this one had its falsehoods written by us, which is the
    limitation the paper flags hardest: the confound is measured on data we
    constructed, and two of our five corpora manufactured the effect we were
    hunting. Here the false cells are statements a disjoint model actually produced
    when asked a factual question and got it wrong, verified against the same
    knowledge base. Nothing about the falsehood is our choice except which errors
    to keep.

    Consequences, both deliberate:
      - There is no edit provenance, so `edited` is False throughout and the edit
        canary and edit-balance gate do not apply. That is not a relaxed gate: the
        confound they exist to catch cannot exist here, because no statement was
        edited by anyone.
      - The false cells are not free to be balanced arbitrarily. Model errors
        concentrate on rare entities, so the false-typical cell is supply limited
        by the model's actual behaviour. If it cannot be filled, that is a finding
        about natural error distribution and the build says so rather than padding.
    """
    from . import harvest as hv

    freq_map = load_entity_freq()

    def _resolved(e):
        v = freq_map.get(e) or {}
        return isinstance(v.get("count"), int) and v["count"] >= 0

    cache_unres = sum(1 for v in freq_map.values()
                      if not isinstance(v.get("count"), int) or v["count"] < 0)
    print(f"[harvested] frequency cache: {len(freq_map)} entities, {cache_unres} "
          f"unresolved ({cache_unres / max(len(freq_map), 1):.1%})", flush=True)

    false_items = hv.load_harvested(topic)
    for _it in false_items:
        _it["domain"] = "cities"
    if len(false_items) < 4 * min_per_cell:
        raise RuntimeError(
            f"only {len(false_items)} harvested errors for '{topic}'; need at least "
            f"{4 * min_per_cell}. Harvest more subjects before building.")

    # Truths come from the SAME gazetteer that adjudicated the errors, so both
    # halves are drawn from one distribution. Sourcing truths from the A&M KB
    # instead would reintroduce a provenance difference between the true and false
    # cells, which is the class of shortcut that produced the v1 domain artifact.
    gold_path = _ROOT / "data" / "corpus" / "wikidata_gold_cities.json"
    gold = {k: sorted(v) for k, v in json.loads(
        gold_path.read_text(encoding="utf-8")).items()}

    # ONE canonical surface form per country on both sides. The gazetteer says
    # "People's Republic of China" and models say "China", so before this the word
    # "China" appeared only in false statements and predicted falsehood perfectly.
    # The composition gate caught it. Answers that canonicalize to nothing are not
    # countries at all ("Arizona", truncated generations) and are dropped.
    canon = hv.canonical_country_map(gold)
    kept_false = []
    for it in false_items:
        c = hv.canonicalize(it["entities"][1], canon)
        if c is None:
            continue
        it["entities"] = [it["entities"][0], c]
        it["text"] = f"{it['entities'][0]} is a city in {c}."
        kept_false.append(it)
    print(f"[harvested] {len(false_items)} errors -> {len(kept_false)} after "
          f"canonicalizing the named country", flush=True)
    false_items = kept_false

    wrong_subjects = {it["entities"][0] for it in false_items}
    true_items = []
    for s, countries in gold.items():
        c = hv.canonicalize(countries[0], canon) if len(countries) == 1 else None
        if s in wrong_subjects or c is None:
            continue   # a subject cannot supply both a truth and an error, and an
                       # ambiguous name has no single true completion
        true_items.append({
            "text": f"{s} is a city in {c}.", "truth": True,
            "edited": False, "entity": s, "entities": [s, c],
            "domain": "cities",
            "provenance": {"source": "wikidata", "strategy": "gazetteer_true",
                           "object_type": "country"}})

    # Gate on the entities the corpus ACTUALLY USES, not on the whole cache.
    # v1's failure was that 91% of USED entities carried fabricated zeros, which is
    # what makes an axis meaningless. A cache holding extra unresolved entries that
    # no item references is not that failure, and gating on it produced a false
    # refusal here. Items with an unresolved entity are dropped outright.
    items = [it for it in (true_items + false_items)
             if all(_resolved(e) for e in it["entities"])]
    dropped = (len(true_items) + len(false_items)) - len(items)
    used = {e for it in items for e in it["entities"]}
    used_unres = sum(1 for e in used if not _resolved(e))
    print(f"[harvested] dropped {dropped} items with an unresolved entity; "
          f"{len(used)} entities in use, {used_unres} unresolved", flush=True)
    if used_unres / max(len(used), 1) > 0.05:
        raise RuntimeError(
            f"{used_unres}/{len(used)} entities used by the corpus have no measured "
            "frequency; the typicality axis is not trustworthy at this coverage")
    n_true_used = sum(1 for it in items if it["truth"])
    n_false_used = len(items) - n_true_used
    if n_false_used < 4 * min_per_cell:
        raise RuntimeError(
            f"only {n_false_used} harvested errors survive frequency resolution; "
            f"need {4 * min_per_cell}. Fetch more entity frequencies.")
    texts = [it["text"] for it in items]
    print(f"[harvested] {len(true_items)} true + {len(false_items)} model errors "
          f"= {len(items)} candidates", flush=True)
    ppl = reference_perplexity(texts, reference_substrate)

    staged, domain_report = _assign_cells_v2(items, ppl, freq_map, min_per_cell)
    full, balance_report = _balance_by_domain(staged, seed)
    raw_counts = {c: sum(1 for it in full if it["cell"] == c)
                  for c in ("TT", "TA", "FT", "FA")}

    L = int(canary_substrate.model.config.num_hidden_layers * CANARY_LAYER_FRAC)
    frag_c = corpus_build.fragmentation_canary(
        corpus_build.fragmentation_features(
            [it["text"] for it in full], [it["entity"] for it in full],
            canary_substrate.tokenizer),
        np.array([it["truth"] for it in full]))
    crossing = corpus_build.verify_crossing(full)
    composition = corpus_build.composition_canary(full)
    edit_c = {"gate": "edit_canary", "pass": True,
              "not_applicable": "falsehoods are harvested model errors, not edits; "
                                "no statement in this corpus was edited"}

    print(f"[harvested] n={len(full)} {raw_counts} | balance={balance_report} | "
          f"crossing pass={crossing.get('pass')} | composition pass={composition.get('pass')} "
          f"failed={composition.get('failed_fields')} | frag {frag_c.get('auroc')}", flush=True)
    for dom, rep in domain_report.items():
        print(f"    domain {dom}: {rep}", flush=True)

    return {
        "items": full,
        "meta": {"version": "harvested", "topic": topic,
                 "reference_model": reference_substrate.model_id,
                 "canary_model": canary_substrate.model_id, "canary_layer": L,
                 "n_full": len(full), "n_true_candidates": len(true_items),
                 "n_harvested_errors": len(false_items),
                 "n_released": len(full), "raw_counts": raw_counts,
                 "released_counts": raw_counts,
                 "domain_report": domain_report, "balance": balance_report,
                 "seed": seed},
        "crossing": crossing, "edit_canary": edit_c,
        "fragmentation_canary": frag_c, "composition_canary": composition,
    }


def score_and_gate(reference_substrate, canary_substrate, edit_rate: float = 0.5,
                   seed: int = 20260712) -> dict:
    """v1 GPU stage. SUPERSEDED — kept only to reproduce the flawed v1 corpus for
    the record. Its cell assignment ranks frequency globally (leaking domain) and
    tolerates unresolved frequencies as zero. Use score_and_gate_v2."""
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
