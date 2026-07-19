"""
Stage 2 — Decorrelation corpus: build + VERIFY the crossed truth x typicality 2x2.

This module currently implements the GATES, not the generation. Generation
(symmetric entity swaps, natural-error harvesting — decisions.md D-003) is
deliberately unimplemented until the Stage-2 design has owner signoff and has
been seen by an external reviewer (the project's standing directive §3 Stage 2 gate, §7).

Gates implemented here (thresholds in configs/corpus.yaml):
  1. crossing gate      the axes must actually be crossed: within each truth
                        value, atypical items must have genuinely higher
                        reference-perplexity than typical items (effect size +
                        significance), and the FT cell must be genuinely low-ppl.
  2. edit-canary gate   a "was-this-edited" probe trained on hidden states must
                        be near chance; otherwise the false cell carries an edit
                        signature and every Stage-3 truth number is contaminated.
  3. signoff gate       finalize() refuses to write a corpus version without an
                        owner signoff recorded in notes/decisions.md.

Corpus item schema (one JSON object per line):
  {text, truth: bool, cell: TT|TA|FT|FA, edited: bool, entity, domain,
   typicality: {reference_ppl, entity_freq_log10, self_ppl, self_ppl_circular},
   provenance: {source, strategy, verified_against}}
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

_ROOT = Path(__file__).resolve().parent.parent


def _cfg() -> dict:
    with open(_ROOT / "configs" / "corpus.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _ci_includes_half(res: dict) -> bool:
    ci = res.get("ci") or [None, None]
    return bool(res.get("auroc") is not None and ci[0] is not None
                and ci[0] <= 0.5 <= ci[1])


# ── Gate 1: the axes are actually crossed ────────────────────────────────────

def verify_crossing(items: list[dict]) -> dict:
    """Hard gate on the PRIMARY typicality axis (D-007 = entity frequency): per
    truth value, typical items must be genuinely more frequent than atypical ones
    (freq(typical) >> freq(atypical)). Perplexity medians are reported alongside
    as the cross-check (the ppl-vs-frequency divergence is itself a finding).

    Cliff's delta (effect size) + Mann-Whitney U; thresholds from corpus.yaml.
    """
    from scipy.stats import mannwhitneyu

    cfg = _cfg()["gates"]["crossing"]
    report: dict = {"gate": "crossing", "axis": "entity_frequency", "checks": {}}

    def freq(cell):
        return np.array([it["typicality"]["entity_freq_log10"] for it in items
                         if it["cell"] == cell])

    def ppl(cell):
        return np.array([it["typicality"]["reference_ppl"] for it in items
                         if it["cell"] == cell])

    for truth_val, typ_cell, atyp_cell in (("true", "TT", "TA"), ("false", "FT", "FA")):
        typ, atyp = freq(typ_cell), freq(atyp_cell)
        if len(typ) < 2 or len(atyp) < 2:
            report["checks"][truth_val] = {"pass": False, "note": "empty cell"}
            continue
        # Cliff's delta on frequency: typical should exceed atypical
        gt = (typ[:, None] > atyp[None, :]).mean()
        lt = (typ[:, None] < atyp[None, :]).mean()
        delta = float(gt - lt)
        _, p = mannwhitneyu(typ, atyp, alternative="greater")
        report["checks"][truth_val] = {
            "cells": f"{typ_cell} (typical) vs {atyp_cell} (atypical)",
            "cliffs_delta_freq": round(delta, 4),
            "mannwhitney_p": float(p),
            "median_log10freq_typical": round(float(np.median(typ)), 3),
            "median_log10freq_atypical": round(float(np.median(atyp)), 3),
            "xcheck_median_ppl_typical": round(float(np.median(ppl(typ_cell))), 2),
            "xcheck_median_ppl_atypical": round(float(np.median(ppl(atyp_cell))), 2),
            "pass": bool(delta >= cfg["min_cliffs_delta"] and p <= cfg["max_p_value"]),
        }
    report["pass"] = all(c.get("pass") for c in report["checks"].values())
    return report


# ── Gate 2: edit-signature canary ────────────────────────────────────────────

def edit_canary(hidden_states: np.ndarray, edited: np.ndarray,
                seed: int = 20260709) -> dict:
    """Train a 'was-this-edited' probe. If it beats chance, the edit pipeline
    left a signature the truth probe can shortcut on (the recursive trap, §4.2).

    hidden_states: [n, d] at the same layer the truth probes use.
    """
    from .probes.saplma import SaplmaProbe  # same capacity as the real probe
    from .stats import auroc_with_ci

    edited = np.asarray(edited, dtype=int)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(edited))
    cut = int(0.8 * len(idx))
    tr, te = idx[:cut], idx[cut:]
    probe = SaplmaProbe(seed=seed).fit(hidden_states[tr], edited[tr])
    res = auroc_with_ci(edited[te], probe.score(hidden_states[te]), seed=seed)
    res.update({"gate": "edit_canary", "pass": _ci_includes_half(res)})
    return res


# ── Gate 2b: tokenization / fragmentation canary (confound C3) ───────────────

def fragmentation_features(texts: list[str], entities: list[str], tokenizer) -> np.ndarray:
    """Tokenization-only features that track atypicality independent of the
    perplexity axis (stage2_self_review.md C3): a probe reading these instead of
    truth is a confound under the confound. Features: entity sub-word count,
    statement sub-word count, entity chars, sub-words-per-word, final-token-is-
    continuation-piece."""
    cont_marks = ("Ġ", "▁", " ")  # GPT-2 'Ġ', sentencepiece '▁', space
    rows = []
    for txt, ent in zip(texts, entities):
        sids = tokenizer(txt)["input_ids"]
        eids = tokenizer(ent)["input_ids"] if ent else []
        last_tok = tokenizer.convert_ids_to_tokens(sids[-1]) if sids else ""
        is_cont = 0 if any(last_tok.startswith(m) for m in cont_marks) else 1
        rows.append([len(eids), len(sids), len(ent or ""),
                     len(eids) / max(len((ent or "").split()), 1), is_cont])
    return np.asarray(rows, dtype=float)


def fragmentation_canary(features: np.ndarray, truth: np.ndarray,
                         seed: int = 20260712) -> dict:
    """A probe predicting TRUTH from tokenization features ALONE must be ≈ chance.
    If it isn't, fragmentation leaks truth and the 'typicality-controlled' AUROC
    is itself confounded (C3). (Fragmentation predicting *typicality* is expected
    and fine; predicting *truth* is the danger — the 2x2 crosses them, so on a
    balanced corpus this should be near 0.5.)"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    from .stats import auroc_with_ci

    truth = np.asarray(truth, dtype=int)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(truth))
    cut = int(0.8 * len(idx))
    tr, te = idx[:cut], idx[cut:]
    sc = StandardScaler().fit(features[tr])
    clf = LogisticRegression(max_iter=500, random_state=seed).fit(sc.transform(features[tr]), truth[tr])
    scores = clf.predict_proba(sc.transform(features[te]))[:, 1]
    res = auroc_with_ci(truth[te], scores, seed=seed)
    res.update({"gate": "fragmentation_canary", "pass": _ci_includes_half(res)})
    return res


# ── Gate 3: signoff + versioned write ────────────────────────────────────────

def finalize(items: list[dict], crossing_report: dict, canary_report: dict,
             fragmentation_report: dict | None = None,
             owner_signoff_decision_id: str | None = None) -> Path:
    """Write the hash-versioned corpus. Refuses without passed gates + signoff."""
    if not crossing_report.get("pass"):
        raise RuntimeError("crossing gate FAILED — corpus must not be finalized")
    if not canary_report.get("pass"):
        raise RuntimeError("edit-canary gate FAILED — truth results would be contaminated")
    if _cfg()["gates"]["owner_signoff_required"] and not owner_signoff_decision_id:
        raise RuntimeError(
            "owner signoff required: record the decision in notes/decisions.md "
            "and pass its ID (e.g. 'D-006') — do not run probes on an unsigned corpus")

    payload = "\n".join(json.dumps(it, sort_keys=True) for it in items)
    h = hashlib.sha256(payload.encode()).hexdigest()[:12]
    out = _ROOT / "data" / "corpus" / f"mirage_2x2_v{h}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(payload + "\n", encoding="utf-8")
    (out.with_suffix(".report.json")).write_text(json.dumps({
        "hash": h, "n": len(items), "signoff": owner_signoff_decision_id,
        "crossing": crossing_report, "edit_canary": canary_report,
        "fragmentation_canary": fragmentation_report,
    }, indent=2), encoding="utf-8")
    return out


# Generation strategies (D-003) — BLOCKED on Stage-2 owner review. Kept as
# explicit stubs so nobody implements them casually inside another module.

def generate_false_typical_symmetric_swap(*a, **k):
    raise NotImplementedError("Stage-2 gate: design needs owner + external-reviewer signoff")


def harvest_natural_confident_errors(*a, **k):
    raise NotImplementedError("Stage-2 gate: design needs owner + external-reviewer signoff")
