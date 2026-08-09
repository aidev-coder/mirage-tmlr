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
from collections import Counter
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


DIAGONAL_CELLS = ("TT", "FA")
OFF_DIAGONAL_CELLS = ("TA", "FT")


def _categorical_fields(items: list[dict], max_card: int = 20) -> dict[str, list]:
    """Low-cardinality categorical metadata, top level and one level into
    `provenance`. Free text and per-item identifiers are excluded by the
    cardinality cap."""
    skip = {"text", "entity", "entities", "cell", "truth", "typicality"}
    vals: dict[str, list] = {}
    for it in items:
        for k, v in it.items():
            if k in skip or not isinstance(v, (str, bool)):
                continue
            vals.setdefault(k, []).append(v)
        for k, v in (it.get("provenance") or {}).items():
            if isinstance(v, (str, bool)):
                vals.setdefault(f"provenance.{k}", []).append(v)
    return {k: v for k, v in vals.items()
            if len(v) == len(items) and 1 < len(set(v)) <= max_card}


def _text_visible(items: list[dict], values: np.ndarray, seed: int = 20260712,
                  threshold: float = 0.65) -> dict:
    """Is this field recoverable from the statement text alone? Character n-gram
    classifier, held-out accuracy against the majority-class baseline. A field the
    probe cannot see in its input cannot be the shortcut it exploits."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    y = np.asarray([str(x) for x in values])
    texts = [it["text"] for it in items]
    try:
        tr_x, te_x, tr_y, te_y = train_test_split(
            texts, y, test_size=0.3, random_state=seed, stratify=y)
    except ValueError:
        return {"text_visible": None, "accuracy": None, "note": "too few per class"}
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2, max_features=20000)
    Xtr = vec.fit_transform(tr_x)
    clf = LogisticRegression(max_iter=1000, random_state=seed).fit(Xtr, tr_y)
    acc = float(clf.score(vec.transform(te_x), te_y))
    base = float(max(np.mean(te_y == c) for c in set(te_y)))
    return {"text_visible": bool(acc > max(threshold, base + 0.05)),
            "accuracy": round(acc, 4), "majority_baseline": round(base, 4)}


def composition_canary(items: list[dict], seed: int = 20260712) -> dict:
    """Can a categorical METADATA field alone predict truth off-diagonal, having
    learned the association on the diagonal? It must not.

    This is the check MIRAGE failed and did not have (2026-08-03). Our own corpus
    put 32% non-cities in diagonal-TRUE and 0% in diagonal-FALSE, while the
    off-diagonal reversed it (0% non-cities in TRUE, 52% in FALSE). A probe could
    score the entire off-diagonal by topic alone, and below chance at that — which
    is exactly the "collapse" we mistook for a typicality confound for three weeks.

    Method (no model, no GPU): for each field, learn P(true | value) on the
    diagonal, score the off-diagonal by that lookup, AUROC against truth. Chance
    means the field carries no diagonal-to-off-diagonal shortcut. Deviation in
    EITHER direction fails: below chance is the inverting case, which is worse
    because it manufactures an apparent collapse."""
    from .stats import auroc_with_ci

    cells = np.array([it["cell"] for it in items])
    truth = np.array([bool(it["truth"]) for it in items], dtype=int)
    diag = np.isin(cells, DIAGONAL_CELLS)
    off = np.isin(cells, OFF_DIAGONAL_CELLS)
    out: dict = {"gate": "composition_canary", "fields": {}}
    if diag.sum() == 0 or off.sum() == 0 or len(set(truth[off])) < 2:
        out.update({"pass": None, "note": "needs both cells populated and both truth values off-diagonal"})
        return out

    for name, values in _categorical_fields(items).items():
        v = np.asarray(values, dtype=object)
        rate = {}
        for val in set(v[diag]):
            m = diag & (v == val)
            rate[val] = float(truth[m].mean()) if m.sum() else 0.5
        scores = np.array([rate.get(x, 0.5) for x in v[off]], dtype=float)
        if len(set(scores)) < 2:
            continue
        res = auroc_with_ci(truth[off], scores, seed=seed)
        # composition table, so a failure is immediately legible
        table = {}
        for lab, mask in (("diagonal", diag), ("off_diagonal", off)):
            table[lab] = {str(val): {
                "n": int(((v == val) & mask).sum()),
                "frac_true": round(float(truth[(v == val) & mask].mean()), 3)
                if ((v == val) & mask).sum() else None}
                for val in sorted(set(v), key=str)}
        vis = _text_visible(items, v, seed=seed)
        res.update({"shortcut_present": not _ci_includes_half(res),
                    "text_visible": vis["text_visible"],
                    "text_recoverability_acc": vis["accuracy"],
                    "composition": table})
        # A field only threatens the probe if BOTH: it carries a diagonal->off-diagonal
        # shortcut AND it is recoverable from the input the model actually sees.
        # `provenance.strategy` names encode truth perfectly but are invisible to the
        # probe, so gating on them would fail every corpus and train users to ignore
        # this check.
        res["pass"] = not (res["shortcut_present"] and vis["text_visible"])
        out["fields"][name] = res

    failed = [k for k, r in out["fields"].items() if r.get("pass") is False]
    out["failed_fields"] = failed
    out["metadata_only_imbalances"] = [
        k for k, r in out["fields"].items()
        if r.get("shortcut_present") and not r.get("text_visible")]
    out["pass"] = not failed
    return out


# ── Gate 3: signoff + versioned write ────────────────────────────────────────

def duplicate_check(items: list[dict]) -> dict:
    """No statement may appear twice (D-016).

    v3 shipped 70 repeated rows concentrated in the two TRUE cells, so its per-cell
    balance held in rows but not in statements, and 44 statements carried
    contradictory `edited` flags — 14.5% of edit-canary rows were unlearnable by
    construction, which biases that canary toward passing.
    """
    texts = [it["text"] for it in items]
    counts = Counter(texts)
    dups = sorted(t for t, n in counts.items() if n > 1)
    return {"gate": "duplicates", "n_rows": len(texts), "n_unique": len(counts),
            "n_repeated_rows": len(texts) - len(counts),
            "n_duplicated_statements": len(dups), "examples": dups[:3],
            "pass": not dups}


def domain_share_check(items: list[dict], max_share: float = 0.5) -> dict:
    """No single domain may be a majority of the corpus.

    Orthogonality does not prevent this: a corpus can have P(true|domain)=0.5
    everywhere and still be 76% one topic, which makes a pooled headline a
    single-domain result in all but name. v1's failure was domain composition, so
    volume dominance is gated separately from correlation.
    """
    counts = Counter(it.get("domain", "?") for it in items)
    top, n = counts.most_common(1)[0] if counts else ("?", 0)
    share = n / len(items) if items else 0.0
    return {"gate": "domain_share", "max_share": max_share,
            "shares": {d: round(c / len(items), 4) for d, c in counts.items()},
            "largest_domain": top, "largest_share": round(share, 4),
            "pass": bool(len(counts) <= 1 or share <= max_share)}


def finalize(items: list[dict], crossing_report: dict, canary_report: dict,
             fragmentation_report: dict | None = None,
             owner_signoff_decision_id: str | None = None,
             composition_report: dict | None = None) -> Path:
    """Write the hash-versioned corpus. Refuses without passed gates + signoff."""
    if not crossing_report.get("pass"):
        raise RuntimeError("crossing gate FAILED — corpus must not be finalized")
    if not canary_report.get("pass"):
        raise RuntimeError("edit-canary gate FAILED — truth results would be contaminated")
    # Composition is a HARD gate as of 2026-08-06. v1 finalized without it and
    # shipped a corpus where `domain` predicted truth off-diagonal at AUROC 0.24,
    # which by itself produced the "collapse" the paper reported. Passing None is
    # allowed only to reproduce pre-v2 corpora for the record.
    if composition_report is not None and not composition_report.get("pass"):
        raise RuntimeError(
            "composition gate FAILED on "
            f"{composition_report.get('failed_fields')} — a text-visible metadata "
            "field predicts truth off-diagonal after learning it on the diagonal. "
            "Any probe result on this corpus would measure that shortcut.")
    if _cfg()["gates"]["owner_signoff_required"] and not owner_signoff_decision_id:
        raise RuntimeError(
            "owner signoff required: record the decision in notes/decisions.md "
            "and pass its ID (e.g. 'D-006') — do not run probes on an unsigned corpus")

    dup = duplicate_check(items)
    if not dup["pass"]:
        raise RuntimeError(
            f"duplicate gate FAILED — {dup['n_repeated_rows']} repeated rows across "
            f"{dup['n_duplicated_statements']} statements, e.g. {dup['examples']}")

    dom = domain_share_check(items)
    if not dom["pass"]:
        raise RuntimeError(
            f"domain-majority gate FAILED — '{dom['largest_domain']}' is "
            f"{dom['largest_share']:.1%} of the corpus")

    texts = [it["text"] for it in items]
    dom_counts = Counter(it.get("domain", "?") for it in items)
    payload = "\n".join(json.dumps(it, sort_keys=True) for it in items)
    h = hashlib.sha256(payload.encode()).hexdigest()[:12]
    out = _ROOT / "data" / "corpus" / f"mirage_2x2_v{h}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(payload + "\n", encoding="utf-8")
    (out.with_suffix(".report.json")).write_text(json.dumps({
        "hash": h, "n": len(items), "signoff": owner_signoff_decision_id,
        "crossing": crossing_report, "edit_canary": canary_report,
        "fragmentation_canary": fragmentation_report,
        "composition_canary": composition_report,
        "duplicates": {"n_rows": len(texts), "n_unique": len(set(texts)), "pass": True},
        "domain_share": {d: round(c / len(items), 4) for d, c in dom_counts.items()},
    }, indent=2), encoding="utf-8")
    return out


# Generation strategies (D-003) — BLOCKED on Stage-2 owner review. Kept as
# explicit stubs so nobody implements them casually inside another module.

def generate_false_typical_symmetric_swap(*a, **k):
    raise NotImplementedError("Stage-2 gate: design needs owner + external-reviewer signoff")


def harvest_natural_confident_errors(*a, **k):
    raise NotImplementedError("Stage-2 gate: design needs owner + external-reviewer signoff")
