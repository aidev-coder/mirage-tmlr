"""
Stage 2 — symmetric-swap corpus generation (decisions.md D-003, signoff D-006).

Builds candidate items for the crossed 2x2 BEFORE typicality scoring (that needs
the GPU reference LMs; done in the Modal build step). The truth x typicality
crossing is completed later by assigning reference-ppl terciles; here we only
produce (text, truth, edited, entity, domain, provenance).

THE LOAD-BEARING INVARIANT (§4.2 recursive trap): edit provenance must be
orthogonal to truth. We enforce, per topic and per truth value,
    P(edited_by_our_pipeline | true) == P(edited_by_our_pipeline | false) == edit_rate
so a "was-this-edited" probe (corpus_build.edit_canary) cannot shortcut to truth.

Item edit-classes:
  true,  unedited  : original A&M true statement
  true,  edited    : truth-PRESERVING swap (another KB-valid true pair)
  false, unedited  : original A&M false statement (natural, not from OUR pipeline)
  false, edited    : truth-BREAKING swap (subject kept, object -> a wrong same-type object)

Only structured topics with a clean (subject, relation, object) template are used
for swaps; freeform topics (facts/companies/animals) are out of scope for v1
(they cannot support controlled truth-preserving edits). O-2 natural-error
harvesting from a disjoint model can later replace the 'false, unedited' half.
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
AM_DIR = _ROOT / "data" / "raw" / "azaria_mitchell"

# topic -> (regex capturing subject, object; template to rebuild; object type tag)
TEMPLATES = {
    "cities":     (re.compile(r"^(.+?) is a city in (.+?)\.?$"),
                   "{s} is a city in {o}.", "country"),
    "generated":  (re.compile(r"^(.+?) is located in (.+?)\.?$"),
                   "{s} is located in {o}.", "place"),
    "inventions": (re.compile(r"^(.+?) invented (.+?)\.?$"),
                   "{s} invented {o}.", "invention"),
    "elements":   (re.compile(r"^(.+?) is used in (.+?)\.?$"),
                   "{s} is used in {o}.", "use"),
    # Added 2026-08-07. The A&M release ships eight topics; v1 used four, which is
    # why the corpus could only ever support two balanced domains. `companies` is
    # structurally identical to `cities` (object is a country, so swaps draw from
    # the same frequency-matched pool); `animals` adds a third object type.
    # `capitals` is deliberately NOT added: its object takes only two values
    # ("city"/"country"), so it carries no frequency variation and a swap between
    # them is trivially detectable. `facts` has no parseable subject-object form.
    "companies":  (re.compile(r"^(.+?) has headquarters in (.+?)\.?$"),
                   "{s} has headquarters in {o}.", "country"),
    "animals":    (re.compile(r"^(.+?) uses (.+?) for locomotion\.?$"),
                   "{s} uses {o} for locomotion.", "locomotion"),
}

# Negated rebuild per topic, reusing TEMPLATES' regexes so the two cannot drift.
NEGATED_TEMPLATES = {
    "cities":     "{s} is not a city in {o}.",
    "generated":  "{s} is not located in {o}.",
    "inventions": "{s} did not invent {o}.",
    "elements":   "{s} is not used in {o}.",
    "companies":  "{s} does not have headquarters in {o}.",
    "animals":    "{s} does not use {o} for locomotion.",
}


def negate(text: str) -> str:
    for topic, (rx, _, _) in TEMPLATES.items():
        m = rx.match(text)
        if m:
            return NEGATED_TEMPLATES[topic].format(s=m.group(1), o=m.group(2))
    raise ValueError(f"no negation rule matches: {text!r}")


def negate_all(texts: list[str]) -> list[str]:
    # A silently un-negated item hands CCS an identical pair, which satisfies its
    # consistency loss trivially and is invisible in the output AUROC.
    out = [negate(t) for t in texts]
    if len(set(out)) != len(set(texts)):
        raise ValueError("negation collapsed distinct statements onto each other")
    return out


def _rows(topic: str) -> list[dict]:
    path = AM_DIR / f"{topic}_true_false.csv"
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _parse_true_pairs(topic: str) -> list[tuple[str, str]]:
    """(subject, object) from TRUE statements that match the topic template."""
    rx, _, _ = TEMPLATES[topic]
    pairs = []
    for r in _rows(topic):
        if r["label"] != "1":
            continue
        m = rx.match(r["statement"].strip())
        if m:
            pairs.append((m.group(1).strip(), m.group(2).strip()))
    return pairs


def _unedited_false(topic: str) -> list[str]:
    """Original A&M FALSE statements matching the template (not our pipeline's edit)."""
    rx, _, _ = TEMPLATES[topic]
    out = []
    for r in _rows(topic):
        if r["label"] == "0" and rx.match(r["statement"].strip()):
            out.append(r["statement"].strip())
    return out


def _object_frequency(topic: str) -> dict[str, int]:
    """Commonness proxy: how many statements each object appears in (self-contained,
    reproducible, no network). 'United States'/'China' are common; 'Swaziland' rare.
    Used to make FT (fluent-typical) vs FA (odd-atypical) falsehoods on purpose —
    D-003's 'frequency-matched swap' spec that v1 under-implemented (it swapped to
    RANDOM same-type objects, so most lies were odd -> FA, starving FT)."""
    from collections import Counter
    rx, _, _ = TEMPLATES[topic]
    c: Counter = Counter()
    for r in _rows(topic):
        m = rx.match(r["statement"].strip())
        if m:
            c[m.group(2).strip()] += 1
    return dict(c)


def generate_topic(topic: str, edit_rate: float = 0.5, seed: int = 20260712) -> list[dict]:
    """Balanced candidate items for one structured topic, edited ⊥ truth."""
    if topic not in TEMPLATES:
        raise ValueError(f"{topic} has no structured template (freeform — out of v1 scope)")
    _, tpl, otype = TEMPLATES[topic]
    rng = np.random.default_rng(seed)

    pairs = _parse_true_pairs(topic)
    if len(pairs) < 4:
        return []
    subjects = [s for s, _ in pairs]
    objects = [o for _, o in pairs]
    true_obj = {}
    for s, o in pairs:
        true_obj.setdefault(s, set()).add(o)
    obj_pool = sorted(set(objects))

    items: list[dict] = []

    def add(text, truth, edited, entity, strategy, obj=None):
        items.append({
            "text": text, "truth": bool(truth), "edited": bool(edited),
            "entity": entity, "entities": [e for e in (entity, obj) if e],
            "domain": topic,
            "provenance": {"source": "azaria_mitchell", "strategy": strategy,
                           "object_type": otype},
        })

    # ── TRUE cell: edit_rate fraction truth-preserving swaps, rest originals ──
    n_true = len(pairs)
    edit_true_idx = set(rng.choice(n_true, int(round(edit_rate * n_true)), replace=False).tolist())
    for i, (s, o) in enumerate(pairs):
        if i in edit_true_idx:
            s2, o2 = pairs[int(rng.integers(len(pairs)))]        # another KB-valid TRUE pair
            add(tpl.format(s=s2, o=o2), True, True, s2, "truth_preserving_swap", o2)
        else:
            add(tpl.format(s=s, o=o), True, False, s, "original_true", o)

    # ── FALSE cell: match n_true; edit_rate fraction truth-breaking swaps ─────
    nat_false = _unedited_false(topic)
    n_edit_false = int(round(edit_rate * n_true))
    n_unedit_false = n_true - n_edit_false
    # frequency-aware swap (D-003): half the truth-breaking swaps use a COMMON
    # wrong object (fluent -> low-ppl -> FT), half a RARE one (odd -> high-ppl ->
    # FA). Deliberate cell coverage, not random (v1's random swap starved FT).
    freq = _object_frequency(topic)
    ranked = sorted(obj_pool, key=lambda o: freq.get(o, 0), reverse=True)
    common = ranked[:max(1, len(ranked) // 3)]      # top-tercile-frequency objects
    rare = ranked[len(ranked) - max(1, len(ranked) // 3):]
    made = 0
    guard = 0
    while made < n_edit_false and guard < n_edit_false * 80:
        guard += 1
        s = subjects[int(rng.integers(len(subjects)))]
        pool = common if (made % 2 == 0) else rare   # alternate FT-lean / FA-lean
        o_wrong = pool[int(rng.integers(len(pool)))]
        if o_wrong in true_obj.get(s, set()):
            continue  # would be true — reject
        lean = "FT_lean" if pool is common else "FA_lean"
        add(tpl.format(s=s, o=o_wrong), False, True, s, f"truth_breaking_swap_{lean}", o_wrong)
        made += 1
    # unedited natural falses (A&M-constructed, not OUR pipeline)
    if nat_false:
        pick = rng.choice(len(nat_false), min(n_unedit_false, len(nat_false)), replace=False)
        for j in pick.tolist():
            txt = nat_false[j]
            rx = TEMPLATES[topic][0]
            m = rx.match(txt)
            subj = m.group(1).strip() if m else ""
            obj = m.group(2).strip() if m else None
            add(txt, False, False, subj, "original_false", obj)

    return items


def build_candidate_items(topics: list[str] | None = None, edit_rate: float = 0.5,
                          seed: int = 20260712) -> list[dict]:
    """All structured topics concatenated. Typicality scored later (Modal)."""
    topics = topics or list(TEMPLATES)
    out: list[dict] = []
    for t in topics:
        if t in TEMPLATES:
            out.extend(generate_topic(t, edit_rate=edit_rate, seed=seed))
    return out


def edited_truth_independence(items: list[dict]) -> dict:
    """Verify the load-bearing invariant: P(edited|true) ≈ P(edited|false)."""
    tru = [it for it in items if it["truth"]]
    fal = [it for it in items if not it["truth"]]
    p_edit_true = np.mean([it["edited"] for it in tru]) if tru else float("nan")
    p_edit_false = np.mean([it["edited"] for it in fal]) if fal else float("nan")
    return {
        "n_true": len(tru), "n_false": len(fal),
        "p_edited_given_true": round(float(p_edit_true), 4),
        "p_edited_given_false": round(float(p_edit_false), 4),
        "gap": round(abs(float(p_edit_true) - float(p_edit_false)), 4),
        "balanced": bool(abs(float(p_edit_true) - float(p_edit_false)) <= 0.05),
    }
