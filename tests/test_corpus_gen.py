"""
Tests for the Stage-2 symmetric-swap generator. Pins the load-bearing §4.2
invariant (edit provenance ⊥ truth) and swap correctness on the REAL A&M data.
No GPU / network. These guard the recursive-trap defense (D-003).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import corpus_gen as cg  # noqa: E402


def test_edited_orthogonal_to_truth():
    """The whole point: edit provenance must not predict truth, or the truth probe
    can shortcut on "was edited" and every Stage-3 number is contaminated.

    This is now enforced at BALANCING rather than at generation, and asserted on the
    released corpus rather than on the candidate pool. Deduplicating (D-016) drops
    items that are all edited+true, since a truth-preserving swap landing on an
    existing original is always both, so the pool itself is left mildly imbalanced.
    `_balance_by_domain` then draws `edited` 50/50 inside every cell (D-017), which
    is strictly stronger than pool-level independence: equal P(edited) per cell
    implies equal P(edited) per truth value, and also kills the diagonal-only
    variant that pool-level balance would have missed.
    """
    items = cg.build_candidate_items(seed=1)
    assert len(items) > 200, len(items)
    inv = cg.edited_truth_independence(items)
    # both classes present (the canary needs 2-class 'edited' AND 2-class 'truth')
    assert 0.0 < inv["p_edited_given_true"] < 1.0
    assert 0.0 < inv["p_edited_given_false"] < 1.0
    assert inv["gap"] < 0.15, inv        # pool may drift; a large gap is still wrong

    import json
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(root))
    from src import corpus_build as cb
    released = root / "data" / "corpus" / "mirage_2x2_v44b4126cba1c.jsonl"
    if released.exists():
        corpus = [json.loads(x) for x in released.read_text(encoding="utf-8").splitlines() if x.strip()]
        eb = cb.edit_balance_check(corpus)
        assert eb["pass"], eb
        assert eb["cell_spread"] == 0.0, eb
        assert eb["truth_gap"] == 0.0, eb


def test_truth_labels_correct():
    """Truth-preserving swaps stay in the KB; truth-breaking swaps leave it."""
    pairs = cg._parse_true_pairs("cities")
    true_obj = {}
    for s, o in pairs:
        true_obj.setdefault(s, set()).add(o)
    items = cg.generate_topic("cities", seed=2)
    rx = cg.TEMPLATES["cities"][0]
    for it in items:
        m = rx.match(it["text"])
        if not m:            # unedited natural falses may not re-parse; skip
            continue
        s, o = m.group(1).strip(), m.group(2).strip()
        if it["provenance"]["strategy"].startswith("truth_breaking_swap"):
            assert o not in true_obj.get(s, set()), f"swap not actually false: {it['text']}"
            assert it["truth"] is False
        if it["provenance"]["strategy"] in ("original_true", "truth_preserving_swap"):
            assert it["truth"] is True
    print("PASS  swap truth labels consistent with the KB")


def test_schema_and_entities():
    items = cg.build_candidate_items(seed=3)
    for it in items[:50]:
        assert set(it) >= {"text", "truth", "edited", "entity", "domain", "provenance"}
        assert isinstance(it["truth"], bool) and isinstance(it["edited"], bool)
        assert it["domain"] in cg.TEMPLATES
    # every structured topic contributes
    domains = {it["domain"] for it in items}
    assert domains == set(cg.TEMPLATES), domains
    print("PASS  schema + all structured topics present:", sorted(domains))


if __name__ == "__main__":
    test_edited_orthogonal_to_truth()
    test_truth_labels_correct()
    test_schema_and_entities()
    print("\nALL PASS — Stage-2 generator (symmetric swap, edited ⊥ truth)")


def test_fragmentation_canary_mechanics():
    """C3 canary computes features + trains on real corpus text (tiny tokenizer)."""
    import random
    import tempfile
    import numpy as np
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from test_substrate_plumbing import build_tiny_substrate
    from src import corpus_build as cb
    with tempfile.TemporaryDirectory() as tmp:
        sub = build_tiny_substrate(tmp)
        items = cg.build_candidate_items(seed=5)
        random.Random(5).shuffle(items)   # items are true-then-false per topic
        items = items[:400]
        feats = cb.fragmentation_features([it["text"] for it in items],
                                          [it["entity"] for it in items], sub.tokenizer)
        assert feats.shape == (len(items), 5)
        res = cb.fragmentation_canary(feats, np.array([it["truth"] for it in items]))
        assert res["gate"] == "fragmentation_canary" and "auroc" in res and "pass" in res
        print("PASS  fragmentation canary mechanics:",
              {k: res[k] for k in ("auroc", "pass")})


def test_negation_templates():
    assert cg.negate("Monywa is a city in Myanmar.") == "Monywa is not a city in Myanmar."
    assert (cg.negate("Anthem has headquarters in France.")
            == "Anthem does not have headquarters in France.")
    assert (cg.negate("Godfrey Hounsfield invented the electromagnetism discovery.")
            == "Godfrey Hounsfield did not invent the electromagnetism discovery.")


def test_negation_covers_every_template():
    assert set(cg.NEGATED_TEMPLATES) == set(cg.TEMPLATES)


def test_negation_rejects_unrecognised_statement():
    import pytest
    with pytest.raises(ValueError):
        cg.negate("The sky is blue")


def test_negation_is_total_over_every_built_corpus():
    import json
    for path in sorted((Path(__file__).resolve().parent.parent
                        / "data" / "corpus").glob("mirage_2x2_v*.jsonl")):
        texts = [json.loads(l)["text"]
                 for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        out = cg.negate_all(texts)
        assert len(out) == len(texts)
        assert all(o != t for o, t in zip(out, texts)), path.name
