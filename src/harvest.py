"""
Stage 2 — natural-error harvest (O-2 / D-008) from a DISJOINT model.

Swap-based false generation left a persistent hidden-state edit signature that
the edit canary detects (v1-v3, ~0.67). Harvesting a disjoint model's GENUINE
confident errors removes that signature by construction: the false items are
real model outputs, not edits of true statements. The harvest model
(Mistral-7B-Instruct) is disjoint from every probed substrate (Llama/Qwen/
Gemma/Pythia), so its error style cannot leak into a probe's train/test on those.

Method (cities shown; analogous per structured topic): ask the disjoint model
the factual question behind each true (subject, object) pair; keep answers that
are (a) a VALID same-type object (in the KB object pool) and (b) WRONG (not the
true object). Those are fluent, confident, natural falsehoods. Typicality (FT
vs FA) is then sorted by the D-007 frequency axis, same as everything else.

Deterministic (greedy) so the harvest is reproducible; cached to the volume.
"""
from __future__ import annotations

import re

from . import corpus_gen as cg

# per-topic factual question + answer-slot (object type)
HARVEST_Q = {
    "cities":     ("Which country is the city {s} in? Reply with ONLY the country name.",
                   "{s} is a city in {o}."),
    "generated":  ("In which country or region is {s} located? Reply with ONLY the place name.",
                   "{s} is located in {o}."),
    "inventions": ("Who invented the {s}? Reply with ONLY the inventor's full name.",
                   "{s} invented the {o}."),  # note: template differs; handled below
}


def _greedy(substrate, prompt: str, max_new_tokens: int = 16) -> str:
    import torch
    tok = substrate.tokenizer
    msgs = [{"role": "user", "content": prompt}]
    try:
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    except Exception:
        text = prompt
    enc = tok(text, return_tensors="pt").to(substrate.model.device)
    with torch.no_grad():
        out = substrate.model.generate(**enc, do_sample=False, max_new_tokens=max_new_tokens,
                                       pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def _clean(ans: str) -> str:
    ans = ans.strip().strip('."\'').split("\n")[0].strip()
    ans = re.sub(r"^(the|it is|it's|located in|in)\s+", "", ans, flags=re.I).strip()
    return ans.rstrip(".")


def harvest_topic(substrate, topic: str, max_subjects: int = 400,
                  seed: int = 20260719) -> dict:
    """Return {items: [natural false items], stats}. Only 'cities'/'generated'
    use the (subject -> object) question form; objects validated against the KB
    object pool so harvested falsehoods are same-type, fluent, and genuinely wrong."""
    import numpy as np
    if topic not in ("cities", "generated"):
        return {"items": [], "stats": {"note": f"{topic} not harvested in v1 O-2"}}

    q_tpl, stmt_tpl = HARVEST_Q[topic]
    pairs = cg._parse_true_pairs(topic)
    true_obj: dict = {}
    for s, o in pairs:
        true_obj.setdefault(s, set()).add(o)
    obj_pool = {o.lower(): o for _, o in pairs}   # canonical-case lookup

    rng = np.random.default_rng(seed)
    subjects = sorted(true_obj)
    rng.shuffle(subjects)
    subjects = subjects[:max_subjects]

    items, n_valid_obj, n_wrong, n_asked = [], 0, 0, 0
    for s in subjects:
        n_asked += 1
        ans = _clean(_greedy(substrate, q_tpl.format(s=s)))
        canon = obj_pool.get(ans.lower())
        if canon is None:
            continue                       # not a same-type object we can KB-check
        n_valid_obj += 1
        if canon in true_obj.get(s, set()):
            continue                       # model was right -> not a false item
        n_wrong += 1
        items.append({
            "text": stmt_tpl.format(s=s, o=canon),
            "truth": False, "edited": False,          # NATURAL output, no edit
            "entity": s, "entities": [s, canon], "domain": topic,
            "provenance": {"source": "harvest_mistral", "strategy": "natural_confident_error",
                           "verified_against": "azaria_mitchell_kb"},
        })
        if n_asked % 50 == 0:
            print(f"    [harvest {topic}] {n_asked}/{len(subjects)} asked, "
                  f"{n_wrong} genuine errors", flush=True)
    return {"items": items,
            "stats": {"topic": topic, "asked": n_asked, "valid_object": n_valid_obj,
                      "genuine_errors": n_wrong, "error_rate": round(n_wrong / max(n_asked, 1), 3)}}
