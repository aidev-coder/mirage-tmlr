"""
Generation-time test: does the confound survive where hallucination actually happens?

Everything so far judged statements WE wrote. Here the model writes them. We prompt
it to complete "<subject> is a city in ___", greedily decode the object, and check
the answer against ground truth taken from the corpus's true items. Whatever it gets
wrong is a natural hallucination: no entity swap, no edit signature, nothing we
authored. That retires the generation-artifact objection outright.

Then, on those self-generated statements, we read three things:

  probe       the fielded probe (trained on the corpus diagonal, the field's recipe)
              scoring the model's own output.
  behavior    the model's own true/false judgment of the statement it just produced.
  truth       whether the generated object is actually right.

Two questions fall out. Does the field's probe catch the model's own hallucinations
(AUROC of probe vs generated-truth)? And does the judgment-time dissociation hold
here, where the model is grading its own work rather than ours?

Stratified by subject frequency, because the confound predicts the probe should fail
specifically on hallucinations about COMMON entities: the natural analogue of the FT
cell, arrived at without us constructing anything.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from .causal import _fit_logreg, _standardize
from .stage3 import load_corpus

DIAGONAL = ("TT", "FA")
# Constrained so the answer is a bare country name. A free completion
# ("<city> is a city in ___") makes the model answer with containing regions and
# demonyms ("Shan State of Myanmar", "the Indian state of ..."), and scoring those
# needs a matcher whose leniency we would end up tuning against the error rate —
# the exact knob the project's standing directive §1.1 forbids. This keeps the model generating the fact
# from parametric memory while leaving ground truth unambiguous.
ASK = ("In which country is {s} located? "
       "Reply with ONLY the country name and nothing else.")
COMPLETE = "{s} is a city in"
JUDGE = ('Is the following statement true or false?\n'
         'Statement: {text}\nAnswer (true or false):')
TRUE_WORDS = [" true", " True", "true", "True"]
FALSE_WORDS = [" false", " False", "false", "False"]


def _first_ids(tokenizer, words):
    ids = set()
    for w in words:
        t = tokenizer(w, add_special_tokens=False)["input_ids"]
        if t:
            ids.add(t[0])
    return sorted(ids)


def _greedy(substrate, prompt: str, max_new_tokens: int = 8) -> str:
    import torch
    tok = substrate.tokenizer
    add_special = not getattr(tok, "chat_template", None)   # template carries its own
    enc = tok(prompt, return_tensors="pt",
              add_special_tokens=add_special).to(substrate.model.device)
    with torch.no_grad():
        out = substrate.model.generate(**enc, do_sample=False,
                                       max_new_tokens=max_new_tokens,
                                       pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)


def _clean_object(ans: str) -> str:
    """First sentence of the continuation. We keep the whole clause rather than the
    first noun: the model often answers with a containing region ("Shan State of
    Myanmar", "north central Ukraine"), which names the right country and must count
    as correct."""
    ans = ans.strip().split("\n")[0]
    ans = re.split(r"[.;!?]", ans)[0]
    return re.sub(r"^(the|a|an)\s+", "", ans.strip(), flags=re.I).strip()


ALIASES = {
    "united states": ["usa", "us", "united states of america", "america"],
    "united kingdom": ["uk", "england", "scotland", "wales", "great britain", "britain"],
    "russia": ["russian federation"], "south korea": ["korea", "republic of korea"],
    "north korea": ["dprk"], "czech republic": ["czechia"], "myanmar": ["burma"],
    "netherlands": ["holland"], "swaziland": ["eswatini"], "ivory coast": ["cote divoire"],
    "east timor": ["timorleste"], "cape verde": ["cabo verde"],
    "democratic republic of the congo": ["drc", "congo kinshasa", "zaire"],
    "uae": ["united arab emirates"], "vatican city": ["vatican", "holy see"],
}


def _norm(s: str) -> str:
    s = re.sub(r"^(the|a|an)\s+", "", s.lower().strip())
    return re.sub(r"\s+", " ", re.sub(r"[^a-z ]", " ", s)).strip()


def _match(generated: str, truth_obj: str) -> bool:
    """Correct if the continuation NAMES the right country anywhere in it. The model
    frequently gives a region that contains the country ("Shan State of Myanmar"), so
    substring containment on word boundaries is the right test, not prefix equality.
    Prefix matching also produced false positives on truncations ("U" ~ "United
    States"), which the length guard rules out."""
    g, t = _norm(generated), _norm(truth_obj)
    if not g or len(t) < 3:
        return False
    cands = [t] + [_norm(a) for a in ALIASES.get(t, [])]
    for c in cands:
        if c and re.search(rf"\b{re.escape(c)}\b", g):
            return True
    return False


def _judge_p_true(substrate, text, true_ids, false_ids):
    import torch
    tok = substrate.tokenizer
    body = JUDGE.format(text=text)
    if getattr(tok, "chat_template", None):
        body = tok.apply_chat_template([{"role": "user", "content": body}],
                                       tokenize=False, add_generation_prompt=True)
        add_special = False
    else:
        add_special = True
    enc = tok(body, return_tensors="pt", add_special_tokens=add_special).to(substrate.model.device)
    with torch.no_grad():
        probs = torch.softmax(substrate.model(**enc).logits[0, -1], dim=-1)
    pt = float(probs[true_ids].sum()); pf = float(probs[false_ids].sum())
    return pt / (pt + pf + 1e-12)


def run(substrate, corpus_path: str | Path, max_subjects: int = 400,
        seed: int = 20260719, commit_fn=None) -> dict:
    from .stats import auroc_with_ci

    items = load_corpus(corpus_path)
    cities = [it for it in items if it["domain"] == "cities"]

    # ground truth: subject -> true country, from the corpus's true items
    truth_map, freq_map = {}, {}
    for it in cities:
        if it["truth"]:
            truth_map[it["entities"][0]] = it["entities"][1]
        freq_map[it["entities"][0]] = it["typicality"]["entity_freq_log10"]
    subjects = sorted(truth_map)
    rng = np.random.default_rng(seed)
    if len(subjects) > max_subjects:
        subjects = list(rng.choice(subjects, max_subjects, replace=False))

    tok = substrate.tokenizer
    true_ids, false_ids = _first_ids(tok, TRUE_WORDS), _first_ids(tok, FALSE_WORDS)

    def _ask(s: str) -> str:
        q = ASK.format(s=s)
        if getattr(tok, "chat_template", None):
            q = tok.apply_chat_template([{"role": "user", "content": q}],
                                        tokenize=False, add_generation_prompt=True)
        return q

    gen_texts, gen_correct, gen_freq, records = [], [], [], []
    for i, s in enumerate(subjects):
        obj = _clean_object(_greedy(substrate, _ask(s), max_new_tokens=12))
        if not obj:
            continue
        text = f"{s} is a city in {obj}."
        ok = _match(obj, truth_map[s])
        gen_texts.append(text); gen_correct.append(ok); gen_freq.append(freq_map[s])
        records.append({"subject": s, "generated": obj, "gold": truth_map[s],
                        "correct": bool(ok), "freq_log10": round(float(freq_map[s]), 3)})
        if commit_fn and i % 50 == 0:
            commit_fn()
    gen_correct = np.array(gen_correct); gen_freq = np.array(gen_freq)
    print(f"  generated {len(gen_texts)}; natural error rate "
          f"{1 - gen_correct.mean():.3f}", flush=True)

    # fielded probe: fit on the corpus diagonal, apply to the model's own generations
    corpus_texts = [it["text"] for it in items]
    cells = np.array([it["cell"] for it in items])
    truth = np.array([bool(it["truth"]) for it in items])
    Hc = substrate.hidden_states_matrix(corpus_texts, batch_size=32)
    Hg = substrate.hidden_states_matrix(gen_texts, batch_size=32)
    L = Hc.shape[1] // 2
    Xc, Xg = Hc[:, L, :].astype(np.float64), Hg[:, L, :].astype(np.float64)
    mu, sd = Xc.mean(0), np.where(Xc.std(0) < 1e-8, 1.0, Xc.std(0))
    diag = np.isin(cells, DIAGONAL)
    w, b = _fit_logreg(_standardize(Xc)[diag], truth[diag], seed)
    probe_gen = 1.0 / (1.0 + np.exp(-(((Xg - mu) / sd) @ w + b)))

    behav = np.array([_judge_p_true(substrate, t, true_ids, false_ids) for t in gen_texts])
    for r, p, bv in zip(records, probe_gen, behav):
        r["probe_p_true"] = round(float(p), 4); r["behavior_p_true"] = round(float(bv), 4)

    hi = gen_freq >= np.median(gen_freq)
    out = {
        "model": substrate.model_id, "corpus": Path(corpus_path).name, "layer": L,
        "n_generated": len(gen_texts),
        "natural_error_rate": round(float(1 - gen_correct.mean()), 4),
        "provenance": "measured", "seed": seed,
    }
    if 0 < gen_correct.sum() < len(gen_correct):
        out["probe_auroc_on_own_generations"] = auroc_with_ci(gen_correct, probe_gen, seed=seed)
        out["behavior_auroc_on_own_generations"] = auroc_with_ci(gen_correct, behav, seed=seed)
    err = ~gen_correct
    out["hallucinations"] = {
        "n": int(err.sum()),
        "probe_mean_p_true": round(float(probe_gen[err].mean()), 4) if err.any() else None,
        "behavior_mean_p_true": round(float(behav[err].mean()), 4) if err.any() else None,
        "n_high_freq": int((err & hi).sum()),
        "probe_mean_p_true_high_freq": round(float(probe_gen[err & hi].mean()), 4) if (err & hi).any() else None,
        "behavior_mean_p_true_high_freq": round(float(behav[err & hi].mean()), 4) if (err & hi).any() else None,
    }
    out["correct_generations"] = {
        "n": int(gen_correct.sum()),
        "probe_mean_p_true": round(float(probe_gen[gen_correct].mean()), 4) if gen_correct.any() else None,
        "behavior_mean_p_true": round(float(behav[gen_correct].mean()), 4) if gen_correct.any() else None,
    }
    out["records"] = records[:200]
    h, c = out["hallucinations"], out["correct_generations"]
    print(f"  hallucinations n={h['n']}: probe P(true)={h['probe_mean_p_true']} "
          f"behavior={h['behavior_mean_p_true']}", flush=True)
    print(f"  correct       n={c['n']}: probe P(true)={c['probe_mean_p_true']} "
          f"behavior={c['behavior_mean_p_true']}", flush=True)
    return out
