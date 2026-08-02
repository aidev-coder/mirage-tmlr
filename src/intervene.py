"""
Model-level intervention: does ablating the frequency manifold from the residual
stream flip the model's OWN stated answer on a fluent falsehood?

We ask the model to judge each statement true/false and read P(true) from the
next-token logits. Then we hook the residual at the headline layer, project the
judgment token off the frequency manifold (the same low-rank subspace the causal
mediation found), and re-read P(true). If the model stops calling fluent lies
(FT) true, the manifold is not just what a probe reads — it is what the MODEL
computes and reports as truth. A random subspace of equal size is the null.

Manifold basis, mean/std, and the FA reference level are all computed from the
model's activations on the SAME judgment prompts, so the reset lives in the space
it is applied to.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .causal import _fit_logreg, _standardize, _typicality_subspace
from .stage3 import load_corpus

DIAGONAL = ("TT", "FA")

_ROOT = Path(__file__).resolve().parent.parent
PROMPT = ('Is the following statement true or false?\n'
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


def _format(substrate, text):
    """Chat-template the judgment for instruct models; plain completion otherwise."""
    tok = substrate.tokenizer
    body = PROMPT.format(text=text)
    if getattr(tok, "chat_template", None):
        return tok.apply_chat_template([{"role": "user", "content": body}],
                                       tokenize=False, add_generation_prompt=True)
    return body


def _p_true(substrate, prompt, true_ids, false_ids, hook_state=None):
    import torch
    tok = substrate.tokenizer
    add_special = not getattr(tok, "chat_template", None)   # chat template already has them
    enc = tok(prompt, return_tensors="pt", add_special_tokens=add_special).to(substrate.model.device)
    with torch.no_grad():
        logits = substrate.model(**enc).logits[0, -1]
    probs = torch.softmax(logits, dim=-1)
    pt = float(probs[true_ids].sum())
    pf = float(probs[false_ids].sum())
    return pt / (pt + pf + 1e-12)


def _make_hook(mu, sd, basis, z_ref):
    """basis: (d, k) orthonormal. Reset the last token's manifold projection to the
    FA reference level, leaving the orthogonal complement untouched."""
    import torch
    mu_t = torch.tensor(mu); sd_t = torch.tensor(sd)
    B = torch.tensor(basis); zr = torch.tensor(z_ref)

    def hook(module, inp, out):
        hs = out[0] if isinstance(out, tuple) else out
        dev, dt = hs.device, hs.dtype
        z = (hs[:, -1, :].to(torch.float64) - mu_t.to(dev)) / sd_t.to(dev)
        proj = (z - zr.to(dev)) @ B.to(dev)          # (b, k)
        z = z - proj @ B.to(dev).T
        hs[:, -1, :] = (z * sd_t.to(dev) + mu_t.to(dev)).to(dt)
        if isinstance(out, tuple):
            return (hs,) + tuple(out[1:])
        return hs
    return hook


def _cell_p_true(substrate, prompts, idx, true_ids, false_ids, hook=None, layer=None, cap=200, seed=0):
    rng = np.random.default_rng(seed)
    if len(idx) > cap:
        idx = rng.choice(idx, cap, replace=False)
    handle = None
    if hook is not None:
        block = substrate.model.model.layers[layer - 1]
        handle = block.register_forward_hook(hook)
    try:
        vals = [_p_true(substrate, prompts[i], true_ids, false_ids) for i in idx]
    finally:
        if handle is not None:
            handle.remove()
    return float(np.mean(vals)), int(len(idx))


def run(substrate, corpus_path: str | Path, k: int = 8, layer: int | None = None,
        seed: int = 20260719, cap: int = 200, commit_fn=None,
        domain: str | None = None) -> dict:
    items = load_corpus(corpus_path)
    if domain:   # pooled corpus confounds domain with truth across cells (2026-08-03)
        items = [it for it in items if it["domain"] == domain]
    texts = [it["text"] for it in items]
    cells = np.array([it["cell"] for it in items])
    freq = np.array([it["typicality"]["entity_freq_log10"] for it in items], dtype=float)
    prompts = [_format(substrate, t) for t in texts]

    tok = substrate.tokenizer
    true_ids = _first_ids(tok, TRUE_WORDS)
    false_ids = _first_ids(tok, FALSE_WORDS)

    Hp = substrate.hidden_states_matrix(prompts, batch_size=32)   # [n, L+1, d]
    n_layers = Hp.shape[1]
    L = layer if layer is not None else n_layers // 2
    Z = _standardize(Hp[:, L, :].astype(np.float64))
    mu = Hp[:, L, :].astype(np.float64).mean(0)
    sd = np.where(Hp[:, L, :].astype(np.float64).std(0) < 1e-8, 1.0, Hp[:, L, :].astype(np.float64).std(0))
    U = _typicality_subspace(Z, freq, k)
    z_ref = Z[cells == "FA"].mean(0)
    d = Z.shape[1]
    U_rand, _ = np.linalg.qr(np.random.default_rng(seed).standard_normal((d, k)))
    U_rand = U_rand[:, :k]

    # dissociation check: the fielded probe read on the SAME judgment-prompt
    # activations, so probe-readout and behavioral output share the input.
    truth = np.array([bool(it["truth"]) for it in items])
    diag = np.isin(cells, DIAGONAL)
    w_field, b_field = _fit_logreg(Z[diag], truth[diag], seed)
    probe_p = 1.0 / (1.0 + np.exp(-(Z @ w_field + b_field)))

    cell_idx = {c: np.flatnonzero(cells == c) for c in ("TT", "TA", "FT", "FA")}
    hook_man = _make_hook(mu, sd, U, z_ref)
    hook_rnd = _make_hook(mu, sd, U_rand, z_ref)

    out = {"model": substrate.model_id, "domain": domain or "all",
           "corpus": Path(corpus_path).name,
           "layer": L, "k": k, "n_by_cell": {c: int(len(v)) for c, v in cell_idx.items()},
           "p_true": {}, "provenance": "measured", "seed": seed}
    for c, idx in cell_idx.items():
        base, n = _cell_p_true(substrate, prompts, idx, true_ids, false_ids, cap=cap, seed=seed)
        man, _ = _cell_p_true(substrate, prompts, idx, true_ids, false_ids, hook_man, L, cap, seed)
        rnd, _ = _cell_p_true(substrate, prompts, idx, true_ids, false_ids, hook_rnd, L, cap, seed)
        out["p_true"][c] = {"n": n, "behavioral_baseline": round(base, 4),
                            "manifold_ablated": round(man, 4), "random_ablated": round(rnd, 4),
                            "probe_readout": round(float(probe_p[idx].mean()), 4)}
        print(f"  {c}: PROBE P(true)={probe_p[idx].mean():.3f}  vs  BEHAVIOR P(true)={base:.3f} "
              f"(manifold-ablated {man:.3f})", flush=True)
        if commit_fn:
            commit_fn()
    return out
