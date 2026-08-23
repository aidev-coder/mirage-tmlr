"""
The contested axis: typicality scoring, triangulated per decisions.md D-002.

Three operationalizations, deliberately kept separate in every output:

  primary    cross-family reference-LM perplexity  -> used for 2x2 cell assignment
  secondary  entity corpus frequency (infini-gram) -> cross-check
  tertiary   substrate model's own perplexity      -> reported, FLAGGED CIRCULAR,
                                                      never used for assignment

The circularity flag exists because probe score and self-perplexity are two
functions of the same forward pass; a mediation analysis conditioning on
self-perplexity partly conditions on the probe itself (see D-002 reasoning).

Axis agreement (Spearman) is computed and reported — disagreement between the
three axes is itself a finding (.6), not a nuisance to hide.
"""

from __future__ import annotations

import time

import numpy as np

INFINIGRAM_API = "https://api.infini-gram.io/"
INFINIGRAM_INDEX = "v4_dolma-v1_7_llama"  # Dolma 1.7 — open pretraining-scale corpus


def reference_perplexity(texts: list[str], reference_substrate) -> np.ndarray:
    """Mean-per-token perplexity of each text under a *different-family* reference LM.

    `reference_substrate` is a src.substrate.Substrate for the model mapped in
    configs/models.yaml `reference_lms` — never the substrate being probed.
    """
    return np.array([float(np.exp(reference_substrate.nll(t))) for t in texts])


def self_perplexity(texts: list[str], probed_substrate) -> np.ndarray:
    """Tertiary axis. CIRCULAR with respect to probes on `probed_substrate` —
    report it, never assign cells with it."""
    return np.array([float(np.exp(probed_substrate.nll(t))) for t in texts])


def entity_frequency(entities: list[str], index: str = INFINIGRAM_INDEX,
                     sleep_s: float = 0.1) -> np.ndarray:
    """Corpus n-gram counts for entity strings via the infini-gram API.

    Returns log10(1 + count). Network-dependent; failures return NaN for that
    entity and are reported, not silently imputed.
    """
    import requests

    out = np.full(len(entities), np.nan)
    for i, ent in enumerate(entities):
        try:
            r = requests.post(INFINIGRAM_API, json={
                "index": index, "query_type": "count", "query": ent,
            }, timeout=20)
            r.raise_for_status()
            payload = r.json()
            if "count" in payload:
                out[i] = np.log10(1 + payload["count"])
        except Exception:
            pass  # stays NaN; caller must report coverage
        time.sleep(sleep_s)
    return out


def axis_agreement(primary: np.ndarray, secondary: np.ndarray,
                   tertiary: np.ndarray | None = None) -> dict:
    """Spearman agreement matrix between the typicality axes.

    Frequency should anti-correlate with perplexity; the matrix reports raw signs.
    NaNs (e.g. infini-gram misses) are dropped pairwise, with coverage reported.
    """
    from scipy.stats import spearmanr

    axes = {"reference_ppl": primary, "entity_freq": secondary}
    if tertiary is not None:
        axes["self_ppl"] = tertiary

    names = list(axes)
    result: dict = {"n": int(len(primary)), "pairs": {}}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            x, y = axes[a], axes[b]
            mask = np.isfinite(x) & np.isfinite(y)
            rho, p = spearmanr(x[mask], y[mask]) if mask.sum() >= 3 else (np.nan, np.nan)
            result["pairs"][f"{a}~{b}"] = {
                "spearman_rho": None if np.isnan(rho) else round(float(rho), 4),
                "p": None if np.isnan(p) else float(p),
                "coverage": round(float(mask.mean()), 3),
            }
    return result


def score_items(texts: list[str], entities: list[str | None],
                reference_substrate, probed_substrate=None) -> list[dict]:
    """Full triangulated typicality record per item, provenance-tagged."""
    ref = reference_perplexity(texts, reference_substrate)
    freq = entity_frequency([e or "" for e in entities])
    self_ppl = (self_perplexity(texts, probed_substrate)
                if probed_substrate is not None else [None] * len(texts))
    return [{
        "reference_ppl": float(r),
        "entity_freq_log10": None if np.isnan(f) else float(f),
        "self_ppl": None if s is None else float(s),
        "self_ppl_circular": True,   # standing flag, D-002
        "provenance": "measured",
    } for r, f, s in zip(ref, freq, self_ppl)]
