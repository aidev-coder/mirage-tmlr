"""
Stage 0 — Substrate: uniform per-layer hidden-state extraction.

Returns, for any (model, text, token position), the stack of hidden states across
all layers as a numpy array [n_layers+1, d_model] (index 0 = embeddings).
Activations are cached to disk keyed by (model, text, position) hash; recomputing
hidden states is the main compute cost (the project's standing directive §5).

torch/transformers are imported lazily so the Stage-3 eval stack and tests stay
runnable on a CPU-only box without them.

Stage 0 gate: `python -m src.substrate --sanity` must pass on the GPU box before
Stage 1 begins — it verifies extraction against architectural invariants on 3
sanity strings per configured model and prints VRAM usage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import yaml

_ROOT = Path(__file__).resolve().parent.parent


def _load_config(path: str | Path | None = None) -> dict:
    with open(path or _ROOT / "configs" / "models.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


class Substrate:
    """One loaded model exposing per-layer hidden states."""

    def __init__(self, model_id: str, device_map: str = "auto",
                 dtype: str = "bfloat16", cache_dir: str | Path | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        # transformers renamed torch_dtype -> dtype in v5; requirements pin <5,
        # but tolerate both so a newer environment degrades gracefully.
        load_kw = dict(device_map=device_map, output_hidden_states=True)
        resolved = getattr(torch, dtype) if dtype != "auto" else "auto"
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id, torch_dtype=resolved, **load_kw)
        except TypeError:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id, dtype=resolved, **load_kw)
        self._finish_init(cache_dir)

    @classmethod
    def from_objects(cls, model, tokenizer, model_id: str,
                     cache_dir: str | Path | None = None) -> "Substrate":
        """Wrap an already-constructed (model, tokenizer) pair — used by the
        plumbing test (random tiny model, no Hub access) and any caller that
        loads weights its own way. `model_id` labels the activation cache."""
        self = cls.__new__(cls)
        self.model_id = model_id
        self.tokenizer = tokenizer
        self.model = model
        self._finish_init(cache_dir)
        return self

    def _finish_init(self, cache_dir: str | Path | None) -> None:
        self.model.eval()
        if getattr(self.model.config, "output_hidden_states", False) is not True:
            self.model.config.output_hidden_states = True
        self.cache_dir = Path(cache_dir or _ROOT / "data" / "activations")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ── extraction ───────────────────────────────────────────────────────

    def _cache_key(self, text: str, position: int) -> Path:
        h = hashlib.sha256(f"{self.model_id}||{position}||{text}".encode()).hexdigest()[:24]
        return self.cache_dir / f"{h}.npz"

    def hidden_states(self, text: str, position: int = -1,
                      use_cache: bool = True) -> np.ndarray:
        """Hidden states at `position` for every layer: [n_layers+1, d_model]."""
        key = self._cache_key(text, position)
        if use_cache and key.exists():
            return np.load(key)["h"]

        import torch
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model(**inputs)
        # out.hidden_states: tuple of [1, seq, d], length n_layers+1
        h = np.stack([hs[0, position].float().cpu().numpy() for hs in out.hidden_states])
        if use_cache:
            np.savez_compressed(key, h=h)
        return h

    def batch_hidden_states(self, texts: list[str], position: int = -1,
                            use_cache: bool = True) -> np.ndarray:
        """[n_texts, n_layers+1, d_model]. Sequential on purpose: cache hits dominate."""
        return np.stack([self.hidden_states(t, position, use_cache) for t in texts])

    def hidden_states_matrix(self, texts: list[str], batch_size: int = 32,
                             position: int = -1, progress_every: int = 25,
                             out_dtype=np.float16, tag: str = "") -> np.ndarray:
        """Batched last-token hidden-state extraction: [n, n_layers+1, d_model].

        GPU-efficient replacement for per-text extraction (20-50x fewer forward
        passes). `position=-1` gathers each row's LAST NON-PAD token (mask-aware,
        right-padded) — correct under batching, unlike a naive [:, -1] which would
        read padding. fp16 output halves the cached activation footprint; the
        probe standardizes anyway.
        """
        import torch
        tok = self.tokenizer
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        tok.padding_side = "right"  # so last real token = attention_mask.sum(1)-1

        rows, n = [], len(texts)
        n_batches = (n + batch_size - 1) // batch_size
        for bi in range(n_batches):
            batch = texts[bi * batch_size:(bi + 1) * batch_size]
            enc = tok(batch, return_tensors="pt", padding=True, truncation=True,
                      max_length=128).to(self.model.device)
            with torch.no_grad():
                hs = self.model(**enc).hidden_states  # tuple(L+1) of [b, seq, d]
            if position == -1:
                last = enc["attention_mask"].sum(dim=1) - 1  # [b]
                b, d = hs[0].shape[0], hs[0].shape[-1]
                gi = last.view(b, 1, 1).expand(b, 1, d)
                picked = torch.stack(
                    [h.gather(1, gi).squeeze(1) for h in hs], dim=1)  # [b, L+1, d]
            else:
                picked = torch.stack([h[:, position, :] for h in hs], dim=1)
            rows.append(picked.float().cpu().numpy().astype(out_dtype))
            if progress_every and (bi % progress_every == 0 or bi == n_batches - 1):
                print(f"    [extract{(' ' + tag) if tag else ''}] "
                      f"batch {bi + 1}/{n_batches} "
                      f"({min((bi + 1) * batch_size, n)}/{n} texts)", flush=True)
        return np.concatenate(rows, axis=0)

    def eigenscore_matrix(self, texts: list[str], batch_size: int = 16,
                          alpha: float = 1e-3, progress_every: int = 25) -> np.ndarray:
        """Per-item EigenScore at every layer: [n, n_layers+1] (INSIDE self variant).

        For each statement, score = mean log-eigenvalue of the covariance of its
        own real-token hidden states at that layer. Higher = more dispersed token
        cloud = the hallucination signal (Chen et al. 2024). No sampling: the
        statement's token spread is the consistency proxy, so this is cheap and
        applies to a fixed corpus item directly.
        """
        import torch

        from .probes.eigenscore import eigenscore
        tok = self.tokenizer
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        tok.padding_side = "right"

        rows, n = [], len(texts)
        n_batches = (n + batch_size - 1) // batch_size
        for bi in range(n_batches):
            batch = texts[bi * batch_size:(bi + 1) * batch_size]
            enc = tok(batch, return_tensors="pt", padding=True, truncation=True,
                      max_length=128).to(self.model.device)
            with torch.no_grad():
                hs = self.model(**enc).hidden_states  # tuple(L+1) of [b, seq, d]
            lens = enc["attention_mask"].sum(dim=1).tolist()
            stacked = torch.stack(hs, dim=1).float().cpu().numpy()  # [b, L+1, seq, d]
            for r, L_real in enumerate(lens):
                m = max(int(L_real), 2)
                rows.append([eigenscore(stacked[r, layer, :m, :], alpha=alpha)
                             for layer in range(stacked.shape[1])])
            if progress_every and (bi % progress_every == 0 or bi == n_batches - 1):
                print(f"    [eigenscore] batch {bi + 1}/{n_batches} "
                      f"({min((bi + 1) * batch_size, n)}/{n} texts)", flush=True)
        return np.asarray(rows, dtype=np.float64)

    def nll(self, text: str) -> float:
        """Mean per-token negative log-likelihood of `text` under this model."""
        import torch
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model(**inputs, labels=inputs["input_ids"])
        return float(out.loss)

    def sample(self, prompt: str, n: int, temperature: float = 1.0,
               max_new_tokens: int = 64) -> list[str]:
        """n sampled continuations (for eigenscore / semantic entropy)."""
        import torch
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            gen = self.model.generate(
                **inputs, do_sample=True, temperature=temperature,
                num_return_sequences=n, max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        prompt_len = inputs["input_ids"].shape[1]
        return [self.tokenizer.decode(g[prompt_len:], skip_special_tokens=True) for g in gen]

    # ── Stage 0 gate ─────────────────────────────────────────────────────

    SANITY_STRINGS = [
        "The capital of France is Paris.",
        "Water boils at 100 degrees Celsius at sea level.",
        "The 1969 moon landing was part of the Apollo program.",
    ]

    def sanity_check(self) -> dict:
        """Architectural invariants + basic signal checks on 3 known strings.

        Not a comparison against stored golden values (those are model-version
        fragile); instead verifies shape against config, non-degeneracy, cache
        round-trip, and that distinct inputs yield distinct states.
        """
        n_layers = self.model.config.num_hidden_layers
        d_model = self.model.config.hidden_size
        report = {"model": self.model_id, "n_layers": n_layers, "d_model": d_model, "checks": {}}

        # bool(...) casts matter: numpy bools (np.bool_) crash json.dumps
        hs = [self.hidden_states(s, use_cache=False) for s in self.SANITY_STRINGS]
        report["checks"]["shape"] = all(h.shape == (n_layers + 1, d_model) for h in hs)
        report["checks"]["finite"] = bool(all(np.isfinite(h).all() for h in hs))
        report["checks"]["nonconstant_across_layers"] = bool(all(
            np.std([np.linalg.norm(h[i]) for i in range(h.shape[0])]) > 0 for h in hs))
        report["checks"]["distinct_inputs_distinct_states"] = bool(
            np.linalg.norm(hs[0][-1] - hs[1][-1]) > 1e-3)
        # cache round-trip
        a = self.hidden_states(self.SANITY_STRINGS[0], use_cache=True)
        b = self.hidden_states(self.SANITY_STRINGS[0], use_cache=True)
        report["checks"]["cache_roundtrip"] = bool(np.allclose(a, b))

        try:
            import torch
            if torch.cuda.is_available():
                report["vram_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
        except Exception:
            pass
        report["pass"] = all(report["checks"].values())
        return report


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 0 substrate gate")
    ap.add_argument("--sanity", action="store_true", help="run the Stage 0 gate on all configured models")
    ap.add_argument("--model", help="single model id override")
    args = ap.parse_args()
    if not args.sanity:
        ap.error("nothing to do; pass --sanity")

    cfg = _load_config()
    ids = [args.model] if args.model else [m["id"] for m in cfg["substrates"]]
    results = []
    for mid in ids:
        print(f"── {mid}")
        rep = Substrate(mid, device_map=cfg["extraction"]["device_map"],
                        dtype=cfg["extraction"]["dtype"],
                        cache_dir=cfg["extraction"]["cache_dir"]).sanity_check()
        print(json.dumps(rep, indent=2))
        results.append(rep)
    out = _ROOT / "results" / "stage0_sanity.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"gate {'PASS' if all(r['pass'] for r in results) else 'FAIL'} -> {out}")


if __name__ == "__main__":
    main()
