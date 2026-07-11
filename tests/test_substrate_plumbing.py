"""
Plumbing test: Substrate -> probes on a TINY random-weights model (CPU, no Hub).

This verifies the mechanical pipeline — tokenization, per-layer extraction,
cache round-trip, probe training over extracted states — with a randomly
initialized 4-layer GPT-2. All AUROC-like numbers here are MEANINGLESS by
construction (random weights, random labels); the test asserts shapes and
invariants only. It exists because the Stage-0 gate proper needs the GPU box,
and broken plumbing should be caught before burning GPU time.

Run: python tests/test_substrate_plumbing.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

N_LAYERS, D_MODEL = 4, 64
SEED = 20260709


def build_tiny_substrate(cache_dir: str):
    """Random 4-layer GPT-2 + a from-scratch BPE tokenizer, no network."""
    import torch
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    from src.substrate import Substrate

    torch.manual_seed(SEED)

    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.BpeTrainer(vocab_size=500,
                                  special_tokens=["<unk>", "<eos>"])
    corpus = [s.lower() for s in Substrate.SANITY_STRINGS] + [
        "the quick brown fox jumps over the lazy dog",
        "paris is the capital of france",
        "water boils at one hundred degrees",
        "abcdefghijklmnopqrstuvwxyz 0123456789",
    ]
    tok.train_from_iterator(corpus, trainer)
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tok,
                                        unk_token="<unk>", eos_token="<eos>",
                                        pad_token="<eos>")

    eos_id = tokenizer.convert_tokens_to_ids("<eos>")
    cfg = GPT2Config(vocab_size=tok.get_vocab_size(), n_layer=N_LAYERS,
                     n_head=4, n_embd=D_MODEL, n_positions=128,
                     bos_token_id=eos_id, eos_token_id=eos_id)
    model = GPT2LMHeadModel(cfg)
    return Substrate.from_objects(model, tokenizer, "tiny-random-gpt2",
                                  cache_dir=cache_dir)


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        sub = build_tiny_substrate(tmp)

        # per-layer extraction: [n_layers+1, d_model]
        h = sub.hidden_states("paris is the capital of france", use_cache=False)
        assert h.shape == (N_LAYERS + 1, D_MODEL), h.shape
        assert np.isfinite(h).all()
        print(f"PASS  hidden_states shape {h.shape}, finite")

        # cache round-trip: second call must hit disk and match exactly
        a = sub.hidden_states("water boils at one hundred degrees")
        n_cached = len(list(Path(tmp).glob("*.npz")))
        b = sub.hidden_states("water boils at one hundred degrees")
        assert n_cached == 1 and np.array_equal(a, b)
        print("PASS  cache round-trip (1 npz, exact match)")

        # distinct inputs -> distinct states
        c = sub.hidden_states("the quick brown fox jumps over the lazy dog")
        assert np.linalg.norm(a[-1] - c[-1]) > 1e-4
        print("PASS  distinct inputs give distinct final-layer states")

        # nll returns a finite positive float
        nll = sub.nll("paris is the capital of france")
        assert np.isfinite(nll) and nll > 0
        print(f"PASS  nll finite ({nll:.2f})")

        # batch extraction feeds the SAPLMA layer sweep end to end
        texts = [f"{w} is the capital of france" for w in
                 ("paris", "water", "fox", "dog", "france", "brown",
                  "lazy", "quick", "degrees", "boils", "capital", "jumps")] * 4
        H = sub.batch_hidden_states(texts)          # [n, L+1, d]
        assert H.shape == (len(texts), N_LAYERS + 1, D_MODEL)

        from src.probes.saplma import layer_sweep
        rng = np.random.default_rng(SEED)
        y = rng.integers(0, 2, len(texts))          # meaningless labels, plumbing only
        idx = rng.permutation(len(texts))
        sweep = layer_sweep(H, y, idx[: int(0.75 * len(idx))],
                            idx[int(0.75 * len(idx)):], seed=SEED, max_iter=50)
        assert len(sweep) == N_LAYERS + 1
        assert all(r["layer"] == i for i, r in enumerate(sweep))
        print(f"PASS  layer_sweep over {len(sweep)} layers "
              "(AUROCs meaningless by construction — random weights, random labels)")

        # full sanity_check report must be json-serializable (regression:
        # np.bool_ leaked into the Stage-0 report and crashed the Modal
        # entrypoint's json.dumps after an otherwise-passing gate run)
        import json
        rep = sub.sanity_check()
        json.dumps(rep)
        assert rep["pass"] is True or rep["pass"] is False
        print(f"PASS  sanity_check json-serializable (pass={rep['pass']})")

        # eigenscore consumes per-token states from the same substrate path
        from src.probes.eigenscore import eigenscore
        embs = np.stack([sub.hidden_states(t, position=p)[N_LAYERS // 2]
                         for t, p in [(texts[0], -1), (texts[1], -1),
                                      (texts[2], -1), (texts[3], -1)]])
        es = eigenscore(embs)
        assert np.isfinite(es)
        print(f"PASS  eigenscore finite ({es:.3f})")

    print("\nALL PASS — substrate/probe plumbing verified on CPU (random tiny model)")


def test_substrate_plumbing():
    main()


if __name__ == "__main__":
    main()
