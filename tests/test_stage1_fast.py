"""
Plumbing test for the fast Stage-1 path (batched GPU extraction + torch probe
+ checkpoint resume), on the tiny random CPU model. Numbers are meaningless by
construction; this pins mechanics and one correctness invariant that MUST hold:

  batched, mask-aware last-token extraction == per-text extraction.

If that fails, the fast path reads padding instead of the real last token and
every Stage-1 AUROC is silently wrong — exactly the "instrument measuring
itself" trap ().
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_substrate_plumbing import build_tiny_substrate, N_LAYERS, D_MODEL, SEED  # noqa: E402

TEXTS = [f"{w} is the capital of france" for w in
         ("paris", "water", "fox", "dog", "france", "brown",
          "lazy", "quick", "degrees", "boils", "capital", "jumps", "the")]


def test_batched_extraction_matches_per_text():
    """Mask-aware batched last-token gather must equal unbatched extraction."""
    with tempfile.TemporaryDirectory() as tmp:
        sub = build_tiny_substrate(tmp)
        ref = sub.batch_hidden_states(TEXTS, use_cache=False)          # [n, L+1, d] fp32
        fast = sub.hidden_states_matrix(TEXTS, batch_size=4, progress_every=0)  # fp16
        assert fast.shape == ref.shape == (len(TEXTS), N_LAYERS + 1, D_MODEL)
        # fp16 tolerance; the batched path pads but must read the SAME real last token
        assert np.allclose(fast.astype(np.float32), ref, atol=1e-2, rtol=1e-2), \
            f"max abs diff {np.abs(fast.astype(np.float32) - ref).max()}"
        print("PASS  batched extraction == per-text (mask-aware last token)")


def test_torch_sweep_and_checkpoint_resume():
    """torch layer_sweep_fast returns the full curve and honours done_layers."""
    from src.probes.torch_mlp import layer_sweep_fast
    with tempfile.TemporaryDirectory() as tmp:
        sub = build_tiny_substrate(tmp)
        H = sub.hidden_states_matrix(TEXTS, batch_size=8, progress_every=0).astype(np.float32)
        H = np.concatenate([H, H, H], axis=0)               # more rows for a val split
        rng = np.random.default_rng(SEED)
        y = rng.integers(0, 2, len(H))
        tr, te = np.arange(0, int(0.75 * len(H))), np.arange(int(0.75 * len(H)), len(H))

        full = layer_sweep_fast(H, y, tr, te, seed=SEED, device="cpu")
        assert len(full) == N_LAYERS + 1
        assert all(r["layer"] == i for i, r in enumerate(full))
        assert all(("auroc" in r and "ci" in r) for r in full)

        # resume: pre-seed 2 layers as "done"; they must be returned verbatim,
        # not recomputed
        done = {0: {**full[0], "_sentinel": 1}, 1: {**full[1], "_sentinel": 1}}
        resumed = layer_sweep_fast(H, y, tr, te, seed=SEED, device="cpu",
                                   done_layers=done)
        assert resumed[0].get("_sentinel") == 1 and resumed[1].get("_sentinel") == 1
        assert len(resumed) == N_LAYERS + 1
        print("PASS  torch sweep full curve + checkpoint resume skips done layers")


def test_stage1_run_end_to_end_smoke(monkeypatch=None):
    """run() wires extraction cache + checkpointed sweep into the artifact schema."""
    import src.stage1 as stage1
    with tempfile.TemporaryDirectory() as tmp:
        sub = build_tiny_substrate(tmp)
        # tiny fake corpus: monkeypatch load_statements to avoid the real CSVs
        topics = ["t0", "t1", "t2"]
        texts = TEXTS * 3
        y = np.array([1, 0] * (len(texts) // 2) + [1] * (len(texts) % 2))[:len(texts)]
        tidx = np.array([i % 3 for i in range(len(texts))])
        stage1.load_statements = lambda t=None: (texts, y, tidx)
        res = stage1.run(sub, topics=topics, fast=True, device="cpu", batch_size=8)
        assert res["detector"] == "saplma" and res["probe_impl"] == "torch_gpu"
        assert len(res["mean_by_layer"]) == N_LAYERS + 1
        assert set(res["per_topic"]) == set(topics)
        assert "best layer-mean" in res["gate_note"]
        print("PASS  stage1.run end-to-end (cache + checkpoint + artifact schema)")


if __name__ == "__main__":
    test_batched_extraction_matches_per_text()
    test_torch_sweep_and_checkpoint_resume()
    test_stage1_run_end_to_end_smoke()
    print("\nALL PASS — fast Stage-1 path verified on CPU")
