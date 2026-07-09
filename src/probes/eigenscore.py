"""
EigenScore (INSIDE-style; Chen et al. 2024): hallucination score from the
eigenvalue spectrum of the covariance of K sampled responses' sentence
embeddings. Diverse (high-volume) embedding clouds indicate inconsistency.

Score per item: (1/K) * sum_i log(lambda_i + alpha) over the regularized
covariance spectrum of the K response embeddings. Higher = more divergent
samples = more likely hallucination.
"""

from __future__ import annotations

import numpy as np


def eigenscore(embeddings: np.ndarray, alpha: float = 1e-3) -> float:
    """embeddings: [K, d] — one sentence embedding per sampled response."""
    Z = np.asarray(embeddings, dtype=np.float64)
    K = Z.shape[0]
    if K < 2:
        raise ValueError("eigenscore needs >= 2 sampled responses")
    J = np.eye(K) - np.ones((K, K)) / K          # centering
    # Gram trick: eigvals of (1/K) Z_c Z_c^T equal those of the d x d covariance
    cov = (J @ Z) @ (J @ Z).T / K
    lam = np.linalg.eigvalsh(cov)
    return float(np.mean(np.log(np.clip(lam, 0, None) + alpha)))


def eigenscore_batch(embedding_sets: list[np.ndarray], alpha: float = 1e-3) -> np.ndarray:
    return np.array([eigenscore(e, alpha) for e in embedding_sets])


def sentence_embedding(hidden_states: np.ndarray, layer: str | int = "middle") -> np.ndarray:
    """Sentence embedding from a response's per-token hidden states [seq, layers+1, d]:
    mean over tokens at the chosen layer (INSIDE uses a middle layer)."""
    n_layers = hidden_states.shape[1]
    idx = n_layers // 2 if layer == "middle" else int(layer)
    return hidden_states[:, idx, :].mean(axis=0)
