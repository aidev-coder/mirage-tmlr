"""
Adapter: TTPD (Truth is Universal, Bürger, Hamprecht & Nadler, NeurIPS 2024),
wrapped to the MIRAGE auditor's fit/score contract.

Fourth external probe and the first from a genuinely independent lineage. MMProbe,
LRProbe and CCS all reach us through one vendored file by one pair of authors;
TTPD is a different group (Heidelberg / Weizmann) with a different model of what
truth looks like in activation space. Where the others fit a single direction,
TTPD fits a two-dimensional subspace and claims the general truth direction inside
it is universal across statement types.

Its model of an activation, with tau in {-1,+1} the truth label, p in {-1,+1} the
polarity (affirmative +1, negated -1) and mu_i a per-topic mean:

    a_ij ~= mu_i + tau_ij * t_G + tau_ij * p_i * t_P

fitted by ordinary least squares on topic-centered activations, then a logistic
regression on the two projections at inference.

REIMPLEMENTATION, not vendored. The authors release MIT-licensed code but the
source could not be retrieved verbatim here, so this follows the published
equations and is validated against a synthetic world where t_G and t_P are known
by construction. That is a weaker provenance than the other three adapters carry,
and the paper should say so.

Requires BOTH polarities in training, which our all-affirmative corpus does not
have. Negations come from corpus_gen.negate_all, the same machinery CCS uses, so
each statement contributes an affirmative row (tau = +1 if true) and a negated row
(tau flipped, p = -1).

Input layout, declared here and nowhere else:
    X = [ affirmative acts (d) | negated acts (d) | topic one-hot (n_topics) ]
with d inferred as (X.shape[1] - n_topics) // 2. Row indexing is unchanged, so
adversarial_split and the cross-fitting loop need no special case.
"""
from __future__ import annotations

import numpy as np


def stack(X_pos: np.ndarray, X_neg: np.ndarray, topics: np.ndarray) -> np.ndarray:
    X_pos, X_neg = np.asarray(X_pos), np.asarray(X_neg)
    if X_pos.shape != X_neg.shape:
        raise ValueError(f"shape mismatch: {X_pos.shape} vs {X_neg.shape}")
    topics = np.asarray(topics)
    levels = sorted(set(topics.tolist()))
    onehot = np.zeros((len(topics), len(levels)), dtype=float)
    for j, lv in enumerate(levels):
        onehot[topics == lv, j] = 1.0
    return np.hstack([X_pos, X_neg, onehot])


class TTPDProbe:
    """fit(X, y) -> self; score(X) -> [n] P(true). X is the stacked matrix above."""

    paired = True
    needs_topics = True

    def __init__(self, seed: int = 0, n_topics: int = 3):
        self.seed = seed
        self.n_topics = n_topics
        self._t_g = None
        self._t_p = None
        self._polarity_dir = None
        self._clf = None

    def _split(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        X = np.asarray(X, dtype=np.float64)
        d2 = X.shape[1] - self.n_topics
        if d2 <= 0 or d2 % 2:
            raise ValueError(f"bad TTPD layout: width {X.shape[1]}, n_topics {self.n_topics}")
        d = d2 // 2
        return X[:, :d], X[:, d:d2], X[:, d2:]

    @staticmethod
    def _expand(pos, neg, topic_oh, y=None):
        """One row per statement becomes two: affirmative and negated."""
        acts = np.vstack([pos, neg])
        topics = np.vstack([topic_oh, topic_oh])
        polarity = np.concatenate([np.ones(len(pos)), -np.ones(len(neg))])
        if y is None:
            return acts, topics, polarity, None
        tau_pos = np.where(np.asarray(y, dtype=bool), 1.0, -1.0)
        tau = np.concatenate([tau_pos, -tau_pos])
        return acts, topics, polarity, tau

    @staticmethod
    def _center_by_topic(acts, topics):
        out = acts.copy()
        for j in range(topics.shape[1]):
            m = topics[:, j] > 0
            if m.any():
                out[m] -= out[m].mean(axis=0)
        return out

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TTPDProbe":
        from sklearn.linear_model import LogisticRegression

        pos, neg, topic_oh = self._split(X)
        acts, topics, polarity, tau = self._expand(pos, neg, topic_oh, y)
        centered = self._center_by_topic(acts, topics)

        D = np.column_stack([tau, tau * polarity])
        coef, *_ = np.linalg.lstsq(D, centered, rcond=None)
        self._t_g, self._t_p = coef[0], coef[1]

        pol_clf = LogisticRegression(max_iter=1000, random_state=self.seed)
        pol_clf.fit(centered, (polarity > 0).astype(int))
        self._polarity_dir = pol_clf.coef_[0]

        feats = np.column_stack([centered @ self._t_g, centered @ self._polarity_dir])
        self._clf = LogisticRegression(max_iter=1000, random_state=self.seed)
        self._clf.fit(feats, (tau > 0).astype(int))
        self._topic_means = None
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        pos, _, topic_oh = self._split(X)
        centered = self._center_by_topic(pos, topic_oh)
        feats = np.column_stack([centered @ self._t_g, centered @ self._polarity_dir])
        return self._clf.predict_proba(feats)[:, 1].astype(np.float64)
