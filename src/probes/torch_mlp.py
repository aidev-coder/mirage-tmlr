"""
GPU SAPLMA probe — faithful architecture (Azaria & Mitchell 2023), fast.

Same net as `saplma.py` (ReLU MLP 256-128-64 -> 1 logit) but in torch on the
GPU, so a full layer sweep (n_layers x leave-one-topic-out folds = hundreds of
fits) runs in minutes instead of the hours sklearn takes on CPU while the GPU
sits idle (, compute discipline).

Device-agnostic: falls back to CPU (used by the plumbing test). Standardization,
early stopping, and the bootstrap-CI AUROC are unchanged from the sklearn path,
so this is an accelerator, not a different method.
"""
from __future__ import annotations

import numpy as np

from ..stats import auroc_with_ci


def _pick_device(device: str | None):
    import torch
    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _train_one(Xtr, ytr, Xte, device, hidden=(256, 128, 64),
               max_epochs=200, patience=12, lr=1e-3, batch=512, seed=0):
    """Train one SAPLMA MLP; return P(true) on Xte. Standardizes on train stats,
    early-stops on a 10% validation split (mirrors sklearn early_stopping)."""
    import torch
    from torch import nn

    torch.manual_seed(seed)
    mu, sd = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-6
    Xtr = (Xtr - mu) / sd
    Xte = (Xte - mu) / sd

    n = len(Xtr)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_val = max(1, int(0.1 * n))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    d = Xtr.shape[1]
    layers, prev = [], d
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.ReLU()]
        prev = h
    layers += [nn.Linear(prev, 1)]
    net = nn.Sequential(*layers).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    Xt = torch.as_tensor(Xtr, dtype=torch.float32, device=device)
    yt = torch.as_tensor(ytr, dtype=torch.float32, device=device)
    tr_t = torch.as_tensor(tr_idx, device=device)
    Xv = Xt[torch.as_tensor(val_idx, device=device)]
    yv = yt[torch.as_tensor(val_idx, device=device)]

    best_val, best_state, bad = float("inf"), None, 0
    for _ in range(max_epochs):
        net.train()
        p = tr_t[torch.randperm(len(tr_t), device=device)]
        for s in range(0, len(p), batch):
            idx = p[s:s + batch]
            opt.zero_grad()
            out = net(Xt[idx]).squeeze(1)
            loss_fn(out, yt[idx]).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            v = loss_fn(net(Xv).squeeze(1), yv).item()
        if v < best_val - 1e-4:
            best_val, best_state, bad = v, {k: t.clone() for k, t in net.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        Xe = torch.as_tensor(Xte, dtype=torch.float32, device=device)
        return torch.sigmoid(net(Xe).squeeze(1)).cpu().numpy()


def layer_sweep_fast(hidden_states: np.ndarray, y: np.ndarray,
                     train_idx: np.ndarray, test_idx: np.ndarray,
                     seed: int = 0, device: str | None = None,
                     done_layers: dict | None = None,
                     on_layer=None) -> list[dict]:
    """Train+eval one probe per layer (full curve, D-004 — no cherry-pick).

    `done_layers`: {layer_int: result_dict} already computed (checkpoint resume);
    those layers are skipped. `on_layer(result)` is called after each new layer so
    the caller can persist a checkpoint. Returns the full ordered curve.
    """
    dev = _pick_device(device)
    done_layers = done_layers or {}
    ytr = y[train_idx].astype(np.float32)
    out = []
    for layer in range(hidden_states.shape[1]):
        if layer in done_layers:
            out.append(done_layers[layer])
            continue
        Xtr = hidden_states[train_idx, layer, :].astype(np.float32)
        Xte = hidden_states[test_idx, layer, :].astype(np.float32)
        scores = _train_one(Xtr, ytr, Xte, dev, seed=seed)
        res = auroc_with_ci(y[test_idx], scores, seed=seed)
        res["layer"] = layer
        out.append(res)
        if on_layer is not None:
            on_layer(res)
    return out
