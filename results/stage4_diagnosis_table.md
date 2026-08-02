| Detector | Model | L | In-dist AUROC (3c) | Off-diagonal AUROC (3c) | Gap [95% CI] | FT mean P(true) | 3a fielded gap [CI] | Recoverable off-diag AUROC | Recoverability (all-cell β) |
|---|---|---|---|---|---|---|---|---|---|
| saplma | Qwen2.5-7B-Instruct | 14 | 0.993 [0.979, 1.000] | 0.965 [0.954, 0.976] | +0.028 [+0.010, +0.042]* | 0.174 | -0.011 [-0.017, -0.005]* | 0.982 [0.971, 0.990] | 0.876 |
| saplma | gemma-2-9b | 21 | 1.000 [1.000, 1.000] | 0.984 [0.978, 0.990] | +0.016 [+0.010, +0.022]* | 0.087 | -0.003 [-0.006, -0.000]* | 0.992 [0.985, 0.997] | 0.928 |
| saplma | Llama-3.1-8B | 16 | 0.993 [0.981, 1.000] | 0.974 [0.963, 0.983] | +0.020 [+0.003, +0.033]* | 0.127 | -0.006 [-0.013, +0.001] | 0.982 [0.970, 0.993] | 0.889 |

`*` = CI excludes zero. Overlapping CIs are not a difference (the project's standing directive §1.5).
In-dist = held-out diagonal (the field's reported number). Off-diagonal = honest truth detection on TA+FT. FT mean P(true) is the fluent-lie cell: BELOW 0.5 means the probe correctly rejects fluent falsehood. Recoverability = truth β under an all-cell (fairly trained) probe, typicality+fragmentation partialled out.
Domain scope: cities. A POOLED (all-domain) row is NOT a valid headline for this corpus: domain is confounded with truth across the diagonal/off-diagonal split, which by itself drives the off-diagonal below chance (see notes/weakness_audit.md A1). Use domain="cities".
