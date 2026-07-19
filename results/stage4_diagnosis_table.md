| Detector | Model | L | In-dist AUROC (3c) | Off-diagonal AUROC (3c) | Gap [95% CI] | FT mean P(true) | 3a fielded gap [CI] | Recoverability (all-cell β) |
|---|---|---|---|---|---|---|---|---|
| saplma | gemma-2-9b | 21 | 1.000 [1.000, 1.000] | 0.484 [0.440, 0.525] | +0.516 [+0.474, +0.557]* | 0.551 | +0.067 [+0.052, +0.080]* | 0.931 |
| saplma | Llama-3.1-8B | 16 | 1.000 [0.998, 1.000] | 0.461 [0.417, 0.499] | +0.539 [+0.496, +0.579]* | 0.560 | +0.060 [+0.045, +0.074]* | 0.919 |
| saplma | Qwen2.5-7B-Instruct | 14 | 0.998 [0.992, 1.000] | 0.472 [0.426, 0.512] | +0.526 [+0.484, +0.568]* | 0.601 | +0.070 [+0.058, +0.084]* | 0.894 |

`*` = CI excludes zero. Overlapping CIs are not a difference (the project's standing directive §1.5).
In-dist = held-out diagonal (the field's reported number). Off-diagonal = honest truth detection on TA+FT. FT mean P(true) near/above 0.5 is the mechanism: the field's probe rates confident fluent falsehoods TRUE. Recoverability = truth β under an all-cell (fairly trained) probe, typicality+fragmentation partialled out.
