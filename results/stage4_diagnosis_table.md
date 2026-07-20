| Detector | Model | L | In-dist AUROC (3c) | Off-diagonal AUROC (3c) | Gap [95% CI] | FT mean P(true) | 3a fielded gap [CI] | Recoverability (all-cell β) |
|---|---|---|---|---|---|---|---|---|
| eigenscore | gemma-2-9b | 21 | 0.622 [0.573, 0.671] | 0.695 [0.665, 0.724] | -0.073 [-0.130, -0.017]* | -6.770 | -0.008 [-0.014, -0.003]* | n/a |
| eigenscore | Llama-3.1-8B | 16 | 0.544 [0.489, 0.597] | 0.449 [0.417, 0.480] | +0.095 [+0.034, +0.157]* | -1.702 | +0.021 [+0.013, +0.028]* | n/a |
| eigenscore | Qwen2.5-7B-Instruct | 14 | 0.580 [0.525, 0.632] | 0.587 [0.557, 0.618] | -0.007 [-0.069, +0.054] | -5.414 | +0.004 [-0.001, +0.009] | n/a |
| saplma | gemma-2-9b | 21 | 1.000 [1.000, 1.000] | 0.484 [0.440, 0.525] | +0.516 [+0.474, +0.557]* | 0.551 | +0.067 [+0.052, +0.080]* | 0.930 |
| saplma | Llama-3.1-8B | 16 | 1.000 [0.998, 1.000] | 0.461 [0.417, 0.499] | +0.539 [+0.496, +0.579]* | 0.560 | +0.060 [+0.045, +0.074]* | 0.918 |
| saplma | Qwen2.5-7B-Instruct | 14 | 0.998 [0.992, 1.000] | 0.472 [0.426, 0.512] | +0.526 [+0.484, +0.568]* | 0.601 | +0.070 [+0.058, +0.084]* | 0.889 |

`*` = CI excludes zero. Overlapping CIs are not a difference (the project's standing directive §1.5).
In-dist = held-out diagonal (the field's reported number). Off-diagonal = honest truth detection on TA+FT. FT mean P(true) near/above 0.5 is the mechanism: the field's probe rates confident fluent falsehoods TRUE. Recoverability = truth β under an all-cell (fairly trained) probe, typicality+fragmentation partialled out.
