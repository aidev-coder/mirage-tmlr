# Provenance

`probes.py` fetched verbatim, unmodified, from:

    https://raw.githubusercontent.com/saprmarks/geometry-of-truth/main/probes.py

Repo: https://github.com/saprmarks/geometry-of-truth (Marks & Tegmark, "The Geometry
of Truth", already cited in the related-work section of the paper
Work as the optimistic baseline MIRAGE audits).

Commit pinned (last commit touching this file as of fetch): `2ac52d739c7fc4550659b11a589f9a04d2e42011`, 2024-02-01.
Fetched: 2026-08-07.

No LICENSE file is present in the source repo. This file is vendored unmodified,
in full, with attribution, solely to run the paper's own published mass-mean
probing method (`MMProbe`) through the MIRAGE probe auditor as an external check —
academic reproducibility use, not redistribution as a product. If the authors
object, remove on request.

We use `MMProbe` only (mass-mean probing with whitened covariance — the paper's
flagship method). `LRProbe` and `CCSProbe` are present in the file but unused here.
