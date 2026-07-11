"""
Regression: Stage-0 sanity artifact must always serialize (MIRAGE §1.2).

The Stage-0 gate crashed AFTER all substrates passed, because
`sanity_check()` stored one check (`distinct_inputs_distinct_states`) as a raw
`np.bool_` and `json.dumps` refuses numpy scalars — so the committed artifact,
a hard precondition for Stage 1, was lost.

These tests pin both defence layers:
  1. source: report check values are native Python bools;
  2. boundary: modal_app._json_default coerces any stray numpy scalar/array.
"""
import json

import numpy as np
import pytest


def _report_with_numpy_leak() -> dict:
    """Mimic the pre-fix sanity report: a numpy bool inside checks."""
    return {
        "model": "dummy/model",
        "n_layers": 4,
        "checks": {
            "shape": True,
            # the exact culprit: an un-cast comparison of a numpy float
            "distinct_inputs_distinct_states": np.linalg.norm(np.array([1.0, 2.0])) > 1e-3,
        },
    }


def test_numpy_bool_breaks_plain_dumps():
    """Documents the failure: raw numpy bool is not JSON serializable."""
    leaky = _report_with_numpy_leak()
    assert isinstance(leaky["checks"]["distinct_inputs_distinct_states"], np.bool_)
    with pytest.raises(TypeError):
        json.dumps(leaky)


def test_source_cast_yields_native_bools():
    """substrate.sanity_check casts checks -> plain bool; round-trips as JSON bool."""
    leaky = _report_with_numpy_leak()
    fixed = {**leaky, "checks": {k: bool(v) for k, v in leaky["checks"].items()}}
    loaded = json.loads(json.dumps(fixed))
    for v in loaded["checks"].values():
        assert isinstance(v, bool)  # real JSON boolean, not a stringified one


def test_boundary_handler_coerces_numpy():
    """modal_app._json_default is the catch-all so an artifact write never fails."""
    modal_app = pytest.importorskip("modal_app", reason="modal not installed")
    d = modal_app._json_default
    assert json.loads(json.dumps(np.bool_(True), default=d)) is True
    assert json.loads(json.dumps(np.int64(5), default=d)) == 5
    assert json.loads(json.dumps(np.float64(1.5), default=d)) == 1.5
    assert json.loads(json.dumps(np.array([1, 2, 3]), default=d)) == [1, 2, 3]
    # a genuinely unserializable object still raises (no silent data loss)
    with pytest.raises(TypeError):
        json.dumps(object(), default=d)


def test_full_sanity_shape_report_serializes():
    """End-to-end: a realistic sanity report serializes via the boundary default."""
    modal_app = pytest.importorskip("modal_app", reason="modal not installed")
    report = {
        "model": "meta-llama/Llama-3.1-8B",
        "n_layers": 32, "d_model": 4096,
        "checks": {
            "shape": np.bool_(True), "finite": True,
            "nonconstant_across_layers": np.bool_(True),
            "distinct_inputs_distinct_states": np.bool_(True),
            "cache_roundtrip": True,
        },
        "vram_gb": np.float64(14.97),
        "pass": np.bool_(True),
    }
    s = json.dumps(report, indent=2, default=modal_app._json_default)
    back = json.loads(s)
    assert back["pass"] is True and back["vram_gb"] == 14.97
