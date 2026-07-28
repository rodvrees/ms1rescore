"""Test the A2 anti-localization helper in scripts/validate_results.py.

The helper decides whether two ion images co-localize (r > 0) or occupy distinct
compartments (r < 0). scripts/ is not a package, so load it by file path.
"""
import importlib.util
import os

import numpy as np

_SCRIPT = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "scripts", "validate_results.py"
)
_spec = importlib.util.spec_from_file_location("validate_results", _SCRIPT)
vr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vr)


def test_masked_pearson_distinguishes_compartments():
    h = w = 20
    mask = np.ones(h * w, dtype=bool)
    left = np.zeros((h, w), dtype=np.float32)
    left[:, : w // 2] = 1.0          # signal in left half
    right = np.zeros((h, w), dtype=np.float32)
    right[:, w // 2 :] = 1.0         # signal in right half (distinct compartment)
    left2 = left + np.zeros_like(left)  # co-located with `left`

    # distinct compartments -> negative correlation
    assert vr._masked_pearson(left, right, mask) < 0
    # co-located -> positive correlation
    assert vr._masked_pearson(left, left2, mask) > 0


def test_masked_pearson_guards_degenerate():
    mask = np.ones(100, dtype=bool)
    flat = np.ones((10, 10), dtype=np.float32)      # zero variance
    other = np.random.default_rng(0).random((10, 10)).astype(np.float32)
    assert np.isnan(vr._masked_pearson(flat, other, mask))
