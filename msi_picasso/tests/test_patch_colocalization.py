"""Tests for patch-level (local) colocalization features.

compute_patch_colocalization_features tiles the ion-image grid into patches and
computes per-pair Pearson r over each patch's on-tissue pixels, aggregating to
protein_patch_colocalization_mean/_max/_frac_above. Purely spatial -> blind to
is_decoy.
"""

import numpy as np
import pandas as pd

from msi_picasso import maldi_features
from msi_picasso.maldi_features import (
    compute_patch_colocalization_features,
    compute_tissue_mask,
)


def _frame(mzs, proteins, n_decoy=0):
    n = len(mzs)
    return pd.DataFrame({
        "feature_mz": mzs,
        "protein": proteins,
        "peptide": [f"P{i}" for i in range(n)],
        "is_decoy": [False] * (n - n_decoy) + [True] * n_decoy,
    })


class TestPatchColocalization:
    def test_adds_three_columns(self):
        rng = np.random.default_rng(0)
        imgs = rng.random((3, 20, 20)).astype(np.float32)
        df = _frame([100.0, 200.0, 300.0], ["A", "A", "A"])
        out = compute_patch_colocalization_features(
            df, imgs, np.array([100.0, 200.0, 300.0]), patch_size=10
        )
        for col in maldi_features._PATCH_COLOC_COLS:
            assert col in out.columns
            assert out[col].notna().all()

    def test_local_cocluster_high_frac_above(self):
        # Two features that are strongly correlated within every patch (one is a
        # scaled copy of the other + tiny noise) -> high frac_above and mean.
        rng = np.random.default_rng(1)
        H = W = 20
        base = rng.random((H, W)).astype(np.float32)
        a = base + 0.01 * rng.random((H, W)).astype(np.float32)
        b = 2.0 * base + 0.01 * rng.random((H, W)).astype(np.float32)
        c = rng.random((H, W)).astype(np.float32)  # unrelated
        imgs = np.stack([a, b, c])
        df = _frame([100.0, 200.0, 300.0], ["A", "A", "B"])  # a,b same protein; c alone
        out = compute_patch_colocalization_features(
            df, imgs, np.array([100.0, 200.0, 300.0]), patch_size=10, threshold=0.5
        ).set_index("feature_mz")
        # a/b co-distribute locally in (almost) every patch
        assert out.loc[100.0, "protein_patch_colocalization_frac_above"] > 0.9
        assert out.loc[100.0, "protein_patch_colocalization_mean"] > 0.8
        # c is alone in protein B -> no partners -> 0
        assert out.loc[300.0, "protein_patch_colocalization_mean"] == 0.0

    def test_partial_overlap_intermediate_frac(self):
        # a,b correlate in the left half only -> frac_above strictly between 0 and 1.
        rng = np.random.default_rng(2)
        H = W = 20
        a = rng.random((H, W)).astype(np.float32)
        b = a.copy()
        b[:, W // 2:] = rng.random((H, W // 2)).astype(np.float32)  # decorrelate right half
        imgs = np.stack([a, b])
        df = _frame([100.0, 200.0], ["A", "A"])
        out = compute_patch_colocalization_features(
            df, imgs, np.array([100.0, 200.0]), patch_size=10, threshold=0.5
        ).set_index("feature_mz")
        frac = out.loc[100.0, "protein_patch_colocalization_frac_above"]
        assert 0.0 < frac < 1.0

    def test_blind_to_is_decoy(self):
        rng = np.random.default_rng(3)
        imgs = rng.random((3, 20, 20)).astype(np.float32)
        mzs = np.array([100.0, 200.0, 300.0])
        out_t = compute_patch_colocalization_features(
            _frame([100.0, 200.0, 300.0], ["A", "A", "A"], n_decoy=0), imgs, mzs, patch_size=10
        )
        out_d = compute_patch_colocalization_features(
            _frame([100.0, 200.0, 300.0], ["A", "A", "A"], n_decoy=3), imgs, mzs, patch_size=10
        )
        for col in maldi_features._PATCH_COLOC_COLS:
            assert np.allclose(out_t[col], out_d[col])

    def test_respects_pixel_mask(self):
        # With a tissue mask that zeroes most pixels, patches below min get skipped;
        # function still returns the columns (filled 0 where no patch qualifies).
        rng = np.random.default_rng(4)
        imgs = rng.random((2, 20, 20)).astype(np.float32)
        mask = np.zeros(20 * 20, dtype=bool)
        mask[:3] = True  # only 3 on-tissue pixels total -> every patch < min_patch_pixels
        df = _frame([100.0, 200.0], ["A", "A"])
        out = compute_patch_colocalization_features(
            df, imgs, np.array([100.0, 200.0]), pixel_mask=mask, patch_size=10
        )
        assert (out["protein_patch_colocalization_mean"] == 0.0).all()
