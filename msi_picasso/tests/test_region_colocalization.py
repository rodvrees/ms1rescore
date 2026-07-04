"""Tests for region-profile colocalization features.

The tissue is segmented into regions (k-means on per-pixel TIC-normalized
composition); each ion image is reduced to a per-region composition fingerprint
and same-protein features are scored by the within-protein Pearson r of those
fingerprints. See ``compute_region_colocalization_features`` in
``maldi_features.py``.
"""

import numpy as np
import pandas as pd

from msi_picasso.maldi_features import (
    _REGION_COLOC_COLS,
    compute_region_colocalization_features,
    compute_tissue_mask,
)


def _images_with_shared_region():
    """6 images on an 8x8 grid.

    Protein A's three features all concentrate in the SAME top-right block
    (shared region). Protein B's three features each occupy a DIFFERENT region,
    so they should share a region fingerprint much less.
    """
    H = W = 8
    imgs = np.zeros((6, H, W), dtype=np.float32)
    for i in range(3):
        imgs[i, 0:3, 4:7] = 1.0 + 0.05 * i      # A0, A1, A2 — same block
    imgs[3, 5:8, 4:6] = 1.0                       # B0 } three
    imgs[4, 0:2, 6:8] = 1.0                       # B1 } disjoint
    imgs[5, 4:6, 6:8] = 1.0                       # B2 } regions
    mzs = np.array([100.0, 110.0, 120.0, 200.0, 210.0, 220.0])
    proteins = ["A", "A", "A", "B", "B", "B"]
    df = pd.DataFrame({"feature_mz": mzs, "protein": proteins})
    return df, imgs, mzs


def _anti_located_pair():
    """2 same-protein features in disjoint left/right halves of the tissue."""
    H = W = 8
    imgs = np.zeros((2, H, W), dtype=np.float32)
    imgs[0, :, 0:4] = 1.0   # left half
    imgs[1, :, 4:8] = 1.0   # right half
    mzs = np.array([100.0, 110.0])
    df = pd.DataFrame({"feature_mz": mzs, "protein": ["P", "P"]})
    return df, imgs, mzs


class TestRegionColocalization:
    def test_adds_three_columns(self):
        df, imgs, mzs = _images_with_shared_region()
        mask = compute_tissue_mask(imgs, 0.0)
        out = compute_region_colocalization_features(df, imgs, mzs, pixel_mask=mask, n_regions=4)
        for col in _REGION_COLOC_COLS:
            assert col in out.columns
            assert out[col].notna().all()

    def test_shared_region_scores_higher(self):
        df, imgs, mzs = _images_with_shared_region()
        mask = compute_tissue_mask(imgs, 0.0)
        out = compute_region_colocalization_features(df, imgs, mzs, pixel_mask=mask, n_regions=4)
        a = out.loc[out["protein"] == "A", "protein_region_colocalization"].mean()
        b = out.loc[out["protein"] == "B", "protein_region_colocalization"].mean()
        assert a > b

    def test_decoy_namespace_not_pooled_with_target(self):
        df, imgs, mzs = _images_with_shared_region()
        df["protein"] = ["P", "P", "P", "DECOY_P", "DECOY_P", "DECOY_P"]
        mask = compute_tissue_mask(imgs, 0.0)
        out = compute_region_colocalization_features(df, imgs, mzs, pixel_mask=mask, n_regions=4)
        t = out.loc[out["protein"] == "P", "protein_region_colocalization"].mean()
        d = out.loc[out["protein"] == "DECOY_P", "protein_region_colocalization"].mean()
        assert t > d

    def test_signed_metric_can_go_negative(self):
        # Unlike a cosine of non-negative loadings, region-profile r is signed:
        # two features occupying disjoint regions correlate negatively.
        df, imgs, mzs = _anti_located_pair()
        mask = compute_tissue_mask(imgs, 0.0)
        out = compute_region_colocalization_features(df, imgs, mzs, pixel_mask=mask, n_regions=2)
        assert out["protein_region_colocalization"].min() < 0.0

    def test_blind_to_is_decoy_and_deterministic(self):
        # is_decoy is not an input; same inputs + seed -> identical output.
        df, imgs, mzs = _images_with_shared_region()
        mask = compute_tissue_mask(imgs, 0.0)
        a = compute_region_colocalization_features(
            df.copy(), imgs, mzs, pixel_mask=mask, n_regions=4, random_state=7
        )["protein_region_colocalization"].to_numpy()
        b = compute_region_colocalization_features(
            df.copy(), imgs, mzs, pixel_mask=mask, n_regions=4, random_state=7
        )["protein_region_colocalization"].to_numpy()
        np.testing.assert_allclose(a, b)

    def test_debug_dict_populated(self):
        df, imgs, mzs = _images_with_shared_region()
        mask = compute_tissue_mask(imgs, 0.0)
        dbg = {}
        compute_region_colocalization_features(
            df, imgs, mzs, pixel_mask=mask, n_regions=4, debug=dbg
        )
        assert "region_labels" in dbg and dbg["region_labels"].shape == (8 * 8,)
        assert "region_profiles" in dbg and "region_profile_mzs" in dbg
