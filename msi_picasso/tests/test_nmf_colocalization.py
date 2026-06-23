"""Tests for NMF substructure-sharing colocalization features.

NMF factorises the on-tissue ion-image matrix into spatial components; each
feature gets a loading vector and same-protein features are scored by the
within-protein cosine similarity of those loadings. See
``compute_nmf_colocalization_features`` in ``maldi_features.py``.
"""

import numpy as np
import pandas as pd

from msi_picasso.maldi_features import (
    _NMF_COLOC_COLS,
    compute_nmf_colocalization_features,
    compute_tissue_mask,
)


def _images_with_shared_substructure():
    """6 images on an 8x8 grid (left half = unmeasured padding).

    Protein A's three features all concentrate in the SAME top-right quadrant
    (shared substructure). Protein B's three features each occupy a DIFFERENT
    region, so they should share substructure much less.
    """
    H = W = 8
    imgs = np.zeros((6, H, W), dtype=np.float32)
    # right half (cols 4:8) is the measured tissue
    # A0, A1, A2 — same top-right block
    for i in range(3):
        imgs[i, 0:3, 4:7] = 1.0 + 0.05 * i
    # B0, B1, B2 — three disjoint regions in the measured half
    imgs[3, 5:8, 4:6] = 1.0
    imgs[4, 0:2, 6:8] = 1.0
    imgs[5, 4:6, 6:8] = 1.0
    mzs = np.array([100.0, 110.0, 120.0, 200.0, 210.0, 220.0])
    proteins = ["A", "A", "A", "B", "B", "B"]
    df = pd.DataFrame({"feature_mz": mzs, "protein": proteins})
    return df, imgs, mzs


class TestNmfColocalization:
    def test_adds_three_columns(self):
        df, imgs, mzs = _images_with_shared_substructure()
        mask = compute_tissue_mask(imgs, 0.0)
        out = compute_nmf_colocalization_features(df, imgs, mzs, pixel_mask=mask, n_components=4)
        for col in _NMF_COLOC_COLS:
            assert col in out.columns
            assert out[col].notna().all()

    def test_shared_substructure_scores_higher(self):
        df, imgs, mzs = _images_with_shared_substructure()
        mask = compute_tissue_mask(imgs, 0.0)
        out = compute_nmf_colocalization_features(df, imgs, mzs, pixel_mask=mask, n_components=4)
        a = out.loc[out["protein"] == "A", "protein_nmf_colocalization"].mean()
        b = out.loc[out["protein"] == "B", "protein_nmf_colocalization"].mean()
        # Protein A's co-located features share a component; B's disjoint ones do not.
        assert a > b

    def test_decoy_namespace_not_pooled_with_target(self):
        # A target protein and its DECOY_ counterpart must aggregate within class.
        df, imgs, mzs = _images_with_shared_substructure()
        df["protein"] = ["P", "P", "P", "DECOY_P", "DECOY_P", "DECOY_P"]
        mask = compute_tissue_mask(imgs, 0.0)
        out = compute_nmf_colocalization_features(df, imgs, mzs, pixel_mask=mask, n_components=4)
        # The two namespaces are scored independently; targets (shared block)
        # should out-colocalize the disjoint decoys.
        t = out.loc[out["protein"] == "P", "protein_nmf_colocalization"].mean()
        d = out.loc[out["protein"] == "DECOY_P", "protein_nmf_colocalization"].mean()
        assert t > d

    def test_cosine_in_unit_range(self):
        df, imgs, mzs = _images_with_shared_substructure()
        mask = compute_tissue_mask(imgs, 0.0)
        out = compute_nmf_colocalization_features(df, imgs, mzs, pixel_mask=mask, n_components=4)
        vals = out[_NMF_COLOC_COLS].to_numpy()
        assert np.all(vals >= -1e-4) and np.all(vals <= 1.0 + 1e-4)
