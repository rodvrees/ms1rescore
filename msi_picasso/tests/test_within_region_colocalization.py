"""Tests for within-region and dominant-region Pearson-r colocalization (O3).

Unlike ``compute_region_colocalization_features`` (per-region MEAN fingerprint
correlation), these features correlate RAW pixel intensities restricted to a
segmented tissue region, asking whether same-protein peptides co-vary
pixel-to-pixel *inside* a shared region rather than merely sharing a region
average. See ``_within_region_corr_matrices`` /
``compute_within_region_colocalization_features`` in ``maldi_features.py``.
"""

import numpy as np
import pandas as pd

from msi_picasso.maldi_features import (
    _DOMINANT_REGION_COLOC_COLS,
    _WITHIN_REGION_COLOC_COLS,
    compute_region_colocalization_features,
    compute_tissue_mask,
    compute_within_region_colocalization_features,
)


def _in_phase_vs_anti_phase():
    """Driver + 2-image pairs on an 8x8 grid, forcing a clean k=2 split.

    A strong non-protein "driver" image splits the tissue into two regions:
    rows 0-5 (region 0, dominant, 48 px) and rows 6-7 (region 1, 16 px).
    Within region 0:
      - A and B rise together with the same row gradient (in phase).
      - A and C share the exact same row VALUES in reverse order, so C has
        the same per-region MEAN as A (indistinguishable to the region-profile
        fingerprint metric) but is anti-phase with A pixel-to-pixel.
    """
    H = W = 8
    driver = np.zeros((H, W), dtype=np.float32)
    driver[0:6, :] = 1.0
    driver[6:8, :] = 5.0
    grad = np.arange(1, 7, dtype=np.float32)  # [1..6] row gradient
    A = np.zeros((H, W), dtype=np.float32)
    A[0:6, :] = grad[:, None]
    A[6:8, :] = 1.0
    B = np.zeros((H, W), dtype=np.float32)
    B[0:6, :] = grad[:, None] * 2 + 1  # in phase with A
    B[6:8, :] = 1.0
    C = np.zeros((H, W), dtype=np.float32)
    C[0:6, :] = grad[::-1][:, None]  # row-reversed: anti-phase, same marginal mean as A
    C[6:8, :] = 1.0
    imgs = np.stack([driver, A, B, C])
    mzs = np.array([1.0, 100.0, 110.0, 120.0])
    return imgs, mzs


def _target_decoy_gradient_pair():
    """Driver + in-phase target pair (P) + anti-phase decoy pair (DECOY_P)."""
    H = W = 8
    driver = np.zeros((H, W), dtype=np.float32)
    driver[0:6, :] = 1.0
    driver[6:8, :] = 5.0
    grad = np.arange(1, 7, dtype=np.float32)

    A = np.zeros((H, W), dtype=np.float32)
    A[0:6, :] = grad[:, None]
    A[6:8, :] = 1.0
    B = np.zeros((H, W), dtype=np.float32)
    B[0:6, :] = grad[:, None] * 2 + 1  # in phase with A
    B[6:8, :] = 1.0

    Ad = np.zeros((H, W), dtype=np.float32)
    Ad[0:6, :] = grad[:, None]
    Ad[6:8, :] = 1.0
    Bd = np.zeros((H, W), dtype=np.float32)
    Bd[0:6, :] = grad[::-1][:, None]  # anti-phase
    Bd[6:8, :] = 1.0

    imgs = np.stack([driver, A, B, Ad, Bd])
    mzs = np.array([1.0, 100.0, 110.0, 200.0, 210.0])
    df = pd.DataFrame({
        "feature_mz": [100.0, 110.0, 200.0, 210.0],
        "protein": ["P", "P", "DECOY_P", "DECOY_P"],
    })
    return df, imgs, mzs


def _dominant_vs_small_region_conflict():
    """Same-protein pair positively correlated in the dominant region (rows 0-5,
    48 px) but negatively correlated in the small region (rows 6-7, 16 px)."""
    H = W = 8
    driver = np.zeros((H, W), dtype=np.float32)
    driver[0:6, :] = 1.0
    driver[6:8, :] = 5.0
    grad = np.arange(1, 7, dtype=np.float32)

    A = np.zeros((H, W), dtype=np.float32)
    A[0:6, :] = grad[:, None]
    A[6, :] = 1.0
    A[7, :] = 5.0
    D = np.zeros((H, W), dtype=np.float32)
    D[0:6, :] = grad[:, None] * 3 + 2  # in phase with A in the dominant region
    D[6, :] = 5.0  # opposite of A in the small region
    D[7, :] = 1.0

    imgs = np.stack([driver, A, D])
    mzs = np.array([1.0, 100.0, 110.0])
    df = pd.DataFrame({"feature_mz": [100.0, 110.0], "protein": ["P", "P"]})
    return df, imgs, mzs


class TestWithinRegionColocalization:
    def test_adds_columns(self):
        imgs, mzs = _in_phase_vs_anti_phase()
        df = pd.DataFrame({"feature_mz": [100.0, 110.0], "protein": ["P", "P"]})
        mask = compute_tissue_mask(imgs, 0.0)
        out = compute_within_region_colocalization_features(
            df, imgs, mzs, pixel_mask=mask, n_regions=2, min_region_pixels=1,
        )
        for col in _WITHIN_REGION_COLOC_COLS + _DOMINANT_REGION_COLOC_COLS:
            assert col in out.columns
            assert out[col].notna().all()

    def test_within_region_distinguishes_in_phase_from_anti_phase(self):
        """The key differentiating case: A-B (in phase) and A-C (anti-phase,
        same per-region mean) are near-identical to the fingerprint metric but
        must be clearly separated by the within-region metric."""
        imgs, mzs = _in_phase_vs_anti_phase()
        mask = compute_tissue_mask(imgs, 0.0)

        df_ab = pd.DataFrame({"feature_mz": [100.0, 110.0], "protein": ["P", "P"]})
        df_ac = pd.DataFrame({"feature_mz": [100.0, 120.0], "protein": ["P", "P"]})

        within_ab = compute_within_region_colocalization_features(
            df_ab, imgs, mzs, pixel_mask=mask, n_regions=2, min_region_pixels=1,
        )["protein_within_region_colocalization"].iloc[0]
        within_ac = compute_within_region_colocalization_features(
            df_ac, imgs, mzs, pixel_mask=mask, n_regions=2, min_region_pixels=1,
        )["protein_within_region_colocalization"].iloc[0]
        assert within_ab - within_ac > 0.3

        fp_ab = compute_region_colocalization_features(
            df_ab, imgs, mzs, pixel_mask=mask, n_regions=2,
        )["protein_region_colocalization"].iloc[0]
        fp_ac = compute_region_colocalization_features(
            df_ac, imgs, mzs, pixel_mask=mask, n_regions=2,
        )["protein_region_colocalization"].iloc[0]
        assert abs(fp_ab - fp_ac) < 0.1

    def test_dominant_region_uses_largest_cluster(self):
        df, imgs, mzs = _dominant_vs_small_region_conflict()
        mask = compute_tissue_mask(imgs, 0.0)
        out = compute_within_region_colocalization_features(
            df, imgs, mzs, pixel_mask=mask, n_regions=2, min_region_pixels=1,
        )
        assert out["protein_dominant_region_colocalization"].iloc[0] > 0.5

    def test_weighted_family_falls_back_when_no_region_clears_floor(self):
        df, imgs, mzs = _dominant_vs_small_region_conflict()
        mask = compute_tissue_mask(imgs, 0.0)
        out = compute_within_region_colocalization_features(
            df, imgs, mzs, pixel_mask=mask, n_regions=2, min_region_pixels=1000,
        )
        for col in _WITHIN_REGION_COLOC_COLS:
            assert (out[col] == 0.0).all()
        # Dominant family is exempt from the floor — still computed normally.
        assert out["protein_dominant_region_colocalization"].iloc[0] > 0.5

    def test_decoy_namespace_not_pooled_with_target(self):
        df, imgs, mzs = _target_decoy_gradient_pair()
        mask = compute_tissue_mask(imgs, 0.0)
        out = compute_within_region_colocalization_features(
            df, imgs, mzs, pixel_mask=mask, n_regions=2, min_region_pixels=1,
        )
        t = out.loc[out["protein"] == "P", "protein_within_region_colocalization"].mean()
        d = out.loc[out["protein"] == "DECOY_P", "protein_within_region_colocalization"].mean()
        assert t > d

    def test_blind_to_is_decoy_and_deterministic(self):
        # is_decoy is not an input; same inputs + seed -> identical output.
        imgs, mzs = _in_phase_vs_anti_phase()
        df = pd.DataFrame({"feature_mz": [100.0, 110.0], "protein": ["P", "P"]})
        mask = compute_tissue_mask(imgs, 0.0)
        a = compute_within_region_colocalization_features(
            df.copy(), imgs, mzs, pixel_mask=mask, n_regions=2, min_region_pixels=1, random_state=7,
        )[_WITHIN_REGION_COLOC_COLS + _DOMINANT_REGION_COLOC_COLS].to_numpy()
        b = compute_within_region_colocalization_features(
            df.copy(), imgs, mzs, pixel_mask=mask, n_regions=2, min_region_pixels=1, random_state=7,
        )[_WITHIN_REGION_COLOC_COLS + _DOMINANT_REGION_COLOC_COLS].to_numpy()
        np.testing.assert_allclose(a, b)

    def test_debug_dict_populated(self):
        imgs, mzs = _in_phase_vs_anti_phase()
        df = pd.DataFrame({"feature_mz": [100.0, 110.0], "protein": ["P", "P"]})
        mask = compute_tissue_mask(imgs, 0.0)
        dbg = {}
        compute_within_region_colocalization_features(
            df, imgs, mzs, pixel_mask=mask, n_regions=2, min_region_pixels=1, debug=dbg,
        )
        assert "region_labels" in dbg and dbg["region_labels"].shape == (8 * 8,)
        assert "region_pixel_counts" in dbg
        assert "dominant_region_id" in dbg and dbg["dominant_region_id"] >= 0
