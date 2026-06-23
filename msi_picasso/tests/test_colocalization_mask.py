"""Tests for TIC-masked colocalization (on-tissue pixel restriction).

Raw MALDI ion images share a dominant on/off-tissue component: every image is
~0 in the unmeasured padding around the acquired grid. A raw Pearson r is
therefore inflated toward the tissue outline for *any* pair of images. The mask
restricts the correlation to on-tissue pixels so r reflects co-distribution
within the tissue. See ``compute_tissue_mask`` / ``_pearson_r_matrix``.
"""

import numpy as np

from msi_picasso.maldi_features import (
    _pearson_r_matrix,
    _pearson_r_pairwise,
    compute_tissue_mask,
)


def _two_layer_images():
    """Build a (2, 4, 4) stack: left half = unmeasured padding (all zero).

    On-tissue (right half) the two images are anti-correlated. Off-tissue both
    are zero. Raw r over all 16 pixels is inflated positive by the shared zeros;
    masked r over the 8 on-tissue pixels recovers the true negative relationship.
    """
    H = W = 4
    a = np.zeros((H, W), dtype=np.float32)
    b = np.zeros((H, W), dtype=np.float32)
    on = np.array([4.0, 1.0, 3.0, 2.0, 5.0, 0.5, 6.0, 1.5], dtype=np.float32)
    a[:, 2:] = on.reshape(4, 2)
    b[:, 2:] = (7.0 - on).reshape(4, 2)  # anti-correlated within tissue
    return np.stack([a, b])


class TestComputeTissueMask:
    def test_drops_zero_tic_padding(self):
        imgs = _two_layer_images()
        mask = compute_tissue_mask(imgs, tic_quantile=0.0)
        assert mask.sum() == 8  # only the right (measured) half
        assert mask.dtype == bool
        assert mask.shape == (16,)

    def test_quantile_trims_low_signal(self):
        imgs = _two_layer_images()
        mask0 = compute_tissue_mask(imgs, tic_quantile=0.0)
        mask_q = compute_tissue_mask(imgs, tic_quantile=0.5)
        assert mask_q.sum() < mask0.sum()
        # every kept pixel must still be a measured pixel
        assert np.all(mask0[mask_q])

    def test_all_measured_when_no_padding(self):
        rng = np.random.RandomState(0)
        imgs = rng.rand(3, 4, 4).astype(np.float32) + 0.1  # strictly > 0
        mask = compute_tissue_mask(imgs, tic_quantile=0.0)
        assert mask.all()


class TestMaskRemovesOnTissueInflation:
    def test_masked_r_is_lower_than_raw_r(self):
        imgs = _two_layer_images()
        mzs = np.array([100.0, 200.0])

        cm_raw, _, _ = _pearson_r_matrix(imgs, mzs, pixel_mask=None)
        mask = compute_tissue_mask(imgs, tic_quantile=0.0)
        cm_masked, _, _ = _pearson_r_matrix(imgs, mzs, pixel_mask=mask)

        r_raw = float(cm_raw[0, 1])
        r_masked = float(cm_masked[0, 1])
        # Shared zeros inflate the raw correlation; masking recovers the true
        # (negative) on-tissue relationship.
        assert r_raw > r_masked
        assert r_masked < 0.0

    def test_pairwise_matches_matrix_under_mask(self):
        imgs = _two_layer_images()
        mask = compute_tissue_mask(imgs, tic_quantile=0.0)
        cm, _, _ = _pearson_r_matrix(imgs, np.array([100.0, 200.0]), pixel_mask=mask)
        r_pair = _pearson_r_pairwise(imgs[:1], imgs[1:], pixel_mask=mask)
        assert np.isclose(float(cm[0, 1]), float(r_pair[0]), atol=1e-5)

    def test_self_correlation_is_unity(self):
        imgs = _two_layer_images()
        mask = compute_tissue_mask(imgs, tic_quantile=0.0)
        cm, _, _ = _pearson_r_matrix(imgs, np.array([100.0, 200.0]), pixel_mask=mask)
        assert np.allclose(np.diag(cm), 1.0, atol=1e-4)
