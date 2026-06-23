"""Tests for the Linear SVM rescoring backend (model="svm").

LinearSVC is a linear, decision_function-based classifier that reuses the same
semi-supervised CV machinery as LDA (`_rescore_linear`). These tests check it
produces scores/importances of the right shape and that C is honored.
"""

import numpy as np
import pandas as pd
import pytest

from msi_picasso.pipeline import _rescore_lda, _rescore_svm


def _synthetic_features(n=400, seed=0):
    """Targets with a separable feature + noise; decoys ~ noise only."""
    rng = np.random.default_rng(seed)
    n_t, n_d = n // 2, n // 2
    # one discriminative feature: targets shifted +1.5, decoys at 0
    good_t = rng.normal(1.5, 1.0, n_t)
    good_d = rng.normal(0.0, 1.0, n_d)
    noise = rng.normal(0.0, 1.0, n)
    df = pd.DataFrame({
        "peptide": [f"PEP{i}" for i in range(n)],
        "is_decoy": np.array([False] * n_t + [True] * n_d),
        "good_feature": np.concatenate([good_t, good_d]),
        "noise_feature": noise,
        "ppm_error_abs": rng.uniform(0, 10, n),
        "n_candidates": np.ones(n, dtype=int),
    })
    return df, ["good_feature", "noise_feature"]


class TestRescoreSvm:
    def test_returns_scores_and_importances(self):
        df, feats = _synthetic_features()
        scores, importances, struct, struct_names, names = _rescore_svm(
            df, feats, init_ppm_threshold=5.0, init_fdr=0.2, train_fdr=0.05, max_iter=5,
        )
        assert scores.shape == (len(df),)
        assert np.isfinite(scores).all()
        assert importances is not None and len(importances) == len(feats)
        assert names == feats

    def test_separates_targets_from_decoys(self):
        df, feats = _synthetic_features()
        scores, *_ = _rescore_svm(df, feats, init_ppm_threshold=5.0)
        d = df["is_decoy"].to_numpy()
        # the discriminative feature should let targets score above decoys on average
        assert scores[~d].mean() > scores[d].mean()

    def test_svm_c_changes_the_fit(self):
        df, feats = _synthetic_features()
        s_lo, imp_lo, *_ = _rescore_svm(df, feats, init_ppm_threshold=5.0, svm_c=0.001)
        s_hi, imp_hi, *_ = _rescore_svm(df, feats, init_ppm_threshold=5.0, svm_c=100.0)
        # different C → different decision boundary → different coefficients
        assert not np.allclose(imp_lo, imp_hi)

    def test_matches_lda_interface_shape(self):
        """SVM and LDA return the same 5-tuple shape so the dispatch is shared."""
        df, feats = _synthetic_features()
        lda_out = _rescore_lda(df, feats, init_ppm_threshold=5.0)
        svm_out = _rescore_svm(df, feats, init_ppm_threshold=5.0)
        assert len(lda_out) == len(svm_out) == 5
        assert lda_out[0].shape == svm_out[0].shape
