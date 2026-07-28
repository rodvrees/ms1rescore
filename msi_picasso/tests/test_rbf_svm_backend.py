"""Tests for the RBF-kernel SVM rescoring backend (model="rbf_svm").

SVC(kernel="rbf") is a nonlinear, decision_function-based classifier that reuses
the same semi-supervised CV machinery as LDA/SVM (`_rescore_linear`). Unlike the
tree backend it produces continuous scores, and unlike the linear backends it has
no coef_ (importances fall back to |structure coefficient|). These tests check the
5-tuple shape, target/decoy separation, that hyperparameters change the fit, that
gamma accepts both strings and floats, and that scores are continuous.
"""

import numpy as np
import pandas as pd

from msi_picasso.pipeline import _rescore_lda, _rescore_rbf_svm


def _synthetic_features(n=400, seed=0):
    """Targets with a separable feature + noise; decoys ~ noise only."""
    rng = np.random.default_rng(seed)
    n_t, n_d = n // 2, n // 2
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


class TestRescoreRbfSvm:
    def test_returns_scores_and_importances(self):
        df, feats = _synthetic_features()
        scores, importances, struct, struct_names, names = _rescore_rbf_svm(
            df, feats, init_ppm_threshold=5.0, init_fdr=0.2, train_fdr=0.05, max_iter=5,
        )
        assert scores.shape == (len(df),)
        assert np.isfinite(scores).all()
        # kernel SVM has no coef_/feature_importances_; importances fall back to
        # |structure coefficient|, so they are still populated and non-negative
        assert importances is not None and len(importances) == len(feats)
        assert (importances >= -1e-9).all()
        assert names == feats

    def test_separates_targets_from_decoys(self):
        df, feats = _synthetic_features()
        scores, *_ = _rescore_rbf_svm(df, feats, init_ppm_threshold=5.0)
        d = df["is_decoy"].to_numpy()
        assert scores[~d].mean() > scores[d].mean()

    def test_scores_are_continuous(self):
        """Unlike the tree backend, the RBF-SVM decision function is continuous:
        nearly every candidate gets a distinct score (no discrete leaf spikes)."""
        df, feats = _synthetic_features()
        scores, *_ = _rescore_rbf_svm(df, feats, init_ppm_threshold=5.0)
        n_unique = len(np.unique(np.round(scores, 8)))
        assert n_unique > 0.9 * len(scores)

    def test_hyperparameters_change_the_fit(self):
        df, feats = _synthetic_features()
        s_lo, *_ = _rescore_rbf_svm(df, feats, init_ppm_threshold=5.0, rbf_svm_c=0.1, rbf_svm_gamma="scale")
        s_hi, *_ = _rescore_rbf_svm(df, feats, init_ppm_threshold=5.0, rbf_svm_c=100.0, rbf_svm_gamma=0.1)
        assert not np.allclose(s_lo, s_hi)

    def test_gamma_accepts_string_and_float(self):
        df, feats = _synthetic_features()
        # both a keyword string and a numeric gamma must run without error
        _rescore_rbf_svm(df, feats, init_ppm_threshold=5.0, rbf_svm_gamma="auto")
        _rescore_rbf_svm(df, feats, init_ppm_threshold=5.0, rbf_svm_gamma=0.05)
        _rescore_rbf_svm(df, feats, init_ppm_threshold=5.0, rbf_svm_gamma="0.05")

    def test_matches_lda_interface_shape(self):
        df, feats = _synthetic_features()
        lda_out = _rescore_lda(df, feats, init_ppm_threshold=5.0)
        rbf_out = _rescore_rbf_svm(df, feats, init_ppm_threshold=5.0)
        assert len(lda_out) == len(rbf_out) == 5
        assert lda_out[0].shape == rbf_out[0].shape
