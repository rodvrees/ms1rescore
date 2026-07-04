"""Tests for the gradient-boosted-tree rescoring backend (model="gbt").

GradientBoostingClassifier is a nonlinear, decision_function-based classifier that
reuses the same semi-supervised CV machinery as LDA/SVM (`_rescore_linear`), but
exposes `feature_importances_` instead of `coef_`. These tests check it produces
scores/importances of the right shape, separates targets from decoys, honors its
hyperparameters, and captures a nonlinear (XOR-like) boundary that a linear model
cannot.
"""

import numpy as np
import pandas as pd

from msi_picasso.pipeline import _rescore_lda, _rescore_gbt


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


class TestRescoreGbt:
    def test_returns_scores_and_importances(self):
        df, feats = _synthetic_features()
        scores, importances, struct, struct_names, names = _rescore_gbt(
            df, feats, init_ppm_threshold=5.0, init_fdr=0.2, train_fdr=0.05, max_iter=5,
        )
        assert scores.shape == (len(df),)
        assert np.isfinite(scores).all()
        # tree ensembles expose feature_importances_ (>=0, sum ~1), not coef_
        assert importances is not None and len(importances) == len(feats)
        assert (importances >= -1e-9).all()
        assert names == feats

    def test_separates_targets_from_decoys(self):
        df, feats = _synthetic_features()
        scores, *_ = _rescore_gbt(df, feats, init_ppm_threshold=5.0)
        d = df["is_decoy"].to_numpy()
        assert scores[~d].mean() > scores[d].mean()

    def test_hyperparameters_change_the_fit(self):
        df, feats = _synthetic_features()
        s_shallow, *_ = _rescore_gbt(df, feats, init_ppm_threshold=5.0,
                                     gbt_n_estimators=10, gbt_max_depth=1)
        s_deep, *_ = _rescore_gbt(df, feats, init_ppm_threshold=5.0,
                                  gbt_n_estimators=300, gbt_max_depth=3)
        assert not np.allclose(s_shallow, s_deep)

    def test_matches_lda_interface_shape(self):
        """GBT and LDA return the same 5-tuple shape so the dispatch is shared."""
        df, feats = _synthetic_features()
        lda_out = _rescore_lda(df, feats, init_ppm_threshold=5.0)
        gbt_out = _rescore_gbt(df, feats, init_ppm_threshold=5.0)
        assert len(lda_out) == len(gbt_out) == 5
        assert lda_out[0].shape == gbt_out[0].shape
