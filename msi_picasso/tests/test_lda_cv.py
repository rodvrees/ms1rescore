"""Cross-validated (out-of-fold) scoring in the LDA/QDA backends.

The key property: when targets and decoys are exchangeable (no real signal), an
in-sample model overfits and manufactures separation, but out-of-fold scoring
does not — so the TDC FDR stays honest.
"""

import numpy as np
import pytest
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from msi_picasso.pipeline import _cv_semisup_scores, _make_fold_ids


def _make_pipe():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("lda", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto", priors=[0.5, 0.5])),
    ])


class TestMakeFoldIds:
    def test_none_when_too_few(self):
        is_decoy = np.array([True, False] * 3)  # 3 per class, need >= 2*folds = 6
        assert _make_fold_ids(is_decoy, cv_folds=3) is None

    def test_valid_folds_stratified(self):
        rng = np.random.default_rng(0)
        is_decoy = rng.integers(0, 2, 300).astype(bool)
        folds = _make_fold_ids(is_decoy, cv_folds=3)
        assert folds is not None
        assert set(np.unique(folds)) == {0, 1, 2}
        # every fold has both targets and decoys
        for k in range(3):
            m = folds == k
            assert is_decoy[m].any() and (~is_decoy[m]).any()


class TestCvScoring:
    def test_oof_does_not_separate_exchangeable_classes(self):
        """Pure-noise features (no real T/D signal): in-sample LDA overfits to a high
        AUC, but out-of-fold scoring stays ~0.5 (chance)."""
        rng = np.random.default_rng(0)
        n, p = 240, 40           # many features, no signal -> easy to overfit
        X = rng.standard_normal((n, p))
        is_decoy = np.zeros(n, dtype=bool)
        is_decoy[n // 2:] = True
        labels = np.where(is_decoy, -1, 1).astype(np.int8)  # all targets +1, decoys -1
        folds = _make_fold_ids(is_decoy, cv_folds=3)
        assert folds is not None

        oof, pipe_full = _cv_semisup_scores(X, labels, folds, _make_pipe)
        insample = pipe_full.decision_function(X).ravel()

        auc_oof = roc_auc_score(is_decoy.astype(int), oof)
        auc_in = roc_auc_score(is_decoy.astype(int), insample)
        auc_oof = max(auc_oof, 1 - auc_oof)
        auc_in = max(auc_in, 1 - auc_in)

        assert auc_in > 0.68, f"in-sample should overfit, got {auc_in:.2f}"
        assert auc_oof < 0.62, f"out-of-fold should be ~chance, got {auc_oof:.2f}"
        assert auc_in - auc_oof > 0.10  # overfitting gap removed by out-of-fold scoring

    def test_oof_recovers_real_signal(self):
        """When there IS genuine signal, out-of-fold scoring still recovers it."""
        rng = np.random.default_rng(1)
        n, p = 240, 10
        X = rng.standard_normal((n, p))
        is_decoy = np.zeros(n, dtype=bool)
        is_decoy[n // 2:] = True
        X[~is_decoy, 0] += 1.5  # real signal in feature 0 for targets
        labels = np.where(is_decoy, -1, 1).astype(np.int8)
        folds = _make_fold_ids(is_decoy, cv_folds=3)
        oof, _ = _cv_semisup_scores(X, labels, folds, _make_pipe)
        auc = roc_auc_score(is_decoy.astype(int), oof)
        assert max(auc, 1 - auc) > 0.75  # real separation survives CV

    def test_in_sample_fallback_when_no_folds(self):
        rng = np.random.default_rng(2)
        X = rng.standard_normal((40, 5))
        is_decoy = np.zeros(40, dtype=bool); is_decoy[20:] = True
        labels = np.where(is_decoy, -1, 1).astype(np.int8)
        # fold_ids=None -> in-sample scoring, returns finite scores
        scores, pipe = _cv_semisup_scores(X, labels, None, _make_pipe)
        assert np.isfinite(scores).all()
        assert np.allclose(scores, pipe.decision_function(X).ravel())

    def test_label_zero_rows_are_scored_not_trained(self):
        """label==0 (unlabelled targets) are scored out-of-fold but never trained."""
        rng = np.random.default_rng(3)
        n, p = 300, 8
        X = rng.standard_normal((n, p))
        is_decoy = np.zeros(n, dtype=bool); is_decoy[n // 2:] = True
        labels = np.where(is_decoy, -1, 1).astype(np.int8)
        # demote a chunk of targets to label 0
        tgt_idx = np.where(~is_decoy)[0]
        labels[tgt_idx[:50]] = 0
        folds = _make_fold_ids(is_decoy, cv_folds=3)
        oof, _ = _cv_semisup_scores(X, labels, folds, _make_pipe)
        assert np.isfinite(oof).all()  # every row (incl label-0) gets a score
