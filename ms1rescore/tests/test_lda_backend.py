"""Tests for the LDA rescoring backend."""
import numpy as np
import pandas as pd
import pytest

from ms1rescore.feature_generator import LDA_FEATURES, MALDI_INTRINSIC_FEATURES
from ms1rescore.pipeline import (
    _find_best_feature_labels,
    _rescore_lda,
    _select_feature_winners,
    _tdc_qvalues,
)

_EXPECTED_RESULT_COLS = [
    "peptide", "protein", "feature_mz", "feature_idx", "is_decoy",
    "lda_score_r1", "lda_score_r2", "q_value", "is_tdc_winner",
    "reweighted_score", "reweighted_q_value",
]


def _make_features_df(n_per_class: int = 50, seed: int = 42) -> pd.DataFrame:
    """Synthetic features_df with two candidates per feature_idx.

    Targets get low ppm / high isotope cosine; decoys get the opposite so LDA
    has a clear signal to learn from.
    """
    rng = np.random.default_rng(seed)
    n = n_per_class * 2  # targets + decoys
    is_decoy = np.array([False] * n_per_class + [True] * n_per_class)

    ppm = np.concatenate([
        rng.uniform(0.0, 1.5, n_per_class),   # targets: good accuracy
        rng.uniform(8.0, 20.0, n_per_class),  # decoys: worse
    ])
    iso = np.concatenate([
        rng.uniform(0.75, 1.0, n_per_class),  # targets: high cosine
        rng.uniform(0.0, 0.4, n_per_class),   # decoys: low cosine
    ])

    # Two candidates per feature_idx (one target, one decoy)
    feature_idx = np.tile(np.arange(n_per_class), 2)
    feature_mz = feature_idx * 10.0 + 500.0

    return pd.DataFrame({
        "peptide": [f"PEP{i}" for i in range(n)],
        "protein": [f"PROT{i % 10}" for i in range(n)],
        "is_decoy": is_decoy,
        "feature_idx": feature_idx,
        "feature_mz": feature_mz,
        "ppm_error_abs": ppm,
        "theo_isotope_cosine": iso,
        "n_candidates": rng.integers(1, 10, n).astype(float),
        "log_n_candidates": rng.uniform(0.0, 2.0, n),
        "peptide_length": rng.integers(7, 25, n).astype(float),
        "n_missed_cleavages": rng.integers(0, 3, n).astype(float),
    })


# ---------------------------------------------------------------------------
# Alias
# ---------------------------------------------------------------------------


class TestLDAAlias:
    def test_lda_features_is_alias_for_maldi_intrinsic(self):
        assert LDA_FEATURES is MALDI_INTRINSIC_FEATURES


# ---------------------------------------------------------------------------
# _rescore_lda
# ---------------------------------------------------------------------------


class TestRescoreLDA:
    @pytest.fixture(scope="class")
    def scores(self):
        df = _make_features_df()
        feat = [f for f in MALDI_INTRINSIC_FEATURES if f in df.columns]
        scores, _, _ = _rescore_lda(df, feat, init_ppm_threshold=2.0)
        return scores

    def test_returns_ndarray(self, scores):
        assert isinstance(scores, np.ndarray)

    def test_correct_shape(self, scores):
        df = _make_features_df()
        assert scores.shape == (len(df),)

    def test_scores_all_finite(self, scores):
        assert np.isfinite(scores).all()

    def test_targets_score_higher_on_average(self, scores):
        df = _make_features_df()
        is_decoy = df["is_decoy"].values
        assert scores[~is_decoy].mean() > scores[is_decoy].mean()


# ---------------------------------------------------------------------------
# Two-pass result_df structure
# ---------------------------------------------------------------------------


def _build_result_df(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full two-pass LDA logic and return the result_df."""
    feat = [f for f in MALDI_INTRINSIC_FEATURES if f in df.columns]
    feature_col = "feature_mz"

    scores1, _, _ = _rescore_lda(df, feat, init_ppm_threshold=2.0)

    winner_pos, winners_df = _select_feature_winners(df, scores1, feature_col)

    scores2, _, _ = _rescore_lda(winners_df, feat, init_ppm_threshold=2.0)

    is_decoy_w = winners_df["is_decoy"].values.astype(bool)
    q2 = _tdc_qvalues(scores2, is_decoy_w)

    is_winner_full = np.zeros(len(df), dtype=bool)
    is_winner_full[winner_pos] = True
    scores2_full = np.full(len(df), np.nan)
    scores2_full[winner_pos] = scores2
    q_full = np.full(len(df), np.nan)
    q_full[winner_pos] = q2
    rw_full = np.full(len(df), np.nan)
    rw_full[winner_pos] = scores2  # no LC-MS/MS prior in this test
    rw_q_full = np.full(len(df), np.nan)
    rw_q_full[winner_pos] = q2

    return pd.DataFrame({
        "peptide": df["peptide"].values,
        "protein": df["protein"].values,
        "feature_mz": df["feature_mz"].values,
        "feature_idx": df["feature_idx"].values,
        "is_decoy": df["is_decoy"].values.astype(bool),
        "lda_score_r1": scores1,
        "lda_score_r2": scores2_full,
        "q_value": q_full,
        "is_tdc_winner": is_winner_full,
        "reweighted_score": rw_full,
        "reweighted_q_value": rw_q_full,
    })


class TestTwoPass:
    @pytest.fixture(scope="class")
    def result_df(self):
        return _build_result_df(_make_features_df())

    def test_result_df_has_expected_columns(self, result_df):
        assert list(result_df.columns) == _EXPECTED_RESULT_COLS

    def test_lda_score_r2_finite_for_winners(self, result_df):
        winner_r2 = result_df.loc[result_df["is_tdc_winner"], "lda_score_r2"]
        assert np.isfinite(winner_r2.values).all()

    def test_non_winners_have_nan_r2(self, result_df):
        non_winner_r2 = result_df.loc[~result_df["is_tdc_winner"], "lda_score_r2"]
        assert non_winner_r2.isna().all()

    def test_at_most_one_winner_per_feature(self, result_df):
        # _select_feature_winners may drop low-confidence features (Q0.02 filter),
        # so counts can be 0 or 1 but never > 1.
        counts = result_df.groupby("feature_idx")["is_tdc_winner"].sum()
        assert counts.le(1).all()


# ---------------------------------------------------------------------------
# _find_best_feature_labels — pairwise combination fallback
# ---------------------------------------------------------------------------


class TestFindBestFeatureLabels:
    """
    Verify that _find_best_feature_labels takes the pairwise path when no
    single feature yields enough pseudo-positives.

    Construction:
    - 100 targets, 100 decoys.
    - Feature A: first 50 targets score 3, second 50 targets score 0, decoys ~0.
    - Feature B: first 50 targets score 0, second 50 targets score 3, decoys ~0.
    - A and B are complementary, so each individually scores the first
      or second target half at the top but cannot reach q <= 0.01 (FDR at
      top-50 = 1/50 = 0.02 > 0.01 → 0 targets pass individually).
    - A + B gives all 100 targets a score of ~3 and all decoys a score of
      ~0. After StandardScaler, sorted descending the top 100 items are all
      targets → FDR at rank 100 = 1/100 = 0.01 → 100 targets at q <= 0.01.
    """

    @pytest.fixture(scope="class")
    def xy(self):
        n = 100
        rng = np.random.default_rng(7)
        is_decoy = np.array([False] * n + [True] * n)

        feat_a = np.concatenate([
            np.full(n // 2, 3.0),
            np.full(n // 2, 0.0),
            rng.uniform(-0.1, 0.1, n),
        ])
        feat_b = np.concatenate([
            np.full(n // 2, 0.0),
            np.full(n // 2, 3.0),
            rng.uniform(-0.1, 0.1, n),
        ])
        X = np.column_stack([feat_a, feat_b])
        feature_names = ["signal_a", "signal_b"]
        return X, is_decoy, feature_names

    def test_single_feature_gives_zero_positives(self, xy):
        X, is_decoy, feature_names = xy
        # Verify the premise: neither single feature reaches q <= 0.01.
        from ms1rescore.pipeline import _tdc_qvalues

        rng = np.random.default_rng(0)
        tiebreak = rng.uniform(-1e-9, 1e-9, X.shape[0])
        for col_idx in range(2):
            col = X[:, col_idx].copy()
            col[~np.isfinite(col)] = np.median(col[np.isfinite(col)])
            for ascending in (True, False):
                scores = (col if ascending else -col) + tiebreak
                q = _tdc_qvalues(scores, is_decoy)
                n_pass = int(((~is_decoy) & (q <= 0.01)).sum())
                assert n_pass == 0, (
                    f"Expected 0 targets at q<=0.01 for feature {col_idx} "
                    f"(ascending={ascending}), got {n_pass}"
                )

    def test_pair_path_is_taken(self, xy):
        X, is_decoy, feature_names = xy
        result = _find_best_feature_labels(
            X, is_decoy, feature_names, train_fdr=0.01, min_pair_threshold=10
        )
        assert result is not None, "Expected pair fallback to find a result"
        labels, best_feat_name, n_passing = result
        # The best feature name must describe a pair (contains a space, indicating
        # the "feat_i + feat_j" or "feat_i - feat_j" format).
        assert "signal_a" in best_feat_name and "signal_b" in best_feat_name, (
            f"Expected pair name containing both features, got '{best_feat_name}'"
        )

    def test_pair_path_returns_enough_positives(self, xy):
        X, is_decoy, feature_names = xy
        result = _find_best_feature_labels(
            X, is_decoy, feature_names, train_fdr=0.01, min_pair_threshold=10
        )
        assert result is not None
        _, _, n_passing = result
        assert n_passing >= 10, f"Expected at least 10 targets at q<=0.01, got {n_passing}"

    def test_labels_consistent_with_decoy_mask(self, xy):
        X, is_decoy, feature_names = xy
        result = _find_best_feature_labels(
            X, is_decoy, feature_names, train_fdr=0.01, min_pair_threshold=10
        )
        assert result is not None
        labels, _, _ = result
        assert labels.shape == (X.shape[0],)
        assert np.all(labels[is_decoy] == -1), "All decoys must have label -1"
        assert np.all(labels[~is_decoy] >= 0), "Targets must have label 0 or +1"

    def test_skips_pair_search_when_single_feature_passes_threshold(self, xy):
        X, is_decoy, feature_names = xy
        # With min_pair_threshold=0, pair search should never be triggered
        # (single-feature result of 0 is not < 0). The function falls back
        # to returning None when no single feature works.
        result = _find_best_feature_labels(
            X, is_decoy, feature_names, train_fdr=0.01, min_pair_threshold=0
        )
        assert result is None, (
            "Expected None when min_pair_threshold=0 and no single feature separates"
        )
