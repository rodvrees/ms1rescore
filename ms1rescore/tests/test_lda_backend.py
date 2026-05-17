"""Tests for the LDA rescoring backend."""
import numpy as np
import pandas as pd
import pytest

from ms1rescore.feature_generator import LDA_FEATURES, MALDI_INTRINSIC_FEATURES
from ms1rescore.pipeline import _rescore_lda, _select_feature_winners, _tdc_qvalues

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
        "has_modifications": rng.integers(0, 2, n).astype(float),
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

    def test_one_winner_per_feature(self, result_df):
        counts = result_df.groupby("feature_idx")["is_tdc_winner"].sum()
        assert counts.eq(1).all()
