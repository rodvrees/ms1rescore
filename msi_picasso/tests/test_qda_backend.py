"""Tests for the QDA rescoring backend (_rescore_qda)."""
import numpy as np
import pandas as pd
import pytest

from msi_picasso.pipeline import _rescore_qda


def _make_features_df(
    n_target: int = 200,
    n_decoy: int = 100,
    seed: int = 42,
) -> pd.DataFrame:
    """Bimodal synthetic feature DataFrame: targets have higher feature_a scores."""
    rng = np.random.default_rng(seed)
    target_a = rng.normal(2.0, 0.6, n_target)
    decoy_a = rng.normal(0.0, 0.6, n_decoy)
    target_b = rng.normal(1.5, 0.8, n_target)
    decoy_b = rng.normal(0.0, 0.8, n_decoy)

    feature_a = np.concatenate([target_a, decoy_a])
    feature_b = np.concatenate([target_b, decoy_b])
    is_decoy = np.array([False] * n_target + [True] * n_decoy)
    ppm = np.concatenate([
        rng.uniform(0, 3, n_target),
        rng.uniform(0, 10, n_decoy),
    ])
    n_cand = np.ones(n_target + n_decoy, dtype=int)

    return pd.DataFrame({
        "feature_a": feature_a,
        "feature_b": feature_b,
        "is_decoy": is_decoy,
        "ppm_error_abs": ppm,
        "n_candidates": n_cand,
    })


class TestRescoreQDA:
    def test_returns_four_tuple(self):
        df = _make_features_df()
        result = _rescore_qda(df, ["feature_a", "feature_b"], init_ppm_threshold=5.0)
        assert len(result) == 4

    def test_scores_length_matches_input(self):
        df = _make_features_df()
        scores, pep_proba, _, _ = _rescore_qda(df, ["feature_a", "feature_b"], init_ppm_threshold=5.0)
        assert len(scores) == len(df)
        assert len(pep_proba) == len(df)

    def test_scores_are_finite(self):
        df = _make_features_df()
        scores, _, _, _ = _rescore_qda(df, ["feature_a", "feature_b"], init_ppm_threshold=5.0)
        assert np.all(np.isfinite(scores))

    def test_pep_proba_in_unit_interval(self):
        """predict_proba returns calibrated probabilities in [0, 1]."""
        df = _make_features_df()
        _, pep_proba, _, _ = _rescore_qda(df, ["feature_a", "feature_b"], init_ppm_threshold=5.0)
        finite = pep_proba[np.isfinite(pep_proba)]
        assert np.all((finite >= 0.0) & (finite <= 1.0))

    def test_pep_proba_targets_lower_than_decoys(self):
        """Targets should have lower P(decoy|x) than decoys on well-separated data."""
        df = _make_features_df()
        _, pep_proba, _, _ = _rescore_qda(df, ["feature_a", "feature_b"], init_ppm_threshold=5.0)
        is_decoy = df["is_decoy"].values
        assert pep_proba[~is_decoy].mean() < pep_proba[is_decoy].mean()

    def test_target_scores_higher_than_decoy_on_average(self):
        """Well-separated bimodal data: mean target score > mean decoy score."""
        df = _make_features_df()
        scores, _, _, _ = _rescore_qda(df, ["feature_a", "feature_b"], init_ppm_threshold=5.0)
        is_decoy = df["is_decoy"].values
        assert scores[~is_decoy].mean() > scores[is_decoy].mean()

    def test_importances_nonnegative(self):
        """t-statistic proxy importances are |...|, so all ≥ 0."""
        df = _make_features_df()
        _, _, importances, feat_names = _rescore_qda(
            df, ["feature_a", "feature_b"], init_ppm_threshold=5.0
        )
        assert importances is not None
        assert len(importances) == len(feat_names)
        assert np.all(importances >= 0)

    def test_feature_names_used_are_subset_of_present(self):
        df = _make_features_df()
        _, _, _, feat_names = _rescore_qda(
            df, ["feature_a", "feature_b", "missing_col"], init_ppm_threshold=5.0
        )
        assert "missing_col" not in feat_names
        assert set(feat_names) <= {"feature_a", "feature_b"}

    def test_seed_mask_overrides_ppm_seed(self):
        """Passing an explicit seed_mask should not raise and should return finite scores."""
        df = _make_features_df()
        is_decoy = df["is_decoy"].values.astype(bool)
        seed = (~is_decoy) & (df["feature_a"].values > 1.5)
        scores, _, _, _ = _rescore_qda(
            df, ["feature_a", "feature_b"],
            init_ppm_threshold=5.0,
            seed_mask=seed,
        )
        assert np.all(np.isfinite(scores))

    def test_runs_with_nan_features(self):
        """NaN values in features are imputed; function should not raise."""
        df = _make_features_df()
        df.loc[df.index[:20], "feature_a"] = np.nan
        scores, _, _, _ = _rescore_qda(df, ["feature_a", "feature_b"], init_ppm_threshold=5.0)
        assert np.all(np.isfinite(scores))


class TestQDABranchIntegration:
    """Light integration test: model='qda' runs end-to-end via _rescore_qda directly."""

    def test_two_pass_does_not_raise(self):
        """Simulate R1 → winner selection → R2 as the pipeline does."""
        from msi_picasso.pipeline import _select_feature_winners

        rng = np.random.default_rng(0)
        n_features = 200
        # One target and one decoy per feature, with overlapping score distributions
        # to guarantee a mix of target and decoy winners after selection.
        feat_a_target = rng.normal(1.0, 1.0, n_features)
        feat_b_target = rng.normal(1.0, 1.0, n_features)
        feat_a_decoy = rng.normal(0.5, 1.0, n_features)
        feat_b_decoy = rng.normal(0.5, 1.0, n_features)

        df = pd.DataFrame({
            "feature_a": np.concatenate([feat_a_target, feat_a_decoy]),
            "feature_b": np.concatenate([feat_b_target, feat_b_decoy]),
            "is_decoy": np.array([False] * n_features + [True] * n_features),
            "ppm_error_abs": np.concatenate([
                rng.uniform(0, 5, n_features),
                rng.uniform(0, 10, n_features),
            ]),
            "n_candidates": np.ones(2 * n_features, dtype=int),
            "feature_idx": np.concatenate([
                np.arange(n_features), np.arange(n_features)
            ]),
        })

        scores1, _pep1, _, _ = _rescore_qda(df, ["feature_a", "feature_b"], init_ppm_threshold=5.0)

        winner_pos, winners_df = _select_feature_winners(df, scores1, "feature_idx")
        is_decoy_w = winners_df["is_decoy"].values.astype(bool)

        if is_decoy_w.all() or (~is_decoy_w).all():
            pytest.skip("Winner set is single-class — data too skewed for this test")

        target_scores_w = scores1[winner_pos][~is_decoy_w]
        score_threshold = np.percentile(target_scores_w, 80)
        r2_seed = (~is_decoy_w) & (scores1[winner_pos] >= score_threshold)

        scores2, pep2, _, _ = _rescore_qda(
            winners_df, ["feature_a", "feature_b"],
            init_ppm_threshold=5.0,
            seed_mask=r2_seed,
        )
        assert len(scores2) == len(winners_df)
        assert np.all(np.isfinite(scores2))
        # pep2 comes from predict_proba — should be in [0, 1]
        finite_pep = pep2[np.isfinite(pep2)]
        assert np.all((finite_pep >= 0.0) & (finite_pep <= 1.0))
