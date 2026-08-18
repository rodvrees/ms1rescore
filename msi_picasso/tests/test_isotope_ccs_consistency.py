"""Tests for the isotope-envelope CCS consistency features.

The feature turns the spread of the *observed* CCS across the M0/M+1/M+2 envelope
into a ranker signal (the CCS analogue of IsoMobil's IPMV).  These tests pin the
symmetry invariant, the spread arithmetic, the min-peak-fraction gate, and the
missing-record behaviour.
"""

import inspect

import numpy as np
import pandas as pd
import pytest

from msi_picasso.maldi_features import compute_isotope_ccs_consistency_features
from msi_picasso.pipeline import _observed_envelope_ccs_by_feature_idx


def _rec(ccs, ints):
    """A 6-element envelope record (ccs_m0..m2, int_m0..m2)."""
    return np.array(list(ccs) + list(ints), dtype=float)


def _df(feature_idx, is_decoy=None, feature_mz=None):
    n = len(feature_idx)
    return pd.DataFrame({
        "feature_idx": feature_idx,
        "feature_mz": feature_mz if feature_mz is not None else np.arange(n, dtype=float) + 800.0,
        "peptide": [f"PEPTIDE{i}" for i in range(n)],
        "is_decoy": is_decoy if is_decoy is not None else [False] * n,
    })


class TestSymmetry:

    def test_signature_has_no_is_decoy(self):
        params = inspect.signature(compute_isotope_ccs_consistency_features).parameters
        assert "is_decoy" not in params

    def test_swapping_labels_changes_nothing(self):
        env = {0: _rec([300.0, 302.0, 301.0], [100.0, 50.0, 20.0]),
               1: _rec([400.0, 430.0, np.nan], [100.0, 50.0, 0.0])}
        df_a = _df([0, 1, 0, 1], is_decoy=[False, False, True, True])
        df_b = _df([0, 1, 0, 1], is_decoy=[True, True, False, False])
        out_a = compute_isotope_ccs_consistency_features(df_a, env)
        out_b = compute_isotope_ccs_consistency_features(df_b, env)
        cols = ["isotope_ccs_n_peaks", "isotope_ccs_spread", "isotope_ccs_spread_rel"]
        for c in cols:
            np.testing.assert_allclose(
                out_a[c].to_numpy(dtype=float), out_b[c].to_numpy(dtype=float),
                equal_nan=True,
            )


class TestSpreadMath:

    def test_three_peaks(self):
        env = {0: _rec([300.0, 306.0, 303.0], [100.0, 50.0, 20.0])}
        out = compute_isotope_ccs_consistency_features(_df([0]), env)
        assert out["isotope_ccs_n_peaks"].iloc[0] == 3
        assert out["isotope_ccs_spread"].iloc[0] == pytest.approx(6.0)
        assert out["isotope_ccs_spread_rel"].iloc[0] == pytest.approx(6.0 / 303.0)

    def test_two_peaks(self):
        env = {0: _rec([300.0, 310.0, np.nan], [100.0, 50.0, 0.0])}
        out = compute_isotope_ccs_consistency_features(_df([0]), env)
        assert out["isotope_ccs_n_peaks"].iloc[0] == 2
        assert out["isotope_ccs_spread"].iloc[0] == pytest.approx(10.0)
        assert out["isotope_ccs_spread_rel"].iloc[0] == pytest.approx(10.0 / 305.0)

    def test_one_peak_spread_is_nan(self):
        env = {0: _rec([300.0, np.nan, np.nan], [100.0, 0.0, 0.0])}
        out = compute_isotope_ccs_consistency_features(_df([0]), env)
        assert out["isotope_ccs_n_peaks"].iloc[0] == 1
        assert np.isnan(out["isotope_ccs_spread"].iloc[0])
        assert np.isnan(out["isotope_ccs_spread_rel"].iloc[0])

    def test_zero_peaks_spread_is_nan(self):
        env = {0: _rec([np.nan, np.nan, np.nan], [0.0, 0.0, 0.0])}
        out = compute_isotope_ccs_consistency_features(_df([0]), env)
        assert out["isotope_ccs_n_peaks"].iloc[0] == 0
        assert np.isnan(out["isotope_ccs_spread"].iloc[0])
        assert np.isnan(out["isotope_ccs_spread_rel"].iloc[0])

    def test_nonfinite_ccs_with_signal_does_not_count(self):
        """Intensity without a finite CCS (no mobility at that peak) is not a peak."""
        env = {0: _rec([300.0, np.nan, 306.0], [100.0, 50.0, 20.0])}
        out = compute_isotope_ccs_consistency_features(_df([0]), env)
        assert out["isotope_ccs_n_peaks"].iloc[0] == 2
        assert out["isotope_ccs_spread"].iloc[0] == pytest.approx(6.0)


class TestMinPeakFraction:

    def test_drops_m1_below_fraction(self):
        # M+1 carries 5% of M0 -> dropped at frac 0.10, kept at the 0.0 default.
        env = {0: _rec([300.0, 350.0, 305.0], [100.0, 5.0, 40.0])}
        loose = compute_isotope_ccs_consistency_features(_df([0]), env)
        assert loose["isotope_ccs_n_peaks"].iloc[0] == 3
        assert loose["isotope_ccs_spread"].iloc[0] == pytest.approx(50.0)

        strict = compute_isotope_ccs_consistency_features(
            _df([0]), env, isotope_ccs_min_peak_frac=0.10
        )
        assert strict["isotope_ccs_n_peaks"].iloc[0] == 2
        assert strict["isotope_ccs_spread"].iloc[0] == pytest.approx(5.0)

    def test_drops_m2_below_fraction(self):
        env = {0: _rec([300.0, 305.0, 400.0], [100.0, 40.0, 1.0])}
        strict = compute_isotope_ccs_consistency_features(
            _df([0]), env, isotope_ccs_min_peak_frac=0.05
        )
        assert strict["isotope_ccs_n_peaks"].iloc[0] == 2
        assert strict["isotope_ccs_spread"].iloc[0] == pytest.approx(5.0)

    def test_no_m0_signal_disables_the_gate(self):
        """With I_0 == 0 there is nothing to take a fraction of, so no peak is dropped."""
        env = {0: _rec([np.nan, 300.0, 308.0], [0.0, 1.0, 2.0])}
        out = compute_isotope_ccs_consistency_features(
            _df([0]), env, isotope_ccs_min_peak_frac=0.5
        )
        assert out["isotope_ccs_n_peaks"].iloc[0] == 2
        assert out["isotope_ccs_spread"].iloc[0] == pytest.approx(8.0)


class TestMissingRecord:

    def test_all_nan_without_raising(self):
        env = {0: _rec([300.0, 302.0, 301.0], [100.0, 50.0, 20.0])}
        out = compute_isotope_ccs_consistency_features(_df([0, 7]), env)
        assert np.isnan(out["isotope_ccs_n_peaks"].iloc[1])
        assert np.isnan(out["isotope_ccs_spread"].iloc[1])
        assert np.isnan(out["isotope_ccs_spread_rel"].iloc[1])

    def test_empty_map_returns_df_unchanged(self):
        df = _df([0, 1])
        assert compute_isotope_ccs_consistency_features(df, None) is df
        assert compute_isotope_ccs_consistency_features(df, {}) is df


class TestColocatedNullIsSymmetric:
    """A co-located target/decoy pair (mz_shuffle-style) shares the feature and
    therefore the envelope, so the feature cannot separate them (AUC ~ 0.5)."""

    def test_colocated_pair_gets_identical_values(self):
        rng = np.random.default_rng(0)
        n_feat = 40
        env = {
            i: _rec(300.0 + rng.normal(0, 5, 3), [100.0, 60.0, 30.0])
            for i in range(n_feat)
        }
        # One target and one decoy per feature, on the identical feature_idx/mz.
        idx = list(range(n_feat)) * 2
        mz = [800.0 + i for i in range(n_feat)] * 2
        out = compute_isotope_ccs_consistency_features(
            _df(idx, is_decoy=[False] * n_feat + [True] * n_feat, feature_mz=mz), env
        )
        t = out.loc[~out["is_decoy"], "isotope_ccs_spread"].to_numpy()
        d = out.loc[out["is_decoy"], "isotope_ccs_spread"].to_numpy()
        np.testing.assert_allclose(t, d)

        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(out["is_decoy"].astype(int), out["isotope_ccs_spread"])
        assert auc == pytest.approx(0.5, abs=1e-9)


class TestEnvelopeBuilder:
    """pipeline._observed_envelope_ccs_by_feature_idx bridges the queried m/z grid
    onto the candidates' own feature_idx (which indexes the digest grid in raw-query)."""

    def test_bridges_via_feature_mz(self):
        maldi_mzs = np.array([800.0, 900.0, 1000.0])
        envelope = {
            "ccs_m0": np.array([300.0, 400.0, np.nan]),
            "ccs_m1": np.array([302.0, 460.0, np.nan]),
            "ccs_m2": np.array([301.0, np.nan, np.nan]),
            "int_m0": np.array([100.0, 100.0, 0.0]),
            "int_m1": np.array([50.0, 50.0, 0.0]),
            "int_m2": np.array([20.0, 0.0, 0.0]),
        }
        candidates = pd.DataFrame({
            "feature_idx": [11, 12, 13],
            "feature_mz": [800.0, 900.0, 1234.0],  # 1234 is not on the queried grid
        })
        out = _observed_envelope_ccs_by_feature_idx(candidates, maldi_mzs, envelope)
        assert set(out) == {11, 12}
        np.testing.assert_allclose(out[11], [300.0, 302.0, 301.0, 100.0, 50.0, 20.0])

        feats = compute_isotope_ccs_consistency_features(
            _df([11, 12, 13], feature_mz=[800.0, 900.0, 1234.0]), out
        )
        assert feats["isotope_ccs_spread"].iloc[0] == pytest.approx(2.0)
        assert feats["isotope_ccs_spread"].iloc[1] == pytest.approx(60.0)
        assert np.isnan(feats["isotope_ccs_spread"].iloc[2])

    def test_none_envelope_returns_none(self):
        candidates = pd.DataFrame({"feature_idx": [1], "feature_mz": [800.0]})
        assert _observed_envelope_ccs_by_feature_idx(candidates, np.array([800.0]), None) is None


class TestNotInLeakList:

    def test_features_are_not_excluded_for_mz_shuffle(self):
        from msi_picasso.feature_generator import MALDI_INTRINSIC_FEATURES
        from msi_picasso.pipeline import _MZ_SHUFFLE_CCS_LEAK_FEATURES

        cols = ["isotope_ccs_n_peaks", "isotope_ccs_spread", "isotope_ccs_spread_rel"]
        for c in cols:
            assert c in MALDI_INTRINSIC_FEATURES
            assert c not in _MZ_SHUFFLE_CCS_LEAK_FEATURES


class TestWeightedMeanReturnsWeight:
    """extract_observed_feature_stats_raw needs the integrated in-window intensity
    alongside the weighted mean; _weighted_mean_in_windows(return_weight=True)."""

    def test_returns_mean_and_intensity_sum(self):
        from msi_picasso.maldi_query import _weighted_mean_in_windows

        query = np.array([1000.0, 1001.003355])
        peak_mzs = np.array([1000.0, 1000.005, 1001.003355])
        peak_ints = np.array([10.0, 30.0, 4.0])
        values = np.array([1.0, 2.0, 5.0])
        mean, isum = _weighted_mean_in_windows(
            peak_mzs, peak_ints, values, query, ppm=25.0, return_weight=True
        )
        assert isum[0] == pytest.approx(40.0)
        assert isum[1] == pytest.approx(4.0)
        assert mean[0] == pytest.approx((10 * 1.0 + 30 * 2.0) / 40.0)
        assert mean[1] == pytest.approx(5.0)

    def test_no_peaks_returns_zero_weight(self):
        from msi_picasso.maldi_query import _weighted_mean_in_windows

        query = np.array([1000.0, 1001.0])
        mean, isum = _weighted_mean_in_windows(
            np.array([]), np.array([]), np.array([]), query, ppm=25.0, return_weight=True
        )
        assert np.isnan(mean).all()
        assert (isum == 0).all()
