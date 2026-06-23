"""Tests for query_raw_maldi() and the raw-query pipeline interaction."""

import sys
import warnings

import numpy as np
import pandas as pd
import pytest

from msi_picasso import maldi_query


def _stub_5tuple(query_mzs):
    """A stand-in for extract_maldi_data's 6-tuple output."""
    n = len(query_mzs)
    ion_images = np.ones((n, 3, 3), dtype=np.float32)
    extra = {k: np.ones((n, 3, 3), dtype=np.float32) for k in ("m1", "m2", "na", "k", "chca")}
    spatial = pd.DataFrame({
        "feature_mz": np.asarray(query_mzs, dtype=np.float64),
        "intensity_p90": np.arange(n, dtype=float),
        "intensity_sum": np.arange(n, dtype=float),
        "mean_intensity": np.arange(n, dtype=float),
    })
    envelopes = {float(mz): [1.0, 0.5, 0.2] for mz in query_mzs}
    xs = np.array([0, 1, 2], dtype=np.int32)
    ys = np.array([0, 1, 2], dtype=np.int32)
    return np.asarray(query_mzs, dtype=np.float64), ion_images, extra, spatial, envelopes, (xs, ys)


class TestQueryRawMaldiAssertions:

    def test_unsorted_query_mzs_raises(self):
        with pytest.raises(AssertionError, match="sorted"):
            maldi_query.query_raw_maldi("dummy.d", np.array([900.0, 800.0, 1000.0]))

    def test_nan_query_mzs_raises(self):
        with pytest.raises(AssertionError, match="NaN"):
            maldi_query.query_raw_maldi("dummy.d", np.array([800.0, np.nan, 1000.0]))


class TestQueryRawMaldiWrapper:

    def test_returns_5tuple_schema(self, monkeypatch):
        query = np.array([800.0, 900.0, 1000.0], dtype=np.float64)

        def _fake(d_path, **kwargs):
            assert kwargs["feature_mzs"] is query or np.array_equal(kwargs["feature_mzs"], query)
            assert kwargs["drop_zero_signal"] is False
            return _stub_5tuple(kwargs["feature_mzs"])

        monkeypatch.setattr(
            "msi_picasso.maldi_extraction.extract_maldi_data", _fake
        )
        out = maldi_query.query_raw_maldi("dummy.d", query, extraction_ppm=25.0)
        assert len(out) == 5
        feature_mzs, ion_images, extra, spatial, envelopes = out
        assert np.array_equal(feature_mzs, query)
        assert ion_images.shape == (3, 3, 3)
        assert set(extra) == {"m1", "m2", "na", "k", "chca"}
        assert list(spatial["feature_mz"]) == list(query)
        assert isinstance(envelopes, dict)

    def test_extra_images_false_returns_none(self, monkeypatch):
        query = np.array([800.0, 900.0], dtype=np.float64)
        monkeypatch.setattr(
            "msi_picasso.maldi_extraction.extract_maldi_data",
            lambda d_path, **kw: _stub_5tuple(kw["feature_mzs"]),
        )
        _, _, extra, _, _ = maldi_query.query_raw_maldi(
            "dummy.d", query, extra_images=False
        )
        assert extra is None

    def test_query_mzs_derivation_is_sorted_nan_free(self):
        """The derivation candidates_df['feature_mz'] -> query grid must satisfy
        the assertions query_raw_maldi enforces."""
        feature_mz = pd.Series([1000.0, np.nan, 800.0, 900.0, 800.0])
        query = np.sort(feature_mz.dropna().to_numpy(dtype=np.float64))
        query = np.unique(query)
        assert not np.any(np.isnan(query))
        assert np.all(np.diff(query) >= 0)


class TestRawQueryPipelineInteraction:

    def test_requires_maldi_d_path(self, tmp_path):
        from msi_picasso import pipeline

        fasta = tmp_path / "t.fasta"
        fasta.write_text(">sp|P1|T_HUMAN x\nMALPVTALLLLAAGLLAHAAGTSQVQVSTQILHQKPEPTIDEKVFGR\n")
        with pytest.raises(ValueError, match="maldi_query_raw=True requires maldi_d_path"):
            pipeline.rescore(
                fasta_path=str(fasta),
                mzml_paths=[],
                maldi_mzs=np.array([], dtype=np.float64),
                digest=True,
                maldi_query_raw=True,
                maldi_d_path=None,
                output_dir=str(tmp_path / "out"),
            )

    def test_mz_shift_low_delta_warns(self, tmp_path):
        """mz_shift + maldi_query_raw + delta_min < 10 emits a UserWarning before
        extraction (extraction then fails on the bogus .d path, which we swallow)."""
        from msi_picasso import pipeline

        fasta = tmp_path / "t.fasta"
        fasta.write_text(">sp|P1|T_HUMAN x\nMALPVTALLLLAAGLLAHAAGTSQVQVSTQILHQKPEPTIDEKVFGR\n")
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            try:
                pipeline.rescore(
                    fasta_path=str(fasta),
                    mzml_paths=[],
                    maldi_mzs=np.array([], dtype=np.float64),
                    digest=True,
                    decoy_method="mz_shift",
                    mz_shift_delta_min=5.0,
                    maldi_query_raw=True,
                    maldi_d_path=str(tmp_path / "nonexistent.d"),
                    output_dir=str(tmp_path / "out"),
                )
            except Exception:
                pass
        assert any(
            issubclass(w.category, UserWarning) and "zero-signal" in str(w.message)
            for w in rec
        ), "expected a zero-signal UserWarning for mz_shift + raw-query with low delta_min"


class TestWeightedMeanInvK0:

    def test_intensity_weighted_mean_per_window(self):
        # Two query m/z; peaks within ppm windows with known mobilities.
        query = np.array([1000.0, 1500.0], dtype=np.float64)
        # peaks: (mz, intensity, mobility)
        peak_mzs = np.array([1000.0, 1000.005, 1500.0, 2000.0])  # last is outside any window
        peak_ints = np.array([3.0, 1.0, 5.0, 9.0])
        peak_mob = np.array([0.80, 0.90, 1.20, 1.50])
        out = maldi_query._weighted_mean_inv_k0(peak_mzs, peak_ints, peak_mob, query, ppm=25.0)
        # window 0: (3*0.80 + 1*0.90)/(3+1) = (2.4+0.9)/4 = 0.825
        assert out[0] == pytest.approx(0.825)
        # window 1: single peak -> 1.20
        assert out[1] == pytest.approx(1.20)

    def test_empty_window_is_nan(self):
        query = np.array([1000.0, 1500.0], dtype=np.float64)
        # only the 1000 window has signal
        out = maldi_query._weighted_mean_inv_k0(
            np.array([1000.0]), np.array([2.0]), np.array([0.95]), query, ppm=25.0
        )
        assert out[0] == pytest.approx(0.95)
        assert np.isnan(out[1])

    def test_no_peaks_all_nan(self):
        query = np.array([800.0, 900.0], dtype=np.float64)
        out = maldi_query._weighted_mean_inv_k0(
            np.array([]), np.array([]), np.array([]), query, ppm=25.0
        )
        assert out.shape == (2,)
        assert np.all(np.isnan(out))

    def test_peak_in_two_overlapping_windows_contributes_to_both(self):
        # Two query m/z closer than the ppm tolerance -> a peak between them
        # falls in both windows.
        query = np.array([1000.000, 1000.010], dtype=np.float64)  # ~10 ppm apart
        out = maldi_query._weighted_mean_inv_k0(
            np.array([1000.005]), np.array([4.0]), np.array([1.0]), query, ppm=25.0
        )
        assert out[0] == pytest.approx(1.0)
        assert out[1] == pytest.approx(1.0)

    def test_ccs_conversion_propagates_nan(self):
        from msi_picasso.maldi_imzml import one_over_k0_to_ccs

        query = np.array([1000.0, 1500.0], dtype=np.float64)
        mean_k0 = np.array([0.9, np.nan])
        ccs = np.asarray(one_over_k0_to_ccs(mean_k0, query, charge=1), dtype=float)
        assert np.isfinite(ccs[0]) and ccs[0] > 0
        assert np.isnan(ccs[1])

    def test_centroid_is_intensity_weighted_mz(self):
        # Reusing the same helper with values = peak m/z gives the observed centroid.
        query = np.array([1000.0], dtype=np.float64)
        # two peaks in the window: heavier intensity slightly above 1000
        peak_mzs = np.array([1000.000, 1000.010])
        peak_ints = np.array([1.0, 3.0])
        centroid = maldi_query._weighted_mean_in_windows(
            peak_mzs, peak_ints, peak_mzs, query, ppm=25.0
        )
        # (1*1000.000 + 3*1000.010)/4 = 1000.0075
        assert centroid[0] == pytest.approx(1000.0075)


class TestExtractObservedFeatureStatsGraceful:

    def test_returns_all_nan_without_alphatims(self, monkeypatch):
        # Setting the submodule to None makes `import alphatims.bruker` raise ImportError.
        monkeypatch.setitem(sys.modules, "alphatims.bruker", None)
        query = np.array([800.0, 900.0, 1000.0], dtype=np.float64)
        ccs, centroid = maldi_query.extract_observed_feature_stats_raw("nonexistent.d", query)
        assert ccs.shape == (3,) and centroid.shape == (3,)
        assert np.all(np.isnan(ccs)) and np.all(np.isnan(centroid))

    def test_empty_query_returns_empty(self):
        ccs, centroid = maldi_query.extract_observed_feature_stats_raw(
            "nonexistent.d", np.array([])
        )
        assert ccs.shape == (0,) and centroid.shape == (0,)


class TestRecomputePpmFromCentroids:

    def test_symmetric_target_and_decoy(self):
        from msi_picasso.pipeline import _recompute_ppm_from_centroids

        # A target anchored at 1000.0 and a decoy anchored at 1500.0, each with an
        # observed centroid offset by the same +5 ppm -> identical ppm_error.
        maldi_mzs = np.array([1000.0, 1500.0])
        centroid = np.array([1000.0 * (1 + 5e-6), 1500.0 * (1 + 5e-6)])
        feature_mz = np.array([1000.0, 1500.0])  # row 0 = target anchor, row 1 = decoy anchor
        ppm = _recompute_ppm_from_centroids(feature_mz, maldi_mzs, centroid)
        assert ppm[0] == pytest.approx(5.0, abs=1e-3)
        assert ppm[1] == pytest.approx(5.0, abs=1e-3)  # decoy treated identically

    def test_no_signal_window_is_nan_by_default(self):
        from msi_picasso.pipeline import _recompute_ppm_from_centroids

        maldi_mzs = np.array([1000.0, 1500.0])
        centroid = np.array([1000.0, np.nan])  # 1500 window had no observed peak
        feature_mz = np.array([1000.0, 1500.0])
        ppm = _recompute_ppm_from_centroids(feature_mz, maldi_mzs, centroid)
        assert ppm[0] == pytest.approx(0.0, abs=1e-6)
        assert np.isnan(ppm[1])

    def test_no_signal_window_gets_worst_case_fill(self):
        from msi_picasso.pipeline import _recompute_ppm_from_centroids

        maldi_mzs = np.array([1000.0, 1500.0])
        centroid = np.array([1000.0, np.nan])  # 1500 window had no observed peak
        feature_mz = np.array([1000.0, 1500.0])
        ppm = _recompute_ppm_from_centroids(
            feature_mz, maldi_mzs, centroid, worst_case_ppm=25.0
        )
        # signal row keeps its measured ppm; empty-window row -> worst-case edge
        assert ppm[0] == pytest.approx(0.0, abs=1e-6)
        assert ppm[1] == pytest.approx(25.0)
        assert np.isfinite(ppm).all()  # no NaN left to median-impute

    def test_decoy_ppm_not_derived_from_peptide_mass(self):
        from msi_picasso.pipeline import _recompute_ppm_from_centroids

        # mz_shift decoy: peptide mass 1000, shifted anchor 1012 (12 Da off). The
        # observed peak sits exactly on the shifted anchor -> ppm ~0, NOT ~12000 ppm
        # (which is what (centroid - peptide_mass) would give, leaking the label).
        maldi_mzs = np.array([1012.0])
        centroid = np.array([1012.0])
        feature_mz = np.array([1012.0])  # decoy's anchor = shifted m/z
        ppm = _recompute_ppm_from_centroids(feature_mz, maldi_mzs, centroid)
        assert ppm[0] == pytest.approx(0.0, abs=1e-6)


class TestObservedCcsByFeatureIdx:

    def test_keyed_by_feature_idx_via_mz(self):
        from msi_picasso.pipeline import _observed_ccs_by_feature_idx

        # candidates carry feature_idx into a (different) digest grid; ccs_arr is
        # aligned to maldi_mzs (the query grid). Bridge must be by feature_mz.
        candidates = pd.DataFrame({
            "feature_idx": [50, 50, 77, 91],
            "feature_mz": [1000.0, 1000.0, 1500.0, 2000.0],
        })
        maldi_mzs = np.array([1000.0, 1500.0, 2000.0])
        ccs_arr = np.array([350.0, 420.0, np.nan])  # 2000.0 has no CCS
        out = _observed_ccs_by_feature_idx(candidates, maldi_mzs, ccs_arr)
        assert out == {50: 350.0, 77: 420.0}  # feature_idx 91 dropped (NaN CCS)

    def test_returns_none_when_no_finite_ccs(self):
        from msi_picasso.pipeline import _observed_ccs_by_feature_idx

        candidates = pd.DataFrame({"feature_idx": [1], "feature_mz": [1000.0]})
        out = _observed_ccs_by_feature_idx(
            candidates, np.array([1000.0]), np.array([np.nan])
        )
        assert out is None
