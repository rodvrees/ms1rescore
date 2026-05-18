"""Tests for DeepLC-anchored apex window features (F1-F4) and MS2 RT filter."""
import numpy as np
import pytest

from ms1rescore.lcms_evidence import LCMSData, compute_all_lcms_evidence


PROTON = 1.007276


def _make_lcms_data(
    rts: list[float],
    peak_signals: list[float],
    ms1_peak_mz: float,
    ms2_rt: float | None = None,
    ms2_precursor_mass: float | None = None,
) -> LCMSData:
    """Build a minimal LCMSData with MS1 scans at given RTs and a single peak."""
    ms1_mz_arrays = [np.array([ms1_peak_mz]) for _ in rts]
    ms1_int_arrays = [np.array([sig]) for sig in peak_signals]

    if ms2_rt is not None and ms2_precursor_mass is not None:
        charge = 2
        prec_mz = (ms2_precursor_mass + charge * PROTON) / charge
        data = LCMSData(
            ms1_rts=np.array(rts),
            ms1_mz_arrays=ms1_mz_arrays,
            ms1_int_arrays=ms1_int_arrays,
            ms2_precursor_mz=np.array([prec_mz]),
            ms2_precursor_charge=np.array([charge], dtype=int),
            ms2_precursor_rt=np.array([ms2_rt]),
            ms2_mz_arrays=[np.array([100.0, 200.0, 300.0, 400.0])],
            ms2_int_arrays=[np.array([0.9, 0.7, 0.5, 0.3])],
        )
    else:
        data = LCMSData(
            ms1_rts=np.array(rts),
            ms1_mz_arrays=ms1_mz_arrays,
            ms1_int_arrays=ms1_int_arrays,
        )
    data.build_index()
    return data


def _run(
    peptide: str,
    mh_mz: float,
    predicted_rt: float,
    lcms_data: LCMSData,
    rt_window_min: float,
    ms2pip_cache: dict | None = None,
    n_C: int = 30,
    n_H: int = 50,
    n_N: int = 8,
    n_O: int = 10,
    n_S: int = 0,
) -> dict:
    import pandas as pd

    candidates_df = pd.DataFrame(
        {
            "peptide": [peptide],
            "feature_mz": [mh_mz],
            "is_decoy": [False],
            "n_C": [n_C],
            "n_H": [n_H],
            "n_N": [n_N],
            "n_O": [n_O],
            "n_S": [n_S],
        },
        index=[0],
    )
    deeplc_cache = {peptide: predicted_rt}
    if ms2pip_cache is None:
        ms2pip_cache = {}

    evidence = compute_all_lcms_evidence(
        candidates_df=candidates_df,
        lcms_data=lcms_data,
        ms2pip_cache=ms2pip_cache,
        deeplc_cache=deeplc_cache,
        rt_window_min=rt_window_min,
        ppm_tolerance=50.0,
    )
    return evidence[0]


class TestApexWindowFeatures:
    """F1-F3: lcms_ms1_apex_rt_delta, lcms_ms1_frac_apex_signal, lcms_ms1_n_scans_with_signal."""

    def _setup(self, apex_rt: float = 12.5, predicted_rt: float = 12.5):
        """5 MS1 scans with Gaussian-like signal peaking at apex_rt=12.5 min.

        No MS2 scans are used here, so best_charge remains 1 and lc_mz = mh_mz.
        The MS1 peak is placed at mh_mz so the apex window search finds signal.
        """
        rts = [10.0, 11.0, 12.5, 14.0, 15.0]
        signals = [10.0, 50.0, 200.0, 50.0, 10.0]
        mh_mz = 1001.5
        lcms_data = _make_lcms_data(rts, signals, ms1_peak_mz=mh_mz)
        return lcms_data, mh_mz, predicted_rt, mh_mz

    def test_apex_rt_delta_near_zero_when_predicted_at_apex(self):
        lcms_data, mh_mz, predicted_rt, _ = self._setup()
        feat = _run("PEPTIDER", mh_mz, predicted_rt, lcms_data, rt_window_min=3.0)
        assert np.isfinite(feat["lcms_ms1_apex_rt_delta"])
        assert feat["lcms_ms1_apex_rt_delta"] < 0.1

    def test_frac_apex_signal_is_one_when_anchor_is_apex(self):
        """When predicted_rt == apex RT, anchor scan IS the apex scan → ratio = 1."""
        lcms_data, mh_mz, predicted_rt, _ = self._setup()
        feat = _run("PEPTIDER", mh_mz, predicted_rt, lcms_data, rt_window_min=3.0)
        assert abs(feat["lcms_ms1_frac_apex_signal"] - 1.0) < 1e-6

    def test_frac_apex_signal_less_than_one_when_anchor_off_apex(self):
        """When predicted_rt != apex RT, anchor != apex → ratio < 1."""
        lcms_data, mh_mz, _, _ = self._setup()
        # Predict at RT=10.0 (low-signal scan) while apex is at 12.5
        feat = _run("PEPTIDER", mh_mz, 10.0, lcms_data, rt_window_min=5.0)
        assert feat["lcms_ms1_frac_apex_signal"] < 1.0

    def test_n_scans_with_signal_counts_nonzero_scans(self):
        lcms_data, mh_mz, predicted_rt, _ = self._setup()
        feat = _run("PEPTIDER", mh_mz, predicted_rt, lcms_data, rt_window_min=3.0)
        # Window ±3 min around RT=12.5 covers all 5 scans [10,11,12.5,14,15].
        # All have signal > 0 at lc_mz.
        assert feat["lcms_ms1_n_scans_with_signal"] == 5.0

    def test_apex_features_are_sentinel_when_rt_window_zero(self):
        """rt_window_min=0 disables the apex window; features remain at sentinels."""
        lcms_data, mh_mz, predicted_rt, _ = self._setup()
        feat = _run("PEPTIDER", mh_mz, predicted_rt, lcms_data, rt_window_min=0.0)
        assert np.isnan(feat["lcms_ms1_apex_rt_delta"])
        assert feat["lcms_ms1_frac_apex_signal"] == 0.0
        assert feat["lcms_ms1_n_scans_with_signal"] == 0.0

    def test_apex_features_are_sentinel_when_no_signal(self):
        """Candidate m/z with no signal in any scan → sentinel values."""
        rts = [10.0, 12.5, 15.0]
        lcms_data = _make_lcms_data(rts, [0.0, 0.0, 0.0], ms1_peak_mz=999.0)
        feat = _run("PEPTIDER", 1001.5, 12.5, lcms_data, rt_window_min=3.0)
        assert np.isnan(feat["lcms_ms1_apex_rt_delta"])
        assert feat["lcms_ms1_frac_apex_signal"] == 0.0
        assert feat["lcms_ms1_n_scans_with_signal"] == 0.0


class TestMS2RTDeltaAndFilter:
    """F4: lcms_ms2_rt_delta and the DeepLC RT filter on MS2 scans."""

    def _make_candidate_with_ms2(self, ms2_rt: float):
        mh_mz = 1001.5
        neutral_mass = mh_mz - PROTON
        rts = [11.0, 12.5, 14.0]
        signals = [50.0, 100.0, 50.0]
        lc_mz_2 = (neutral_mass + 2 * PROTON) / 2
        lcms_data = _make_lcms_data(
            rts, signals,
            ms1_peak_mz=lc_mz_2,
            ms2_rt=ms2_rt,
            ms2_precursor_mass=neutral_mass,
        )
        return lcms_data, mh_mz, neutral_mass

    def _make_ms2pip_cache(self, peptide: str, neutral_mass: float):
        charge = 2
        pred_mz = np.array([100.0, 200.0, 300.0, 400.0])
        pred_int = np.array([1.0, 0.8, 0.6, 0.4])
        return {(peptide, charge): (pred_mz, pred_int)}

    def test_ms2_rt_delta_near_zero_when_ms2_at_predicted_rt(self):
        predicted_rt = 12.5
        lcms_data, mh_mz, neutral_mass = self._make_candidate_with_ms2(ms2_rt=12.5)
        ms2pip = self._make_ms2pip_cache("PEPTIDER", neutral_mass)
        feat = _run("PEPTIDER", mh_mz, predicted_rt, lcms_data,
                    rt_window_min=3.0, ms2pip_cache=ms2pip)
        assert np.isfinite(feat["lcms_ms2_rt_delta"])
        assert feat["lcms_ms2_rt_delta"] < 0.1

    def test_ms2_rt_filter_excludes_distant_scan(self):
        """MS2 scan at RT=20.0 (far from predicted_rt=12.5) should be excluded."""
        predicted_rt = 12.5
        lcms_data, mh_mz, neutral_mass = self._make_candidate_with_ms2(ms2_rt=20.0)
        ms2pip = self._make_ms2pip_cache("PEPTIDER", neutral_mass)
        feat = _run("PEPTIDER", mh_mz, predicted_rt, lcms_data,
                    rt_window_min=3.0, ms2pip_cache=ms2pip)
        # Scan at RT=20 is >3 min away → filtered out → no match → n_matches=0
        assert feat["lcms_ms2_n_matches"] == 0.0
        assert np.isnan(feat["lcms_ms2_rt_delta"])

    def test_ms2_rt_filter_disabled_when_rt_window_zero(self):
        """When rt_window_min=0, the MS2 scan is not filtered by RT."""
        predicted_rt = 12.5
        lcms_data, mh_mz, neutral_mass = self._make_candidate_with_ms2(ms2_rt=20.0)
        ms2pip = self._make_ms2pip_cache("PEPTIDER", neutral_mass)
        feat = _run("PEPTIDER", mh_mz, predicted_rt, lcms_data,
                    rt_window_min=0.0, ms2pip_cache=ms2pip)
        # No RT filter → the MS2 scan at RT=20 is still considered
        assert feat["lcms_ms2_n_matches"] == 1.0

    def test_ms2_rt_delta_nan_when_no_ms2_match(self):
        rts = [11.0, 12.5, 14.0]
        lcms_data = _make_lcms_data(rts, [50.0, 100.0, 50.0], ms1_peak_mz=501.0)
        feat = _run("PEPTIDER", 1001.5, 12.5, lcms_data, rt_window_min=3.0)
        assert np.isnan(feat["lcms_ms2_rt_delta"])
