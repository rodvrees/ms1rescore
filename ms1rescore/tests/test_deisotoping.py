"""Unit tests for the ms_deisotope-based _deisotope_intervals implementation."""

import numpy as np
import pytest

from ms1rescore.maldi_imzml import _deisotope_intervals, _merge_duplicate_intervals

NEUTRON = 1.003355


# ---------------------------------------------------------------------------
# _merge_duplicate_intervals
# ---------------------------------------------------------------------------

def test_merge_no_duplicates():
    ivs = [(100.0, 101.0, 100.5), (200.0, 201.0, 200.5)]
    ints = np.array([10.0, 20.0])
    merged_ivs, merged_ints = _merge_duplicate_intervals(ivs, ints, tol_da=0.001)
    assert len(merged_ivs) == 2
    assert len(merged_ints) == 2


def test_merge_duplicates_combined():
    ivs = [(100.0, 101.0, 100.5000), (100.0, 101.0, 100.5005)]
    ints = np.array([10.0, 5.0])
    merged_ivs, merged_ints = _merge_duplicate_intervals(ivs, ints, tol_da=0.001)
    assert len(merged_ivs) == 1
    assert merged_ints[0] == pytest.approx(10.0)   # max intensity kept


def test_merge_empty():
    ivs, ints = _merge_duplicate_intervals([], np.array([]), tol_da=0.001)
    assert ivs == []


# ---------------------------------------------------------------------------
# _deisotope_intervals
# ---------------------------------------------------------------------------

def _make_intervals(mz_values):
    """Build (mz-0.5, mz+0.5, mz) interval tuples for a list of apex m/z."""
    return [(mz - 0.5, mz + 0.5, mz) for mz in mz_values]


def test_empty_input():
    result = _deisotope_intervals([], np.array([]))
    assert result == []


def test_single_peak_kept():
    ivs = _make_intervals([1000.0])
    ints = np.array([100.0])
    result = _deisotope_intervals(ivs, ints)
    assert len(result) == 1
    assert result[0][2] == pytest.approx(1000.0)


def test_clear_isotope_removed():
    """M0 at 1000 Da (high intensity), M+1 at 1001.003 Da (60%), M+2 at 1002.006 Da (20%).
    M+1 and M+2 should be removed; M0 kept."""
    mzs = [1000.0, 1000.0 + NEUTRON, 1000.0 + 2 * NEUTRON]
    ints = np.array([100.0, 60.0, 20.0])
    ivs = _make_intervals(mzs)
    result = _deisotope_intervals(ivs, ints, error_ppm=50.0, min_score=5.0)
    apex_mzs = [iv[2] for iv in result]
    assert pytest.approx(1000.0, abs=0.01) in apex_mzs
    assert len(result) < 3, "At least one isotope satellite should have been removed"


def test_inverted_ratio_pair_kept():
    """M+1 10× more intense than M0 — no valid averagine fit; both kept."""
    # ms_deisotope cannot fit an envelope where M+1 >> M0 above the min_score threshold.
    mzs = [1000.0, 1000.0 + NEUTRON]
    ints = np.array([10.0, 100.0])   # M+1 10× M0 → ms_deisotope returns empty peak_set
    ivs = _make_intervals(mzs)
    result = _deisotope_intervals(ivs, ints, error_ppm=50.0, min_score=5.0)
    assert len(result) == 2, "Both peaks should be kept when M+1 >> M0"


def test_penalized_scorer():
    """PenalizedMSDeconVFitter should also remove a clear isotope satellite."""
    mzs = [1000.0, 1000.0 + NEUTRON, 1000.0 + 2 * NEUTRON]
    ints = np.array([100.0, 60.0, 20.0])
    ivs = _make_intervals(mzs)
    result = _deisotope_intervals(
        ivs, ints,
        scorer="PenalizedMSDeconVFitter",
        error_ppm=50.0,
        min_score=5.0,
    )
    assert len(result) < 3


def test_return_diagnostics_structure():
    """return_diagnostics=True should return (intervals, list[dict]) with correct keys."""
    mzs = [1000.0, 1000.0 + NEUTRON]
    ints = np.array([100.0, 60.0])
    ivs = _make_intervals(mzs)
    result = _deisotope_intervals(ivs, ints, return_diagnostics=True, error_ppm=50.0, min_score=5.0)
    assert isinstance(result, tuple) and len(result) == 2
    kept, diags = result
    assert len(diags) == len(ivs)
    for d in diags:
        assert "apex_mz" in d
        assert "removed" in d
        assert "score" in d
        assert "charge" in d


def test_invalid_scorer_raises():
    ivs = _make_intervals([1000.0])
    ints = np.array([100.0])
    with pytest.raises(ValueError, match="Unknown deisotope scorer"):
        _deisotope_intervals(ivs, ints, scorer="BogusScorer")


def test_invalid_averagine_raises():
    ivs = _make_intervals([1000.0])
    ints = np.array([100.0])
    with pytest.raises(ValueError, match="Unknown averagine model"):
        _deisotope_intervals(ivs, ints, averagine="bogus_model")


def test_secondary_ratio_pass_removes_m1_when_m0_more_intense():
    """Secondary ratio pass: M+1 removed when M0 > M+1 but ms_deisotope declined to form envelope.

    This mimics the mean-spectrum heterogeneity case where M0/M+1 ≈ 1.05 (observed)
    vs ~1.4 (averagine expected), causing ms_deisotope to pick M+1 as M0 of a better
    downstream triplet while leaving M0 and M+1 both in the kept set.
    """
    # Simplified reproduction of the GT 1045.56 case:
    # ms_deisotope prefers 1001.003 as M0 of (1001.003, 1002.006, 1003.009) triplet,
    # leaving 1000.0 and 1001.003 both in kept. Secondary pass: M0=1000 > M+1=1001.003
    # → remove M+1.
    mzs = [1000.0, 1000.0 + NEUTRON, 1000.0 + 2 * NEUTRON, 1000.0 + 3 * NEUTRON]
    ints = np.array([100.0, 95.0, 80.0, 60.0])  # M0/M+1 = 1.05, close to 1:1
    ivs = _make_intervals(mzs)
    result = _deisotope_intervals(ivs, ints, error_ppm=25.0, min_score=2.0)
    # Either ms_deisotope or secondary pass must remove 1001.003 (M+1 of 1000.0)
    apex_mzs = [iv[2] for iv in result]
    assert pytest.approx(1000.0, abs=0.01) in apex_mzs, "M0 must be kept"
    assert not any(abs(mz - (1000.0 + NEUTRON)) < 0.01 for mz in apex_mzs), \
        "M+1 must be removed (either by ms_deisotope or secondary ratio pass)"


def test_phantom_m1_prevents_m2_removal():
    """M0 and M+2 present but no M+1 — M+2 should NOT be removed (phantom M+1 guard)."""
    # 1169.58 and 1171.57 are ~2 Da apart, mimicking a GT false-positive found in
    # PXD056528 data where ms_deisotope fit a phantom M+1 and removed the M+2 peak.
    mzs = [1169.58, 1169.58 + 2 * NEUTRON]   # no M+1 in between
    ints = np.array([100.0, 70.0])
    ivs = _make_intervals(mzs)
    result = _deisotope_intervals(ivs, ints, error_ppm=25.0, min_score=5.0)
    assert len(result) == 2, "M+2 must not be removed when M+1 is absent from the peak list"


def test_unrelated_peaks_one_da_apart_kept():
    """Two peaks ~1 Da apart but with M+1 more intense than expected → both kept."""
    # At 1648 Da, averagine M0/M+1 ≈ 1.2. If obs M0/M+1 < 1 → bad fit → kept.
    mzs = [1648.0, 1648.0 + NEUTRON]
    ints = np.array([10.0, 18.0])   # M+1 more intense than M0 — physically inconsistent
    ivs = _make_intervals(mzs)
    result = _deisotope_intervals(ivs, ints, error_ppm=50.0, min_score=5.0)
    assert len(result) == 2, "Both peaks should be kept when M+1 > M0"
