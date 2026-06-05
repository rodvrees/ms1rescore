"""Tests for _select_calibration_peptides() (DeepLC/IM2Deep calibration anchors)."""

import numpy as np
import pandas as pd

from ms1rescore.pipeline import _select_calibration_peptides


def _make_candidates(n_target=100, n_decoy=100, seed=0):
    rng = np.random.default_rng(seed)
    # Targets: spread of ppm error and isotope cosine.
    tgt = pd.DataFrame({
        "peptide": [f"TGTPEP{i}" for i in range(n_target)],
        "is_decoy": False,
        "ppm_error_abs": rng.uniform(0, 20, n_target),
        "theo_isotope_cosine": rng.uniform(0, 1, n_target),
    })
    dec = pd.DataFrame({
        "peptide": [f"DECPEP{i}" for i in range(n_decoy)],
        "is_decoy": True,
        "ppm_error_abs": rng.uniform(0, 20, n_decoy),
        "theo_isotope_cosine": rng.uniform(0, 1, n_decoy),
    })
    return pd.concat([tgt, dec], ignore_index=True)


def test_decoys_never_selected():
    cand = _make_candidates()
    keep = _select_calibration_peptides(cand, percentile=0.5)
    assert not cand.loc[keep, "is_decoy"].any()


def test_selects_top_percentile_of_targets():
    cand = _make_candidates(n_target=100, n_decoy=100)
    keep = _select_calibration_peptides(cand, percentile=0.10)
    n_sel = int(keep.sum())
    # ~10% of 100 targets, allowing for ties at the quantile boundary.
    assert 8 <= n_sel <= 15


def test_prefers_low_ppm_high_isotope():
    # One clearly best target (ppm 0, cosine 1) and one clearly worst.
    cand = pd.DataFrame({
        "peptide": ["BEST", "WORST", "MID1", "MID2", "MID3"],
        "is_decoy": [False] * 5,
        "ppm_error_abs": [0.0, 19.0, 10.0, 8.0, 12.0],
        "theo_isotope_cosine": [0.99, 0.05, 0.5, 0.6, 0.4],
    })
    keep = _select_calibration_peptides(cand, percentile=0.20)  # top 1 of 5
    assert keep[cand["peptide"] == "BEST"].all()
    assert not keep[cand["peptide"] == "WORST"].any()


def test_percentile_zero_selects_nothing():
    cand = _make_candidates()
    keep = _select_calibration_peptides(cand, percentile=0.0)
    assert keep.sum() == 0


def test_falls_back_to_ppm_when_isotope_constant():
    # theo_isotope_cosine all zero (e.g. no maldi_envelopes) → ranking by ppm only.
    cand = pd.DataFrame({
        "peptide": ["A", "B", "C", "D"],
        "is_decoy": [False] * 4,
        "ppm_error_abs": [1.0, 2.0, 18.0, 19.0],
        "theo_isotope_cosine": [0.0, 0.0, 0.0, 0.0],
    })
    keep = _select_calibration_peptides(cand, percentile=0.50)  # top 2 of 4
    assert keep[cand["peptide"].isin(["A", "B"])].all()
    assert not keep[cand["peptide"].isin(["C", "D"])].any()


def test_missing_isotope_column_uses_ppm():
    cand = pd.DataFrame({
        "peptide": ["A", "B", "C", "D"],
        "is_decoy": [False] * 4,
        "ppm_error_abs": [1.0, 2.0, 18.0, 19.0],
    })
    keep = _select_calibration_peptides(cand, percentile=0.50)
    assert keep[cand["peptide"].isin(["A", "B"])].all()


def test_no_targets_returns_all_false():
    cand = pd.DataFrame({
        "peptide": ["D1", "D2"],
        "is_decoy": [True, True],
        "ppm_error_abs": [1.0, 2.0],
        "theo_isotope_cosine": [0.9, 0.8],
    })
    keep = _select_calibration_peptides(cand, percentile=0.5)
    assert keep.sum() == 0


def test_decoy_count_does_not_shrink_target_selection():
    """The selection size depends only on the target count, not on how many
    decoys are paired (the core fix vs the old n_candidates == 1 heuristic)."""
    base = _make_candidates(n_target=100, n_decoy=10, seed=1)
    many_decoys = _make_candidates(n_target=100, n_decoy=400, seed=1)
    k1 = int(_select_calibration_peptides(base, percentile=0.10).sum())
    k2 = int(_select_calibration_peptides(many_decoys, percentile=0.10).sum())
    assert k1 == k2
