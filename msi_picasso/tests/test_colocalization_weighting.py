"""Tests for intensity-weighted and rank-weighted (top-k) colocalization.

compute_colocalization_features adds, alongside the plain mean/max/median:
- protein_colocalization_weighted      = Σ(w·r) / Σw,  w = sqrt(I_a·I_b)
- protein_colocalization_weighted_max  = max(w·r)
- protein_colocalization_top{2,3,5}    = mean r over the k highest-weight pairs
All blind to is_decoy. These tests use a hand-built correlation matrix so the
expected values are known exactly.
"""

import numpy as np
import pandas as pd

from msi_picasso import maldi_features
from msi_picasso.maldi_features import compute_colocalization_features


def _run(df, corr, mzs):
    """Drive compute_colocalization_features with an injected corr-cache."""
    mz_to_idx = {float(m): i for i, m in enumerate(mzs)}
    cache = (np.asarray(corr, dtype=np.float32), np.asarray(mzs, dtype=float), mz_to_idx)
    # ion_images unused when _corr_cache is supplied
    return compute_colocalization_features(df, None, None, _corr_cache=cache)


def _frame(intensities):
    # one protein "A" with 3 features at mz 100/200/300
    return pd.DataFrame({
        "feature_mz": [100.0, 200.0, 300.0],
        "protein": ["A", "A", "A"],
        "peptide": ["P1", "P2", "P3"],
        "is_decoy": [False, False, False],
        "feature_intensity_p90": intensities,
    })


class TestWeightedAggregation:
    def test_uniform_intensity_weighted_equals_mean(self):
        df = _frame([1.0, 1.0, 1.0])
        corr = [[1.0, 0.2, 0.8],
                [0.2, 1.0, 0.4],
                [0.8, 0.4, 1.0]]
        out = _run(df, corr, [100.0, 200.0, 300.0])
        assert np.allclose(out["protein_colocalization_weighted"],
                           out["protein_colocalization"])

    def test_weighted_mean_matches_formula(self):
        # feature 100: partners 200 (r=0.2, I=4) and 300 (r=0.8, I=9); I_100=1
        df = _frame([1.0, 4.0, 9.0])
        corr = [[1.0, 0.2, 0.8],
                [0.2, 1.0, 0.4],
                [0.8, 0.4, 1.0]]
        out = _run(df, corr, [100.0, 200.0, 300.0]).set_index("feature_mz")
        w12 = np.sqrt(1.0 * 4.0); w13 = np.sqrt(1.0 * 9.0)
        expected = (w12 * 0.2 + w13 * 0.8) / (w12 + w13)
        assert np.isclose(out.loc[100.0, "protein_colocalization_weighted"], expected, atol=1e-5)

    def test_weighted_max_is_max_of_wr(self):
        df = _frame([1.0, 4.0, 9.0])
        corr = [[1.0, 0.2, 0.8],
                [0.2, 1.0, 0.4],
                [0.8, 0.4, 1.0]]
        out = _run(df, corr, [100.0, 200.0, 300.0]).set_index("feature_mz")
        w12 = np.sqrt(1.0 * 4.0) * 0.2; w13 = np.sqrt(1.0 * 9.0) * 0.8
        assert np.isclose(out.loc[100.0, "protein_colocalization_weighted_max"], max(w12, w13), atol=1e-5)


class TestTopK:
    def test_top2_uses_two_highest_weight_pairs(self):
        # 4 features so feature 100 has 3 partners with distinct weights
        df = pd.DataFrame({
            "feature_mz": [100.0, 200.0, 300.0, 400.0],
            "protein": ["A"] * 4,
            "peptide": ["P1", "P2", "P3", "P4"],
            "is_decoy": [False] * 4,
            "feature_intensity_p90": [1.0, 1.0, 4.0, 9.0],  # partners of 100: w∝sqrt(I)
        })
        corr = np.array([
            [1.0, 0.1, 0.5, 0.9],
            [0.1, 1.0, 0.3, 0.3],
            [0.5, 0.3, 1.0, 0.3],
            [0.9, 0.3, 0.3, 1.0],
        ])
        out = _run(df, corr, [100.0, 200.0, 300.0, 400.0]).set_index("feature_mz")
        # feature 100 partners by weight desc: 400 (I=9, r=0.9), 300 (I=4, r=0.5), 200 (I=1, r=0.1)
        assert np.isclose(out.loc[100.0, "protein_colocalization_top2"], (0.9 + 0.5) / 2, atol=1e-5)
        assert np.isclose(out.loc[100.0, "protein_colocalization_top3"], (0.9 + 0.5 + 0.1) / 3, atol=1e-5)

    def test_topk_uses_all_when_fewer_than_k(self):
        # only 2 partners but k=5 requested → mean over the 2 available
        df = _frame([1.0, 1.0, 1.0])
        corr = [[1.0, 0.2, 0.8],
                [0.2, 1.0, 0.4],
                [0.8, 0.4, 1.0]]
        out = _run(df, corr, [100.0, 200.0, 300.0]).set_index("feature_mz")
        assert np.isclose(out.loc[100.0, "protein_colocalization_top5"], (0.2 + 0.8) / 2, atol=1e-5)


class TestUndefinedColocalization:
    """Candidates with no within-protein partner get NaN coloc summaries (so the
    ranker median-imputes them) + has_coloc=0, not a 0.0 floor."""

    def test_singleton_protein_has_nan_summaries_and_zero_indicator(self):
        # protein "A" has partners (2 features); "B" is a singleton (no partner)
        df = pd.DataFrame({
            "feature_mz": [100.0, 200.0, 300.0],
            "protein": ["A", "A", "B"],
            "peptide": ["P1", "P2", "P3"],
            "is_decoy": [False, False, False],
            "feature_intensity_p90": [1.0, 1.0, 1.0],
        })
        corr = [[1.0, 0.5, 0.0],
                [0.5, 1.0, 0.0],
                [0.0, 0.0, 1.0]]
        out = _run(df, corr, [100.0, 200.0, 300.0]).set_index("feature_mz")
        # singleton B (mz 300): all r-summaries NaN, indicator 0, count 0
        r_cols = [c for c in maldi_features._COLOC_FEATURE_COLS
                  if c != "protein_colocalization_n_partners"]
        for c in r_cols:
            assert np.isnan(out.loc[300.0, c]), c
        assert out.loc[300.0, "protein_colocalization_n_partners"] == 0.0
        assert out.loc[300.0, "has_coloc"] == 0.0
        # colocalizable A: defined summaries + indicator 1
        assert not np.isnan(out.loc[100.0, "protein_colocalization"])
        assert out.loc[100.0, "has_coloc"] == 1.0

    def test_no_pairs_at_all(self):
        # every protein is a singleton -> no within-protein pairs
        df = pd.DataFrame({
            "feature_mz": [100.0, 200.0],
            "protein": ["A", "B"],
            "peptide": ["P1", "P2"],
            "is_decoy": [False, False],
            "feature_intensity_p90": [1.0, 1.0],
        })
        corr = [[1.0, 0.3], [0.3, 1.0]]
        out = _run(df, corr, [100.0, 200.0])
        assert out["has_coloc"].eq(0.0).all()
        assert out["protein_colocalization"].isna().all()
        assert out["protein_colocalization_n_partners"].eq(0.0).all()


class TestSymmetry:
    def test_blind_to_is_decoy(self):
        corr = [[1.0, 0.2, 0.8],
                [0.2, 1.0, 0.4],
                [0.8, 0.4, 1.0]]
        df_t = _frame([1.0, 4.0, 9.0])
        df_d = _frame([1.0, 4.0, 9.0]); df_d["is_decoy"] = True
        out_t = _run(df_t, corr, [100.0, 200.0, 300.0])
        out_d = _run(df_d, corr, [100.0, 200.0, 300.0])
        for c in maldi_features._COLOC_FEATURE_COLS:
            assert np.allclose(out_t[c], out_d[c])
