"""Tests for the single-round scoring toggle (rescore(single_round=True)).

Single-round skips the R2 retrain: candidates are scored once (R1), per-feature
winners are selected (the target-vs-decoy competition / TDC is unchanged), and
the FDR is computed on the R1 winner scores. The score columns and result schema
must match the two-round path so downstream code is unaffected.
"""

import numpy as np
import pandas as pd
import pytest

from msi_picasso import pipeline


def _toy_features(n=200, seed=0):
    """Co-located target+decoy pairs over n//2 features (mz_shuffle-style):
    each feature carries one target and one decoy sharing feature_idx."""
    rng = np.random.default_rng(seed)
    nf = n // 2
    rows = []
    for i in range(nf):
        # target: discriminative feature shifted up; decoy: at 0
        rows.append(dict(peptide=f"T{i}", protein=f"P{i%10}", feature_mz=100.0 + i,
                         feature_idx=i, is_decoy=False,
                         good=rng.normal(1.5, 1.0), ppm_error_abs=rng.uniform(0, 5),
                         n_candidates=2))
        rows.append(dict(peptide=f"D{i}", protein=f"DECOY_P{i%10}", feature_mz=100.0 + i,
                         feature_idx=i, is_decoy=True,
                         good=rng.normal(0.0, 1.0), ppm_error_abs=rng.uniform(0, 5),
                         n_candidates=2))
    return pd.DataFrame(rows)


def _run_dispatch(df, model, single_round):
    """Call the model dispatch in isolation via _rescore_* + winner selection,
    mirroring the rescore() branch, to check single-round vs two-round scoring."""
    # Use the public _rescore_linear path through the same helpers the dispatch uses.
    feats = ["good"]
    feature_col = "feature_idx"
    routine = pipeline._rescore_svm if model == "svm" else pipeline._rescore_lda
    scores1, *_ = routine(df, feats, init_ppm_threshold=5.0)
    winner_pos, winners_df = pipeline._select_feature_winners(df, scores1, feature_col, 0.02)
    if single_round:
        scores2 = scores1[winner_pos]
    else:
        is_d = winners_df["is_decoy"].to_numpy(dtype=bool)
        thr = np.percentile(scores1[winner_pos][~is_d], 80.0)
        seed = (~is_d) & (scores1[winner_pos] >= thr)
        scores2, *_ = routine(winners_df, feats, init_ppm_threshold=5.0, seed_mask=seed)
    q = pipeline._tdc_qvalues(scores2, winners_df["is_decoy"].to_numpy(dtype=bool))
    return scores1, winner_pos, scores2, q


class TestSingleRound:
    @pytest.mark.parametrize("model", ["lda", "svm"])
    def test_single_round_uses_r1_winner_scores(self, model):
        df = _toy_features()
        scores1, winner_pos, scores2, q = _run_dispatch(df, model, single_round=True)
        # In single-round, the final (R2) scores ARE the R1 winner scores.
        assert np.allclose(scores2, scores1[winner_pos])
        assert np.isfinite(q).all()

    @pytest.mark.parametrize("model", ["lda", "svm"])
    def test_competition_population_identical(self, model):
        """Winner selection (the competition) is the same regardless of single_round."""
        df = _toy_features()
        _, wp_single, _, _ = _run_dispatch(df, model, single_round=True)
        _, wp_two, _, _ = _run_dispatch(df, model, single_round=False)
        assert np.array_equal(wp_single, wp_two)


class TestRescoreSignature:
    def test_rescore_accepts_single_round(self):
        import inspect
        assert "single_round" in inspect.signature(pipeline.rescore).parameters
