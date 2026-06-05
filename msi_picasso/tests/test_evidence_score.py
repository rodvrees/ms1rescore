"""Unit tests for evidence score computation."""

import numpy as np
import pytest
from psm_utils import PSM, PSMList

from ms2rescore.evidence_score import EvidenceBreakdown, EvidenceScorer


@pytest.fixture
def sample_psms():
    return PSMList(psm_list=[
        PSM(peptidoform="TGAQELLR/1", spectrum_id="s1", run="test", precursor_mz=887.5, score=10.0),
        PSM(peptidoform="ASGPPVSELITK/1", spectrum_id="s2", run="test", precursor_mz=1198.7, score=20.0),
    ])


class TestEvidenceScorer:
    def test_weights_must_sum_to_one(self):
        with pytest.raises(ValueError):
            EvidenceScorer(ccs_weight=0.5, ms2_weight=0.3)

    def test_basic_scoring(self, sample_psms):
        scorer = EvidenceScorer(ccs_error_mean=0, ccs_error_std=10)
        breakdowns = scorer.score_psms(
            sample_psms,
            ccs_observed=np.array([300.0, 330.0]),
            ccs_predicted=np.array([300.0, 330.0]),  # Perfect match
        )
        assert len(breakdowns) == 2
        # Perfect CCS match should give high score
        assert breakdowns[0].ccs_score == pytest.approx(1.0)
        assert breakdowns[0].composite_score == pytest.approx(1.0)

    def test_with_spectral_angles(self, sample_psms):
        scorer = EvidenceScorer(ccs_weight=0.6, ms2_weight=0.4, ccs_error_mean=0, ccs_error_std=10)
        breakdowns = scorer.score_psms(
            sample_psms,
            ccs_observed=np.array([300.0, 330.0]),
            ccs_predicted=np.array([300.0, 330.0]),
            spectral_angles=np.array([0.9, 0.8]),
        )
        # Composite = 0.6 * 1.0 + 0.4 * 0.9 = 0.96
        assert breakdowns[0].composite_score == pytest.approx(0.96)
        assert breakdowns[0].ms2_score == pytest.approx(0.9)

    def test_large_ccs_error_gives_low_score(self, sample_psms):
        scorer = EvidenceScorer(ccs_error_mean=0, ccs_error_std=5)
        breakdowns = scorer.score_psms(
            sample_psms,
            ccs_observed=np.array([300.0, 330.0]),
            ccs_predicted=np.array([320.0, 330.0]),  # 20 Å² error on first
        )
        assert breakdowns[0].ccs_score < 0.1  # z = 4, should be very low
        assert breakdowns[1].ccs_score == pytest.approx(1.0)

    def test_breakdowns_to_dataframe(self, sample_psms):
        scorer = EvidenceScorer(ccs_error_mean=0, ccs_error_std=10)
        breakdowns = scorer.score_psms(
            sample_psms,
            ccs_observed=np.array([300.0, 330.0]),
            ccs_predicted=np.array([305.0, 325.0]),
        )
        df = EvidenceScorer.breakdowns_to_dataframe(breakdowns)
        assert len(df) == 2
        assert "composite_score" in df.columns
        assert "ccs_z_score" in df.columns

    def test_format_report(self, sample_psms):
        scorer = EvidenceScorer(ccs_error_mean=0, ccs_error_std=10)
        breakdowns = scorer.score_psms(
            sample_psms,
            ccs_observed=np.array([300.0, 330.0]),
            ccs_predicted=np.array([305.0, 325.0]),
        )
        report = EvidenceScorer.format_report(breakdowns)
        assert "Evidence Score Report" in report
        assert "TGAQELLR" in report
