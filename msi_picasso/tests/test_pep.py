"""Tests for estimate_pep and _pep_qvalues."""
import numpy as np
import pytest

from msi_picasso.pipeline import _pep_qvalues, estimate_pep


def _make_bimodal(n_target: int = 300, n_decoy: int = 150, seed: int = 42):
    rng = np.random.default_rng(seed)
    target_scores = rng.normal(2.0, 0.6, n_target)
    decoy_scores = rng.normal(0.0, 0.6, n_decoy)
    scores = np.concatenate([target_scores, decoy_scores])
    is_decoy = np.array([False] * n_target + [True] * n_decoy)
    return scores, is_decoy


def _make_skewed(n_target: int = 300, n_decoy: int = 150, seed: int = 7):
    """Heavy-tailed / skewed score distributions (representative of QDA output)."""
    rng = np.random.default_rng(seed)
    # Log-normal targets (right-skewed); exponential decoys
    target_scores = rng.lognormal(mean=1.0, sigma=0.8, size=n_target)
    decoy_scores = rng.exponential(scale=0.5, size=n_decoy)
    scores = np.concatenate([target_scores, decoy_scores])
    is_decoy = np.array([False] * n_target + [True] * n_decoy)
    return scores, is_decoy


class TestEstimatePep:
    def test_pep_in_unit_interval(self):
        scores, is_decoy = _make_bimodal()
        pep = estimate_pep(scores, is_decoy)
        assert np.all((pep >= 0.0) & (pep <= 1.0))

    def test_pep_monotonically_decreasing_with_score(self):
        scores, is_decoy = _make_bimodal()
        pep = estimate_pep(scores, is_decoy)
        sort_idx = np.argsort(scores)
        pep_sorted = pep[sort_idx]
        # Allow small numerical tolerance; the Gaussian ratio is smooth and
        # monotone when mu1 > mu0, so differences should be non-positive.
        diffs = np.diff(pep_sorted.astype(float))
        assert np.all(diffs <= 1e-9), (
            f"PEP not monotonically decreasing: max positive diff = {diffs.max():.3e}"
        )

    def test_high_scoring_targets_have_low_pep(self):
        scores, is_decoy = _make_bimodal()
        pep = estimate_pep(scores, is_decoy)
        target_scores = scores[~is_decoy]
        high_threshold = np.percentile(target_scores, 90)
        high_target_mask = (~is_decoy) & (scores >= high_threshold)
        assert np.all(pep[high_target_mask] < 0.1), (
            f"High-scoring targets should have PEP < 0.1; "
            f"got max PEP = {pep[high_target_mask].max():.3f}"
        )

    def test_decoys_have_high_pep(self):
        scores, is_decoy = _make_bimodal()
        pep = estimate_pep(scores, is_decoy)
        # Decoys that score well below zero (deep null) should have PEP near 1
        low_decoys = is_decoy & (scores < -0.5)
        if low_decoys.sum() > 0:
            assert np.all(pep[low_decoys] > 0.5)

    def test_returns_nan_with_too_few_decoys(self):
        scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        is_decoy = np.array([False, False, False, False, True])  # only 1 decoy
        pep = estimate_pep(scores, is_decoy)
        assert np.all(np.isnan(pep))

    def test_returns_nan_with_no_targets(self):
        scores = np.array([1.0, 2.0, 3.0])
        is_decoy = np.array([True, True, True])
        pep = estimate_pep(scores, is_decoy)
        assert np.all(np.isnan(pep))

    def test_output_length_matches_input(self):
        scores, is_decoy = _make_bimodal(n_target=100, n_decoy=50)
        pep = estimate_pep(scores, is_decoy)
        assert len(pep) == len(scores)


class TestEstimatePepKDE:
    """Tests for method='kde' path used by the QDA backend."""

    def test_kde_pep_in_unit_interval(self):
        scores, is_decoy = _make_skewed()
        pep = estimate_pep(scores, is_decoy, method="kde")
        assert np.all((pep >= 0.0) & (pep <= 1.0))

    def test_kde_output_length_matches_input(self):
        scores, is_decoy = _make_skewed()
        pep = estimate_pep(scores, is_decoy, method="kde")
        assert len(pep) == len(scores)

    def test_kde_high_scoring_targets_have_low_pep(self):
        scores, is_decoy = _make_skewed()
        pep = estimate_pep(scores, is_decoy, method="kde")
        target_scores = scores[~is_decoy]
        high_threshold = np.percentile(target_scores, 90)
        high_target_mask = (~is_decoy) & (scores >= high_threshold)
        assert np.all(pep[high_target_mask] < 0.3), (
            f"High-scoring targets should have low PEP; "
            f"got max PEP = {pep[high_target_mask].max():.3f}"
        )

    def test_kde_returns_nan_with_too_few_decoys(self):
        scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        is_decoy = np.array([False, False, False, False, True])
        pep = estimate_pep(scores, is_decoy, method="kde")
        assert np.all(np.isnan(pep))

    def test_kde_gaussian_agree_on_gaussian_data(self):
        """KDE and Gaussian methods should produce similar PEP on well-separated Gaussian data."""
        scores, is_decoy = _make_bimodal(n_target=500, n_decoy=250)
        pep_g = estimate_pep(scores, is_decoy, method="gaussian")
        pep_k = estimate_pep(scores, is_decoy, method="kde")
        # Ranking correlation: Spearman rank order should be very high
        from scipy.stats import spearmanr
        rho, _ = spearmanr(pep_g, pep_k)
        assert rho > 0.97, f"KDE and Gaussian PEP ranks diverge: Spearman rho = {rho:.4f}"


class TestPepQvalues:
    def test_qvalues_monotone_nondecreasing(self):
        # PEP sorted ascending → cumulative mean is non-decreasing
        scores, is_decoy = _make_bimodal()
        pep = estimate_pep(scores, is_decoy)
        q = _pep_qvalues(pep)
        finite_q = q[np.isfinite(q)]
        sort_idx = np.argsort(pep[np.isfinite(pep)])
        q_sorted = finite_q[sort_idx]
        assert np.all(np.diff(q_sorted) >= -1e-12)

    def test_qvalues_in_unit_interval(self):
        scores, is_decoy = _make_bimodal()
        pep = estimate_pep(scores, is_decoy)
        q = _pep_qvalues(pep)
        finite_q = q[np.isfinite(q)]
        assert np.all((finite_q >= 0.0) & (finite_q <= 1.0))

    def test_nan_propagated_for_nan_pep(self):
        pep = np.array([0.1, np.nan, 0.3, 0.05])
        q = _pep_qvalues(pep)
        assert np.isnan(q[1])
        assert np.all(np.isfinite(q[[0, 2, 3]]))

    def test_all_nan_input(self):
        pep = np.full(5, np.nan)
        q = _pep_qvalues(pep)
        assert np.all(np.isnan(q))
