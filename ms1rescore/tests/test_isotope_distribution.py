"""Tests for theoretical_isotope_distribution (brainpy-backed)."""

import numpy as np
import pytest

from ms1rescore.utils import theoretical_isotope_distribution


# ---------------------------------------------------------------------------
# Reference compositions
# ---------------------------------------------------------------------------

# VHRIIR: C35 H64 N14 O7 S0, [M+H]+ = 793.5 Da
VHRIIR = (35, 64, 14, 7, 0)

# GG (Gly-Gly): C4 H8 N2 O3 S0, ~132 Da — small molecule, M0 dominant
GG = (4, 8, 2, 3, 0)

# Large peptide: C89 H140 N25 O27 S0, ~2000 Da — M+1 > M0 at this mass
LARGE = (89, 140, 25, 27, 0)

# Sulfur-containing: used to test S-34 contribution to M+2
# Same heavy-atom composition as VHRIIR except for 2 sulfur atoms
WITH_S = (35, 64, 14, 7, 2)
WITHOUT_S = (35, 64, 14, 7, 0)  # identical but S=0


# ---------------------------------------------------------------------------
# Output shape and basic properties
# ---------------------------------------------------------------------------


class TestOutputShape:
    @pytest.mark.parametrize("n", [1, 2, 3, 4, 6])
    def test_returns_n_peaks_elements(self, n):
        dist = theoretical_isotope_distribution(*VHRIIR, n_peaks=n)
        assert len(dist) == n

    def test_returns_numpy_array(self):
        dist = theoretical_isotope_distribution(*VHRIIR, n_peaks=3)
        assert isinstance(dist, np.ndarray)

    def test_all_values_nonnegative(self):
        for comp in (VHRIIR, GG, LARGE, WITH_S):
            dist = theoretical_isotope_distribution(*comp, n_peaks=4)
            assert np.all(dist >= 0), f"Negative value in distribution for {comp}"

    def test_values_bounded_by_one(self):
        for comp in (VHRIIR, GG, LARGE, WITH_S):
            dist = theoretical_isotope_distribution(*comp, n_peaks=4)
            assert np.all(dist <= 1.0), f"Value > 1 in distribution for {comp}"


# ---------------------------------------------------------------------------
# Normalization (sum-to-1 over the truncated n_peaks)
# ---------------------------------------------------------------------------


class TestNormalization:
    def test_sum_equals_one_for_truncated_peaks(self):
        # Distribution is normalised sum-to-1 over the truncated n_peaks so that
        # comparisons against sum-to-1 observed envelopes (chi², KL) are unbiased.
        dist = theoretical_isotope_distribution(*VHRIIR, n_peaks=3)
        assert dist.sum() == pytest.approx(1.0, abs=1e-6)

    def test_sum_equals_one_for_small_molecule(self):
        dist = theoretical_isotope_distribution(*GG, n_peaks=3)
        assert dist.sum() == pytest.approx(1.0, abs=1e-6)

    def test_sum_equals_one_for_many_peaks(self):
        dist = theoretical_isotope_distribution(*VHRIIR, n_peaks=6)
        assert dist.sum() == pytest.approx(1.0, abs=1e-6)

    def test_monotone_trailing_zeros_for_small_molecule(self):
        # GG has negligible M+5/M+6; requesting 8 peaks should yield zeros at the end.
        dist = theoretical_isotope_distribution(*GG, n_peaks=8)
        assert dist[6] == pytest.approx(0.0, abs=1e-4)
        assert dist[7] == pytest.approx(0.0, abs=1e-4)


# ---------------------------------------------------------------------------
# Physical properties of the distribution
# ---------------------------------------------------------------------------


class TestPhysicalProperties:
    def test_monoisotopic_dominant_for_small_peptide(self):
        # GG (~132 Da): nearly monoisotopic; M0 >> M1 >> M2.
        dist = theoretical_isotope_distribution(*GG, n_peaks=3)
        assert dist[0] > dist[1] > dist[2]

    def test_monoisotopic_dominant_for_mid_mass_peptide(self):
        # VHRIIR (792 Da): M0 still largest.
        dist = theoretical_isotope_distribution(*VHRIIR, n_peaks=3)
        assert dist[0] > dist[1]

    def test_monoisotopic_not_dominant_for_large_peptide(self):
        # At ~1600 Da the M+1 peak exceeds M0.
        dist = theoretical_isotope_distribution(*LARGE, n_peaks=3)
        assert dist[1] > dist[0], "Expected M+1 > M0 for ~1600 Da peptide"

    def test_m0_fraction_decreases_with_mass(self):
        dist_small = theoretical_isotope_distribution(*GG, n_peaks=3)
        dist_mid = theoretical_isotope_distribution(*VHRIIR, n_peaks=3)
        dist_large = theoretical_isotope_distribution(*LARGE, n_peaks=3)
        assert dist_small[0] > dist_mid[0] > dist_large[0]

    def test_m1_m2_fractions_increase_with_mass(self):
        dist_small = theoretical_isotope_distribution(*GG, n_peaks=3)
        dist_large = theoretical_isotope_distribution(*LARGE, n_peaks=3)
        assert dist_large[1] > dist_small[1]
        assert dist_large[2] > dist_small[2]

    def test_sulfur_elevates_m2_not_m1(self):
        # S-34 natural abundance (~4.25%) contributes primarily to M+2.
        # Adding 2 sulfur atoms should raise M+2 noticeably.
        dist_s = theoretical_isotope_distribution(*WITH_S, n_peaks=3)
        dist_no_s = theoretical_isotope_distribution(*WITHOUT_S, n_peaks=3)
        # M+2 should be higher with S present
        assert dist_s[2] > dist_no_s[2] + 0.03
        # M+1 is barely affected by sulfur (S-33 abundance is only ~0.75%)
        assert abs(dist_s[1] - dist_no_s[1]) < 0.02


# ---------------------------------------------------------------------------
# Known numerical values (brainpy 1.5.x, normalised sum-to-1 over n_peaks=3)
# ---------------------------------------------------------------------------


class TestKnownValues:
    def test_vhriir_distribution(self):
        # Verified empirically against brainpy 1.5.19 with sum-to-1 over 3 peaks.
        dist = theoretical_isotope_distribution(35, 64, 14, 7, 0, n_peaks=3)
        assert dist[0] == pytest.approx(0.6457, abs=0.001)  # M0
        assert dist[1] == pytest.approx(0.2839, abs=0.001)  # M+1
        assert dist[2] == pytest.approx(0.0703, abs=0.001)  # M+2

    def test_brainpy_differs_from_poisson_approximation(self):
        # The Poisson approximation gives M+1 ≈ 0.294 for VHRIIR.
        # brainpy gives ~0.280 — a meaningful difference (>1%).
        dist = theoretical_isotope_distribution(*VHRIIR, n_peaks=3)
        poisson_m1_approx = 0.294
        assert abs(dist[1] - poisson_m1_approx) > 0.005


# ---------------------------------------------------------------------------
# Caching behaviour
# ---------------------------------------------------------------------------


class TestCaching:
    def test_repeated_call_returns_equal_result(self):
        dist1 = theoretical_isotope_distribution(*VHRIIR, n_peaks=3)
        dist2 = theoretical_isotope_distribution(*VHRIIR, n_peaks=3)
        assert np.array_equal(dist1, dist2)

    def test_different_n_peaks_cached_independently(self):
        dist3 = theoretical_isotope_distribution(*VHRIIR, n_peaks=3)
        dist4 = theoretical_isotope_distribution(*VHRIIR, n_peaks=4)
        assert len(dist3) == 3
        assert len(dist4) == 4
        # Each distribution is normalised sum-to-1 over its own n_peaks, so
        # dist3 and dist4[:3] differ by a constant factor (dist4 puts some
        # mass on M+3).  The relative shape is identical.
        assert np.allclose(dist3, dist4[:3] / dist4[:3].sum())

    def test_numpy_int_and_python_int_give_same_result(self):
        nc, nh, nn, no_, ns = VHRIIR
        dist_py = theoretical_isotope_distribution(int(nc), int(nh), int(nn), int(no_), int(ns), n_peaks=3)
        dist_np = theoretical_isotope_distribution(np.int64(nc), np.int64(nh), np.int64(nn), np.int64(no_), np.int64(ns), n_peaks=3)
        assert np.allclose(dist_py, dist_np)
