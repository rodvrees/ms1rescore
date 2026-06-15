"""Tests for the spatial ranker feature gating (Phase 3)."""

import warnings

import pytest

from msi_picasso.feature_generator import (
    PROTEIN_LEVEL_FEATURES,
    SPATIAL_RANKER_FEATURES,
)
from msi_picasso.pipeline import _resolve_spatial_ranker_features


class TestSpatialRankerGate:

    @pytest.mark.parametrize("decoy_method", ["entrapment", "mz_shift"])
    def test_permitted_decoys_keep_flag(self, decoy_method):
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # no warning expected
            assert _resolve_spatial_ranker_features(True, decoy_method) is True

    @pytest.mark.parametrize(
        "decoy_method", ["shuffle", "balanced_shuffle", "paired_shuffle"]
    )
    def test_blocked_decoys_disable_and_warn(self, decoy_method):
        with pytest.warns(UserWarning, match="use-spatial-ranker-features"):
            assert _resolve_spatial_ranker_features(True, decoy_method) is False

    def test_flag_off_is_noop(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert _resolve_spatial_ranker_features(False, "shuffle") is False
            assert _resolve_spatial_ranker_features(False, "entrapment") is False


class TestSpatialRankerFeatureList:

    def test_contains_expected_features(self):
        assert "spatial_morans_i" in SPATIAL_RANKER_FEATURES
        assert "spatial_gearys_c" in SPATIAL_RANKER_FEATURES
        assert "fraction_detected" in SPATIAL_RANKER_FEATURES
        assert "intensity_cv" in SPATIAL_RANKER_FEATURES

    def test_protein_coloc_overlaps_protein_level(self):
        """The protein_colocalization_* members overlap PROTEIN_LEVEL_FEATURES;
        the pipeline dedups them when both flags are active."""
        overlap = set(SPATIAL_RANKER_FEATURES) & set(PROTEIN_LEVEL_FEATURES)
        assert overlap == {
            "protein_colocalization",
            "protein_colocalization_max",
            "protein_colocalization_median",
            "protein_colocalization_n_partners",
        }

    def test_dedup_when_both_flags_active(self):
        """Emulate the pipeline pool assembly: no duplicates when both lists merge."""
        base = ["ppm_error_abs", "theo_isotope_cosine"]
        pool = list(base) + PROTEIN_LEVEL_FEATURES + SPATIAL_RANKER_FEATURES
        seen: set[str] = set()
        deduped = [f for f in pool if not (f in seen or seen.add(f))]
        assert len(deduped) == len(set(deduped))
        # All spatial ranker features present exactly once.
        for f in SPATIAL_RANKER_FEATURES:
            assert deduped.count(f) == 1
