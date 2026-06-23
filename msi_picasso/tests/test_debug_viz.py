"""Tests for debug_viz helpers (FDR-coloured ion-image frames)."""

import numpy as np

from msi_picasso.debug_viz import _fdr_frame_color


class TestFdrFrameColor:
    def test_dark_green_at_or_below_1pct(self):
        assert _fdr_frame_color(0.0) == "#006400"
        assert _fdr_frame_color(0.01) == "#006400"
        assert _fdr_frame_color(0.005) == "#006400"

    def test_light_green_between_1_and_5pct(self):
        assert _fdr_frame_color(0.0101) == "#90EE90"
        assert _fdr_frame_color(0.05) == "#90EE90"

    def test_white_above_5pct(self):
        assert _fdr_frame_color(0.0501) == "white"
        assert _fdr_frame_color(0.5) == "white"

    def test_white_for_nan_or_missing(self):
        assert _fdr_frame_color(float("nan")) == "white"
        assert _fdr_frame_color(None) == "white"
