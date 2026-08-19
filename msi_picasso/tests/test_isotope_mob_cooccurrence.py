"""Tests for the per-pixel isotope-envelope mobility co-occurrence features.

The measurement asks "does this candidate's M+1/M+2 appear at M0's mobility AND in
M0's pixels?" rather than "do two tissue-summed windows have the same mean mobility?"
(which is blind — every window returns the acquisition's global peptide corridor).
These tests pin the symmetry invariant, the band/pixel gating, and the bounds.
"""

import inspect

import numpy as np
import pandas as pd
import pytest

from msi_picasso.maldi_query import _envelope_mobility_cooccurrence
from msi_picasso.maldi_features import compute_isotope_mobility_cooccurrence_features
from msi_picasso.pipeline import _observed_envelope_cooc_by_feature_idx
from msi_picasso.utils import NEUTRON


def _df(feature_idx, is_decoy=None, feature_mz=None):
    n = len(feature_idx)
    return pd.DataFrame({
        "feature_idx": feature_idx,
        "feature_mz": feature_mz if feature_mz is not None else np.arange(n, dtype=float) + 800.0,
        "peptide": [f"PEPTIDE{i}" for i in range(n)],
        "is_decoy": is_decoy if is_decoy is not None else [False] * n,
    })


def _peaks(spec):
    """Build (mz, intensity, mobility, frame) arrays from (mz, int, mob, frame) tuples."""
    a = np.asarray(spec, dtype=float)
    return a[:, 0], a[:, 1], a[:, 2], a[:, 3].astype(np.int64)


class TestCooccurrenceExtraction:
    """_envelope_mobility_cooccurrence — enrichment over the independence null.

    The reported value is log2( P(band & pixel) / (P(band) * P(pixel)) ) within the
    M+k window: 0 = mobility and space are independent (diffuse background), > 0 =
    they co-occur the way a real isotopologue of M0 does.
    """

    def test_pure_background_scores_chance(self):
        """Band- and pixel-membership independent within the window -> log2 lift 0.

        Four equal peaks forming a 2x2 in/out design: P(band)=P(pixel)=0.5 and
        P(band & pixel)=0.25 = 0.5*0.5, exactly the null.
        """
        q = np.array([1000.0])
        mz, i, mob, fr = _peaks([
            (1000.0, 100.0, 0.90, 5),            # M0 defines band 0.90, pixel 5
            (1000.0 + NEUTRON, 25.0, 0.90, 5),   # in band,  in pixel
            (1000.0 + NEUTRON, 25.0, 0.90, 9),   # in band,  out pixel
            (1000.0 + NEUTRON, 25.0, 1.40, 5),   # out band, in pixel
            (1000.0 + NEUTRON, 25.0, 1.40, 9),   # out band, out pixel
        ])
        out = _envelope_mobility_cooccurrence(
            mz, i, mob, fr, q, ppm=25.0, k0_tol=0.02, n_frames=10, pixel_frac=0.0
        )
        assert out["cooc_m1"][0] == pytest.approx(0.0, abs=1e-12)

    def test_coherent_isotopologue_is_enriched(self):
        """A real isotopologue on top of diffuse background -> clear positive lift.

        M+1 = 30 units concentrated in M0's band AND pixel, plus 70 units of
        background spread independently at marginals 0.2/0.2.  Joint 0.328 against
        a null of 0.44*0.44 = 0.194 -> lift 1.69, log2 0.76.
        """
        q = np.array([1000.0])
        mz, i, mob, fr = _peaks([
            (1000.0, 100.0, 0.90, 5),
            (1000.0 + NEUTRON, 32.8, 0.90, 5),   # isotopologue + in/in background
            (1000.0 + NEUTRON, 11.2, 0.90, 9),   # in band,  out pixel
            (1000.0 + NEUTRON, 11.2, 1.40, 5),   # out band, in pixel
            (1000.0 + NEUTRON, 44.8, 1.40, 9),   # out both
        ])
        out = _envelope_mobility_cooccurrence(
            mz, i, mob, fr, q, ppm=25.0, k0_tol=0.02, n_frames=10, pixel_frac=0.0
        )
        assert out["cooc_m1"][0] == pytest.approx(np.log2(1.694), abs=0.01)

    def test_lift_saturates_when_the_isotopologue_dominates(self):
        """Known limitation, pinned deliberately.

        The independence null is estimated from the window's own marginals, so a
        window that is almost *pure* isotopologue drives P(band) and P(pixel) to 1
        along with the joint, and the lift collapses back toward chance.  The
        measurement therefore has its dynamic range in the mixed regime, which is
        where real MALDI windows sit (background is always present).  If a future
        change needs the pure-signal regime too, the null has to come from a
        separate off-peak window instead of the marginals.
        """
        q = np.array([1000.0])
        mz, i, mob, fr = _peaks([
            (1000.0, 100.0, 0.90, 5),
            (1000.0 + NEUTRON, 50.0, 0.90, 5),   # dominant
            (1000.0 + NEUTRON, 1.0, 0.90, 9),
            (1000.0 + NEUTRON, 1.0, 1.40, 5),
            (1000.0 + NEUTRON, 1.0, 1.40, 9),
        ])
        out = _envelope_mobility_cooccurrence(
            mz, i, mob, fr, q, ppm=25.0, k0_tol=0.02, n_frames=10, pixel_frac=0.0
        )
        assert 0.0 < out["cooc_m1"][0] < 0.1   # positive, but nearly chance

    def test_anti_colocated_is_depleted(self):
        """Marginals present but never together -> lift 0, floored, strongly negative."""
        q = np.array([1000.0])
        mz, i, mob, fr = _peaks([
            (1000.0, 100.0, 0.90, 5),
            (1000.0 + NEUTRON, 50.0, 0.90, 9),   # in band,  out pixel
            (1000.0 + NEUTRON, 50.0, 1.40, 5),   # out band, in pixel
        ])
        out = _envelope_mobility_cooccurrence(
            mz, i, mob, fr, q, ppm=25.0, k0_tol=0.02, n_frames=10, pixel_frac=0.0
        )
        assert out["cooc_m1"][0] < 0
        assert np.isfinite(out["cooc_m1"][0])   # floored, never -inf

    def test_background_level_is_cancelled(self):
        """The point of the normalisation: two windows with very different raw
        co-occurrence fractions but the same dependence structure score the same."""
        q = np.array([1000.0])

        def lift(frac_in_band, frac_in_pixel):
            # independent design at the requested marginals
            spec = [(1000.0, 100.0, 0.90, 5)]
            for mb, pb, w in ((True, True, frac_in_band * frac_in_pixel),
                              (True, False, frac_in_band * (1 - frac_in_pixel)),
                              (False, True, (1 - frac_in_band) * frac_in_pixel),
                              (False, False, (1 - frac_in_band) * (1 - frac_in_pixel))):
                spec.append((1000.0 + NEUTRON, 100.0 * w, 0.90 if mb else 1.40, 5 if pb else 9))
            mz, i, mob, fr = _peaks(spec)
            return _envelope_mobility_cooccurrence(
                mz, i, mob, fr, q, ppm=25.0, k0_tol=0.02, n_frames=10, pixel_frac=0.0
            )["cooc_m1"][0]

        # raw fractions differ 4x (0.16 vs 0.64); the lift is 0 for both
        assert lift(0.4, 0.4) == pytest.approx(0.0, abs=1e-9)
        assert lift(0.8, 0.8) == pytest.approx(0.0, abs=1e-9)

    def test_wrong_pixel_reduces_the_lift(self):
        """M+1 at the right mobility but only in pixels where M0 is absent."""
        q = np.array([1000.0])
        mz, i, mob, fr = _peaks([
            (1000.0, 100.0, 0.90, 5),
            (1000.0 + NEUTRON, 50.0, 0.90, 7),   # right mobility, wrong pixel
            (1000.0 + NEUTRON, 50.0, 1.40, 5),
        ])
        out = _envelope_mobility_cooccurrence(
            mz, i, mob, fr, q, ppm=25.0, k0_tol=0.02, n_frames=10
        )
        assert out["cooc_m1"][0] < 0

    def test_no_isotopologue_signal_is_nan(self):
        """An empty M+1 window is unmeasurable, not zero."""
        q = np.array([1000.0])
        mz, i, mob, fr = _peaks([(1000.0, 100.0, 0.90, 5)])
        out = _envelope_mobility_cooccurrence(
            mz, i, mob, fr, q, ppm=25.0, k0_tol=0.02, n_frames=10
        )
        assert np.isnan(out["cooc_m1"][0])
        assert np.isnan(out["cooc_m2"][0])

    def test_empty_marginal_is_nan(self):
        """No M0 at all: no band and no pixel set, so the null is undefined."""
        q = np.array([1000.0])
        mz, i, mob, fr = _peaks([(1000.0 + NEUTRON, 50.0, 0.90, 5)])
        out = _envelope_mobility_cooccurrence(
            mz, i, mob, fr, q, ppm=25.0, k0_tol=0.02, n_frames=10
        )
        assert np.isnan(out["cooc_m1"][0])

    def test_pixel_frac_gates_weak_pixels(self):
        """A pixel holding only a trace of M0 stops counting as on-signal, which
        moves the M+1 intensity out of the joint term."""
        q = np.array([1000.0])
        mz, i, mob, fr = _peaks([
            (1000.0, 100.0, 0.90, 5),            # strong M0 pixel
            (1000.0, 1.0, 0.90, 6),              # trace M0 pixel (1% of the max)
            (1000.0 + NEUTRON, 50.0, 0.90, 6),   # M+1 sits in the trace pixel
            (1000.0 + NEUTRON, 50.0, 1.40, 9),
        ])
        loose = _envelope_mobility_cooccurrence(
            mz, i, mob, fr, q, ppm=25.0, k0_tol=0.02, n_frames=10, pixel_frac=0.0
        )["cooc_m1"][0]
        strict = _envelope_mobility_cooccurrence(
            mz, i, mob, fr, q, ppm=25.0, k0_tol=0.02, n_frames=10, pixel_frac=0.10
        )["cooc_m1"][0]
        assert loose > strict or np.isnan(strict)

    def test_chunking_does_not_change_the_result(self):
        """Chunked accumulation is required at raw-query scale; it must be exact."""
        rng = np.random.default_rng(7)
        q = np.array([1000.0, 1500.0])
        spec = []
        for base in q:
            for _ in range(60):
                k = rng.integers(0, 3)
                spec.append((base + k * NEUTRON, rng.uniform(1, 100),
                             rng.uniform(0.8, 1.0), rng.integers(0, 8)))
        mz, i, mob, fr = _peaks(spec)
        o = np.argsort(mz)
        mz, i, mob, fr = mz[o], i[o], mob[o], fr[o]
        whole = _envelope_mobility_cooccurrence(
            mz, i, mob, fr, q, ppm=25.0, k0_tol=0.02, n_frames=8, chunk=10_000
        )
        chunked = _envelope_mobility_cooccurrence(
            mz, i, mob, fr, q, ppm=25.0, k0_tol=0.02, n_frames=8, chunk=7
        )
        for k in ("cooc_m1", "cooc_m2"):
            np.testing.assert_allclose(whole[k], chunked[k], equal_nan=True)

    def test_empty_inputs(self):
        out = _envelope_mobility_cooccurrence(
            np.array([]), np.array([]), np.array([]), np.array([], dtype=np.int64),
            np.array([1000.0]), ppm=25.0, k0_tol=0.02, n_frames=4,
        )
        assert np.isnan(out["cooc_m1"]).all()


class TestFeatureFunction:

    def test_signature_has_no_is_decoy(self):
        params = inspect.signature(compute_isotope_mobility_cooccurrence_features).parameters
        assert "is_decoy" not in params

    def test_swapping_labels_changes_nothing(self):
        rec = {0: np.array([0.8, 0.6]), 1: np.array([0.2, np.nan])}
        a = compute_isotope_mobility_cooccurrence_features(
            _df([0, 1, 0, 1], is_decoy=[False, False, True, True]), rec)
        b = compute_isotope_mobility_cooccurrence_features(
            _df([0, 1, 0, 1], is_decoy=[True, True, False, False]), rec)
        for c in ("isotope_mob_cooc_m1", "isotope_mob_cooc_m2", "isotope_mob_cooc_mean"):
            np.testing.assert_allclose(
                a[c].to_numpy(float), b[c].to_numpy(float), equal_nan=True
            )

    def test_columns_and_mean(self):
        rec = {0: np.array([1.5, 0.5])}
        out = compute_isotope_mobility_cooccurrence_features(_df([0]), rec)
        assert out["isotope_mob_cooc_m1"].iloc[0] == pytest.approx(1.5)
        assert out["isotope_mob_cooc_m2"].iloc[0] == pytest.approx(0.5)
        assert out["isotope_mob_cooc_mean"].iloc[0] == pytest.approx(1.0)

    def test_mean_ignores_unmeasurable_isotopologue(self):
        rec = {0: np.array([1.5, np.nan])}
        out = compute_isotope_mobility_cooccurrence_features(_df([0]), rec)
        assert out["isotope_mob_cooc_mean"].iloc[0] == pytest.approx(1.5)

    def test_missing_record_all_nan_without_raising(self):
        rec = {0: np.array([1.5, 0.5])}
        out = compute_isotope_mobility_cooccurrence_features(_df([0, 9]), rec)
        for c in ("isotope_mob_cooc_m1", "isotope_mob_cooc_m2", "isotope_mob_cooc_mean"):
            assert np.isnan(out[c].iloc[1])

    def test_empty_map_returns_df_unchanged(self):
        df = _df([0, 1])
        assert compute_isotope_mobility_cooccurrence_features(df, None) is df
        assert compute_isotope_mobility_cooccurrence_features(df, {}) is df

    def test_colocated_pair_gets_identical_values(self):
        """Co-located target/decoy (mz_shuffle-style) share the feature, so the
        feature cannot separate them — AUC 0.5 by construction."""
        rng = np.random.default_rng(0)
        nf = 40
        rec = {i: np.array([rng.uniform(-2, 2), rng.uniform(-2, 2)]) for i in range(nf)}
        idx = list(range(nf)) * 2
        mz = [800.0 + i for i in range(nf)] * 2
        out = compute_isotope_mobility_cooccurrence_features(
            _df(idx, is_decoy=[False] * nf + [True] * nf, feature_mz=mz), rec)
        t = out.loc[~out["is_decoy"], "isotope_mob_cooc_m1"].to_numpy()
        d = out.loc[out["is_decoy"], "isotope_mob_cooc_m1"].to_numpy()
        np.testing.assert_allclose(t, d)

        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(out["is_decoy"].astype(int), out["isotope_mob_cooc_m1"])
        assert auc == pytest.approx(0.5, abs=1e-9)


class TestBridge:

    def test_bridges_via_feature_mz(self):
        maldi_mzs = np.array([800.0, 900.0])
        envelope = {
            "cooc_m1": np.array([1.9, -0.4]),
            "cooc_m2": np.array([0.8, np.nan]),
            # the CCS keys coexist in the same dict and must be ignored here
            "ccs_m0": np.array([300.0, 400.0]),
        }
        candidates = pd.DataFrame({
            "feature_idx": [11, 12, 13],
            "feature_mz": [800.0, 900.0, 1234.0],  # 1234 is off-grid
        })
        out = _observed_envelope_cooc_by_feature_idx(candidates, maldi_mzs, envelope)
        assert set(out) == {11, 12}
        np.testing.assert_allclose(out[11], [1.9, 0.8])
        np.testing.assert_allclose(out[12], [-0.4, np.nan], equal_nan=True)

    def test_missing_key_returns_none(self):
        """A cache written before this feature existed must degrade, not crash."""
        candidates = pd.DataFrame({"feature_idx": [1], "feature_mz": [800.0]})
        legacy = {"ccs_m0": np.array([300.0]), "int_m0": np.array([1.0])}
        assert _observed_envelope_cooc_by_feature_idx(
            candidates, np.array([800.0]), legacy) is None

    def test_none_envelope_returns_none(self):
        candidates = pd.DataFrame({"feature_idx": [1], "feature_mz": [800.0]})
        assert _observed_envelope_cooc_by_feature_idx(
            candidates, np.array([800.0]), None) is None


class TestRankerRegistration:

    def test_features_are_in_the_pool_and_not_leak_listed(self):
        from msi_picasso.feature_generator import MALDI_INTRINSIC_FEATURES
        from msi_picasso.pipeline import _MZ_SHUFFLE_CCS_LEAK_FEATURES

        for c in ("isotope_mob_cooc_m1", "isotope_mob_cooc_m2", "isotope_mob_cooc_mean"):
            assert c in MALDI_INTRINSIC_FEATURES
            assert c not in _MZ_SHUFFLE_CCS_LEAK_FEATURES

    def test_gated_on_observed_ccs(self):
        from msi_picasso.feature_generator import get_feature_names

        without = get_feature_names(has_ccs=False)
        with_ = get_feature_names(has_ccs=True)
        assert "isotope_mob_cooc_m1" not in without
        assert "isotope_mob_cooc_m1" in with_
