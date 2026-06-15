"""Tests for generate_mz_shuffle_candidates() (derangement decoys)."""

import numpy as np
import pytest

from msi_picasso.candidates import (
    digest_fasta,
    generate_mz_shuffle_candidates,
    match_to_maldi_features,
)

# ~90 AA protein with several K/R sites; produces a handful of tryptic peptides.
_PROTEIN_SEQ = (
    "MALPVTALLLLAAGLLAHAAGTSQVQVSTQILHQK"
    "PEPTIDEKVFGRCELAAAMKRHGLDNYRTESTVLGTGFLSR"
    "AAATESTPEPTIDEK"
)


def _write_fasta(tmp_path, seq=_PROTEIN_SEQ, accession="P00001"):
    f = tmp_path / "test.fasta"
    f.write_text(f">sp|{accession}|TEST_HUMAN Test protein\n{seq}\n")
    return str(f)


def _run(tmp_path, matching_ppm=20.0, **kwargs):
    fasta_path = _write_fasta(tmp_path)
    target_db = digest_fasta(fasta_path, generate_decoys=False)
    # Raw-query-style grid: the candidate peptide masses themselves.
    features = np.sort(target_db["mh_mz"].to_numpy(dtype=np.float64))
    result = generate_mz_shuffle_candidates(
        target_db, features, matching_ppm=matching_ppm, random_state=42, **kwargs
    )
    return result, target_db, features


class TestGenerateMzShuffleCandidates:

    def test_flags_and_source(self, tmp_path):
        result, _, _ = _run(tmp_path)
        dec = result[result["is_decoy"]]
        assert len(dec) > 0
        assert result["is_decoy"].dtype == bool
        assert (dec["source"] == "decoy_mz_shuffle").all()
        assert (result[~result["is_decoy"]]["source"] == "target").all()

    def test_decoy_features_are_real_target_features(self, tmp_path):
        """Every decoy sits on a feature that belongs to some target (co-location)."""
        result, _, _ = _run(tmp_path)
        tgt = result[~result["is_decoy"]]
        dec = result[result["is_decoy"]]
        target_feats = set(np.round(tgt["feature_mz"].to_numpy(), 6))
        decoy_feats = set(np.round(dec["feature_mz"].to_numpy(), 6))
        assert decoy_feats.issubset(target_feats)

    def test_every_decoy_feature_is_contested(self, tmp_path):
        """Each decoy feature also carries a target candidate -> contested feature."""
        result, _, _ = _run(tmp_path)
        has_t = result.groupby("feature_mz")["is_decoy"].agg(lambda s: (~s).any())
        dec_feats = result[result["is_decoy"]]["feature_mz"].unique()
        assert all(has_t.loc[f] for f in dec_feats)

    def test_no_decoy_on_its_own_feature(self, tmp_path):
        """Derangement: a peptide is never relocated onto its own m/z (no fixed point)."""
        result, _, _ = _run(tmp_path)
        dec = result[result["is_decoy"]]
        # decoy_delta_da = assigned_feature_mz - mh_mz must be non-zero (relocated)
        assert (dec["decoy_delta_da"].abs() > 1e-6).all()

    def test_decoy_not_near_isobaric_with_assignment(self, tmp_path):
        """The mass-sorted rotation keeps the assigned feature far from the peptide's
        own mass (not a near-isobaric, uninformative pairing)."""
        result, _, _ = _run(tmp_path)
        dec = result[result["is_decoy"]]
        # |delta| should exceed the matching window for essentially all decoys
        frac_far = (dec["decoy_delta_da"].abs() > 0.5).mean()
        assert frac_far > 0.8

    def test_ppm_inherited_not_from_assigned_feature(self, tmp_path):
        """Decoy ppm is the peptide's own (small) match ppm, NOT (assigned_mz - mass)
        which would be huge and leak the label."""
        result, _, _ = _run(tmp_path)
        dec = result[result["is_decoy"]]
        # inherited ppm stays within the matching window, never ~thousands of ppm
        assert (dec["ppm_error_abs"] <= 20.0 + 1e-6).all()

    def test_schema_matches_match_to_maldi_features(self, tmp_path):
        result, target_db, features = _run(tmp_path)
        ref = match_to_maldi_features(features, target_db, 20.0)
        extra = set(result.columns) - set(ref.columns) - {"decoy_delta_da", "source"}
        assert not extra, f"Unexpected extra columns: {extra}"
        missing = set(ref.columns) - set(result.columns)
        assert not missing, f"Missing columns: {missing}"

    def test_roughly_balanced_td(self, tmp_path):
        result, _, _ = _run(tmp_path)
        nt = int((~result["is_decoy"]).sum())
        nd = int(result["is_decoy"].sum())
        assert 0.5 <= nt / nd <= 2.0

    def test_one_target_row_per_peptide_under_multiplicity(self, tmp_path):
        """When peptides match multiple near-isobaric features, targets must be
        deduplicated to one representative row per peptide so the target:decoy
        count stays 1:1 (no multiplicity imbalance like the 5901:2895 seen in
        production). Build a feature grid that duplicates every peptide m/z with
        a +5 ppm twin so each peptide matches two features."""
        fasta_path = _write_fasta(tmp_path)
        target_db = digest_fasta(fasta_path, generate_decoys=False)
        mzs = target_db["mh_mz"].to_numpy(dtype=np.float64)
        # each peptide m/z plus a within-tolerance twin (+5 ppm) → 2 features/peptide
        twin = mzs * (1 + 5e-6)
        features = np.sort(np.concatenate([mzs, twin]))
        result = generate_mz_shuffle_candidates(
            target_db, features, matching_ppm=20.0, random_state=42
        )
        tgt = result[~result["is_decoy"]]
        nt = int((~result["is_decoy"]).sum())
        nd = int(result["is_decoy"].sum())
        # exactly one target row per unique peptide, and 1:1 with decoys
        assert nt == tgt["peptide"].nunique()
        assert nt == nd

    def test_reproducible_same_seed(self, tmp_path):
        r1, _, _ = _run(tmp_path)
        r2, _, _ = _run(tmp_path)
        d1 = r1[r1["is_decoy"]][["peptide", "feature_mz"]].sort_values("peptide").to_numpy()
        d2 = r2[r2["is_decoy"]][["peptide", "feature_mz"]].sort_values("peptide").to_numpy()
        assert np.array_equal(d1, d2)

    def test_decoy_protein_namespace_is_separate(self, tmp_path):
        """Decoys must NOT share protein names with targets, else protein-level
        features pool target+decoy peptides under one protein (invalid null)."""
        result, _, _ = _run(tmp_path)
        tgt_prot = set(result.loc[~result["is_decoy"], "protein"])
        dec_prot = set(result.loc[result["is_decoy"], "protein"])
        assert all(p.startswith("DECOY_") for p in dec_prot)
        assert tgt_prot.isdisjoint(dec_prot)

    def test_decoy_inherits_source_protein_tryptic_count(self, tmp_path):
        result, _, _ = _run(tmp_path)
        tgt_tc = (
            result.loc[~result["is_decoy"]]
            .drop_duplicates("protein").set_index("protein")["protein_tryptic_count"]
        )
        dec = result[result["is_decoy"]].drop_duplicates("protein")
        for _, r in dec.iterrows():
            base = r["protein"][len("DECOY_"):]
            if base in tgt_tc.index:
                assert r["protein_tryptic_count"] == tgt_tc[base]


class TestMzShuffleSpatialGate:

    def test_mz_shuffle_permitted_with_spatial_ranker(self):
        from msi_picasso.pipeline import _resolve_spatial_ranker_features
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error")  # no warning -> permitted
            assert _resolve_spatial_ranker_features(True, "mz_shuffle") is True


class TestCcsMzDetrend:
    """The m/z-detrended CCS residual removes the m/z-baseline leak."""

    def _trend(self, n=60):
        mz = np.linspace(700.0, 2500.0, n)
        ccs = 3.0 * mz ** 0.62  # synthetic power-law CCS↔m/z trend
        return mz, ccs

    def test_target_baseline_is_zero(self):
        from msi_picasso.maldi_features import _ccs_mz_baseline
        fitmz, fitccs = self._trend()
        # feature_mz == mh_mz (a target) -> baseline 0 -> residual == raw delta
        b = _ccs_mz_baseline(np.array([1234.0]), np.array([1234.0]), fitmz, fitccs)
        assert abs(b[0]) < 1e-6

    def test_far_mass_decoy_baseline_is_large(self):
        from msi_picasso.maldi_features import _ccs_mz_baseline
        fitmz, fitccs = self._trend()
        # decoy: light peptide (900) relocated onto a heavy feature (2000)
        b = _ccs_mz_baseline(np.array([2000.0]), np.array([900.0]), fitmz, fitccs)
        assert b[0] > 50.0  # large m/z-gap baseline that must be removed

    def test_pure_baseline_decoy_residual_is_zero(self):
        """A decoy whose CCS difference is PURELY the m/z baseline (no conformational
        mismatch) must get residual ≈ 0 — i.e. the leak is fully removed."""
        from msi_picasso.maldi_features import _ccs_mz_baseline
        fitmz, fitccs = self._trend()
        A, B = 3.0, 0.62
        feat_mz, pep_mz = 2000.0, 900.0
        observed = A * feat_mz ** B   # peak sits exactly on the trend at the feature
        predicted = A * pep_mz ** B   # peptide CCS exactly on the trend at its own m/z
        raw_delta = observed - predicted
        base = _ccs_mz_baseline(np.array([feat_mz]), np.array([pep_mz]), fitmz, fitccs)[0]
        resid = raw_delta - base
        assert abs(raw_delta) > 50.0      # raw delta is huge (leaks m/z gap)
        assert abs(resid) < 1.0           # detrended residual ~0 (no real mismatch)

    def test_few_points_returns_none(self):
        from msi_picasso.maldi_features import _ccs_mz_baseline
        fitmz, fitccs = self._trend()
        assert _ccs_mz_baseline(np.array([1000.0]), np.array([1000.0]), fitmz[:3], fitccs[:3]) is None


class TestMzShuffleExcludesRawCcs:
    """For mz_shuffle the ranker keeps only the detrended *_resid CCS features."""

    def test_resid_registered_and_gated(self):
        from msi_picasso.feature_generator import MALDI_INTRINSIC_FEATURES, get_feature_names
        resid = ["im2deep_delta_ccs_resid", "im2deep_abs_delta_ccs_pct_resid",
                 "im2deep_ccs_zscore_resid", "im2deep_ccs_rank_resid"]
        assert all(f in MALDI_INTRINSIC_FEATURES for f in resid)
        assert "im2deep_abs_delta_ccs_pct_resid" in get_feature_names(has_ccs=True)
        assert "im2deep_abs_delta_ccs_pct_resid" not in get_feature_names(has_ccs=False)

    def test_leak_feature_set_covers_raw_ccs_and_mob_coloc_not_resid(self):
        from msi_picasso.pipeline import _MZ_SHUFFLE_CCS_LEAK_FEATURES as L
        # raw CCS scalars + mobility-gated colocalizations are excluded for mz_shuffle
        assert {"im2deep_delta_ccs", "im2deep_abs_delta_ccs_pct"} <= L
        assert {"isotope_colocalization_mean_mob", "adduct_colocalization_k_mob"} <= L
        # the detrended residuals are NOT excluded (they are the kept CCS signal)
        assert not any(f.endswith("_resid") for f in L)
        # non-mobility colocalizations stay (they're co-located/symmetric)
        assert "isotope_image_colocalization_mean" not in L
        assert "adduct_colocalization_k" not in L
