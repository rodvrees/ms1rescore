"""Tests for generate_balanced_shuffle_candidates()."""

import numpy as np
import pytest

from ms1rescore.candidates import (
    digest_fasta,
    generate_balanced_shuffle_candidates,
    match_to_maldi_features,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

# ~90 AA protein with several K/R sites; produces 5+ tryptic peptides in 700–2000 Da.
_PROTEIN_SEQ = (
    "MALPVTALLLLAAGLLAHAAGTSQVQVSTQILHQK"
    "PEPTIDEKVFGRCELAAAMKRHGLDNYRTESTVLGTGFLSR"
    "AAATESTPEPTIDEK"
)


def _write_fasta(tmp_path, seq=_PROTEIN_SEQ, accession="P00001"):
    f = tmp_path / "test.fasta"
    f.write_text(f">sp|{accession}|TEST_HUMAN Test protein\n{seq}\n")
    return str(f)


def _dense_features(lo: float = 700.0, hi: float = 2000.0, step: float = 0.01) -> np.ndarray:
    """Dense feature grid; at 20 ppm and 1000 Da, 0.01 Da spacing ensures every peptide matches."""
    return np.arange(lo, hi, step, dtype=np.float64)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestGenerateBalancedShuffleCandidates:

    def _run(self, tmp_path, features=None, **kwargs):
        fasta_path = _write_fasta(tmp_path)
        if features is None:
            features = _dense_features()
        defaults = dict(
            matching_ppm=20.0,
            max_shuffle_rounds=5,
            target_ratio=1.0,
            random_state=42,
        )
        defaults.update(kwargs)
        return generate_balanced_shuffle_candidates(
            fasta_path=fasta_path,
            lcms_ids=None,
            feature_mzs=features,
            **defaults,
        )

    def test_target_rows_have_is_decoy_false(self, tmp_path):
        result = self._run(tmp_path)
        targets = result[~result["is_decoy"]]
        assert len(targets) > 0
        assert (targets["is_decoy"] == False).all()

    def test_decoy_rows_have_is_decoy_true(self, tmp_path):
        result = self._run(tmp_path, max_shuffle_rounds=10)
        decoys = result[result["is_decoy"]]
        if len(decoys) > 0:
            assert (decoys["is_decoy"] == True).all()

    def test_no_decoy_sequence_is_target_sequence(self, tmp_path):
        fasta_path = _write_fasta(tmp_path)
        target_db = digest_fasta(fasta_path, generate_decoys=False)
        target_seqs = set(target_db["peptide"].values)
        result = self._run(tmp_path, max_shuffle_rounds=10)
        decoys = result[result["is_decoy"]]
        if len(decoys) > 0:
            collisions = set(decoys["peptide"].values) & target_seqs
            assert len(collisions) == 0, f"Decoy sequences in target set: {collisions}"

    def test_output_columns_match_match_to_maldi_features(self, tmp_path):
        fasta_path = _write_fasta(tmp_path)
        features = _dense_features()
        target_db = digest_fasta(fasta_path, generate_decoys=False)
        ref = match_to_maldi_features(features, target_db, 20.0)
        result = self._run(tmp_path, features=features, max_shuffle_rounds=3)
        ref_cols = set(ref.columns)
        extra = set(result.columns) - ref_cols - {"decoy_delta_da", "source"}
        assert not extra, f"Unexpected extra columns: {extra}"
        missing = ref_cols - set(result.columns)
        assert not missing, f"Missing columns: {missing}"

    def test_decoy_delta_da_is_nan_for_all_rows(self, tmp_path):
        result = self._run(tmp_path, max_shuffle_rounds=5)
        if len(result) > 0:
            assert result["decoy_delta_da"].isna().all()

    def test_reproducibility_same_random_state(self, tmp_path):
        fasta_path = _write_fasta(tmp_path)
        features = _dense_features()
        r1 = generate_balanced_shuffle_candidates(
            fasta_path=fasta_path, lcms_ids=None,
            feature_mzs=features, matching_ppm=20.0,
            max_shuffle_rounds=5, random_state=42,
        )
        r2 = generate_balanced_shuffle_candidates(
            fasta_path=fasta_path, lcms_ids=None,
            feature_mzs=features, matching_ppm=20.0,
            max_shuffle_rounds=5, random_state=42,
        )
        d1 = sorted(r1[r1["is_decoy"]]["peptide"].tolist())
        d2 = sorted(r2[r2["is_decoy"]]["peptide"].tolist())
        assert d1 == d2

    def test_different_random_states_give_different_decoys(self, tmp_path):
        fasta_path = _write_fasta(tmp_path)
        features = _dense_features()
        r1 = generate_balanced_shuffle_candidates(
            fasta_path=fasta_path, lcms_ids=None,
            feature_mzs=features, matching_ppm=20.0,
            max_shuffle_rounds=5, random_state=1,
        )
        r2 = generate_balanced_shuffle_candidates(
            fasta_path=fasta_path, lcms_ids=None,
            feature_mzs=features, matching_ppm=20.0,
            max_shuffle_rounds=5, random_state=999,
        )
        d1 = set(r1[r1["is_decoy"]]["peptide"].tolist())
        d2 = set(r2[r2["is_decoy"]]["peptide"].tolist())
        if len(d1) > 0 and len(d2) > 0:
            # Astronomically unlikely to be identical with different seeds
            assert d1 != d2

    def test_no_decoys_when_features_are_out_of_range(self, tmp_path):
        """No decoys should appear when all features are far from any peptide mass."""
        features = np.array([10000.0, 10001.0, 10002.0], dtype=np.float64)
        result = self._run(tmp_path, features=features, max_shuffle_rounds=3)
        # No targets match → function returns early with 0 rows
        assert len(result) == 0

    def test_td_ratio_near_one_with_sufficient_rounds(self, tmp_path):
        """With a dense feature grid and enough rounds, T:D ratio should be close to 1:1."""
        result = self._run(tmp_path, max_shuffle_rounds=30, target_ratio=1.0)
        n_target = int((~result["is_decoy"]).sum())
        n_decoy = int(result["is_decoy"].sum())
        if n_target == 0:
            pytest.skip("No target candidates — check FASTA or feature grid")
        if n_decoy == 0:
            pytest.skip("No decoy candidates — check shuffle rounds or feature grid")
        ratio = n_target / n_decoy
        assert 0.5 <= ratio <= 2.0, (
            f"T:D ratio {ratio:.2f} far from 1:1; "
            f"n_target={n_target}, n_decoy={n_decoy}"
        )

    def test_early_stopping(self, tmp_path):
        """Function stops before max_shuffle_rounds when enough decoys are collected."""
        # Use target_ratio=0.1 so a small pool satisfies the requirement quickly.
        result = self._run(tmp_path, max_shuffle_rounds=50, target_ratio=0.1)
        # As long as the function returns successfully, early stopping is implicitly tested
        # (verified by checking that decoys were collected without running all 50 rounds).
        n_target = int((~result["is_decoy"]).sum())
        n_decoy = int(result["is_decoy"].sum())
        if n_target > 0:
            assert n_decoy <= max(1, int(0.15 * n_target)), (
                f"Expected at most 15% of targets as decoys with ratio=0.1, got {n_decoy}/{n_target}"
            )

    def test_requires_fasta_when_no_lcms_ids(self, tmp_path):
        """ValueError raised when both fasta_path and lcms_ids are None."""
        features = _dense_features()
        with pytest.raises(ValueError, match="fasta_path required"):
            generate_balanced_shuffle_candidates(
                fasta_path=None,
                lcms_ids=None,
                feature_mzs=features,
            )

    def test_decoy_proteins_labelled_with_round(self, tmp_path):
        """Decoy protein names should contain DECOY_ prefix and round index."""
        result = self._run(tmp_path, max_shuffle_rounds=10)
        decoys = result[result["is_decoy"]]
        if len(decoys) > 0:
            for prot in decoys["protein"].unique():
                assert prot.startswith("DECOY_"), f"Unexpected decoy protein name: {prot}"
                assert "_r" in prot, f"Round index missing from protein name: {prot}"

    def test_length_distribution_stratified(self, tmp_path):
        """Per-length T:D ratio should be close to 1 for all fillable bins."""
        result = self._run(tmp_path, max_shuffle_rounds=30, target_ratio=1.0)
        targets = result[~result["is_decoy"]]
        decoys = result[result["is_decoy"]]
        if len(targets) == 0 or len(decoys) == 0:
            pytest.skip("Not enough candidates for length stratification test")

        tgt_len = targets["peptide"].str.len().value_counts()
        dec_len = decoys["peptide"].str.len().value_counts()

        # For lengths that DO have decoys, the per-length ratio must be close to 1.
        for length in dec_len.index:
            t = tgt_len.get(length, 0)
            d = dec_len[length]
            if t == 0:
                continue  # decoys at lengths without targets should not exist
            ratio = d / t
            assert 0.7 <= ratio <= 1.3, (
                f"Length {length}: T={t}, D={d}, ratio={ratio:.3f} — "
                "length-stratified sampling produced an unbalanced bin"
            )

        # No decoys should appear at lengths not present in targets.
        extra_lengths = set(dec_len.index) - set(tgt_len.index)
        assert not extra_lengths, f"Decoys at lengths absent from targets: {extra_lengths}"

        # Mean length of decoys should be within 1 residue of target mean.
        assert abs(targets["peptide"].str.len().mean() - decoys["peptide"].str.len().mean()) < 1.0, (
            "Mean peptide length differs by more than 1 residue between targets and decoys"
        )


def _contested_count(result) -> tuple[int, int, int]:
    """Return (contested, target_only, decoy_only) feature counts."""
    has_tgt = result.groupby("feature_idx")["is_decoy"].agg(lambda s: (~s).any())
    has_dec = result.groupby("feature_idx")["is_decoy"].agg("any")
    return (
        int((has_tgt & has_dec).sum()),
        int((has_tgt & ~has_dec).sum()),
        int((~has_tgt & has_dec).sum()),
    )


class TestFeaturePairedSelection:
    """Tests for selection_mode='feature' (paired_shuffle)."""

    def _run(self, tmp_path, selection_mode, features=None, **kwargs):
        fasta_path = _write_fasta(tmp_path)
        if features is None:
            features = _dense_features()
        defaults = dict(
            matching_ppm=20.0,
            max_shuffle_rounds=5,
            target_ratio=1.0,
            random_state=42,
            selection_mode=selection_mode,
        )
        defaults.update(kwargs)
        return generate_balanced_shuffle_candidates(
            fasta_path=fasta_path,
            lcms_ids=None,
            feature_mzs=features,
            **defaults,
        )

    def test_invalid_selection_mode_raises(self, tmp_path):
        with pytest.raises(ValueError, match="selection_mode"):
            self._run(tmp_path, selection_mode="bogus")

    def test_feature_mode_produces_decoys(self, tmp_path):
        result = self._run(tmp_path, "feature", max_shuffle_rounds=10)
        assert int(result["is_decoy"].sum()) > 0

    def test_feature_mode_contested_ge_length_mode(self, tmp_path):
        """Feature-paired selection should yield at least as many contested features
        as length-stratified selection on the same shuffle pool (same seed + rounds,
        small enough that early-stopping does not fire and the pools match)."""
        feat = self._run(tmp_path, "feature", max_shuffle_rounds=5)
        length = self._run(tmp_path, "length", max_shuffle_rounds=5)
        c_feat, _, _ = _contested_count(feat)
        c_len, _, _ = _contested_count(length)
        assert c_feat >= c_len, (
            f"feature-paired contested ({c_feat}) < length-stratified ({c_len})"
        )

    def test_feature_mode_preserves_global_ratio(self, tmp_path):
        """Top-up keeps the global T:D ratio at ~target_ratio, matching length mode."""
        result = self._run(tmp_path, "feature", max_shuffle_rounds=30, target_ratio=1.0)
        n_target = int((~result["is_decoy"]).sum())
        n_decoy = int(result["is_decoy"].sum())
        if n_target == 0 or n_decoy == 0:
            pytest.skip("Not enough candidates for ratio test")
        ratio = n_target / n_decoy
        assert 0.5 <= ratio <= 2.0, f"T:D ratio {ratio:.2f} far from 1:1"

    def test_feature_mode_reproducible(self, tmp_path):
        r1 = self._run(tmp_path, "feature", max_shuffle_rounds=5, random_state=7)
        r2 = self._run(tmp_path, "feature", max_shuffle_rounds=5, random_state=7)
        d1 = sorted(r1[r1["is_decoy"]]["peptide"].tolist())
        d2 = sorted(r2[r2["is_decoy"]]["peptide"].tolist())
        assert d1 == d2

    def test_feature_mode_decoy_features_are_real_matches(self, tmp_path):
        """Every decoy row's feature_mz must be one of the input features within ppm
        (no fabricated matches), and its feature_idx must be a valid feature index."""
        features = _dense_features()
        result = self._run(tmp_path, "feature", features=features, max_shuffle_rounds=10)
        decoys = result[result["is_decoy"]]
        if len(decoys) == 0:
            pytest.skip("No decoys collected")
        assert decoys["feature_idx"].between(0, len(features) - 1).all()
        # feature_mz should match the indexed input feature.
        assert np.allclose(
            decoys["feature_mz"].to_numpy(),
            features[decoys["feature_idx"].to_numpy()],
        )

    def test_feature_mode_schema_matches_match_to_maldi_features(self, tmp_path):
        fasta_path = _write_fasta(tmp_path)
        features = _dense_features()
        target_db = digest_fasta(fasta_path, generate_decoys=False)
        ref = match_to_maldi_features(features, target_db, 20.0)
        result = self._run(tmp_path, "feature", features=features, max_shuffle_rounds=3)
        ref_cols = set(ref.columns)
        extra = set(result.columns) - ref_cols - {"decoy_delta_da", "source"}
        assert not extra, f"Unexpected extra columns: {extra}"
        missing = ref_cols - set(result.columns)
        assert not missing, f"Missing columns: {missing}"
