"""Tests for generate_mz_shift_candidates()."""

import numpy as np
import pytest

from msi_picasso.candidates import (
    digest_fasta,
    generate_mz_shift_candidates,
    match_to_maldi_features,
)

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
    """Dense feature grid; at 20 ppm every peptide (and every shifted query) matches."""
    return np.arange(lo, hi, step, dtype=np.float64)


class TestGenerateMzShiftCandidates:

    def _run(self, tmp_path, features=None, **kwargs):
        fasta_path = _write_fasta(tmp_path)
        target_db = digest_fasta(fasta_path, generate_decoys=False)
        if features is None:
            features = _dense_features()
        defaults = dict(
            matching_ppm=20.0,
            delta_min=5.0,
            delta_max=20.0,
            snap_tolerance_ppm=50.0,
            random_state=42,
        )
        defaults.update(kwargs)
        result = generate_mz_shift_candidates(target_db, features, **defaults)
        return result, target_db, features

    def test_decoy_rows_flagged_and_sourced(self, tmp_path):
        result, _, _ = self._run(tmp_path)
        decoys = result[result["is_decoy"]]
        assert len(decoys) > 0
        assert (decoys["is_decoy"] == True).all()  # noqa: E712
        assert (decoys["source"] == "decoy_mz_shift").all()

    def test_is_decoy_dtype_is_bool(self, tmp_path):
        result, _, _ = self._run(tmp_path)
        assert result["is_decoy"].dtype == bool

    def test_target_rows_present_and_not_decoy(self, tmp_path):
        result, _, _ = self._run(tmp_path)
        targets = result[~result["is_decoy"]]
        assert len(targets) > 0
        assert (targets["source"] == "target").all()

    def test_decoy_feature_mz_is_shifted_not_original(self, tmp_path):
        """feature_mz on decoy rows must be the snapped shifted m/z, at least
        delta_min away from the peptide's own [M+H]+ m/z — never the original."""
        result, _, features = self._run(tmp_path, delta_min=5.0, delta_max=20.0)
        decoys = result[result["is_decoy"]]
        assert len(decoys) > 0
        # decoy_delta_da is the actual offset to the chosen feature.
        assert (decoys["decoy_delta_da"].abs() >= 5.0 - 1e-6).all()
        assert (decoys["decoy_delta_da"].abs() <= 20.0 + 0.1).all()
        # feature_mz equals mh_mz + decoy_delta_da (i.e. the shifted anchor).
        assert np.allclose(
            decoys["feature_mz"].to_numpy(),
            (decoys["mh_mz"] + decoys["decoy_delta_da"]).to_numpy(),
        )
        # Every decoy feature must be a real input feature.
        assert decoys["feature_idx"].between(0, len(features) - 1).all()
        assert np.allclose(
            decoys["feature_mz"].to_numpy(),
            features[decoys["feature_idx"].to_numpy().astype(int)],
        )

    def test_decoy_feature_does_not_collide_with_targets(self, tmp_path):
        """Snapped decoy feature must not lie within matching_ppm of any target
        peptide m/z (collision check)."""
        result, target_db, _ = self._run(tmp_path, matching_ppm=20.0)
        decoys = result[result["is_decoy"]]
        target_mzs = np.sort(target_db["mh_mz"].to_numpy())
        tol = 20.0 * 1e-6
        for mz in decoys["feature_mz"].to_numpy():
            lo = np.searchsorted(target_mzs, mz * (1 - tol), side="left")
            hi = np.searchsorted(target_mzs, mz * (1 + tol), side="right")
            assert lo >= hi, f"Decoy feature {mz:.4f} collides with a target peptide m/z"

    def test_decoy_ppm_error_inherited_from_target(self, tmp_path):
        """ppm_error on a decoy row is copied from its peptide's best target match,
        not computed from the (far-off) decoy feature — keeps it non-discriminative."""
        result, _, _ = self._run(tmp_path)
        decoys = result[result["is_decoy"]]
        # Inherited ppm_error must stay within the matching window, never ~thousands of ppm.
        assert (decoys["ppm_error_abs"] <= 20.0 + 1e-6).all()

    def test_schema_matches_match_to_maldi_features(self, tmp_path):
        fasta_path = _write_fasta(tmp_path)
        features = _dense_features()
        target_db = digest_fasta(fasta_path, generate_decoys=False)
        ref = match_to_maldi_features(features, target_db, 20.0)
        result, _, _ = self._run(tmp_path, features=features)
        ref_cols = set(ref.columns)
        extra = set(result.columns) - ref_cols - {"decoy_delta_da", "source"}
        assert not extra, f"Unexpected extra columns: {extra}"
        missing = ref_cols - set(result.columns)
        assert not missing, f"Missing columns: {missing}"

    def test_decoy_delta_da_nan_for_targets(self, tmp_path):
        result, _, _ = self._run(tmp_path)
        targets = result[~result["is_decoy"]]
        assert targets["decoy_delta_da"].isna().all()

    def test_reproducible_same_seed(self, tmp_path):
        r1, _, _ = self._run(tmp_path, random_state=42)
        r2, _, _ = self._run(tmp_path, random_state=42)
        d1 = r1[r1["is_decoy"]][["peptide", "feature_mz"]].sort_values("peptide").to_numpy()
        d2 = r2[r2["is_decoy"]][["peptide", "feature_mz"]].sort_values("peptide").to_numpy()
        assert np.array_equal(d1, d2)

    def test_decoy_protein_namespace_is_separate(self, tmp_path):
        """Decoys carry a DECOY_-prefixed protein, disjoint from target proteins, so
        protein-level features are computed within class (not pooled with targets)."""
        result, _, _ = self._run(tmp_path)
        dec = result[result["is_decoy"]]
        if len(dec) == 0:
            pytest.skip("no decoys produced")
        tgt_prot = set(result.loc[~result["is_decoy"], "protein"])
        dec_prot = set(dec["protein"])
        assert all(p.startswith("DECOY_") for p in dec_prot)
        assert tgt_prot.isdisjoint(dec_prot)

    def test_no_valid_decoys_returns_targets_only(self, tmp_path):
        """When the only feature collides with every target (e.g. a single feature
        right on a target mass), no decoy is produced and target-only is returned."""
        fasta_path = _write_fasta(tmp_path)
        target_db = digest_fasta(fasta_path, generate_decoys=False)
        # One feature exactly on a target mass: shift snaps nowhere valid.
        features = np.array([float(target_db["mh_mz"].iloc[0])], dtype=np.float64)
        result = generate_mz_shift_candidates(target_db, features, matching_ppm=20.0)
        assert int(result["is_decoy"].sum()) == 0


class TestNoSnapMzShift:
    """Raw-query mode: snap_to_features=False places decoys at the exact shifted m/z."""

    def _run_nosnap(self, tmp_path, **kwargs):
        fasta_path = _write_fasta(tmp_path)
        target_db = digest_fasta(fasta_path, generate_decoys=False)
        # Raw-query grid = the candidate peptide masses themselves (targets match
        # exactly; the spacing is sparse so snapping shifted queries would fail).
        features = np.sort(target_db["mh_mz"].to_numpy(dtype=np.float64))
        defaults = dict(
            matching_ppm=20.0, delta_min=5.0, delta_max=20.0,
            snap_to_features=False, random_state=42,
        )
        defaults.update(kwargs)
        result = generate_mz_shift_candidates(target_db, features, **defaults)
        return result, target_db, features

    def test_decoy_feature_mz_is_exact_shift(self, tmp_path):
        """feature_mz == mh_mz ± delta exactly (not snapped to a grid value)."""
        result, _, _ = self._run_nosnap(tmp_path)
        decoys = result[result["is_decoy"]]
        assert len(decoys) > 0
        # decoy_delta_da == feature_mz - mh_mz, and |delta| in [delta_min, delta_max].
        assert np.allclose(
            decoys["feature_mz"].to_numpy(),
            (decoys["mh_mz"] + decoys["decoy_delta_da"]).to_numpy(),
        )
        assert (decoys["decoy_delta_da"].abs() >= 5.0 - 1e-9).all()
        assert (decoys["decoy_delta_da"].abs() <= 20.0 + 1e-9).all()

    def test_decoy_features_are_distinct_no_clustering(self, tmp_path):
        """Every decoy occupies a distinct feature (the whole point of the fix)."""
        result, _, _ = self._run_nosnap(tmp_path)
        decoys = result[result["is_decoy"]]
        assert decoys["feature_mz"].nunique() == len(decoys)
        assert decoys["feature_idx"].nunique() == len(decoys)

    def test_decoy_feature_idx_disjoint_from_targets(self, tmp_path):
        """No-snap decoy feature_idx are assigned past the grid range, never colliding
        with target (grid) indices."""
        result, _, features = self._run_nosnap(tmp_path)
        dec = result[result["is_decoy"]]
        tgt = result[~result["is_decoy"]]
        assert (dec["feature_idx"] >= len(features)).all()
        assert set(dec["feature_idx"]).isdisjoint(set(tgt["feature_idx"]))

    def test_decoys_not_isobaric_with_targets(self, tmp_path):
        """Collision filter still rejects shifts within matching_ppm of a target mass."""
        result, target_db, _ = self._run_nosnap(tmp_path, matching_ppm=20.0)
        decoys = result[result["is_decoy"]]
        target_mzs = np.sort(target_db["mh_mz"].to_numpy())
        tol = 20.0 * 1e-6
        for mz in decoys["feature_mz"].to_numpy():
            lo = np.searchsorted(target_mzs, mz * (1 - tol), side="left")
            hi = np.searchsorted(target_mzs, mz * (1 + tol), side="right")
            assert lo >= hi, f"no-snap decoy {mz:.4f} is isobaric with a target"

    def test_more_decoys_than_snap_on_sparse_grid(self, tmp_path):
        """With the raw-query grid (the sparse set of peptide masses), snapping
        yields few/no valid decoys (no feature within snap_tolerance of a shift),
        while no-snap yields ~one decoy per target peptide."""
        fasta_path = _write_fasta(tmp_path)
        target_db = digest_fasta(fasta_path, generate_decoys=False)
        grid = np.sort(target_db["mh_mz"].to_numpy(dtype=np.float64))
        snapped = generate_mz_shift_candidates(
            target_db, grid, snap_to_features=True, random_state=42
        )
        nosnap = generate_mz_shift_candidates(
            target_db, grid, snap_to_features=False, random_state=42
        )
        assert int(nosnap["is_decoy"].sum()) > int(snapped["is_decoy"].sum())
