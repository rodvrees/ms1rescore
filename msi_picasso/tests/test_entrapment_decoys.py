"""Tests for load_entrapment_candidates()."""

import logging

import numpy as np
import pytest

from msi_picasso.candidates import (
    digest_fasta,
    load_entrapment_candidates,
    match_to_maldi_features,
)

# Target organism protein (human-like).
_TARGET_SEQ = (
    "MALPVTALLLLAAGLLAHAAGTSQVQVSTQILHQK"
    "PEPTIDEKVFGRCELAAAMKRHGLDNYRTESTVLGTGFLSR"
    "AAATESTPEPTIDEK"
)

# Entrapment organism protein (unrelated sequence, e.g. a plant/bacterium).
_ENTRAP_SEQ = (
    "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGK"
    "QLEDGRTLSDYNIQKESTLHLVLRLRGGNNWADIYK"
    "GSAVMHEACLNPDFFTRWMK"
)


def _write_fasta(tmp_path, seq, accession, name="test.fasta"):
    f = tmp_path / name
    f.write_text(f">sp|{accession}|TEST_ORG {accession}\n{seq}\n")
    return str(f)


def _dense_features(lo: float = 700.0, hi: float = 2500.0, step: float = 0.01) -> np.ndarray:
    return np.arange(lo, hi, step, dtype=np.float64)


def _target_candidates(tmp_path, features):
    fasta_path = _write_fasta(tmp_path, _TARGET_SEQ, "P00001", "target.fasta")
    target_db = digest_fasta(fasta_path, generate_decoys=False)
    return match_to_maldi_features(features, target_db, 20.0), target_db


class TestLoadEntrapmentCandidates:

    def _run(self, tmp_path, features=None, **kwargs):
        if features is None:
            features = _dense_features()
        tgt_cand, target_db = _target_candidates(tmp_path, features)
        ent_path = _write_fasta(tmp_path, _ENTRAP_SEQ, "Q99999", "entrap.fasta")
        defaults = dict(matching_ppm=20.0, missed_cleavages=2, min_length=7, max_length=30)
        defaults.update(kwargs)
        decoy = load_entrapment_candidates(ent_path, tgt_cand, features, **defaults)
        return decoy, tgt_cand, target_db, features

    def test_all_rows_flagged_decoy(self, tmp_path):
        decoy, _, _, _ = self._run(tmp_path)
        assert len(decoy) > 0
        assert (decoy["is_decoy"] == True).all()  # noqa: E712

    def test_is_decoy_dtype_is_bool(self, tmp_path):
        decoy, _, _, _ = self._run(tmp_path)
        assert decoy["is_decoy"].dtype == bool

    def test_protein_prefix_and_source(self, tmp_path):
        decoy, _, _, _ = self._run(tmp_path)
        assert decoy["protein"].str.startswith("ENTRAPMENT_").all()
        assert (decoy["source"] == "entrapment").all()

    def test_schema_matches_match_to_maldi_features(self, tmp_path):
        decoy, tgt_cand, _, _ = self._run(tmp_path)
        ref_cols = set(tgt_cand.columns)
        extra = set(decoy.columns) - ref_cols - {"source"}
        assert not extra, f"Unexpected extra columns: {extra}"
        missing = ref_cols - set(decoy.columns)
        # `source` may be present on targets too; only structural match columns required.
        assert not (missing - {"source"}), f"Missing columns: {missing}"

    def test_contamination_filter_removes_isobaric(self, tmp_path):
        """No surviving entrapment peptide may be within matching_ppm of a target m/z."""
        decoy, _, target_db, _ = self._run(tmp_path, matching_ppm=20.0)
        target_mzs = np.sort(target_db["mh_mz"].to_numpy())
        tol = 20.0 * 1e-6
        for mz in decoy["mh_mz"].unique():
            lo = np.searchsorted(target_mzs, mz * (1 - tol), side="left")
            hi = np.searchsorted(target_mzs, mz * (1 + tol), side="right")
            assert lo >= hi, f"Entrapment peptide m/z {mz:.4f} is isobaric with a target"

    def test_high_collision_rate_warns(self, tmp_path, caplog):
        """Using the SAME sequence as both target and entrapment forces ~100% collision."""
        features = _dense_features()
        fasta_path = _write_fasta(tmp_path, _TARGET_SEQ, "P00001", "target.fasta")
        target_db = digest_fasta(fasta_path, generate_decoys=False)
        tgt_cand = match_to_maldi_features(features, target_db, 20.0)
        ent_path = _write_fasta(tmp_path, _TARGET_SEQ, "P00001", "entrap_same.fasta")
        with caplog.at_level(logging.WARNING, logger="msi_picasso.candidates"):
            load_entrapment_candidates(ent_path, tgt_cand, features, matching_ppm=20.0)
        assert any("collision rate" in r.message and "> 10%" in r.message for r in caplog.records)

    def test_feature_mz_is_real_feature(self, tmp_path):
        decoy, _, _, features = self._run(tmp_path)
        assert decoy["feature_idx"].between(0, len(features) - 1).all()
        assert np.allclose(
            decoy["feature_mz"].to_numpy(),
            features[decoy["feature_idx"].to_numpy().astype(int)],
        )


class TestEntrapmentPipelineGuard:

    def test_missing_entrapment_fasta_raises(self, tmp_path):
        """rescore(decoy_method='entrapment') without entrapment_fasta raises ValueError."""
        from msi_picasso import pipeline

        # Stub out everything past the candidate branch by invoking only the guard:
        # the guard fires before any MALDI/LCMS work, so a minimal call must raise.
        with pytest.raises(ValueError, match="entrapment_fasta is required"):
            pipeline.rescore(
                fasta_path=_write_fasta(tmp_path, _TARGET_SEQ, "P00001"),
                mzml_paths=[],
                maldi_mzs=_dense_features(700.0, 1000.0, 0.5),
                digest=True,
                decoy_method="entrapment",
                entrapment_fasta=None,
                output_dir=str(tmp_path / "out"),
            )
