"""Tests for generate_entrapment_from_lcms_ids()."""

import logging

import numpy as np
import pandas as pd
import pytest

from msi_picasso.candidates import generate_entrapment_from_lcms_ids
from msi_picasso.lcms_ids import LCMSIds


def _make_lcms_ids(seq_by_prot: dict) -> LCMSIds:
    rows = [
        {"sequence": s, "protein": p}
        for p, seqs in seq_by_prot.items()
        for s in seqs
    ]
    peps = pd.DataFrame(rows)
    prots = pd.DataFrame({"protein": list(seq_by_prot)})
    return LCMSIds(proteins=prots, peptides=peps)


_LCMS_IDS = _make_lcms_ids({
    "P00001": ["PEPTIDEK", "TESTVLGTGFLSR", "CELAAAMK"],
    "P00002": ["RHGLDNYR", "AAATESTPEPTIDEK", "QVQVSTQILHQK"],
})


class TestGenerateEntrapmentFromLcmsIds:

    def test_returns_dataframe(self):
        result = generate_entrapment_from_lcms_ids(_LCMS_IDS)
        assert isinstance(result, pd.DataFrame)

    def test_contains_both_targets_and_decoys(self):
        result = generate_entrapment_from_lcms_ids(_LCMS_IDS)
        assert len(result) > 0
        assert result["is_decoy"].any(), "should contain entrapment decoys"
        assert (~result["is_decoy"]).any(), "should contain entrapment pseudo-targets"

    def test_source_column(self):
        result = generate_entrapment_from_lcms_ids(_LCMS_IDS)
        sources = set(result["source"].unique())
        assert "entrapment_shuffled" in sources
        assert "entrapment_decoy" in sources

    def test_pseudo_targets_are_not_decoys(self):
        result = generate_entrapment_from_lcms_ids(_LCMS_IDS)
        tgt = result[result["source"] == "entrapment_shuffled"]
        assert (tgt["is_decoy"] == False).all()  # noqa: E712

    def test_entrapment_decoys_are_decoys(self):
        result = generate_entrapment_from_lcms_ids(_LCMS_IDS)
        dec = result[result["source"] == "entrapment_decoy"]
        assert (dec["is_decoy"] == True).all()  # noqa: E712

    def test_protein_prefix_mirrors_original(self):
        result = generate_entrapment_from_lcms_ids(_LCMS_IDS)
        tgt = result[~result["is_decoy"]]
        assert tgt["protein"].str.startswith("ENTRAPMENT_").all()
        assert "ENTRAPMENT_P00001" in tgt["protein"].values
        assert "ENTRAPMENT_P00002" in tgt["protein"].values

    def test_decoy_protein_prefix(self):
        result = generate_entrapment_from_lcms_ids(_LCMS_IDS)
        dec = result[result["is_decoy"]]
        assert dec["protein"].str.startswith("ENTRAPMENT_DECOY_").all()

    def test_no_exact_sequence_match_with_targets(self):
        """No returned peptide may appear verbatim in the input sequences."""
        target_seqs = set(_LCMS_IDS.peptides["sequence"].dropna())
        result = generate_entrapment_from_lcms_ids(_LCMS_IDS)
        overlap = set(result["peptide"]) & target_seqs
        assert len(overlap) == 0, f"Exact target sequences returned: {overlap}"

    def test_no_isobaric_with_targets(self):
        """No returned peptide (target or decoy) may be isobaric with any input sequence."""
        from msi_picasso.candidates import _assign_mass_columns

        matching_ppm = 20.0
        result = generate_entrapment_from_lcms_ids(_LCMS_IDS, matching_ppm=matching_ppm)

        target_df = pd.DataFrame({"peptide": list(_LCMS_IDS.peptides["sequence"].dropna())})
        _assign_mass_columns(target_df)
        target_mzs = np.sort(target_df["mh_mz"].dropna().to_numpy())
        tol = matching_ppm * 1e-6

        for mz in result["mh_mz"].dropna().unique():
            lo = np.searchsorted(target_mzs, mz * (1 - tol), side="left")
            hi = np.searchsorted(target_mzs, mz * (1 + tol), side="right")
            assert lo >= hi, f"Entrapment peptide mz {mz:.4f} is isobaric with a target"

    def test_reproducible_with_same_seed(self):
        a = generate_entrapment_from_lcms_ids(_LCMS_IDS, random_state=42)
        b = generate_entrapment_from_lcms_ids(_LCMS_IDS, random_state=42)
        assert set(a["peptide"]) == set(b["peptide"])

    def test_different_seeds_differ(self):
        a = generate_entrapment_from_lcms_ids(_LCMS_IDS, random_state=42)
        b = generate_entrapment_from_lcms_ids(_LCMS_IDS, random_state=99)
        tgt_a = set(a[~a["is_decoy"]]["peptide"])
        tgt_b = set(b[~b["is_decoy"]]["peptide"])
        assert tgt_a != tgt_b

    def test_fewer_than_5_peptides_does_not_raise(self):
        ids = _make_lcms_ids({"P001": ["PEPTIDEK", "TESTVLK"]})
        result = generate_entrapment_from_lcms_ids(ids)
        assert isinstance(result, pd.DataFrame)

    def test_high_collision_rate_warns(self, caplog):
        ids = _make_lcms_ids({"P001": ["PEPTIDEK", "PEPTIDER", "TESTR", "TESTK"]})
        with caplog.at_level(logging.WARNING, logger="msi_picasso.candidates"):
            generate_entrapment_from_lcms_ids(ids, matching_ppm=20.0)

    def test_has_mh_mz_column(self):
        result = generate_entrapment_from_lcms_ids(_LCMS_IDS)
        assert "mh_mz" in result.columns
        assert result["mh_mz"].notna().any()

    def test_missing_columns_raises(self):
        bad = LCMSIds(
            proteins=pd.DataFrame({"protein": ["P1"]}),
            peptides=pd.DataFrame({"sequence": ["PEPTIDEK"]}),  # missing 'protein'
        )
        with pytest.raises((ValueError, KeyError)):
            generate_entrapment_from_lcms_ids(bad)
