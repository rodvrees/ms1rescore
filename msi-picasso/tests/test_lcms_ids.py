"""Tests for the psm_utils path in parse_lcms_ids."""
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from MSI-PICASSO.lcms_ids import LCMSIds, _PEP_COLS, parse_lcms_ids


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakePeptidoform:
    """Minimal peptidoform stand-in for tests."""

    def __init__(self, sequence: str, charge: int = 2):
        self.sequence = sequence
        self._charge = charge

    def __str__(self) -> str:
        return f"{self.sequence}/{self._charge}"


def _make_psm(
    sequence: str,
    charge: int = 2,
    q_value: float | None = 0.001,
    pep: float | None = 0.005,
    score: float | None = 10.0,
    rt: float | None = 30.0,
    protein_list: list | None = None,
    intensity: float | None = None,
) -> MagicMock:
    psm = MagicMock()
    psm.peptidoform = _FakePeptidoform(sequence, charge)
    psm.qvalue = q_value
    psm.pep = pep
    psm.score = score
    psm.retention_time = rt
    psm.protein_list = protein_list if protein_list is not None else [f"sp|P{sequence[:5]:5}|GENE"]
    psm.get_precursor_charge.return_value = charge
    psm.metadata = {"precursor_intensity": str(intensity)} if intensity is not None else {}
    return psm


# ---------------------------------------------------------------------------
# Shared PSM fixture
# ---------------------------------------------------------------------------

# Two PSMs for the same peptide — q_value min=0.001, score max=15, n_psms=2,
# intensity max=5000, protein_list union includes P99999 from the first PSM.
_PSMS = [
    _make_psm(
        "PEPTIDEK", charge=2, q_value=0.001, score=15.0, rt=20.0,
        protein_list=["sp|P12345|GENE_HUMAN", "sp|P99999|GENE2_HUMAN"],
        intensity=5000.0,
    ),
    _make_psm(
        "PEPTIDEK", charge=2, q_value=0.005, score=12.0, rt=22.0,
        protein_list=["sp|P12345|GENE_HUMAN"],
        intensity=4000.0,
    ),
    _make_psm(
        "ACDEFGHIK", charge=3, q_value=0.008, score=9.0, rt=35.0,
        protein_list=["sp|P67890|OTHER_HUMAN"],
    ),
    # Fails peptide FDR=0.01
    _make_psm(
        "NOPEPFDR", charge=2, q_value=0.05, score=5.0,
        protein_list=["sp|P00000|BAD_HUMAN"],
    ),
    # No q_value
    _make_psm("NOQVAL", charge=1, q_value=None, protein_list=["sp|P11111|NQ_HUMAN"]),
    # No protein
    _make_psm("NOPROT", charge=2, q_value=0.002, protein_list=[]),
]


def _call(psms=_PSMS, fdr=0.01, reader=None):
    with patch("psm_utils.io.read_file", return_value=psms):
        return parse_lcms_ids(
            peptides_path="dummy.tsv",
            peptide_fdr=fdr,
            format="psm_utils",
            psm_utils_reader=reader,
        )


# ---------------------------------------------------------------------------
# Output shape and types
# ---------------------------------------------------------------------------


class TestOutputShape:
    def test_returns_lcmsids(self):
        assert isinstance(_call(), LCMSIds)

    def test_peptide_dataframe_has_required_columns(self):
        result = _call()
        assert list(result.peptides.columns) == _PEP_COLS

    def test_proteins_is_set(self):
        assert isinstance(_call().proteins, set)


# ---------------------------------------------------------------------------
# FDR filtering
# ---------------------------------------------------------------------------


class TestFDRFilter:
    def test_high_qvalue_peptide_excluded(self):
        result = _call(fdr=0.01)
        assert "NOPEPFDR" not in set(result.peptides["sequence"])

    def test_passing_peptides_present(self):
        result = _call(fdr=0.01)
        seqs = set(result.peptides["sequence"])
        assert "PEPTIDEK" in seqs
        assert "ACDEFGHIK" in seqs

    def test_nan_qvalue_psm_excluded_when_others_have_qvalue(self):
        # NOQVAL has q_value=None; with some passing PSMs the FDR filter is applied
        # and NOQVAL is excluded (NaN <= 0.01 is False in pandas)
        result = _call(fdr=0.01)
        assert "NOQVAL" not in set(result.peptides["sequence"])

    def test_empty_after_fdr_raises(self):
        psms = [_make_psm("PEPTIDEK", q_value=0.5)]
        with patch("psm_utils.io.read_file", return_value=psms):
            with pytest.raises(ValueError, match="No peptides remain"):
                parse_lcms_ids(
                    peptides_path="dummy.tsv", peptide_fdr=0.01, format="psm_utils"
                )


# ---------------------------------------------------------------------------
# Aggregation correctness
# ---------------------------------------------------------------------------


class TestAggregation:
    def test_min_q_value(self):
        result = _call()
        row = result.peptides[result.peptides["sequence"] == "PEPTIDEK"].iloc[0]
        assert row["q_value"] == pytest.approx(0.001)

    def test_max_score(self):
        result = _call()
        row = result.peptides[result.peptides["sequence"] == "PEPTIDEK"].iloc[0]
        assert row["score"] == pytest.approx(15.0)

    def test_n_psms_count(self):
        result = _call()
        row = result.peptides[result.peptides["sequence"] == "PEPTIDEK"].iloc[0]
        assert row["n_psms"] == 2

    def test_max_intensity(self):
        result = _call()
        row = result.peptides[result.peptides["sequence"] == "PEPTIDEK"].iloc[0]
        assert row["lcms_intensity"] == pytest.approx(5000.0)

    def test_rt_mean(self):
        result = _call()
        row = result.peptides[result.peptides["sequence"] == "PEPTIDEK"].iloc[0]
        assert row["rt_mean"] == pytest.approx(21.0)  # mean of 20.0 and 22.0

    def test_no_protein_yields_empty_string(self):
        result = _call()
        noprot = result.peptides[result.peptides["sequence"] == "NOPROT"]
        if len(noprot):
            assert noprot.iloc[0]["protein"] == ""

    def test_one_row_per_sequence(self):
        result = _call()
        assert result.peptides["sequence"].is_unique


# ---------------------------------------------------------------------------
# Protein set collection
# ---------------------------------------------------------------------------


class TestProteinSet:
    def test_multi_protein_peptide_all_accessions_collected(self):
        result = _call()
        assert "P12345" in result.proteins
        assert "P99999" in result.proteins

    def test_single_protein_peptide_included(self):
        result = _call()
        assert "P67890" in result.proteins

    def test_failed_fdr_protein_excluded(self):
        result = _call(fdr=0.01)
        assert "P00000" not in result.proteins

    def test_proteins_are_normalised_accessions(self):
        result = _call()
        for acc in result.proteins:
            assert "|" not in acc, f"Un-normalised accession in proteins: {acc!r}"


# ---------------------------------------------------------------------------
# Skipping FDR filter when q_value is unavailable
# ---------------------------------------------------------------------------


class TestNoQValue:
    def test_all_nan_q_value_skips_filter_and_returns_all(self):
        psms = [
            _make_psm("PEPTIDEK", q_value=None),
            _make_psm("ACDEFGHIK", q_value=None),
        ]
        with patch("psm_utils.io.read_file", return_value=psms):
            result = parse_lcms_ids(
                peptides_path="dummy.tsv", format="psm_utils"
            )
        assert len(result.peptides) == 2


# ---------------------------------------------------------------------------
# Reader resolution
# ---------------------------------------------------------------------------


class TestReaderResolution:
    def test_unknown_reader_raises_valueerror(self):
        with patch("psm_utils.io.read_file", return_value=[]):
            with pytest.raises(ValueError, match="Unknown psm_utils reader"):
                parse_lcms_ids(
                    peptides_path="dummy.tsv",
                    format="psm_utils",
                    psm_utils_reader="NonExistentReader123",
                )

    def test_error_message_lists_available_readers(self):
        with patch("psm_utils.io.read_file", return_value=[]):
            with pytest.raises(ValueError, match="tsv"):
                parse_lcms_ids(
                    peptides_path="dummy.tsv",
                    format="psm_utils",
                    psm_utils_reader="NonExistentReader123",
                )

    def test_filetype_key_is_passed_to_read_file(self):
        with patch("psm_utils.io.read_file", return_value=_PSMS[:3]) as mock_read:
            parse_lcms_ids(
                peptides_path="dummy.tsv", format="psm_utils", psm_utils_reader="tsv"
            )
        _, kwargs = mock_read.call_args
        assert kwargs.get("filetype") == "tsv"

    def test_class_name_resolves_to_filetype_key(self):
        with patch("psm_utils.io.read_file", return_value=_PSMS[:3]) as mock_read:
            parse_lcms_ids(
                peptides_path="dummy.tsv",
                format="psm_utils",
                psm_utils_reader="TSVReader",
            )
        _, kwargs = mock_read.call_args
        assert kwargs.get("filetype") == "tsv"

    def test_none_reader_passes_infer(self):
        with patch("psm_utils.io.read_file", return_value=_PSMS[:3]) as mock_read:
            parse_lcms_ids(
                peptides_path="dummy.tsv", format="psm_utils", psm_utils_reader=None
            )
        _, kwargs = mock_read.call_args
        assert kwargs.get("filetype") == "infer"
