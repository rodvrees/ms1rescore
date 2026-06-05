"""Tests for the concatenated pseudo-protein decoy strategy in LC-only mode."""
import pandas as pd
import pytest
from pyteomics import mass

from msi_picasso.candidates import digest_identified_proteins, _shuffle_protein
from msi_picasso.lcms_ids import LCMSIds, _PEP_COLS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_lcms_ids(sequences: list[str], protein: str = "P12345") -> LCMSIds:
    """Build a minimal LCMSIds namedtuple from a list of peptide sequences."""
    rows = []
    for seq in sequences:
        rows.append({
            "sequence": seq,
            "peptidoform": f"{seq}/2",
            "protein": protein,
            "q_value": 0.001,
            "pep": 0.01,
            "score": 10.0,
            "n_psms": 1,
            "charge": 2,
            "rt_mean": 20.0,
            "lcms_intensity": 1000.0,
            "lcms_ccs": float("nan"),
        })
    pep_df = pd.DataFrame(rows, columns=_PEP_COLS)
    proteins = {protein}
    return LCMSIds(proteins=proteins, peptides=pep_df)


# A set of tryptic peptides large enough to produce a meaningful pseudo-protein.
# All end in K or R (proper tryptic termini).
_TARGET_SEQS = [
    "PEPTIDEK",
    "ACDEFGHIK",
    "LMNPQSTVR",
    "YWACDEFGR",
    "GHIKLMNPK",
    "QRSTVWACR",
    "DEFGHIKLR",
    "MNPQRSTVK",
    "ACGHILMNR",
    "PQSTVWYAK",
]


@pytest.fixture(scope="module")
def lc_only_result():
    """Run digest_identified_proteins in LC-only mode (fasta_path=None)."""
    lcms_ids = _make_lcms_ids(_TARGET_SEQS)
    return digest_identified_proteins(
        fasta_path=None,
        lcms_ids=lcms_ids,
        missed_cleavages=1,
        min_length=6,
        max_length=30,
    )


# ---------------------------------------------------------------------------
# Basic schema
# ---------------------------------------------------------------------------

class TestSchema:
    def test_has_required_columns(self, lc_only_result):
        required = {"peptide", "protein", "is_decoy", "mass", "mh_mz",
                    "n_C", "n_H", "n_N", "n_O", "n_S", "source"}
        assert required.issubset(set(lc_only_result.columns))

    def test_targets_are_confirmed_source(self, lc_only_result):
        targets = lc_only_result[~lc_only_result["is_decoy"]]
        assert (targets["source"] == "lcms_confirmed").all()

    def test_decoys_have_decoy_source(self, lc_only_result):
        decoys = lc_only_result[lc_only_result["is_decoy"]]
        assert (decoys["source"] == "decoy").all()

    def test_decoy_protein_is_concat(self, lc_only_result):
        decoys = lc_only_result[lc_only_result["is_decoy"]]
        assert (decoys["protein"] == "DECOY_concat").all()

    def test_all_targets_present(self, lc_only_result):
        targets = set(lc_only_result.loc[~lc_only_result["is_decoy"], "peptide"])
        assert set(_TARGET_SEQS).issubset(targets)


# ---------------------------------------------------------------------------
# Correctness: no overlap between targets and decoys
# ---------------------------------------------------------------------------

class TestNoOverlap:
    def test_no_decoy_matches_any_target(self, lc_only_result):
        targets = set(lc_only_result.loc[~lc_only_result["is_decoy"], "peptide"])
        decoys = set(lc_only_result.loc[lc_only_result["is_decoy"], "peptide"])
        overlap = targets & decoys
        assert len(overlap) == 0, f"Overlap between targets and decoys: {overlap}"


# ---------------------------------------------------------------------------
# Tryptic validity: decoys end in K or R (except possibly the last C-terminal
# peptide of the pseudo-protein, which may not have a tryptic terminus)
# ---------------------------------------------------------------------------

class TestTrypticTermini:
    def test_majority_of_decoys_end_in_K_or_R(self, lc_only_result):
        decoys = lc_only_result.loc[lc_only_result["is_decoy"], "peptide"].tolist()
        if not decoys:
            pytest.skip("No decoys generated")
        kr_terminated = sum(1 for p in decoys if p[-1] in "KR")
        fraction = kr_terminated / len(decoys)
        # At least 80% should be K/R-terminated; the final pseudo-protein
        # fragment may not be.
        assert fraction >= 0.80, (
            f"Only {fraction:.0%} of decoys end in K or R — "
            f"expected ≥80% for a tryptic pseudo-protein digest"
        )


# ---------------------------------------------------------------------------
# Composition: decoys are NOT isobaric with targets
# ---------------------------------------------------------------------------

class TestCompositionDiversity:
    def _composition(self, seq: str) -> dict:
        try:
            return dict(mass.Composition(sequence=seq))
        except Exception:
            return {}

    def test_decoy_compositions_differ_from_targets(self, lc_only_result):
        """
        Verify the concatenated-protein strategy breaks elemental composition
        conservation. With per-peptide shuffle the composition is identical to
        the target; with pseudo-protein digestion it changes for the majority.
        """
        targets = lc_only_result[~lc_only_result["is_decoy"]]
        decoys = lc_only_result[lc_only_result["is_decoy"]]

        if len(decoys) == 0:
            pytest.skip("No decoys generated")

        target_comps = {
            row["peptide"]: self._composition(row["peptide"])
            for _, row in targets.iterrows()
        }
        decoy_seqs = decoys["peptide"].tolist()

        # Build the set of target composition tuples (frozen for comparison)
        target_comp_set = {
            frozenset(c.items()) for c in target_comps.values()
        }

        isobaric_count = sum(
            1 for dec in decoy_seqs
            if frozenset(self._composition(dec).items()) in target_comp_set
        )
        isobaric_fraction = isobaric_count / len(decoy_seqs)

        # Allow up to 20% coincidental isobaric matches (short peptides can
        # share composition by chance), but the majority must differ.
        assert isobaric_fraction <= 0.20, (
            f"{isobaric_fraction:.0%} of decoys are isobaric with some target — "
            f"expected ≤20% (per-peptide shuffle would give 100%)"
        )

    def test_per_peptide_shuffle_would_be_isobaric(self):
        """
        Regression guard: confirm the OLD per-peptide approach IS isobaric,
        so we know our test above is actually measuring the right thing.
        """
        seq = "PEPTIDEK"
        dec = _shuffle_protein(seq, random_state=42)
        if dec == seq:
            pytest.skip("Shuffle returned same sequence — trivial case")
        comp_target = mass.Composition(sequence=seq)
        comp_decoy = mass.Composition(sequence=dec)
        assert dict(comp_target) == dict(comp_decoy), (
            "Per-peptide shuffle should be isobaric but wasn't — "
            "test assumption violated"
        )
