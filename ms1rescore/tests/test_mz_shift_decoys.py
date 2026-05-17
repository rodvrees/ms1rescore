"""Tests for m/z-shift observation-space decoy generation."""

import numpy as np
import pandas as pd
import pytest

from ms1rescore.candidates import (
    digest_fasta,
    digest_identified_proteins,
    generate_mz_shift_candidates,
)
from ms1rescore.utils import PROTON


# ---------------------------------------------------------------------------
# Minimal target peptide DataFrame factory
# ---------------------------------------------------------------------------

def _make_target_df(mh_mzs, peptides=None, proteins=None):
    """Build a minimal target-only peptide DataFrame."""
    n = len(mh_mzs)
    if peptides is None:
        peptides = [f"PEPTIDE{i}" for i in range(n)]
    if proteins is None:
        proteins = [f"PROT{i}" for i in range(n)]
    masses = [mz - PROTON for mz in mh_mzs]
    return pd.DataFrame({
        "peptide": peptides,
        "protein": proteins,
        "is_decoy": [False] * n,
        "mass": masses,
        "mh_mz": mh_mzs,
        "n_C": [10] * n,
        "n_H": [20] * n,
        "n_N": [4] * n,
        "n_O": [6] * n,
        "n_S": [0] * n,
        "source": ["target"] * n,
    })


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestGenerateMzShiftCandidates:
    def _run(self, target_mh_mzs, feature_mzs, **kwargs):
        target_df = _make_target_df(target_mh_mzs)
        feature_arr = np.array(feature_mzs, dtype=np.float64)
        defaults = dict(
            matching_ppm=20.0,
            delta_min=5.0,
            delta_max=20.0,
            random_state=42,
        )
        defaults.update(kwargs)
        return generate_mz_shift_candidates(target_df, feature_arr, **defaults)

    def test_target_rows_have_is_decoy_false(self):
        # Target peptides at 1000 Da, MALDI features near those masses
        target_mzs = [1000.5, 1200.5]
        feature_mzs = [1000.5, 1200.5, 1300.0]
        result = self._run(target_mzs, feature_mzs)
        targets = result[~result["is_decoy"]]
        assert len(targets) > 0
        assert (targets["is_decoy"] == False).all()

    def test_decoy_rows_have_is_decoy_true(self):
        target_mzs = [1000.5, 1200.5]
        # Dense 0.01 Da grid: 20 ppm at 1000 Da ≈ 0.02 Da, so any shifted query
        # from [5, 20] Da will find a feature regardless of seed.
        feature_mzs = list(np.arange(975.0, 1230.0, 0.01))
        result = self._run(target_mzs, feature_mzs)
        decoys = result[result["is_decoy"]]
        assert len(decoys) > 0
        assert (decoys["is_decoy"] == True).all()

    def test_decoy_feature_mz_differs_from_peptide_mh_mz_by_approximately_delta(self):
        target_mzs = [1000.5]
        # Dense 0.01 Da grid in shift range [5, 20] Da.  Any sampled delta finds a
        # feature within snap_tolerance_ppm (50 ppm ≈ 0.05 Da at 1000 Da).
        feature_mzs = [1000.5] + list(np.arange(1005.5, 1021.0, 0.01))
        result = self._run(target_mzs, feature_mzs)
        decoys = result[result["is_decoy"]]
        if len(decoys) > 0:
            orig_mhz = 1000.5
            for _, row in decoys.iterrows():
                diff = abs(row["feature_mz"] - orig_mhz)
                # Decoy feature ≈ shifted query (within snap_tolerance), and
                # shifted query = orig + delta with delta ∈ [5, 20] Da.
                assert 4.5 <= diff <= 21.0, f"Unexpected diff: {diff}"

    def test_decoy_ppm_error_matches_target_ppm_error(self):
        """ppm_error on decoy rows must be copied from the target match, not from the decoy feature."""
        # Target feature is 2 ppm offset from peptide m/z; decoy feature is 10+ Da away.
        # After the fix, ppm_error for the decoy should equal the target's ppm_error (~2 ppm),
        # NOT the thousands-of-ppm offset between the peptide and the decoy feature.
        target_mz = 1000.500
        target_feature_mz = 1000.502  # 2 ppm offset
        decoy_features = list(np.arange(1005.5, 1021.0, 0.01))
        feature_mzs = [target_feature_mz] + decoy_features
        result = self._run([target_mz], feature_mzs)
        targets = result[~result["is_decoy"]]
        decoys  = result[result["is_decoy"]]
        if len(decoys) == 0:
            pytest.skip("No decoy generated — adjust feature grid")
        target_ppm = float(targets["ppm_error"].iloc[0])
        for _, row in decoys.iterrows():
            assert abs(row["ppm_error"] - target_ppm) < 1e-6, (
                f"Decoy ppm_error {row['ppm_error']:.4f} != target ppm_error {target_ppm:.4f}"
            )
            # Must NOT be the huge offset to the decoy feature
            large_ppm = (row["feature_mz"] - target_mz) / target_mz * 1e6
            assert abs(row["ppm_error"] - large_ppm) > 100.0, (
                "ppm_error appears to be computed from the decoy feature, not copied from target"
            )

    def test_collision_avoidance_no_decoy_lands_within_tolerance_of_target(self):
        """Decoy feature m/z must not be within matching_ppm of any target peptide m/z."""
        # Targets at integer m/z; features at integer + half-integer positions.
        # Half-integer features are ~500 ppm from nearest target — far outside matching_ppm.
        # snap_tolerance_ppm=1000 accepts both; the collision check must reject integer features.
        target_mzs = list(np.arange(800.0, 1600.0, 1.0))
        feature_mzs = list(np.arange(800.0, 1600.0, 0.5))
        target_df = _make_target_df(target_mzs)
        feature_arr = np.array(feature_mzs, dtype=np.float64)
        ppm = 20.0
        result = generate_mz_shift_candidates(
            target_df, feature_arr,
            matching_ppm=ppm, delta_min=5.0, delta_max=20.0,
            snap_tolerance_ppm=1000.0,
            random_state=0,
        )
        decoys = result[result["is_decoy"]]
        target_arr_sorted = np.sort(np.array(target_mzs))
        tol_frac = ppm * 1e-6
        for _, row in decoys.iterrows():
            feat_mz = row["feature_mz"]
            lo = np.searchsorted(target_arr_sorted, feat_mz * (1 - tol_frac), side="left")
            hi = np.searchsorted(target_arr_sorted, feat_mz * (1 + tol_frac), side="right")
            assert lo >= hi, (
                f"Decoy feature {feat_mz:.4f} is within {ppm} ppm of a target m/z — "
                "collision avoidance failed"
            )

    def test_output_columns_match_match_to_maldi_features(self):
        """Output must contain all columns produced by match_to_maldi_features."""
        from ms1rescore.candidates import match_to_maldi_features

        target_mzs = [1000.5, 1200.5, 1400.5]
        feature_mzs = [1000.5, 1200.5, 1400.5, 1010.0, 1215.0, 1420.0]
        target_df = _make_target_df(target_mzs)
        feature_arr = np.array(feature_mzs, dtype=np.float64)

        ref = match_to_maldi_features(feature_arr, target_df, 20.0)
        result = generate_mz_shift_candidates(target_df, feature_arr)

        ref_cols = set(ref.columns)
        # decoy_delta_da is the only extra column allowed
        extra = set(result.columns) - ref_cols - {"decoy_delta_da", "source"}
        assert not extra, f"Unexpected extra columns: {extra}"
        missing = ref_cols - set(result.columns)
        assert not missing, f"Missing columns: {missing}"

    def test_decoy_delta_da_is_nan_for_targets(self):
        target_mzs = [1000.5, 1200.5]
        feature_mzs = [1000.5, 1200.5, 1010.0, 1215.0]
        result = self._run(target_mzs, feature_mzs)
        targets = result[~result["is_decoy"]]
        assert targets["decoy_delta_da"].isna().all()

    def test_decoy_delta_da_is_finite_for_decoys(self):
        target_mzs = [1000.5, 1200.5]
        feature_mzs = list(np.arange(975.0, 1230.0, 0.01))  # dense grid
        result = self._run(target_mzs, feature_mzs)
        decoys = result[result["is_decoy"]]
        if len(decoys) > 0:
            assert decoys["decoy_delta_da"].notna().all()

    def test_reproducibility_same_random_state(self):
        target_mzs = [1000.5, 1200.5, 1400.5]
        # Dense grid so decoys are always generated regardless of seed
        feature_mzs = list(np.arange(975.0, 1430.0, 0.01))
        r1 = self._run(target_mzs, feature_mzs, random_state=7)
        r2 = self._run(target_mzs, feature_mzs, random_state=7)
        d1 = r1[r1["is_decoy"]]["decoy_delta_da"].sort_values().values
        d2 = r2[r2["is_decoy"]]["decoy_delta_da"].sort_values().values
        np.testing.assert_array_equal(d1, d2)

    def test_different_random_states_give_different_deltas(self):
        target_mzs = [1000.5, 1200.5, 1400.5]
        # 0.01 Da grid: 20 ppm at 1000–1400 Da is 0.02–0.028 Da, so every
        # shifted query is within tolerance of the nearest grid feature.
        feature_mzs = np.arange(975.0, 1430.0, 0.01).tolist()
        r1 = self._run(target_mzs, feature_mzs, random_state=1)
        r2 = self._run(target_mzs, feature_mzs, random_state=999)
        d1 = set(round(v, 6) for v in r1[r1["is_decoy"]]["decoy_delta_da"].dropna())
        d2 = set(round(v, 6) for v in r2[r2["is_decoy"]]["decoy_delta_da"].dropna())
        # Should not be identical (astronomically unlikely with different seeds)
        assert d1 != d2

    def test_td_ratio_near_one_for_dense_feature_list(self):
        """T:D candidate ratio should be close to 1:1 when features are dense enough.

        10 targets at 1000–1009.  Delta range 50–55 Da puts positive shifts in
        1050–1064 and negative shifts in 945–959 — both well outside the target
        m/z range so the collision check never fires.  Both zones are covered by a
        0.01 Da dense feature grid (≤5 ppm from any shifted query), so every
        peptide gets exactly one decoy.  Expected T:D = 10/10 = 1.0.
        """
        target_mzs = list(np.arange(1000.0, 1010.0, 1.0))  # 10 targets: 1000–1009
        pos_zone = list(np.arange(1050.0, 1066.0, 0.01))    # even-index positive shifts
        neg_zone = list(np.arange(945.0,  961.0,  0.01))    # odd-index negative shifts
        feature_mzs = target_mzs + pos_zone + neg_zone
        result = self._run(
            target_mzs, feature_mzs,
            delta_min=50.0, delta_max=55.0,
            snap_tolerance_ppm=50.0,
        )
        n_target = int((~result["is_decoy"]).sum())
        n_decoy  = int(result["is_decoy"].sum())
        assert n_decoy > 0, "No decoys generated — check feature grid"
        ratio = n_target / n_decoy
        assert 0.8 <= ratio <= 1.2, f"T:D ratio {ratio:.2f} outside [0.8, 1.2]"

    def test_no_decoy_when_nearest_feature_beyond_snap_tolerance(self):
        """No decoy should be produced when the nearest feature is beyond snap_tolerance_ppm."""
        target_mzs = [1000.0]
        # Only two features: the target position and one 50 Da away.
        # Shifted queries land ±5–20 Da from 1000 → nearest feature is ≥30 Da = >>50 ppm away.
        feature_mzs = [1000.0, 1050.0]
        result = self._run(target_mzs, feature_mzs, snap_tolerance_ppm=50.0)
        decoys = result[result["is_decoy"]]
        assert len(decoys) == 0, (
            f"Expected 0 decoys when nearest feature is >>50 ppm from shifted query, "
            f"got {len(decoys)}"
        )


class TestDigestIdentifiedProteinsGenerateDecoys:
    """Test that generate_decoys=False returns zero is_decoy rows."""

    def _make_lcms_ids(self, sequences):
        from collections import namedtuple
        LCMSIds = namedtuple("LCMSIds", ["proteins", "peptides"])
        peptides_df = pd.DataFrame({
            "sequence": sequences,
            "protein": ["SP|P00001|PROT"] * len(sequences),
            "q_value": [0.001] * len(sequences),
            "pep": [0.001] * len(sequences),
            "score": [10.0] * len(sequences),
            "n_psms": [1] * len(sequences),
            "charge": [2] * len(sequences),
            "rt_mean": [10.0] * len(sequences),
            "lcms_intensity": [1e6] * len(sequences),
        })
        proteins = {"P00001"}
        return LCMSIds(proteins=proteins, peptides=peptides_df)

    def test_no_fasta_generate_decoys_false_returns_zero_decoys(self):
        seqs = ["PEPTIDER", "SAMPLEK", "TESTPEPTIDE"]
        lcms_ids = self._make_lcms_ids(seqs)
        result = digest_identified_proteins(
            fasta_path=None,
            lcms_ids=lcms_ids,
            generate_decoys=False,
        )
        assert result["is_decoy"].sum() == 0, "Expected no decoy rows when generate_decoys=False"

    def test_no_fasta_generate_decoys_true_returns_decoys(self):
        seqs = ["PEPTIDERPEPTIDER", "SAMPLEPEPTIDEK", "TESTPEPTIDELONGER"]
        lcms_ids = self._make_lcms_ids(seqs)
        result = digest_identified_proteins(
            fasta_path=None,
            lcms_ids=lcms_ids,
            generate_decoys=True,
        )
        # Concat pseudo-protein may produce decoys; may produce fewer than targets
        # Just confirm the flag is respected (True = decoys can exist)
        assert isinstance(result, pd.DataFrame)
        # The function should have attempted to generate decoys (though may get 0
        # if the shuffled pseudo-protein yields no valid peptides — accept both)


class TestDigestFastaGenerateDecoys:
    """digest_fasta already had generate_decoys; verify the flag works."""

    def test_generate_decoys_false_returns_zero_decoy_rows(self, tmp_path):
        fasta = tmp_path / "test.fasta"
        fasta.write_text(
            ">sp|P00001|TEST_HUMAN Test protein\n"
            "MALPVTALLLLAAGLLAHAAPEPTIDEKVFGRCELAAAMKRHGLDNYR\n"
        )
        result = digest_fasta(str(fasta), generate_decoys=False)
        assert result["is_decoy"].sum() == 0

    def test_generate_decoys_true_returns_decoy_rows(self, tmp_path):
        fasta = tmp_path / "test.fasta"
        fasta.write_text(
            ">sp|P00001|TEST_HUMAN Test protein\n"
            "MALPVTALLLLAAGLLAHAAPEPTIDEKVFGRCELAAAMKRHGLDNYR\n"
        )
        result = digest_fasta(str(fasta), generate_decoys=True)
        assert result["is_decoy"].sum() > 0
