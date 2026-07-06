"""Tests for the is_single_peptide_protein indicator (O8).

Single-peptide proteins have no within-protein colocalization partner, so their
protein-colocalization features are undefined (median-imputed) while
log_protein_n_features sits at its floor. The indicator lets a ranker treat this
group separately. It must be a pure structural flag (1.0 iff the protein has one
observed peptide) and symmetric between a protein and its DECOY_ namespace.
"""

import numpy as np
import pandas as pd

from msi_picasso.maldi_features import compute_protein_consistency_features
from msi_picasso.feature_generator import PROTEIN_LEVEL_FEATURES


def _frame():
    # A: 2 peptides (+ DECOY_A: 2), B: 1 peptide singleton (+ DECOY_B: 1), C: 3 peptides
    rows = [
        dict(protein="A", peptide="p1", is_decoy=False, feature_mz=100.0, protein_n_features=2),
        dict(protein="A", peptide="p2", is_decoy=False, feature_mz=101.0, protein_n_features=2),
        dict(protein="DECOY_A", peptide="d1", is_decoy=True, feature_mz=100.5, protein_n_features=2),
        dict(protein="DECOY_A", peptide="d2", is_decoy=True, feature_mz=101.5, protein_n_features=2),
        dict(protein="B", peptide="p3", is_decoy=False, feature_mz=200.0, protein_n_features=1),
        dict(protein="DECOY_B", peptide="d3", is_decoy=True, feature_mz=200.5, protein_n_features=1),
        dict(protein="C", peptide="p4", is_decoy=False, feature_mz=300.0, protein_n_features=3),
        dict(protein="C", peptide="p5", is_decoy=False, feature_mz=301.0, protein_n_features=3),
        dict(protein="C", peptide="p6", is_decoy=False, feature_mz=302.0, protein_n_features=3),
    ]
    return pd.DataFrame(rows)


class TestSinglePeptideProtein:
    def test_flag_is_one_only_for_singletons(self):
        out = compute_protein_consistency_features(_frame())
        flagged = set(out.loc[out["is_single_peptide_protein"] == 1.0, "protein"])
        assert flagged == {"B", "DECOY_B"}
        assert (out.loc[out["protein"].isin(["A", "DECOY_A", "C"]), "is_single_peptide_protein"] == 0.0).all()

    def test_symmetric_target_vs_decoy(self):
        out = compute_protein_consistency_features(_frame())
        # a protein and its DECOY_ namespace share the peptide set -> equal flag
        for base in ["A", "B"]:
            t = out.loc[out["protein"] == base, "is_single_peptide_protein"].iloc[0]
            d = out.loc[out["protein"] == "DECOY_" + base, "is_single_peptide_protein"].iloc[0]
            assert t == d

    def test_is_float_and_binary(self):
        out = compute_protein_consistency_features(_frame())
        vals = out["is_single_peptide_protein"]
        assert vals.dtype == np.float32
        assert set(np.unique(vals.values)) <= {0.0, 1.0}

    def test_registered_in_protein_level_features(self):
        # so it enters the ranker under --use-protein-level-feats
        assert "is_single_peptide_protein" in PROTEIN_LEVEL_FEATURES
