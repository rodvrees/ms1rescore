"""Tests for protein_coverage symmetry (target/decoy label-leak fix).

protein_coverage previously counted distinct *features* in the numerator. Decoys
are placed on exactly one feature per peptide by construction, so decoy
n_features == tryptic_count → coverage was pinned to 1.0 for every decoy, a
perfect target/decoy separator (label leak). The fix counts distinct observed
*peptides* (symmetric: a protein and its DECOY_ namespace share the same peptide
set) over the true full-digest count. See compute_protein_consistency_features.
"""

import numpy as np
import pandas as pd

from msi_picasso.maldi_features import compute_protein_consistency_features


def _target_decoy_frame():
    """Protein A: 3 peptides, each matched to 2 near-isobaric features
    (protein_n_features=6). DECOY_A: the same 3 peptides, one feature each
    (protein_n_features=3). Equal full-digest count (10)."""
    rows = []
    for pep in ["PEPA1", "PEPA2", "PEPA3"]:
        for f in ("_f1", "_f2"):
            rows.append(dict(protein="A", peptide=pep, feature_mz=pep + f,
                             is_decoy=False, protein_n_features=6, protein_tryptic_count=10))
    for pep in ["PEPA1", "PEPA2", "PEPA3"]:
        rows.append(dict(protein="DECOY_A", peptide=pep, feature_mz=pep + "_d",
                         is_decoy=True, protein_n_features=3, protein_tryptic_count=10))
    return pd.DataFrame(rows)


class TestProteinCoverageSymmetry:
    def test_coverage_is_symmetric_target_vs_decoy(self):
        out = compute_protein_consistency_features(_target_decoy_frame())
        cov = out.groupby("is_decoy")["protein_coverage"].mean()
        assert np.isclose(cov[False], cov[True])

    def test_decoys_not_pinned_to_one(self):
        out = compute_protein_consistency_features(_target_decoy_frame())
        dec = out.loc[out["is_decoy"], "protein_coverage"]
        # 3 observed peptides / 10 full-digest = 0.3, not the old leaky 1.0
        assert np.allclose(dec, 0.3)
        assert (dec == 1.0).sum() == 0

    def test_uses_peptides_not_features(self):
        out = compute_protein_consistency_features(_target_decoy_frame())
        # numerator is unique peptides (3), not distinct features (6 for target)
        tgt = out.loc[~out["is_decoy"]]
        assert (tgt["protein_n_peptides"] == 3).all()
        assert np.allclose(tgt["protein_coverage"], 0.3)

    def test_coverage_capped_at_one(self):
        # If observed peptides exceed the (mis-set) denominator, clip to 1.0.
        df = _target_decoy_frame()
        df["protein_tryptic_count"] = 2  # smaller than 3 observed peptides
        out = compute_protein_consistency_features(df)
        assert (out["protein_coverage"] <= 1.0).all()
