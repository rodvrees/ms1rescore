"""
MS1Rescore feature generator: orchestrates all feature categories.

Converts a candidate DataFrame (from match_to_maldi_features) into a PSMList
with all rescoring features populated.
"""

import logging

import numpy as np
import pandas as pd
from psm_utils import PSM, PSMList, Peptidoform

from ms1rescore.maldi_features import (
    compute_candidate_ambiguity_features,
    compute_colocalization_features,
    compute_envelope_similarity,
    compute_maldi_ionization_features,
    compute_maldi_signal_features,
    compute_mass_accuracy_features,
    compute_peptide_properties,
    compute_protein_consistency_features,
    compute_spatial_features,
    compute_theoretical_isotope_features,
)

logger = logging.getLogger(__name__)

# Feature name groups
_MASS_ACCURACY = ["ppm_error_abs", "ppm_rank", "ppm_best_ratio"]
_AMBIGUITY = ["n_candidates", "log_n_candidates"]
_PROTEIN = [
    "protein_n_features", "log_protein_n_features", "protein_coverage",
    "protein_rank", "protein_best_ratio",
]
_PEPTIDE = ["peptide_length", "n_missed_cleavages", "has_modifications"]
_MALDI_SIGNAL = ["log_maldi_intensity"]
_LCMS_EVIDENCE = [
    "lcms_ms2_spectral_angle", "lcms_ms2_n_matches",
    "lcms_xic_max_intensity", "lcms_xic_n_scans", "lcms_xic_snr",
    "lcms_xic_best_charge",
    "lcms_rt_residual", "lcms_ms1_isotope_cosine",
]
_THEO_ISOTOPE = [
    "theo_isotope_cosine", "theo_isotope_chi2", "theo_isotope_kl",
    "theo_has_sulfur", "averagine_deviation", "averagine_deviation_sulfur",
    "theo_m1_ratio_diff", "theo_m2_ratio_diff",
    "theo_m1_ratio_diff_lcms", "theo_m2_ratio_diff_lcms",
]
_IONIZATION = [
    "n_arginine", "n_basic_residues", "n_phenylalanine",
    "n_aromatic", "gravy_score", "charge_proxy",
]
_SPATIAL = [
    "spatial_autocorrelation", "fraction_detected", "intensity_cv",
    "log_mean_intensity", "spatial_entropy",
]
_COLOCALIZATION = [
    "protein_colocalization", "protein_colocalization_max",
    "protein_colocalization_median", "protein_colocalization_n_partners",
]
_ENVELOPE = [
    "isotope_envelope_cosine", "isotope_envelope_pearson", "isotope_envelope_mse",
    "isotope_m1_ratio_diff", "isotope_m2_ratio_diff", "isotope_n_matched",
]


def get_feature_names(
    has_spatial: bool = False,
    has_ion_images: bool = False,
    has_envelopes: bool = False,
) -> list[str]:
    """Return the list of feature names based on available data."""
    names = (
        _MASS_ACCURACY + _AMBIGUITY + _PROTEIN + _PEPTIDE + _MALDI_SIGNAL
        + _LCMS_EVIDENCE + _THEO_ISOTOPE + _IONIZATION
    )
    if has_spatial:
        names += _SPATIAL
    if has_ion_images:
        names += _COLOCALIZATION
    if has_envelopes:
        names += _ENVELOPE
    return names


def candidates_to_psm_list(candidates_df: pd.DataFrame) -> PSMList:
    """Convert a candidate DataFrame to a PSMList for mokapot."""
    meta_cols = [
        c for c in candidates_df.columns if c not in ("peptide", "protein", "is_decoy")
    ]
    psms = []
    for row in candidates_df.itertuples(index=False):
        peptide = row.peptide
        psm = PSM(
            peptidoform=Peptidoform(f"{peptide}/1"),
            spectrum_id=f"maldi_feature_{getattr(row, 'feature_idx', 0)}",
            run="maldi",
            is_decoy=bool(row.is_decoy),
            protein_list=[str(getattr(row, "protein", ""))],
            precursor_mz=float(getattr(row, "feature_mz", getattr(row, "mh_mz", 0))),
            score=float(-getattr(row, "ppm_error_abs", 0)),
            metadata={c: getattr(row, c) for c in meta_cols},
        )
        psms.append(psm)

    psm_list = PSMList(psm_list=psms)
    n_target = sum(not p.is_decoy for p in psm_list)
    n_decoy = sum(p.is_decoy for p in psm_list)
    logger.info(f"Built PSMList: {n_target} target + {n_decoy} decoy PSMs")
    return psm_list


def compute_all_features(
    candidates_df: pd.DataFrame,
    lcms_evidence: dict[int, dict[str, float]] | None = None,
    spatial_features: pd.DataFrame | None = None,
    ion_images: np.ndarray | None = None,
    ion_image_mzs: np.ndarray | None = None,
    maldi_envelopes: dict | None = None,
    lcms_envelopes: dict | None = None,
) -> pd.DataFrame:
    """
    Compute all features on the candidate DataFrame.

    Returns the DataFrame with all feature columns added.
    """
    df = candidates_df.copy()

    # Mass accuracy
    df = compute_mass_accuracy_features(df)
    df = compute_candidate_ambiguity_features(df)
    df = compute_protein_consistency_features(df)
    df = compute_peptide_properties(df)
    df = compute_maldi_signal_features(df)

    # LC-MS/MS evidence (pre-computed)
    if lcms_evidence is not None:
        for feat in _LCMS_EVIDENCE:
            df[feat] = df.index.map(
                lambda idx: lcms_evidence.get(idx, {}).get(feat, 0.0)
            )
        # Fill NaN for rt_residual and isotope_cosine
        for feat in ["lcms_rt_residual", "lcms_ms1_isotope_cosine"]:
            valid = df[feat].dropna()
            fill = valid.median() if len(valid) > 0 else 0.0
            df[feat] = df[feat].fillna(fill)
    else:
        for feat in _LCMS_EVIDENCE:
            df[feat] = 0.0

    # Theoretical isotope
    df = compute_theoretical_isotope_features(
        df,
        maldi_envelopes=maldi_envelopes,
        lcms_envelopes=lcms_envelopes,
    )

    # MALDI ionization
    df = compute_maldi_ionization_features(df)

    # Spatial (optional)
    if spatial_features is not None:
        df = compute_spatial_features(df, spatial_features)

    # Co-localization (optional)
    if ion_images is not None and ion_image_mzs is not None:
        df = compute_colocalization_features(df, ion_images, ion_image_mzs)

    # Envelope similarity MALDI vs LC-MS/MS (optional)
    if maldi_envelopes is not None and lcms_envelopes is not None:
        df = compute_envelope_similarity(df, maldi_envelopes, lcms_envelopes)

    return df


def populate_psm_features(
    psm_list: PSMList,
    features_df: pd.DataFrame,
    feature_names: list[str],
) -> None:
    """Write feature values from DataFrame into PSMList rescoring_features."""
    for i, psm in enumerate(psm_list):
        psm.rescoring_features = {
            feat: float(features_df.iloc[i].get(feat, 0.0))
            for feat in feature_names
        }
