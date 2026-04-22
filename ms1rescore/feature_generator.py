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
    compute_adduct_colocalization,
    compute_calibrated_ppm_features,
    compute_candidate_ambiguity_features,
    compute_chca_cluster_features,
    compute_colocalization_features,
    compute_envelope_similarity,
    compute_im2deep_features,
    compute_isotopologue_colocalization,
    compute_maldi_ionization_features,
    compute_maldi_signal_features,
    compute_mass_accuracy_features,
    compute_mass_defect_features,
    compute_peptide_properties,
    compute_peptide_property_features,
    compute_protein_consistency_features,
    compute_spatial_autocorrelation_full,
    compute_spatial_features,
    compute_theoretical_isotope_features,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature groups
# ---------------------------------------------------------------------------

# MALDI-intrinsic features: computed entirely from MALDI data and in-silico
# properties. These are the sole input to the ranker/SVM so that the model
# scores MALDI match quality, not LC-MS/MS identification quality.
#
# Optional features are included in the list but filtered out by
# get_feature_names() when the required data was not provided.
MALDI_INTRINSIC_FEATURES = [
    # --- mass accuracy (A-group) ---
    "ppm_error_abs", "ppm_rank", "ppm_best_ratio",
    "ppm_error_calibrated_z",        # A3  — optional, requires pixel_coords
    # --- ambiguity ---
    "n_candidates", "log_n_candidates",
    # --- protein consistency ---
    "protein_n_features", "log_protein_n_features", "protein_coverage",
    "protein_rank", "protein_best_ratio",
    # --- peptide properties (basic) ---
    "peptide_length", "n_missed_cleavages", "has_modifications",
    # --- peptide properties (extended, C-group) ---
    "nterm_basic",                   # C2
    "peptide_pi",                    # C8
    "has_oxidized_met", "has_cys", "n_proline", "nterm_pyroglu_risk",  # C9
    "acidic_residue_density",        # C12
    "n_tryptophan", "n_tyrosine",    # C15
    # --- MALDI signal ---
    "log_maldi_intensity_p90", "log_maldi_intensity_sum",
    "log_maldi_intensity",     # backwards-compatible alias for log_maldi_intensity_p90
    # --- mass defect features (A-group) ---
    "kendrick_mass_defect",          # A10 — computed in match_to_maldi_features
    "mass_defect_residual",          # A11
    # --- CHCA matrix interference ---
    "chca_cluster_distance_ppm",     # A12
    # --- theoretical isotope ---
    "theo_isotope_cosine", "theo_isotope_chi2", "theo_isotope_kl",
    "theo_has_sulfur", "averagine_deviation", "averagine_deviation_sulfur",
    "theo_m1_ratio_diff", "theo_m2_ratio_diff",
    "monoisotopic_confidence",       # A8
    # --- ionization priors ---
    "n_arginine", "n_basic_residues", "n_phenylalanine", "n_aromatic",
    "gravy_score", "charge_proxy",
    # --- ion mobility (B-group) — optional, requires im2deep + observed CCS ---
    "im2deep_delta_ccs", "im2deep_abs_delta_ccs_pct",
    "im2deep_ccs_zscore", "im2deep_ccs_rank", "im2deep_mahalanobis",
    # --- spatial (optional — requires spatial_features DataFrame) ---
    "spatial_autocorrelation", "fraction_detected", "intensity_cv",
    "log_mean_intensity", "spatial_entropy",
    # --- full spatial autocorrelation (E5/E6) — optional, requires ion_images ---
    "spatial_morans_i", "spatial_gearys_c",
    # --- protein co-localization — optional, requires ion_images ---
    "protein_colocalization", "protein_colocalization_max",
    "protein_colocalization_median", "protein_colocalization_n_partners",
    # --- isotopologue co-localization (E1) — optional, requires ion_images ---
    "isotope_image_colocalization_m1", "isotope_image_colocalization_m2",
    "isotope_image_colocalization_mean",
    # --- adduct co-localization (E2) — optional, requires ion_images ---
    "adduct_colocalization_na", "adduct_colocalization_k", "adduct_colocalization_chca",
    # --- observed isotope envelope similarity MALDI vs LC-MS/MS — optional ---
    "isotope_envelope_cosine", "isotope_envelope_pearson",
    "isotope_envelope_mse", "isotope_m1_ratio_diff", "isotope_m2_ratio_diff",
    "isotope_n_matched",
]

# LC-MS/MS prior features: NOT passed to the ranker/SVM — doing so would cause
# the model to score LC-MS/MS identification quality rather than MALDI match
# quality. Applied as a multiplicative Bayesian prior after MALDI-intrinsic
# scoring (see pipeline.compute_lcms_prior).
#
# Split into two sub-groups so compute_all_features can handle them differently:
#   _LCMS_MZML_FEATURES  — derived from raw mzML (always populated by lcms_evidence)
#   _LCMS_ID_FEATURES    — derived from LC-MS/MS IDs (Strategy C, via lcms_ids.py)

_LCMS_MZML_FEATURES = [
    "lcms_ms2_spectral_angle", "lcms_ms2_n_matches",
    "lcms_xic_max_intensity", "lcms_xic_n_scans", "lcms_xic_snr",
    "lcms_xic_best_charge", "lcms_rt_residual",
    "lcms_ms1_isotope_cosine",
    "theo_m1_ratio_diff_lcms", "theo_m2_ratio_diff_lcms",
]

_LCMS_ID_FEATURES = [
    "lcms_q_value", "lcms_pep", "lcms_score",
    "n_psms", "lcms_intensity", "source_lcms_confirmed",
]

LCMS_PRIOR_FEATURES = _LCMS_MZML_FEATURES + _LCMS_ID_FEATURES

# ---------------------------------------------------------------------------
# Optional-feature membership sets (used by get_feature_names)
# ---------------------------------------------------------------------------
_SPATIAL_FEATS = frozenset([
    "spatial_autocorrelation", "fraction_detected", "intensity_cv",
    "log_mean_intensity", "spatial_entropy",
])
_COLOC_FEATS = frozenset([
    "protein_colocalization", "protein_colocalization_max",
    "protein_colocalization_median", "protein_colocalization_n_partners",
])
_ENVELOPE_FEATS = frozenset([
    "isotope_envelope_cosine", "isotope_envelope_pearson",
    "isotope_envelope_mse", "isotope_m1_ratio_diff", "isotope_m2_ratio_diff",
    "isotope_n_matched",
])
_PIXEL_FEATS = frozenset(["ppm_error_calibrated_z"])
_CCS_FEATS = frozenset([
    "im2deep_delta_ccs", "im2deep_abs_delta_ccs_pct",
    "im2deep_ccs_zscore", "im2deep_ccs_rank", "im2deep_mahalanobis",
])
_ISOTOPOLOGUE_COLOC_FEATS = frozenset([
    "isotope_image_colocalization_m1", "isotope_image_colocalization_m2",
    "isotope_image_colocalization_mean",
])
_ADDUCT_COLOC_FEATS = frozenset([
    "adduct_colocalization_na", "adduct_colocalization_k", "adduct_colocalization_chca",
])
_MORANS_FEATS = frozenset(["spatial_morans_i", "spatial_gearys_c"])


def get_feature_names(
    has_spatial: bool = False,
    has_ion_images: bool = False,
    has_envelopes: bool = False,
    has_pixel_coords: bool = False,
    has_ccs: bool = False,
) -> list[str]:
    """Return the full list of feature names based on available data.

    Optional feature groups are included only when the corresponding data
    was computed. ``MALDI_INTRINSIC_FEATURES + LCMS_PRIOR_FEATURES`` is the
    superset; this function selects the applicable subset.
    """
    intrinsic = [
        f for f in MALDI_INTRINSIC_FEATURES
        if (f not in _SPATIAL_FEATS or has_spatial)
        and (f not in _COLOC_FEATS or has_ion_images)
        and (f not in _ISOTOPOLOGUE_COLOC_FEATS or has_ion_images)
        and (f not in _ADDUCT_COLOC_FEATS or has_ion_images)
        and (f not in _MORANS_FEATS or has_ion_images)
        and (f not in _ENVELOPE_FEATS or has_envelopes)
        and (f not in _PIXEL_FEATS or has_pixel_coords)
        and (f not in _CCS_FEATS or has_ccs)
    ]
    return intrinsic + LCMS_PRIOR_FEATURES


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
    pixel_coords: np.ndarray | None = None,
    maldi_mzs: np.ndarray | None = None,
    observed_ccs_per_feature: dict | None = None,
    im2deep_calibration_slope: float = 1.0,
    im2deep_calibration_intercept: float = 0.0,
) -> pd.DataFrame:
    """
    Compute all features on the candidate DataFrame.

    Parameters
    ----------
    candidates_df
        Output of match_to_maldi_features().
    lcms_evidence
        Pre-computed LC-MS/MS evidence dict from compute_all_lcms_evidence().
    spatial_features
        Pre-computed per-feature spatial statistics DataFrame (optional).
    ion_images
        MALDI ion images array, shape (n_features, H, W) (optional).
    ion_image_mzs
        m/z values aligned with ion_images (optional).
    maldi_envelopes
        MALDI isotope envelopes: feature_mz → array (optional).
    lcms_envelopes
        LC-MS/MS isotope envelopes: feature_mz → array (optional).
    pixel_coords
        (N_features,) or (N_features, 2) pixel coordinates aligned with
        maldi_mzs, used for LOWESS ppm calibration (A3, optional).
    maldi_mzs
        Array of MALDI feature m/z values in feature-index order (required
        for A3 if pixel_coords is provided).
    observed_ccs_per_feature
        Dict mapping feature_idx → observed CCS value for IM2Deep features (optional).
    im2deep_calibration_slope / im2deep_calibration_intercept
        Linear calibration parameters for IM2Deep predictions.

    Returns
    -------
    DataFrame with all feature columns added.
    """
    df = candidates_df.copy()

    # --- Always-computed features ---
    df = compute_mass_accuracy_features(df)
    df = compute_candidate_ambiguity_features(df)
    df = compute_protein_consistency_features(df)
    df = compute_peptide_properties(df)
    df = compute_peptide_property_features(df)          # C-group
    df = compute_maldi_signal_features(df)
    df = compute_mass_defect_features(df)               # A11
    df = compute_chca_cluster_features(df)              # A12

    # --- LC-MS/MS mzML evidence (pre-computed per candidate) ---
    if lcms_evidence is not None:
        for feat in _LCMS_MZML_FEATURES:
            df[feat] = df.index.map(
                lambda idx: lcms_evidence.get(idx, {}).get(feat, 0.0)
            )
        for feat in ["lcms_rt_residual", "lcms_ms1_isotope_cosine"]:
            valid = df[feat].dropna()
            fill = valid.median() if len(valid) > 0 else 0.0
            df[feat] = df[feat].fillna(fill)
    else:
        for feat in _LCMS_MZML_FEATURES:
            df[feat] = 0.0

    # --- LC-MS/MS ID features (Strategy C) ---
    # These are pre-populated by digest_identified_proteins(); default to 0.0
    # if using the full-FASTA Strategy A path (no lcms_ids).
    for feat in _LCMS_ID_FEATURES:
        if feat == "source_lcms_confirmed":
            continue  # computed from source column below
        if feat not in df.columns:
            df[feat] = 0.0

    # source_lcms_confirmed: 1.0 for Strategy C confirmed peptides, 0.0 otherwise
    if "source" in df.columns:
        df["source_lcms_confirmed"] = (df["source"] == "lcms_confirmed").astype(float)
    else:
        df["source_lcms_confirmed"] = 0.0

    # --- Theoretical isotope (adds monoisotopic_confidence, A8) ---
    df = compute_theoretical_isotope_features(
        df,
        maldi_envelopes=maldi_envelopes,
        lcms_envelopes=lcms_envelopes,
    )

    # --- MALDI ionization ---
    df = compute_maldi_ionization_features(df)

    # --- A3: LOWESS ppm calibration (optional) ---
    if pixel_coords is not None:
        df = compute_calibrated_ppm_features(df, maldi_mzs=maldi_mzs, pixel_coords=pixel_coords)

    # --- B: IM2Deep CCS features (optional) ---
    if observed_ccs_per_feature is not None:
        df = compute_im2deep_features(
            df,
            observed_ccs_per_feature=observed_ccs_per_feature,
            calibration_slope=im2deep_calibration_slope,
            calibration_intercept=im2deep_calibration_intercept,
        )

    # --- Spatial (optional) ---
    if spatial_features is not None:
        df = compute_spatial_features(df, spatial_features)

    # --- Ion-image-based features (optional) ---
    if ion_images is not None and ion_image_mzs is not None:
        df = compute_colocalization_features(df, ion_images, ion_image_mzs)
        df = compute_isotopologue_colocalization(df, ion_images, ion_image_mzs)  # E1
        df = compute_adduct_colocalization(df, ion_images, ion_image_mzs)        # E2
        df = compute_spatial_autocorrelation_full(df, ion_images, ion_image_mzs) # E5/E6

    # --- Envelope similarity MALDI vs LC-MS/MS (optional) ---
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
