"""
MSI-PICASSO feature generator: orchestrates all feature categories.

Converts a candidate DataFrame (from match_to_maldi_features) into a PSMList
with all rescoring features populated.
"""

import logging

import numpy as np
import pandas as pd
from psm_utils.psm import PSM
from psm_utils.psm_list import PSMList
from psm_utils.peptidoform import Peptidoform

from msi_picasso.maldi_features import (
    _pearson_r_matrix,
    compute_tissue_mask,
    compute_region_colocalization_features,
    compute_within_region_colocalization_features,
    compute_adduct_colocalization,
    compute_calibrated_ppm_features,
    compute_candidate_ambiguity_features,
    compute_chca_cluster_features,
    compute_colocalization_features,
    compute_im2deep_features,
    compute_lcms_ccs_features,
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
    "ppm_error_abs", "ppm_rank", "ppm_best_ratio", "log_ppm_best_ratio",
    "ppm_error_calibrated_z",  # A3 — optional, requires pixel_coords
    "ppm_error_pct", "ppm_error_squared",
    # --- ambiguity ---
    "n_candidates", "log_n_candidates",
    # --- peptide properties ---
    "peptide_length", "n_missed_cleavages",
    "has_oxidized_met", "has_cys", "n_proline",
    "acidic_residue_density",
    # --- MALDI signal ---
    "log_maldi_intensity_p90", "log_maldi_intensity_sum",
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
    "n_arginine", "n_basic_residues", "n_aromatic",
    "gravy_score", "charge_proxy",
    # --- ion mobility (B-group) — optional, requires im2deep + observed CCS ---
    "im2deep_delta_ccs", "im2deep_abs_delta_ccs_pct",
    "im2deep_ccs_zscore", "im2deep_ccs_rank",
    # m/z-detrended (conformational) CCS residuals — for mz_shuffle these REPLACE
    # the raw CCS features above (pipeline excludes the raw ones); for other decoy
    # methods they supplement them.
    "im2deep_delta_ccs_resid", "im2deep_abs_delta_ccs_pct_resid",
    "im2deep_ccs_zscore_resid", "im2deep_ccs_rank_resid",
    # --- isotopologue co-localization (E1) — optional, requires ion_images ---
    "isotope_image_colocalization_m1", "isotope_image_colocalization_m2",
    "isotope_image_colocalization_mean",
    # --- adduct co-localization (E2) — optional, requires ion_images ---
    "adduct_colocalization_na", "adduct_colocalization_k", "adduct_colocalization_chca",
    # --- per-candidate mobility-filtered colocalization — optional, requires tdf_path + im2deep ---
    "isotope_colocalization_m1_mob", "isotope_colocalization_m2_mob",
    "isotope_colocalization_mean_mob",
    "adduct_colocalization_na_mob", "adduct_colocalization_k_mob",
    "adduct_colocalization_chca_mob",
]

# Protein-level features: aggregate signal across all candidates sharing a protein,
# including decoys. This breaks the TDC null model (decoys inherit inflated counts
# from target co-occurring proteins), so these features are excluded from the ranker
# by default and only used when --use-protein-level-feats is explicitly requested.
PROTEIN_LEVEL_FEATURES = [
    # protein consistency (from compute_protein_consistency_features)
    "protein_n_features", "log_protein_n_features", "protein_coverage",
    "protein_rank", "protein_best_ratio",
    # structural indicator: 1.0 when the protein has a single observed peptide, so
    # within-protein colocalization is undefined (median-imputed) and
    # log_protein_n_features is at its floor.  Lets the ranker treat these peptides
    # as a separate group and rely on their intrinsic evidence (see O8).
    "is_single_peptide_protein",
    # protein co-localization (from compute_colocalization_features; requires ion_images)
    "protein_colocalization", "protein_colocalization_max",
    "protein_colocalization_median", "protein_colocalization_n_partners",
    # mobility-gated protein co-localization (from compute_mobility_colocalization_features)
    "protein_colocalization_mob", "protein_colocalization_mob_max",
    "protein_colocalization_mob_n_partners",
    # indicator: 1.0 when within-protein colocalization is defined (>=1 partner),
    # 0.0 otherwise.  Lets the ranker separate "not colocalizable" (small protein,
    # coloc median-imputed) from "colocalizes poorly" (see compute_colocalization_features).
    "has_coloc",
    # intensity-weighted and rank-weighted (top-k) within-protein colocalization
    "protein_colocalization_weighted", "protein_colocalization_weighted_max",
    "protein_colocalization_top2", "protein_colocalization_top3", "protein_colocalization_top5",
]

# Region-profile colocalization (opt-in via --region-coloc, requires ion_images).
# Within-protein Pearson r of per-region composition fingerprints — asks whether
# same-protein peptides occupy the same tissue regions, a sharper question than
# global ion-image Pearson r (which is dominated by the shared tissue envelope).
# Protein-level, so valid only because decoys occupy a separate protein namespace.
# Appended to the ranker pool at runtime in pipeline.py when the flag is set.
REGION_COLOCALIZATION_FEATURES = [
    "protein_region_colocalization",
]

# Within-region and dominant-region Pearson-r colocalization (opt-in via
# --within-region-coloc, requires ion_images). O3 (FLAWS_AND_OPPORTUNITIES.md):
# unlike REGION_COLOCALIZATION_FEATURES (per-region MEAN fingerprint), these
# correlate RAW pixel intensities restricted to a region, asking whether
# same-protein peptides co-vary pixel-to-pixel inside a shared region rather
# than merely sharing a region average. Experimental / unvalidated — see O3's
# validation protocol before removing these from features-exclude in any config.
WITHIN_REGION_COLOCALIZATION_FEATURES = [
    "protein_within_region_colocalization",
    "protein_dominant_region_colocalization",
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
    # DeepLC-anchored MS1 signal features
    "lcms_ms1_intensity", "lcms_ms1_snr",
    # DeepLC-anchored MS1 isotope features (per-candidate, fully symmetric)
    "lcms_ms1_isotope_cosine",
    "theo_m1_ratio_diff_lcms", "theo_m2_ratio_diff_lcms",
    "log_theo_m1_ratio_diff_lcms", "log_theo_m2_ratio_diff_lcms",
    # DeepLC-anchored RT-consistency and apex features
    "lcms_ms1_apex_rt_delta",
    "lcms_ms1_frac_apex_signal",
    "lcms_ms1_n_scans_with_signal",
    "lcms_ms2_rt_delta",
    "isotope_envelope_cosine",
    "isotope_envelope_pearson",
    "isotope_envelope_mse",
    "isotope_n_matched",
    "isotope_absolute_diff",
    "log_isotope_m1_ratio_diff",
    "log_isotope_m2_ratio_diff",
]

_LCMS_ID_FEATURES = [
    "lcms_q_value", "lcms_pep", "lcms_score",
    "n_psms", "lcms_intensity",
]

# MALDI-vs-LC-MS/MS CCS comparison (optional: requires ion mobility in LC-MS/MS data).
_LCMS_CCS_FEATURES = ["lcms_ccs_delta", "lcms_ccs_abs_pct"]

LCMS_PRIOR_FEATURES = _LCMS_MZML_FEATURES + _LCMS_CCS_FEATURES 

# Spatial prior features: ion-image-level quality signals applied as a
# multiplicative prior after scoring (analogous to LCMS_PRIOR_FEATURES).
# Excluded from the ranker because they are feature-level, not candidate-level
# — all candidates at the same m/z feature share identical values, so they
# cannot discriminate between candidate sequences within a feature.
SPATIAL_PRIOR_FEATURES = [
    "spatial_autocorrelation", "fraction_detected", "intensity_cv",
    "log_mean_intensity", "spatial_entropy",
    "spatial_morans_i", "spatial_gearys_c",
    # per-candidate mobility-filtered spatial quality (optional, requires tdf_path + im2deep)
    "fraction_detected_mob", "intensity_cv_mob",
    "log_mean_intensity_mob", "spatial_morans_i_mob",
]

# Spatial ranker features: opt-in (--use-spatial-ranker-features), permitted ONLY
# with decoy methods whose decoys land on real MALDI features (entrapment, mz_shift).
# With shuffle/balanced_shuffle/paired_shuffle there is no consistent spatial anchor
# for decoys, so including these features produces an asymmetric null (guarded in
# pipeline.py).  The protein_colocalization_* members overlap PROTEIN_LEVEL_FEATURES;
# pipeline.py deduplicates when both opt-in flags are active.
SPATIAL_RANKER_FEATURES = [
    # Feature-level spatial quality
    "spatial_autocorrelation",
    "spatial_morans_i",
    "spatial_gearys_c",
    "fraction_detected",
    "intensity_cv",
    # Protein-level colocalization
    "protein_colocalization",
    "protein_colocalization_max",
    "protein_colocalization_median",
    "protein_colocalization_n_partners",
]

# Alias kept separate so LDA-specific feature selection can diverge later.
LDA_FEATURES = MALDI_INTRINSIC_FEATURES

# Reduced feature set: one representative per collinear group.
# Use with --only-main-features to cut the feature count from ~46 to ~19
# and remove inter-feature redundancy before training.
MAIN_FEATURES = [
    # ppm (from 6 → 1)
    "ppm_error_abs",
    # isotope_theo (from 5 → 1)
    "theo_isotope_cosine",
    # sequence_comp (from 8 → 2)
    "gravy_score",
    "peptide_length",
    # adduct (keep all 3, not collinear across adduct types)
    "adduct_colocalization_na",
    "adduct_colocalization_k",
    "adduct_colocalization_chca",
    # chca
    "chca_cluster_distance_ppm",
    # maldi_intensity (from 2 → 1)
    "log_maldi_intensity_p90",
    # im2deep (from 4 → 1)
    "im2deep_ccs_zscore",
    # averagine (from 2 → 1)
    "averagine_deviation",
    # monoisotopic
    "monoisotopic_confidence",
    # isotope_image (from 3 → 1)
    "isotope_image_colocalization_m2",
    # candidates (from 2 → 1)
    "n_candidates",
    # sulfur (from 2 → 1)
    "theo_has_sulfur",
    # mass_defect (from 2 → 1)
    "kendrick_mass_defect",
    # other (keep both, not collinear)
    "has_oxidized_met",
    "n_missed_cleavages",
]

# Feature-specific NaN fill values applied before the generic median imputer.
# Missing values for these features have a known structural meaning, so filling
# with the column median would be misleading.
#
# Values:
#   float      — fill with that constant
#   "col_max"  — fill with np.nanmax of that column (worst-case penalty)
#   "col_min"  — fill with np.nanmin of that column
FEATURE_NAN_FILL: dict[str, float | str] = {
    # No LC-MS/MS envelope match → worst ratio deviation (largest observed error)
    "log_isotope_m1_ratio_diff": "col_max",
    "log_isotope_m2_ratio_diff": "col_max",
}

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
    "has_coloc",
])
_PIXEL_FEATS = frozenset(["ppm_error_calibrated_z"])
_CCS_FEATS = frozenset([
    "im2deep_delta_ccs", "im2deep_abs_delta_ccs_pct",
    "im2deep_ccs_zscore", "im2deep_ccs_rank",
    "im2deep_delta_ccs_resid", "im2deep_abs_delta_ccs_pct_resid",
    "im2deep_ccs_zscore_resid", "im2deep_ccs_rank_resid",
])
_ISOTOPOLOGUE_COLOC_FEATS = frozenset([
    "isotope_image_colocalization_m1", "isotope_image_colocalization_m2",
    "isotope_image_colocalization_mean",
])
_ADDUCT_COLOC_FEATS = frozenset([
    "adduct_colocalization_na", "adduct_colocalization_k", "adduct_colocalization_chca",
])
_MORANS_FEATS = frozenset(["spatial_morans_i", "spatial_gearys_c"])
_MOB_COLOC_FEATS = frozenset([
    "isotope_colocalization_m1_mob", "isotope_colocalization_m2_mob",
    "isotope_colocalization_mean_mob",
    "adduct_colocalization_na_mob", "adduct_colocalization_k_mob",
    "adduct_colocalization_chca_mob",
])
_MOB_SPATIAL_FEATS = frozenset([
    "fraction_detected_mob", "intensity_cv_mob",
    "log_mean_intensity_mob", "spatial_morans_i_mob",
])


def get_feature_names(
    has_spatial: bool = False,  # kept for backwards compatibility; no longer used
    has_ion_images: bool = False,
    has_envelopes: bool = False,  # kept for backwards compatibility; no longer used
    has_pixel_coords: bool = False,
    has_ccs: bool = False,
    has_mob_coloc: bool = False,
) -> list[str]:
    """Return the full list of feature names based on available data.

    Optional feature groups are included only when the corresponding data
    was computed. ``MALDI_INTRINSIC_FEATURES + LCMS_PRIOR_FEATURES`` is the
    superset; this function selects the applicable subset.
    """
    intrinsic = [
        f for f in MALDI_INTRINSIC_FEATURES
        if (f not in _COLOC_FEATS or has_ion_images)
        and (f not in _ISOTOPOLOGUE_COLOC_FEATS or has_ion_images)
        and (f not in _ADDUCT_COLOC_FEATS or has_ion_images)
        and (f not in _PIXEL_FEATS or has_pixel_coords)
        and (f not in _CCS_FEATS or has_ccs)
        and (f not in _MOB_COLOC_FEATS or has_mob_coloc)
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
            metadata={c: str(getattr(row, c)) for c in meta_cols},
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
    extra_ion_images: dict | None = None,
    maldi_envelopes: dict | None = None,
    pixel_coords: np.ndarray | None = None,
    maldi_mzs: np.ndarray | None = None,
    observed_ccs_per_feature: dict | None = None,
    im2deep_calibration: str = "linear",
    im2deep_kwargs: dict | None = None,
    coloc_tic_quantile: float = 0.0,
    coloc_measured_pixel_mask: "np.ndarray | None" = None,
    coloc_tic_normalize: bool = False,
    coloc_common_mode: bool = False,
    region_coloc: bool = False,
    region_coloc_k: int = 20,
    region_coloc_debug: dict | None = None,
    within_region_coloc: bool = False,
    within_region_coloc_debug: dict | None = None,
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
    pixel_coords
        (N_features,) or (N_features, 2) pixel coordinates aligned with
        maldi_mzs, used for LOWESS ppm calibration (A3, optional).
    maldi_mzs
        Array of MALDI feature m/z values in feature-index order (required
        for A3 if pixel_coords is provided).
    observed_ccs_per_feature
        Dict mapping feature_idx → observed CCS value for IM2Deep features (optional).

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

    # --- LC-MS/MS mzML evidence (pre-computed per candidate via compute_all_lcms_evidence) ---
    # All features in _LCMS_MZML_FEATURES are computed symmetrically per-candidate
    # (targets and decoys receive identical treatment; no is_decoy branching).
    # Isotope envelope similarity features (isotope_envelope_*) are included here
    # when maldi_envelopes was passed to compute_all_lcms_evidence.
    # All evidence columns that need to be in features_df:
    #   _LCMS_MZML_FEATURES      → applied as prior (lcms_present in pipeline)
    #   _LCMS_RANKER_FROM_EVIDENCE → also in MALDI_INTRINSIC_FEATURES (ranker)
    _ALL_EVIDENCE_COLS = _LCMS_MZML_FEATURES
    if lcms_evidence is not None:
        ev_df = pd.DataFrame.from_dict(lcms_evidence, orient="index", dtype=float)
        ev_df = ev_df.reindex(columns=_ALL_EVIDENCE_COLS)
        df = df.join(ev_df, how="left")
        df[_ALL_EVIDENCE_COLS] = df[_ALL_EVIDENCE_COLS].fillna(0.0)
        # For similarity/ratio features and NaN-sentinel RT-delta features, replace
        # 0-fill with column median so candidates without signal are not penalised.
        _median_fill_feats = [
            "lcms_ms1_isotope_cosine",
            "theo_m1_ratio_diff_lcms", "theo_m2_ratio_diff_lcms",
            "log_theo_m1_ratio_diff_lcms", "log_theo_m2_ratio_diff_lcms",
            "isotope_envelope_pearson",
            "lcms_ms1_apex_rt_delta",
            "lcms_ms2_rt_delta",
        ]
        for feat in _median_fill_feats:
            if feat not in df.columns:
                continue
            valid = df[feat].replace(0.0, np.nan).dropna()
            fill = float(valid.median()) if len(valid) > 0 else 0.0
            df[feat] = df[feat].replace(0.0, fill)
    else:
        for feat in _ALL_EVIDENCE_COLS:
            df[feat] = 0.0

    # --- MALDI vs LC-MS/MS CCS features (optional) ---
    df = compute_lcms_ccs_features(df, observed_ccs_per_feature=observed_ccs_per_feature)
    for feat in _LCMS_CCS_FEATURES:
        if feat not in df.columns:
            df[feat] = 0.0

    # --- Theoretical isotope (adds monoisotopic_confidence, A8) ---
    df = compute_theoretical_isotope_features(
        df,
        maldi_envelopes=maldi_envelopes,
    )

    # --- MALDI ionization ---
    df = compute_maldi_ionization_features(df)

    # --- A3: LOWESS ppm calibration (optional) ---
    if pixel_coords is not None:
        logger.debug("Computing LOWESS ppm calibration features (A3) using pixel coordinates")
        df = compute_calibrated_ppm_features(df, maldi_mzs=maldi_mzs, pixel_coords=pixel_coords)

    # --- B: IM2Deep CCS features (optional) ---
    if observed_ccs_per_feature is not None:
        logger.debug("Computing IM2Deep CCS features (B-group) using observed CCS values")
        df = compute_im2deep_features(
            df,
            observed_ccs_per_feature=observed_ccs_per_feature,
            calibration_method=im2deep_calibration,
            im2deep_kwargs=im2deep_kwargs,
        )

    # --- Spatial (optional) ---
    if spatial_features is not None:
        logger.debug("Computing spatial features (A3/A4) using pre-computed spatial_features DataFrame")
        df = compute_spatial_features(df, spatial_features)

    # --- Ion-image-based features (optional) ---
    if ion_images is not None and ion_image_mzs is not None:
        logger.debug("Computing ion-image-based features (co-localization, full spatial autocorrelation, etc.) using ion_images and ion_image_mzs")
        # On-tissue pixel mask (TIC proxy): raw MALDI ion images share a dominant
        # on/off-tissue component that inflates every pairwise r toward the tissue
        # outline. Restricting the correlation to on-tissue pixels removes it so
        # colocalization reflects co-distribution within the tissue (see
        # compute_tissue_mask). TIC == 0 padding is always dropped.
        pixel_mask = compute_tissue_mask(ion_images, tic_quantile=coloc_tic_quantile)
        if coloc_measured_pixel_mask is not None:
            pixel_mask = pixel_mask & coloc_measured_pixel_mask
        _mask_suffix = ", measured-coord mask applied" if coloc_measured_pixel_mask is not None else ""
        logger.info(
            f"Colocalization on-tissue mask: {int(pixel_mask.sum())}/{pixel_mask.size} "
            f"pixels kept (tic_quantile={coloc_tic_quantile}{_mask_suffix})"
        )
        # Compute the full Pearson correlation matrix once (single BLAS call) and
        # share it across the isotopologue/adduct colocalization functions to avoid
        # redundant work.  Those correlate an M0 image against its own
        # isotopologue/adduct image, where common-mode removal is not meaningful, so
        # they always use the raw cache.
        corr_cache = _pearson_r_matrix(ion_images, ion_image_mzs, pixel_mask=pixel_mask)
        # The cross-feature protein colocalization optionally uses a preprocessed
        # cache (per-pixel TIC normalization / common-mode removal) to strip the
        # shared tissue envelope; only built when a toggle is set to avoid a second
        # BLAS pass in the default case.
        if coloc_tic_normalize or coloc_common_mode:
            protein_corr_cache = _pearson_r_matrix(
                ion_images, ion_image_mzs, pixel_mask=pixel_mask,
                tic_normalize=coloc_tic_normalize, common_mode_removal=coloc_common_mode,
            )
        else:
            protein_corr_cache = corr_cache
        df = compute_colocalization_features(df, ion_images, ion_image_mzs, _corr_cache=protein_corr_cache)
        df = compute_isotopologue_colocalization(df, ion_images, ion_image_mzs, _corr_cache=corr_cache, extra_ion_images=extra_ion_images, pixel_mask=pixel_mask)  # E1
        df = compute_adduct_colocalization(df, ion_images, ion_image_mzs, _corr_cache=corr_cache, extra_ion_images=extra_ion_images, pixel_mask=pixel_mask)        # E2
        df = compute_spatial_autocorrelation_full(df, ion_images, ion_image_mzs)                         # E5/E6
        if region_coloc:
            df = compute_region_colocalization_features(
                df, ion_images, ion_image_mzs, pixel_mask=pixel_mask,
                n_regions=region_coloc_k, debug=region_coloc_debug,
            )
        if within_region_coloc:
            # Reuse protein_corr_cache's preprocessing recipe (coloc_tic_normalize /
            # coloc_common_mode) so the per-region correlation uses the same
            # best-performing recipe as compute_colocalization_features (O2,
            # FLAWS_AND_OPPORTUNITIES.md) rather than raw pixels.
            df = compute_within_region_colocalization_features(
                df, ion_images, ion_image_mzs, pixel_mask=pixel_mask,
                n_regions=region_coloc_k, tic_normalize=coloc_tic_normalize,
                common_mode_removal=coloc_common_mode,
                _global_corr_cache=protein_corr_cache, debug=within_region_coloc_debug,
            )

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
