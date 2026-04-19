"""End-to-end symmetric MALDI-MSI rescoring pipeline."""

import logging
import os

import numpy as np
import pandas as pd

from ms1rescore.candidates import digest_fasta, match_to_maldi_features
from ms1rescore.feature_generator import (
    candidates_to_psm_list,
    compute_all_features,
    get_feature_names,
    populate_psm_features,
)
from ms1rescore.lcms_evidence import (
    compute_all_lcms_evidence,
    extract_all_xics,
    finetune_deeplc,
    get_deeplc_predictions,
    get_ms2pip_predictions,
    load_lcms_data,
)

logger = logging.getLogger(__name__)


def rescore(
    fasta_path: str,
    maldi_mzs: np.ndarray,
    mzml_paths: list[str],
    spatial_features: pd.DataFrame | None = None,
    ion_images: np.ndarray | None = None,
    ion_image_mzs: np.ndarray | None = None,
    maldi_envelopes: dict | None = None,
    msf_path: str | None = None,
    ppm_tolerance: float = 20.0,
    train_fdr: float = 0.01,
    cache_dir: str | None = None,
    missed_cleavages: int = 2,
    min_length: int = 7,
    max_length: int = 30,
):
    """
    End-to-end symmetric MALDI-MSI rescoring pipeline.

    Parameters
    ----------
    fasta_path
        Path to protein FASTA file (forward sequences only, decoys are generated).
    maldi_mzs
        Array of MALDI feature m/z values.
    mzml_paths
        Paths to LC-MS/MS mzML files.
    spatial_features
        Pre-computed spatial features DataFrame (optional).
    ion_images
        MALDI ion images array, shape (n_features, height, width) (optional).
    ion_image_mzs
        m/z values corresponding to ion_images (optional).
    maldi_envelopes
        MALDI isotope envelopes: feature_mz → normalized envelope (optional).
    msf_path
        Path to PD .msf file for DeepLC finetuning (optional).
    ppm_tolerance
        Mass tolerance for MALDI-to-database matching in ppm.
    train_fdr
        FDR threshold for mokapot training.
    cache_dir
        Directory for caching intermediate results.

    Returns
    -------
    tuple of (psm_list, confidence_estimates, feature_names)
    """
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)

    def _cache(name):
        return os.path.join(cache_dir, name) if cache_dir else None

    # --- Step 1: Candidate generation ---
    logger.info("Step 1: Digesting FASTA and matching to MALDI features...")
    peptide_db = digest_fasta(
        fasta_path,
        missed_cleavages=missed_cleavages,
        min_length=min_length,
        max_length=max_length,
        generate_decoys=True,
    )
    # Compute per-feature MALDI intensities from ion images if available
    maldi_intensities = None
    if ion_images is not None:
        maldi_intensities = np.array([
            img[img > 0].mean() if (img > 0).any() else 0.0
            for img in ion_images
        ])
    candidates = match_to_maldi_features(
        maldi_mzs, peptide_db, ppm_tolerance, maldi_intensities=maldi_intensities,
    )
    if len(candidates) == 0:
        raise ValueError("No candidates matched any MALDI features")

    logger.info(
        f"  {len(candidates)} candidates ({(~candidates['is_decoy']).sum()} target, "
        f"{candidates['is_decoy'].sum()} decoy) across "
        f"{candidates['feature_mz'].nunique()} features"
    )

    # --- Step 2: Load LC-MS/MS data ---
    logger.info("Step 2: Loading LC-MS/MS data...")
    lcms_data = load_lcms_data(mzml_paths, cache_path=_cache("lcms_data.pkl"))

    # --- Step 3: MS2PIP predictions ---
    # Only predict for (peptide, charge) pairs where a matching MS2 scan exists.
    logger.info("Step 3: Finding MS2 matches and running MS2PIP...")
    from ms1rescore.lcms_evidence import _find_matching_ms2_scans
    from ms1rescore.utils import mz_to_mass

    feature_ms2_charges = {}
    for mz in candidates["feature_mz"].unique():
        neutral_mass = mz_to_mass(mz, charge=1)
        scan_idxs = _find_matching_ms2_scans(neutral_mass, lcms_data, ppm_tolerance)
        if scan_idxs:
            feature_ms2_charges[mz] = set(
                int(lcms_data.ms2_precursor_charge[i]) for i in scan_idxs
            )
    logger.info(
        f"  {len(feature_ms2_charges)}/{candidates['feature_mz'].nunique()} features have MS2 matches"
    )

    peptide_charge_pairs = set()
    for mz, charges in feature_ms2_charges.items():
        peps = candidates[candidates["feature_mz"] == mz]["peptide"].unique()
        for pep in peps:
            for c in charges:
                peptide_charge_pairs.add((pep, c))
    logger.info(f"  {len(peptide_charge_pairs)} unique (peptide, charge) pairs for MS2PIP")

    ms2pip_cache = get_ms2pip_predictions(
        list(peptide_charge_pairs),
        model="HCD",
        cache_path=_cache("ms2pip_predictions.pkl"),
    )

    # --- Step 4: DeepLC predictions ---
    logger.info("Step 4: Computing DeepLC predictions...")
    unique_peptides = candidates["peptide"].unique().tolist()
    deeplc_model = None
    if msf_path:
        deeplc_model = finetune_deeplc(msf_path, cache_path=_cache("deeplc_model.pt"))
    deeplc_cache = get_deeplc_predictions(
        unique_peptides,
        model=deeplc_model,
        cache_path=_cache("deeplc_predictions.pkl"),
    )

    # --- Step 5: Compute LC-MS/MS evidence ---
    logger.info("Step 5: Computing LC-MS/MS evidence features...")
    lcms_evidence = compute_all_lcms_evidence(
        candidates,
        lcms_data,
        ms2pip_cache,
        deeplc_cache,
        ppm_tolerance=ppm_tolerance,
    )

    # --- Step 6: Extract LC-MS/MS envelopes from XIC best scans ---
    # Build per-feature LC-MS/MS envelopes from XIC data (symmetric)
    lcms_envelopes_xic = None
    if maldi_envelopes is not None:
        logger.info("Step 6: Extracting LC-MS/MS envelopes from XIC scans...")
        from ms1rescore.lcms_evidence import _extract_ms1_envelope

        unique_feature_mzs = candidates["feature_mz"].unique()
        xic_cache = extract_all_xics(unique_feature_mzs, lcms_data, ppm_tolerance)
        lcms_envelopes_xic = {}
        for mz in unique_feature_mzs:
            rts, ints = xic_cache.get(mz, (np.array([]), np.array([])))
            if len(ints) > 0 and ints.max() > 0:
                best_xic_idx = np.argmax(ints)
                best_rt = rts[best_xic_idx]
                best_ms1_idx = np.argmin(np.abs(lcms_data.ms1_rts - best_rt))
                env = _extract_ms1_envelope(mz, best_ms1_idx, lcms_data, charge=1, n_peaks=3)
                if env.sum() > 0:
                    lcms_envelopes_xic[mz] = env

    # --- Step 7: Compute all features ---
    logger.info("Step 7: Computing all features...")
    features_df = compute_all_features(
        candidates,
        lcms_evidence=lcms_evidence,
        spatial_features=spatial_features,
        ion_images=ion_images,
        ion_image_mzs=ion_image_mzs,
        maldi_envelopes=maldi_envelopes,
        lcms_envelopes=lcms_envelopes_xic,
    )

    feature_names = get_feature_names(
        has_spatial=spatial_features is not None,
        has_ion_images=ion_images is not None,
        has_envelopes=maldi_envelopes is not None and lcms_envelopes_xic is not None,
    )

    # --- Step 8: Build PSMList ---
    logger.info("Step 8: Building PSMList...")
    psm_list = candidates_to_psm_list(features_df)
    populate_psm_features(psm_list, features_df, feature_names)

    logger.info(f"  {len(feature_names)} features: {feature_names}")

    # --- Step 9: Mokapot rescoring ---
    logger.info("Step 9: Running mokapot rescoring...")
    from mokapot import brew
    from mokapot.model import PercolatorModel
    from ms2rescore.rescoring_engines.mokapot import convert_psm_list

    lin = convert_psm_list(psm_list, feature_names=feature_names)
    model = PercolatorModel(train_fdr=train_fdr, max_iter=10)
    result = brew(lin, model=model)
    conf = (result[0] if isinstance(result, tuple) else result).confidence_estimates

    psm_conf = conf["psms"]
    for fdr_threshold in [0.01, 0.05, 0.10]:
        n = (psm_conf["mokapot q-value"] <= fdr_threshold).sum()
        logger.info(f"  At {fdr_threshold*100:.0f}% FDR: {n} target features")

    return psm_list, conf, feature_names
