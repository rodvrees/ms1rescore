"""End-to-end symmetric MALDI-MSI rescoring pipeline."""

import logging
import os

import numpy as np
import pandas as pd

from ms1rescore.candidates import digest_fasta, match_to_maldi_features
from ms1rescore.feature_generator import (
    LCMS_PRIOR_FEATURES,
    MALDI_INTRINSIC_FEATURES,
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


def compute_lcms_prior(
    candidates_df: pd.DataFrame,
    present_lcms_features: list[str],
) -> np.ndarray:
    """
    Compute a per-candidate multiplicative weight in (0, 1] based on
    available LC-MS/MS evidence.

    Each present feature is min-max normalized to [0, 1] across all candidates.
    Features where all values are zero (no evidence for any candidate) are
    excluded from the average. The returned weight is the mean of the
    normalized non-zero features, or 1.0 if no LC-MS/MS features are present.

    LC-MS/MS features are NOT passed to the ranker so that the model scores
    MALDI match quality rather than LC-MS/MS identification quality. The prior
    applies LC-MS/MS evidence as a Bayesian weight after MALDI-intrinsic
    scoring, keeping the two inference steps conceptually separate.
    """
    normed = []
    for feat in present_lcms_features:
        if feat not in candidates_df.columns:
            continue
        col = candidates_df[feat].fillna(0.0).values.astype(float)
        col_min, col_max = col.min(), col.max()
        if col_max - col_min < 1e-12:
            # All values identical (e.g. all zero) — feature carries no information
            continue
        normed.append((col - col_min) / (col_max - col_min))

    if not normed:
        return np.ones(len(candidates_df))

    return np.stack(normed, axis=0).mean(axis=0)


def _rescore_svm(
    psm_list,
    intrinsic_feature_names: list[str],
    train_fdr: float,
):
    """Run mokapot PercolatorModel on MALDI-intrinsic features."""
    from mokapot import brew
    from mokapot.model import PercolatorModel
    from ms2rescore.rescoring_engines.mokapot import convert_psm_list

    lin = convert_psm_list(psm_list, feature_names=intrinsic_feature_names)
    model = PercolatorModel(train_fdr=train_fdr, max_iter=10)
    result = brew(lin, model=model)
    conf_obj = result[0] if isinstance(result, tuple) else result
    return conf_obj


def _rescore_catboost(
    features_df: pd.DataFrame,
    intrinsic_feature_names: list[str],
    train_fdr: float,
    init_ppm_threshold: float,
    init_isotope_threshold: float,
) -> np.ndarray:
    """
    Semi-supervised CatBoostRanker on MALDI-intrinsic features.

    Pseudo-label iteration:
      1. Seed positives: ppm_error_abs < init_ppm_threshold AND
         theo_isotope_cosine > init_isotope_threshold (targets only).
      2. Train CatBoostRanker on seed positives + all decoys.
      3. Score all candidates; compute provisional TDC q-values.
      4. Expand positives to candidates with q <= 0.05 (targets only).
      5. Repeat until convergence (<1% change in positive set size) or 5 iters.

    Returns a score array (higher = more likely correct).
    """
    try:
        from catboost import CatBoostRanker, Pool
    except ImportError as e:
        raise ImportError(
            "CatBoost is required for model='catboost'. "
            "Install with: pip install catboost>=1.2"
        ) from e

    df = features_df.reset_index(drop=True)
    present = [f for f in intrinsic_feature_names if f in df.columns]
    X = df[present].fillna(0.0).values.astype(np.float32)
    is_decoy = df["is_decoy"].values.astype(bool)
    is_target = ~is_decoy

    # Initial positive seed: stringent mass accuracy + isotope filter
    seed_mask = (
        is_target
        & (df.get("ppm_error_abs", pd.Series(np.inf, index=df.index)) < init_ppm_threshold)
        & (df.get("theo_isotope_cosine", pd.Series(0.0, index=df.index)) > init_isotope_threshold)
    ).values

    n_seed = seed_mask.sum()
    logger.info(f"  CatBoost: seed positives = {n_seed}, decoys = {is_decoy.sum()}")
    if n_seed == 0:
        logger.warning("  CatBoost: no seed positives — falling back to top-ppm init")
        ppm_col = df.get("ppm_error_abs", pd.Series(np.inf, index=df.index))
        seed_mask = is_target & (ppm_col < ppm_col[is_target].quantile(0.10)).values

    catboost_params = dict(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        loss_function="YetiRank",
        verbose=False,
        random_seed=42,
    )

    scores = np.zeros(len(df))
    prev_pos_size = -1

    for iteration in range(5):
        # Build training set: pseudo-positives (label=1) + decoys (label=0)
        pos_idx = np.where(seed_mask)[0]
        dec_idx = np.where(is_decoy)[0]
        train_idx = np.concatenate([pos_idx, dec_idx])
        labels = np.concatenate([np.ones(len(pos_idx)), np.zeros(len(dec_idx))])

        # CatBoostRanker requires group_id; use a single group
        group_ids = np.zeros(len(train_idx), dtype=np.int32)

        pool = Pool(
            data=X[train_idx],
            label=labels,
            group_id=group_ids,
        )
        model_cb = CatBoostRanker(**catboost_params)
        model_cb.fit(pool)

        scores = model_cb.predict(X)

        # Provisional TDC q-values (higher score = better)
        q_values = _tdc_qvalues(scores, is_decoy)

        # Expand pseudo-positives: all targets with q <= 0.05
        new_seed = is_target & (q_values <= 0.05)
        n_new = new_seed.sum()

        logger.info(
            f"  CatBoost iter {iteration + 1}: "
            f"pseudo-positives = {n_new} (prev = {prev_pos_size})"
        )

        change = abs(n_new - prev_pos_size) / max(prev_pos_size, 1)
        if prev_pos_size >= 0 and change < 0.01:
            logger.info("  CatBoost: converged")
            break

        prev_pos_size = n_new
        seed_mask = new_seed.values if hasattr(new_seed, "values") else new_seed

    return scores


def _tdc_qvalues(scores: np.ndarray, is_decoy: np.ndarray) -> np.ndarray:
    """
    Compute per-candidate target-decoy q-values.

    Uses standard TDC: sort by descending score, compute cumulative
    FDR = n_decoy / n_target at each position, then take the minimum
    FDR seen at or below each score (q-value = rolling min from the bottom).
    """
    order = np.argsort(-scores)
    n_target_cum = np.cumsum(~is_decoy[order]).astype(float)
    n_decoy_cum = np.cumsum(is_decoy[order]).astype(float)

    # Avoid division by zero
    with np.errstate(invalid="ignore", divide="ignore"):
        fdr = np.where(n_target_cum > 0, n_decoy_cum / n_target_cum, 1.0)

    # q-value: minimum FDR at or below this score (monotone from the tail)
    qval_ordered = np.minimum.accumulate(fdr[::-1])[::-1]

    # Map back to original order
    q_values = np.empty_like(qval_ordered)
    q_values[order] = qval_ordered
    return q_values


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
    model: str = "svm",
    init_ppm_threshold: float = 2.0,
    init_isotope_threshold: float = 0.7,
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
        FDR threshold for mokapot training (SVM backend only).
    cache_dir
        Directory for caching intermediate results.
    missed_cleavages
        Number of missed cleavages for in-silico digest.
    min_length
        Minimum peptide length.
    max_length
        Maximum peptide length.
    model
        Rescoring backend: "svm" (mokapot PercolatorModel, default) or
        "catboost" (semi-supervised CatBoostRanker). Both backends use only
        MALDI_INTRINSIC_FEATURES for training. LC-MS/MS evidence is applied
        as a multiplicative prior after scoring.
    init_ppm_threshold
        CatBoost only: ppm_error_abs threshold for the initial positive seed.
    init_isotope_threshold
        CatBoost only: theo_isotope_cosine threshold for the initial positive seed.

    Returns
    -------
    tuple of (psm_list, result, feature_names)
        result is a confidence estimates dict (SVM) or a DataFrame with
        columns [peptide, feature_idx, score, reweighted_score, q_value,
        reweighted_q_value] (CatBoost).
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

    # Intrinsic features that are actually present in the DataFrame
    intrinsic_present = [
        f for f in MALDI_INTRINSIC_FEATURES if f in features_df.columns
    ]
    lcms_present = [
        f for f in LCMS_PRIOR_FEATURES if f in features_df.columns
    ]

    # --- Step 8: Build PSMList ---
    logger.info("Step 8: Building PSMList...")
    psm_list = candidates_to_psm_list(features_df)
    populate_psm_features(psm_list, features_df, feature_names)

    logger.info(f"  {len(feature_names)} features ({len(intrinsic_present)} intrinsic, {len(lcms_present)} LC-MS/MS prior)")

    # --- Step 9: Rescoring ---
    logger.info(f"Step 9: Running rescoring (model='{model}')...")

    if model == "svm":
        # Pass only intrinsic features to the SVM
        populate_psm_features(psm_list, features_df, intrinsic_present)
        conf_obj = _rescore_svm(psm_list, intrinsic_present, train_fdr)
        conf = conf_obj.confidence_estimates
        psm_conf = conf["psms"]

        # LC-MS/MS prior reweight
        lcms_prior = compute_lcms_prior(features_df, lcms_present)
        if "mokapot score" in psm_conf.columns:
            psm_conf = psm_conf.copy()
            psm_conf["reweighted_score"] = psm_conf["mokapot score"] * lcms_prior[
                psm_conf.index if "index" not in psm_conf.columns else psm_conf["index"]
            ]

        for fdr_threshold in [0.01, 0.05, 0.10]:
            n = (psm_conf["mokapot q-value"] <= fdr_threshold).sum()
            logger.info(f"  At {fdr_threshold*100:.0f}% FDR: {n} target features")

        return psm_list, conf, feature_names

    elif model == "catboost":
        scores = _rescore_catboost(
            features_df,
            intrinsic_present,
            train_fdr=train_fdr,
            init_ppm_threshold=init_ppm_threshold,
            init_isotope_threshold=init_isotope_threshold,
        )

        is_decoy = features_df["is_decoy"].values.astype(bool)
        q_values = _tdc_qvalues(scores, is_decoy)

        # LC-MS/MS prior reweight
        lcms_prior = compute_lcms_prior(features_df, lcms_present)
        reweighted_scores = scores * lcms_prior
        reweighted_q = _tdc_qvalues(reweighted_scores, is_decoy)

        result_df = pd.DataFrame({
            "peptide": features_df["peptide"].values,
            "feature_idx": features_df.get("feature_idx", pd.Series(range(len(features_df)))).values,
            "is_decoy": is_decoy,
            "catboost_score": scores,
            "q_value": q_values,
            "reweighted_score": reweighted_scores,
            "reweighted_q_value": reweighted_q,
        })

        for fdr_threshold in [0.01, 0.05, 0.10]:
            n = ((result_df["q_value"] <= fdr_threshold) & ~is_decoy).sum()
            n_rw = ((result_df["reweighted_q_value"] <= fdr_threshold) & ~is_decoy).sum()
            logger.info(
                f"  At {fdr_threshold*100:.0f}% FDR: {n} targets (base), "
                f"{n_rw} targets (reweighted)"
            )

        return psm_list, result_df, feature_names

    else:
        raise ValueError(f"Unknown model '{model}'. Choose 'svm' or 'catboost'.")
