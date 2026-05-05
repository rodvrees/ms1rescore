"""End-to-end symmetric MALDI-MSI rescoring pipeline."""

import logging
import os
import pickle

import numpy as np
import pandas as pd

from ms1rescore.candidates import (
    digest_fasta,
    digest_identified_proteins,
    match_to_maldi_features,
)
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

# Features where lower = better: invert (1 - value) before normalization
_LCMS_INVERT_FEATURES = frozenset(["lcms_q_value", "lcms_pep"])

# NaN fill values before inversion/normalization (1.0 = worst for lower-is-better)
_LCMS_NAN_FILL: dict[str, float] = {"lcms_q_value": 1.0, "lcms_pep": 1.0}

# Relative weights for the weighted average in compute_lcms_prior
_LCMS_PRIOR_WEIGHTS: dict[str, float] = {"source_lcms_confirmed": 2.0}


def compute_lcms_prior(
    candidates_df: pd.DataFrame,
    present_lcms_features: list[str],
) -> np.ndarray:
    """
    Compute a per-candidate multiplicative weight in (0, 1] based on
    available LC-MS/MS evidence.

    Each feature is min-max normalized to [0, 1]. Features where all values
    are identical (no information) are skipped. Features in
    ``_LCMS_INVERT_FEATURES`` (q-value, PEP) are inverted so that
    lower-is-better becomes higher-is-better before normalization, with NaN
    filled at 1.0 (worst possible value). ``source_lcms_confirmed`` receives
    2x weight (``_LCMS_PRIOR_WEIGHTS``). Returns the weighted mean of
    normalized features, or 1.0 if no informative features are present.
    """
    normed: list[np.ndarray] = []
    weights: list[float] = []

    for feat in present_lcms_features:
        if feat not in candidates_df.columns:
            continue
        nan_fill = _LCMS_NAN_FILL.get(feat, 0.0)
        col = candidates_df[feat].fillna(nan_fill).values.astype(float)
        if feat in _LCMS_INVERT_FEATURES:
            col = 1.0 - np.clip(col, 0.0, 1.0)
        col_min, col_max = col.min(), col.max()
        if col_max - col_min < 1e-12:
            continue
        normed.append((col - col_min) / (col_max - col_min))
        weights.append(_LCMS_PRIOR_WEIGHTS.get(feat, 1.0))

    if not normed:
        return np.ones(len(candidates_df))

    weights_arr = np.array(weights, dtype=float)
    stacked = np.stack(normed, axis=0)  # (n_features, n_candidates)
    return (weights_arr[:, None] * stacked).sum(axis=0) / weights_arr.sum()


def _rescore_svm(
    psm_list,
    features_df: pd.DataFrame,
    intrinsic_feature_names: list[str],
    train_fdr: float,
):
    """Run mokapot PercolatorModel on MALDI-intrinsic features.

    Returns
    -------
    (conf_obj, all_scores)
        ``conf_obj`` is the mokapot LinearConfidence object (for peptide/protein
        level results). ``all_scores`` is a numpy array of SVM decision scores
        for ALL candidates (targets + decoys) in ``features_df`` row order,
        or ``None`` if score extraction fails.
    """
    from mokapot import brew
    from mokapot.model import PercolatorModel
    from ms2rescore.rescoring_engines.mokapot import convert_psm_list

    lin = convert_psm_list(psm_list, feature_names=intrinsic_feature_names)
    model = PercolatorModel(train_fdr=train_fdr, max_iter=10)
    result = brew(lin, model=model, test_fdr=0.05)
    # brew always returns (confidence, [fold_models])
    conf_obj, trained_models = result if isinstance(result, tuple) else (result, [])

    # Score ALL candidates (targets + decoys) for TDC reweighting.
    # brew trains k fold models; average their scores across folds for stability.
    all_scores = None
    try:
        from mokapot.model import _get_scores
        fold_scores = []
        for fm in trained_models:
            if not fm.is_trained:
                continue
            # convert_psm_list prefixes feature names with "feature:"
            raw_names = [f.removeprefix("feature:") for f in fm.features]
            present_raw = [f for f in raw_names if f in features_df.columns]
            if not present_raw:
                continue
            X_all = features_df[present_raw].fillna(0.0).values.astype(float)
            X_scaled = fm.scaler.transform(X_all)
            fold_scores.append(_get_scores(fm.estimator, X_scaled))
        if fold_scores:
            all_scores = np.mean(fold_scores, axis=0)
        else:
            logger.warning("No trained fold models found — LC-MS/MS prior reweighting will be skipped.")
    except Exception as exc:
        logger.warning(
            f"Could not extract SVM scores ({exc}). "
            "LC-MS/MS prior reweighting will be skipped."
        )
    return conf_obj, all_scores


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
        & (
            df.get("ppm_error_abs", pd.Series(np.inf, index=df.index))
            < init_ppm_threshold
        )
        & (
            df.get("theo_isotope_cosine", pd.Series(0.0, index=df.index))
            > init_isotope_threshold
        )
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


def _feature_level_tdc(
    features_df: pd.DataFrame,
    scores: np.ndarray,
    feature_col: str = "feature_mz",
) -> tuple[np.ndarray, np.ndarray]:
    """Per-feature TDC q-values.

    For each MALDI feature the winner is the highest-scoring candidate
    regardless of target/decoy status. TDC is applied over features ranked
    by their winning score. Q-values are propagated to all candidates at
    each feature (non-winners inherit their feature's q-value so the full
    candidate table remains annotated).

    Returns
    -------
    q_values : np.ndarray, shape (n_candidates,)
    is_tdc_winner : np.ndarray[bool], shape (n_candidates,)
    """
    df = pd.DataFrame({"_score": scores, "_feat": features_df[feature_col].values})
    df["_is_decoy"] = features_df["is_decoy"].values.astype(bool)

    winner_pos = df.groupby("_feat")["_score"].idxmax()
    winner_scores = df.loc[winner_pos, "_score"].values
    winner_is_decoy = df.loc[winner_pos, "_is_decoy"].values

    order = np.argsort(-winner_scores)
    n_target_cum = np.cumsum(~winner_is_decoy[order]).astype(float)
    n_decoy_cum = np.cumsum(winner_is_decoy[order]).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        fdr = np.where(n_target_cum > 0, n_decoy_cum / n_target_cum, 1.0)
    qval_sorted = np.minimum.accumulate(fdr[::-1])[::-1].clip(max=1.0)
    feat_qvals = np.empty_like(qval_sorted)
    feat_qvals[order] = qval_sorted

    # q-values are only meaningful for the per-feature winner; NaN for all others
    q_values = np.full(len(df), np.nan)
    q_values[winner_pos.values] = feat_qvals

    is_tdc_winner = np.zeros(len(df), dtype=bool)
    is_tdc_winner[winner_pos.values] = True

    return q_values, is_tdc_winner


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
    compute_generative: bool = True,
    init_ppm_threshold: float = 2.0,
    init_isotope_threshold: float = 0.7,
    lcms_proteins_path: str | None = None,
    lcms_peptides_path: str | None = None,
    lcms_psms_path: str | None = None,
    lcms_id_format: str = "percolator",
    protein_fdr: float = 0.01,
    peptide_fdr: float = 0.01,
    extra_fasta_path: str | None = None,
    verbose: bool = False,
    output_dir: str = "ms1rescore_output",
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
        Rescoring backend: "svm" (mokapot PercolatorModel, default),
        "catboost" (semi-supervised CatBoostRanker), or "generative"
        (probabilistic generative scorer, no training). SVM and CatBoost use
        only MALDI_INTRINSIC_FEATURES for training. LC-MS/MS evidence is
        applied as a multiplicative prior after scoring.
    compute_generative
        When True and model is "svm" or "catboost", run the generative scorer
        first and add its ranking features (generative_score,
        generative_score_rank, generative_score_gap, generative_score_z) to
        MALDI_INTRINSIC_FEATURES before training. Default True.
    init_ppm_threshold
        CatBoost only: ppm_error_abs threshold for the initial positive seed.
    init_isotope_threshold
        CatBoost only: theo_isotope_cosine threshold for the initial positive seed.
    lcms_peptides_path
        Path to LC-MS/MS peptide-level identification results. When provided,
        activates Strategy C: candidates are generated by digesting only the
        identified proteins plus the directly identified peptide sequences,
        instead of the full FASTA (Strategy A). Falls back to Strategy A if no
        identified proteins are found in the FASTA.
    lcms_proteins_path
        Path to LC-MS/MS protein-level results (optional; proteins derived from
        peptide table when omitted).
    lcms_psms_path
        Path to LC-MS/MS PSM-level file for RT/intensity aggregation (optional).
    lcms_id_format
        Format of the LC-MS/MS ID files: ``"percolator"`` (default),
        ``"mzidentml"``, or ``"psm_utils"``.
    protein_fdr
        Protein FDR threshold for Strategy C protein filtering (default 0.01).
    peptide_fdr
        Peptide FDR threshold for Strategy C candidate inclusion (default 0.01).
    extra_fasta_path
        Optional additional FASTA file (e.g. contaminants). All proteins in
        this file are always included in the candidate database regardless of
        LC-MS/MS identification status. Works with both Strategy A and C.
        Proteins already present in the primary database (by peptide sequence)
        are not duplicated; when the same peptide appears in both, the entry
        from the primary digest (with any LC-MS/MS evidence) is kept.

    Returns
    -------
    tuple of (psm_list, result_df, feature_names)
        ``result_df`` is a DataFrame with columns ``[peptide, feature_idx,
        is_decoy, svm_score/catboost_score, q_value, reweighted_score,
        reweighted_q_value]`` for both backends. ``q_value`` and
        ``reweighted_q_value`` are TDC q-values computed over all candidates
        (targets + decoys). The reweighted values incorporate the LC-MS/MS
        prior multiplicative weight.
    """
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)

    def _cache(name):
        return os.path.join(cache_dir, name) if cache_dir else None

    # --- Step 1: Candidate generation ---
    if lcms_peptides_path is not None:
        from ms1rescore.lcms_ids import parse_lcms_ids

        logger.info(
            "Step 1: Strategy C — parsing LC-MS/MS IDs and digesting identified proteins..."
        )
        lcms_ids = parse_lcms_ids(
            proteins_path=lcms_proteins_path,
            peptides_path=lcms_peptides_path,
            psms_path=lcms_psms_path,
            protein_fdr=protein_fdr,
            peptide_fdr=peptide_fdr,
            format=lcms_id_format,
        )
        if verbose:
            logger.debug("Writing parsed LC-MS/MS IDs to debug_lcms_ids.tsv")
            lcms_ids.peptides.to_csv(
                f"{output_dir}/5_debug_lcms_ids.tsv", sep="\t", index=False
            )
        peptide_db = digest_identified_proteins(
            fasta_path,
            lcms_ids,
            missed_cleavages=missed_cleavages,
            min_length=min_length,
            max_length=max_length,
        )
        if verbose:
            logger.debug(
                f"Writing digested peptide database from identified proteins to {output_dir}/6_debug_peptide_db.tsv"
            )
            pd.DataFrame(peptide_db).to_csv(
                f"{output_dir}/6_debug_peptide_db.tsv", sep="\t", index=False
            )
        if len(peptide_db) == 0:
            logger.warning(
                "  Strategy C returned 0 candidates — falling back to Strategy A (full FASTA)"
            )
            peptide_db = digest_fasta(
                fasta_path,
                missed_cleavages=missed_cleavages,
                min_length=min_length,
                max_length=max_length,
                generate_decoys=True,
            )
            if verbose:
                logger.debug(
                    f"Writing digested peptide database from full FASTA to {output_dir}/7_debug_peptide_db_full.tsv"
                )
                pd.DataFrame(peptide_db).to_csv(
                    f"{output_dir}/7_debug_peptide_db_full.tsv", sep="\t", index=False
                )
    else:
        logger.info("Step 1: Strategy A — digesting full FASTA...")
        peptide_db = digest_fasta(
            fasta_path,
            missed_cleavages=missed_cleavages,
            min_length=min_length,
            max_length=max_length,
            generate_decoys=True,
        )
        if verbose:
            logger.debug(
                f"Writing digested peptide database from full FASTA to {output_dir}/8_debug_peptide_db_full.tsv"
            )
            pd.DataFrame(peptide_db).to_csv(
                f"{output_dir}/8_debug_peptide_db_full.tsv", sep="\t", index=False
            )

    # --- Step 1b: Merge extra FASTA (contaminants / spike-ins) ---
    if extra_fasta_path is not None:
        logger.info(f"Step 1b: Merging extra FASTA: {extra_fasta_path}")
        extra_db = digest_fasta(
            extra_fasta_path,
            missed_cleavages=missed_cleavages,
            min_length=min_length,
            max_length=max_length,
            generate_decoys=True,
        )
        existing_seqs = set(peptide_db["peptide"].values)
        extra_db = extra_db[~extra_db["peptide"].isin(existing_seqs)].copy()
        n_target_extra = int((~extra_db["is_decoy"]).sum())
        n_decoy_extra = int(extra_db["is_decoy"].sum())
        peptide_db = pd.concat([peptide_db, extra_db], ignore_index=True)
        logger.info(
            f"  Added {n_target_extra} target + {n_decoy_extra} decoy peptide entries "
            f"from {extra_fasta_path!r}"
        )

    maldi_intensities = None
    maldi_intensities_p90 = None
    maldi_intensities_sum = None
    if spatial_features is not None:
        if "intensity_p90" in spatial_features.columns:
            maldi_intensities_p90 = spatial_features["intensity_p90"].to_numpy(dtype=np.float32)
        if "intensity_sum" in spatial_features.columns:
            maldi_intensities_sum = spatial_features["intensity_sum"].to_numpy(dtype=np.float32)
        if "mean_intensity" in spatial_features.columns:
            maldi_intensities = spatial_features["mean_intensity"].to_numpy(dtype=np.float32)
    elif ion_images is not None:
        maldi_intensities = np.array(
            [img[img > 0].mean() if (img > 0).any() else 0.0 for img in ion_images]
        )
    candidates = match_to_maldi_features(
        maldi_mzs,
        peptide_db,
        ppm_tolerance,
        maldi_intensities=maldi_intensities,
        maldi_intensities_p90=maldi_intensities_p90,
        maldi_intensities_sum=maldi_intensities_sum,
    )
    if verbose:
        logger.debug(f"Writing matched candidates to {output_dir}/9_debug_candidates.tsv")
        candidates.to_csv(f"{output_dir}/9_debug_candidates.tsv", sep="\t", index=False)

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
    # Write to file in human-readable format for debugging (in case of issues with the LC-MS/MS loading)
    if verbose:
        logger.debug(f"Writing LC-MS/MS data to {output_dir}/10_debug_lcms_data.pkl")
        with open(f"{output_dir}/10_debug_lcms_data.pkl", "wb") as f:
            pickle.dump(lcms_data, f)

    # --- Step 3: MS2PIP predictions ---
    logger.info("Step 3: Finding MS2 matches and running MS2PIP...")
    from ms1rescore.lcms_evidence import _find_matching_ms2_scans
    from ms1rescore.utils import mz_to_mass

    # For each unique MALDI feature m/z, find matching MS2 scans within ppm tolerance.
    feature_ms2_charges = {}
    for mz in candidates["feature_mz"].unique():
        neutral_mass = mz_to_mass(mz, charge=1)
        scan_idxs = _find_matching_ms2_scans(neutral_mass, lcms_data, ppm_tolerance)
        if scan_idxs:
            # Store the set of observed precursor charges for this feature's MS2 matches
            feature_ms2_charges[mz] = set(
                int(lcms_data.ms2_precursor_charge[i]) for i in scan_idxs
            )
    logger.info(
        f"  {len(feature_ms2_charges)}/{candidates['feature_mz'].nunique()} features have MS2 matches"
    )

    peptide_charge_pairs = set()
    # For each feature with MS2 matches, get the candidate peptides and observed charges to run MS2PIP on.
    for mz, charges in feature_ms2_charges.items():
        peps = candidates[candidates["feature_mz"] == mz]["peptide"].unique()
        for pep in peps:
            for c in charges:
                peptide_charge_pairs.add((pep, c))
    logger.info(
        f"  {len(peptide_charge_pairs)} unique (peptide, charge) pairs for MS2PIP"
    )

    ms2pip_cache = get_ms2pip_predictions(
        list(peptide_charge_pairs),
        model="HCD",
        cache_path=_cache("ms2pip_predictions.pkl"),
    )

    # --- Step 4: DeepLC predictions ---
    logger.info("Step 4: Computing DeepLC predictions...")
    unique_peptides = candidates["peptide"].unique().tolist()
    deeplc_model = None
    # If a PD .msf file is provided, finetune DeepLC on the identified peptides from that file. Otherwise, use the pretrained model.
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

    if verbose:
        logger.debug(
            f"Writing LC-MS/MS evidence to {output_dir}/11_debug_lcms_evidence.tsv"
        )
        pd.DataFrame(lcms_evidence).T.to_csv(
            f"{output_dir}/11_debug_lcms_evidence.tsv", sep="\t", index=True
        )

    # --- Step 6: Extract LC-MS/MS envelopes from XIC best scans ---
    lcms_envelopes_xic = None
    if maldi_envelopes is not None:
        logger.info("Step 6: Extracting LC-MS/MS envelopes from XIC scans...")
        from ms1rescore.lcms_evidence import _extract_ms1_envelope

        unique_feature_mzs = candidates["feature_mz"].unique()
        xic_cache = extract_all_xics(unique_feature_mzs, lcms_data, ppm_tolerance)
        if verbose:
            logger.debug(f"Writing extracted XICs to {output_dir}/12_debug_xic_cache.pkl")
            with open(f"{output_dir}/12_debug_xic_cache.pkl", "wb") as f:
                pickle.dump(xic_cache, f)
        lcms_envelopes_xic = {}
        for mz in unique_feature_mzs:
            rts, ints = xic_cache.get(mz, (np.array([]), np.array([])))
            if len(ints) > 0 and ints.max() > 0:
                best_xic_idx = np.argmax(ints)
                best_rt = rts[best_xic_idx]
                best_ms1_idx = np.argmin(np.abs(lcms_data.ms1_rts - best_rt))
                env = _extract_ms1_envelope(
                    mz, best_ms1_idx, lcms_data, charge=1, n_peaks=3
                )
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
    if verbose:
        logger.debug(f"Writing computed features to {output_dir}/13_debug_features.tsv")
        features_df.to_csv(f"{output_dir}/13_debug_features.tsv", sep="\t", index=False)

    # --- Step 7b: Generative scoring (optional pre-step) ---
    has_generative = False
    if (model in ("svm", "catboost") and compute_generative) or model == "generative":
        from ms1rescore.probabilistic_scorer import run_generative_scoring

        logger.info("Step 7b: Running generative scorer...")
        feature_col = "feature_idx" if "feature_idx" in features_df.columns else "feature_mz"
        features_df = run_generative_scoring(features_df, feature_col=feature_col)
        has_generative = True
        if verbose:
            logger.debug(
                f"Writing features with generative scores to "
                f"{output_dir}/13b_debug_features_generative.tsv"
            )
            features_df.to_csv(
                f"{output_dir}/13b_debug_features_generative.tsv", sep="\t", index=False
            )

    feature_names = get_feature_names(
        has_spatial=spatial_features is not None,
        has_ion_images=ion_images is not None,
        has_envelopes=maldi_envelopes is not None and lcms_envelopes_xic is not None,
        has_generative=has_generative,
    )
    logger.debug(f"Selected feature names: {feature_names}")

    # Intrinsic features that are actually present in the DataFrame
    intrinsic_present = [
        f for f in MALDI_INTRINSIC_FEATURES if f in features_df.columns
    ]
    lcms_present = [f for f in LCMS_PRIOR_FEATURES if f in features_df.columns]

    # --- Generative-only backend: return without training ---
    if model == "generative":
        from ms1rescore.probabilistic_scorer import estimate_fdr as _gen_fdr

        logger.info("Step 9: Generative backend — margin-based TDC FDR...")
        feature_col = "feature_idx" if "feature_idx" in features_df.columns else "feature_mz"

        # Base q-values already computed by run_generative_scoring (margin-based)
        q_values = features_df["generative_q_value"].values
        gen_scores = features_df["generative_score"].values

        # Reweighted: apply LC-MS/MS prior to raw scores, re-run margin TDC
        lcms_prior = compute_lcms_prior(features_df, lcms_present)
        reweighted_scores = gen_scores * lcms_prior
        rw_df = features_df.copy()
        rw_df["_rw_score"] = reweighted_scores
        rw_df = _gen_fdr(rw_df, score_col="_rw_score", feature_col=feature_col)
        reweighted_q = rw_df["generative_q_value"].values

        is_decoy = features_df["is_decoy"].values.astype(bool)
        result_df = pd.DataFrame(
            {
                "peptide": features_df["peptide"].values,
                "feature_idx": features_df.get(
                    "feature_idx", pd.Series(range(len(features_df)))
                ).values,
                "is_decoy": is_decoy,
                "generative_score": gen_scores,
                "Delta_m": features_df.get("Delta_m", pd.Series(np.nan, index=features_df.index)).values,
                "q_value": q_values,
                "reweighted_score": reweighted_scores,
                "reweighted_q_value": reweighted_q,
                "is_tdc_winner": features_df.get("is_tdc_winner", pd.Series(False, index=features_df.index)).values,
            }
        )

        for fdr_threshold in [0.01, 0.05, 0.10]:
            n = (result_df["is_tdc_winner"] & (result_df["q_value"] <= fdr_threshold)).sum()
            n_rw = (result_df["is_tdc_winner"] & (result_df["reweighted_q_value"] <= fdr_threshold)).sum()
            logger.info(
                f"  At {fdr_threshold*100:.0f}% FDR: {n} winners (base), "
                f"{n_rw} winners (reweighted)"
            )

        psm_list = candidates_to_psm_list(features_df)
        return psm_list, result_df, feature_names

    # --- Step 8: Build PSMList ---
    logger.info("Step 8: Building PSMList...")
    psm_list = candidates_to_psm_list(features_df)
    if verbose:
        logger.debug(
            f"Writing PSM list to {output_dir}/14_debug_psm_list.pkl for mokapot input"
        )
        psm_list_df = psm_list.to_dataframe()
        psm_list_df.to_csv(f"{output_dir}/14_debug_psm_list.tsv", sep="\t", index=False)

    populate_psm_features(psm_list, features_df, feature_names)

    logger.info(
        f"  {len(feature_names)} features ({len(intrinsic_present)} intrinsic, {len(lcms_present)} LC-MS/MS prior)"
    )

    # --- Step 9: Rescoring ---
    logger.info(f"Step 9: Running rescoring (model='{model}')...")

    if model == "svm":
        # Pass only intrinsic features to the SVM
        populate_psm_features(psm_list, features_df, intrinsic_present)
        if verbose:
            logger.debug(
                f"Writing PSM list with intrinsic features to {output_dir}/14_debug_psm_list_after_intrinsic.pkl for mokapot input"
            )
            psm_list_intrinsic_df = psm_list.to_dataframe()
            psm_list_intrinsic_df.to_csv(
                f"{output_dir}/14_debug_psm_list_after_intrinsic.tsv",
                sep="\t",
                index=False,
            )
        conf_obj, all_scores = _rescore_svm(
            psm_list, features_df, intrinsic_present, train_fdr
        )
        if verbose:
            logger.debug(
                f"Writing mokapot confidence object and scores to {output_dir}/15_debug_mokapot_conf.pkl"
            )
            with open(f"{output_dir}/15_debug_mokapot_conf.pkl", "wb") as f:
                pickle.dump(conf_obj, f)
            # all_scores may be None if score extraction fails; write an empty array in that case for debugging
            with open(f"{output_dir}/15_debug_mokapot_scores.pkl", "wb") as f:
                pickle.dump(all_scores if all_scores is not None else np.array([]), f)

        is_decoy = features_df["is_decoy"].values.astype(bool)
        lcms_prior = compute_lcms_prior(features_df, lcms_present)
        feature_col = "feature_mz" if "feature_mz" in features_df.columns else "feature_idx"

        if all_scores is not None:
            q_values, is_winner = _feature_level_tdc(features_df, all_scores, feature_col)
            reweighted_scores = all_scores * lcms_prior
            reweighted_q, _ = _feature_level_tdc(features_df, reweighted_scores, feature_col)
        else:
            logger.warning(
                "SVM score extraction failed — q_value and reweighted_q_value will be NaN."
            )
            all_scores = np.zeros(len(features_df))
            reweighted_scores = np.zeros(len(features_df))
            q_values = np.full(len(features_df), np.nan)
            reweighted_q = np.full(len(features_df), np.nan)
            is_winner = np.zeros(len(features_df), dtype=bool)

        result_df = pd.DataFrame(
            {
                "peptide": features_df["peptide"].values,
                "feature_mz": features_df["feature_mz"].values if "feature_mz" in features_df.columns else np.nan,
                "feature_idx": features_df.get(
                    "feature_idx", pd.Series(range(len(features_df)))
                ).values,
                "is_decoy": is_decoy,
                "svm_score": all_scores,
                "q_value": q_values,
                "is_tdc_winner": is_winner,
                "reweighted_score": reweighted_scores,
                "reweighted_q_value": reweighted_q,
            }
        )

        for fdr_threshold in [0.01, 0.05, 0.10]:
            n = (is_winner & ~is_decoy & (q_values <= fdr_threshold)).sum()
            n_rw = (is_winner & ~is_decoy & (reweighted_q <= fdr_threshold)).sum()
            logger.info(
                f"  At {fdr_threshold*100:.0f}% FDR: {n} target features (base), "
                f"{n_rw} target features (reweighted)"
            )

        return psm_list, result_df, feature_names

    elif model == "catboost":
        scores = _rescore_catboost(
            features_df,
            intrinsic_present,
            train_fdr=train_fdr,
            init_ppm_threshold=init_ppm_threshold,
            init_isotope_threshold=init_isotope_threshold,
        )
        if verbose:
            logger.debug(
                f"Writing CatBoost scores to {output_dir}/16_debug_catboost_scores.pkl"
            )
            with open(f"{output_dir}/16_debug_catboost_scores.pkl", "wb") as f:
                pickle.dump(scores, f)

        is_decoy = features_df["is_decoy"].values.astype(bool)
        feature_col = "feature_mz" if "feature_mz" in features_df.columns else "feature_idx"
        q_values, is_winner = _feature_level_tdc(features_df, scores, feature_col)

        # LC-MS/MS prior reweight
        lcms_prior = compute_lcms_prior(features_df, lcms_present)
        reweighted_scores = scores * lcms_prior
        reweighted_q, _ = _feature_level_tdc(features_df, reweighted_scores, feature_col)

        result_df = pd.DataFrame(
            {
                "peptide": features_df["peptide"].values,
                "feature_mz": features_df["feature_mz"].values if "feature_mz" in features_df.columns else np.nan,
                "feature_idx": features_df.get(
                    "feature_idx", pd.Series(range(len(features_df)))
                ).values,
                "is_decoy": is_decoy,
                "catboost_score": scores,
                "q_value": q_values,
                "is_tdc_winner": is_winner,
                "reweighted_score": reweighted_scores,
                "reweighted_q_value": reweighted_q,
            }
        )

        for fdr_threshold in [0.01, 0.05, 0.10]:
            n = (is_winner & ~is_decoy & (q_values <= fdr_threshold)).sum()
            n_rw = (is_winner & ~is_decoy & (reweighted_q <= fdr_threshold)).sum()
            logger.info(
                f"  At {fdr_threshold*100:.0f}% FDR: {n} target features (base), "
                f"{n_rw} target features (reweighted)"
            )

        return psm_list, result_df, feature_names

    else:
        raise ValueError(f"Unknown model '{model}'. Choose 'svm', 'catboost', or 'generative'.")
