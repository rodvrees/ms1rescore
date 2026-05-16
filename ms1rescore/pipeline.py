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
    PROTEIN_LEVEL_FEATURES,
    SPATIAL_PRIOR_FEATURES,
    candidates_to_psm_list,
    compute_all_features,
    get_feature_names,
    populate_psm_features,
)
from ms1rescore.lcms_evidence import (
    compute_all_lcms_evidence,
    finetune_deeplc,
    finetune_deeplc_from_df,
    get_deeplc_predictions,
    get_ms2pip_predictions,
    load_lcms_data,
)

logger = logging.getLogger(__name__)

# Spatial prior: Geary's C is lower-is-better (< 1 = positive autocorrelation)
_SPATIAL_INVERT_FEATURES = frozenset(["spatial_gearys_c"])


def compute_lcms_prior(
    candidates_df: pd.DataFrame,
    present_lcms_features: list[str],
) -> np.ndarray:
    """
    Compute a per-candidate multiplicative weight in (0, 1] based on
    available LC-MS/MS evidence.

    Each feature is min-max normalized to [0, 1]. Features where all values
    are identical (no information) are skipped. Returns the mean of normalized
    features, or 1.0 if no informative features are present.
    """
    normed: list[np.ndarray] = []

    for feat in present_lcms_features:
        if feat not in candidates_df.columns:
            continue
        col = candidates_df[feat].fillna(0.0).values.astype(float)
        col_min, col_max = col.min(), col.max()
        if col_max - col_min < 1e-12:
            continue
        normed.append((col - col_min) / (col_max - col_min))

    if not normed:
        return np.ones(len(candidates_df))

    return np.stack(normed, axis=0).mean(axis=0)


def compute_spatial_prior(
    candidates_df: pd.DataFrame,
    present_spatial_features: list[str],
) -> np.ndarray:
    """
    Compute a per-candidate multiplicative weight in (0, 1] based on
    spatial quality of the ion image.

    All spatial features are feature-level (identical for all candidates at the
    same m/z), so they cannot discriminate within a feature. Applied as a
    post-scoring prior rather than as ranker inputs. Each feature is min-max
    normalized; ``spatial_gearys_c`` is negated before normalization (lower
    Geary's C = positive autocorrelation = better). Returns 1.0 if no
    informative spatial features are present.
    """
    normed: list[np.ndarray] = []

    for feat in present_spatial_features:
        if feat not in candidates_df.columns:
            continue
        col = candidates_df[feat].fillna(0.0).values.astype(float)
        col = np.where(np.isfinite(col), col, 0.0)
        if feat in _SPATIAL_INVERT_FEATURES:
            col = -col
        col_min, col_max = col.min(), col.max()
        if col_max - col_min < 1e-12:
            continue
        normed.append((col - col_min) / (col_max - col_min))

    if not normed:
        return np.ones(len(candidates_df))

    return np.stack(normed, axis=0).mean(axis=0)


def _rescore_svm(
    psm_list,
    features_df: pd.DataFrame,
    intrinsic_feature_names: list[str],
    train_fdr: float,
):
    """Run mokapot PercolatorModel on MALDI-intrinsic features.

    Returns
    -------
    (conf_obj, all_scores, importances, importance_names)
        ``conf_obj`` is the mokapot LinearConfidence object. ``all_scores`` is
        a numpy array of SVM decision scores for ALL candidates, or ``None``.
        ``importances`` is the mean absolute SVM coefficient vector (or None).
        ``importance_names`` is the aligned feature name list (or None).
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
    importances = None
    importance_names = None
    try:
        from mokapot.model import _get_scores
        fold_scores = []
        fold_coefs = []
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
            try:
                fold_coefs.append(fm.estimator.coef_[0])
                if importance_names is None:
                    importance_names = present_raw
            except AttributeError:
                pass
        if fold_scores:
            all_scores = np.mean(fold_scores, axis=0)
        else:
            logger.warning("No trained fold models found — LC-MS/MS prior reweighting will be skipped.")
        if fold_coefs:
            importances = np.mean(fold_coefs, axis=0)
    except Exception as exc:
        logger.warning(
            f"Could not extract SVM scores ({exc}). "
            "LC-MS/MS prior reweighting will be skipped."
        )
    return conf_obj, all_scores, importances, importance_names


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

    Returns ``(scores, importances, feature_names_used)`` where ``scores`` is
    a 1-D array (higher = more likely correct), ``importances`` is the feature
    importance vector from the last iteration's model (or None), and
    ``feature_names_used`` is the aligned feature name list.
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
    model_cb = None

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

    importances = None
    if model_cb is not None:
        try:
            importances = model_cb.get_feature_importance()
        except Exception:
            pass
    return scores, importances, present


def _rescore_lda(
    features_df: pd.DataFrame,
    intrinsic_feature_names: list[str],
    init_ppm_threshold: float,
    seed_mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Semi-supervised LDA on MALDI-intrinsic features.

    Pre-processing: ±inf replaced with NaN, then median imputation and
    StandardScaler inside a sklearn Pipeline.  Pseudo-label iteration
    follows the same structure as _rescore_catboost.

    Seed positives: if ``seed_mask`` is provided (a boolean array aligned to
    ``features_df``) it is used directly as the initial positive set — intended
    for Round-2 where the seed is inherited from Round-1 q-values.  Otherwise
    targets with ppm_error_abs < init_ppm_threshold OR n_candidates == 1 are
    used.

    Returns ``(scores, importances, feature_names_used)`` where ``scores`` is
    a 1-D array (higher = more likely correct), ``importances`` is
    ``|coef_[0]|`` from the final LDA pipeline (or None), and
    ``feature_names_used`` is the aligned feature name list.
    """
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    df = features_df.reset_index(drop=True)
    present = [f for f in intrinsic_feature_names if f in df.columns]
    X_raw = df[present].values.astype(np.float64)
    X = np.where(np.isfinite(X_raw), X_raw, np.nan)  # ±inf → nan for imputer

    is_decoy = df["is_decoy"].values.astype(bool)
    is_target = ~is_decoy

    if seed_mask is None:
        ppm_col = df.get("ppm_error_abs", pd.Series(np.inf, index=df.index))
        n_cand_col = df.get("n_candidates", pd.Series(np.inf, index=df.index))
        seed_mask = (
            is_target
            & (
                (ppm_col < init_ppm_threshold)
                | (n_cand_col == 1)
            )
        ).values
        if not seed_mask.any():
            logger.warning("  LDA: no seed positives — falling back to top-ppm init")
            seed_mask = (is_target & (ppm_col < ppm_col[is_target].quantile(0.10))).values

    n_seed = seed_mask.sum()
    logger.info(f"  LDA: seed positives = {n_seed}, decoys = {is_decoy.sum()}")

    scores = np.zeros(len(df))
    prev_pos_size = -1
    pipe = None

    for iteration in range(5):
        pos_idx = np.where(seed_mask)[0]
        dec_idx = np.where(is_decoy)[0]
        train_idx = np.concatenate([pos_idx, dec_idx])
        y_train = np.concatenate([np.ones(len(pos_idx)), np.zeros(len(dec_idx))])

        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("lda", LinearDiscriminantAnalysis()),
        ])
        pipe.fit(X[train_idx], y_train)
        scores = pipe.decision_function(X).ravel()  # ensure 1-D for binary case

        q_values = _tdc_qvalues(scores, is_decoy)
        new_seed = is_target & (q_values <= 0.05)
        n_new = new_seed.sum()

        logger.info(
            f"  LDA iter {iteration + 1}: pseudo-positives = {n_new} (prev = {prev_pos_size})"
        )

        if n_new == 0:
            logger.warning("  LDA: no pseudo-positives at q≤0.05 — stopping early")
            break

        change = abs(n_new - prev_pos_size) / max(prev_pos_size, 1)
        if prev_pos_size >= 0 and change < 0.01:
            logger.info("  LDA: converged")
            break

        prev_pos_size = n_new
        seed_mask = new_seed.values if hasattr(new_seed, "values") else new_seed

    importances = None
    if pipe is not None:
        try:
            importances = np.abs(pipe["lda"].coef_[0])
        except Exception:
            pass
    return scores, importances, present


# def _feature_level_tdc(
#     features_df: pd.DataFrame,
#     scores: np.ndarray,
#     feature_col: str = "feature_mz",
# ) -> tuple[np.ndarray, np.ndarray]:
#     """Per-feature TDC q-values.

#     For each MALDI feature the winner is the highest-scoring candidate
#     regardless of target/decoy status. TDC is applied over features ranked
#     by their winning score. Q-values are propagated to all candidates at
#     each feature (non-winners inherit their feature's q-value so the full
#     candidate table remains annotated).

#     Returns
#     -------
#     q_values : np.ndarray, shape (n_candidates,)
#     is_tdc_winner : np.ndarray[bool], shape (n_candidates,)
#     """
#     df = pd.DataFrame({"_score": scores, "_feat": features_df[feature_col].values})
#     df["_is_decoy"] = features_df["is_decoy"].values.astype(bool)

#     winner_pos = df.groupby("_feat")["_score"].idxmax()
#     winner_scores = df.loc[winner_pos, "_score"].values
#     winner_is_decoy = df.loc[winner_pos, "_is_decoy"].values

#     order = np.argsort(-winner_scores)
#     n_target_cum = np.cumsum(~winner_is_decoy[order]).astype(float)
#     n_decoy_cum = np.cumsum(winner_is_decoy[order]).astype(float)
#     with np.errstate(invalid="ignore", divide="ignore"):
#         fdr = np.where(n_target_cum > 0, n_decoy_cum / n_target_cum, 1.0)
#     qval_sorted = np.minimum.accumulate(fdr[::-1])[::-1].clip(max=1.0)
#     feat_qvals = np.empty_like(qval_sorted)
#     feat_qvals[order] = qval_sorted

#     # q-values are only meaningful for the per-feature winner; NaN for all others
#     q_values = np.full(len(df), np.nan)
#     q_values[winner_pos.values] = feat_qvals

#     is_tdc_winner = np.zeros(len(df), dtype=bool)
#     is_tdc_winner[winner_pos.values] = True

#     return q_values, is_tdc_winner


def _tdc_qvalues(scores: np.ndarray, is_decoy: np.ndarray) -> np.ndarray:
    """
    Compute per-candidate target-decoy q-values (Storey/Käll TDC).

    Sort by descending score with a stable sort, compute cumulative
    FDR = (1 + n_decoy) / max(n_target, 1) at each position (the +1
    correction is the standard Storey/Käll adjustment for small-N), then
    take the minimum FDR seen at or below each score (rolling min from
    the tail).
    """
    scores = np.asarray(scores)
    is_decoy = np.asarray(is_decoy).astype(bool)
    order = np.argsort(-scores, kind="stable")
    n_target_cum = np.cumsum(~is_decoy[order]).astype(float)
    n_decoy_cum = np.cumsum(is_decoy[order]).astype(float)

    fdr = (n_decoy_cum + 1.0) / np.maximum(n_target_cum, 1.0)

    # q-value: minimum FDR at or below this score (monotone from the tail)
    qval_ordered = np.minimum.accumulate(fdr[::-1])[::-1]

    # Map back to original order
    q_values = np.empty_like(qval_ordered)
    q_values[order] = qval_ordered
    return q_values


def _select_feature_winners(
    features_df: pd.DataFrame,
    scores: np.ndarray,
    feature_col: str,
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    For each MALDI feature select the candidate with the highest round-1 score.

    Returns
    -------
    winner_pos : np.ndarray[int]
        Integer positions (iloc-style) in ``features_df`` of the selected winners.
    winners_df : pd.DataFrame
        Subset of ``features_df`` with one row per feature, reset index.
    """
    score_series = pd.Series(scores, index=features_df.index)
    winner_idx = score_series.groupby(features_df[feature_col].values).idxmax().values
    # Convert label-based index values to positional indices
    winner_pos = features_df.index.get_indexer(winner_idx)
    winners_df = features_df.loc[winner_idx].copy().reset_index(drop=True)
    return winner_pos, winners_df


def rescore(
    fasta_path: str,
    maldi_mzs: np.ndarray,
    mzml_paths: list[str],
    spatial_features: pd.DataFrame | None = None,
    ion_images: np.ndarray | None = None,
    ion_image_mzs: np.ndarray | None = None,
    extra_ion_images: dict | None = None,
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
    init_ppm_threshold: float = 5.0,
    init_isotope_threshold: float = 0.7,
    lcms_proteins_path: str | None = None,
    lcms_peptides_path: str | None = None,
    lcms_psms_path: str | None = None,
    lcms_id_format: str = "percolator",
    psm_utils_reader: str | None = None,
    protein_fdr: float = 0.01,
    peptide_fdr: float = 0.01,
    extra_fasta_path: str | None = None,
    use_protein_level_features: bool = False,
    verbose: bool = False,
    output_dir: str = "ms1rescore_output",
    debug_dir: str | None = None,
    n_debug: int = 50,
    debug_seed: int = 42,
    observed_ccs_per_feature: dict | None = None,
    im2deep_calibration: str = "linear",
    digest: bool = False,
    gt_peptides: list[str] | None = None,
    maldi_intensities: np.ndarray | None = None,
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
    extra_ion_images
        Dict of ion images at shifted m/z positions for colocalization features (optional).
        Keys: "m1", "m2" (isotopologue) and "na", "k", "chca" (adduct).
        Each value has shape (n_features, H, W). When provided, Pearson r is computed
        directly without requiring these peaks to be in the feature list.
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
        ``"mzidentml"``, ``"psm_utils"``, or ``"msf"``.
    psm_utils_reader
        Only used when ``lcms_id_format="psm_utils"``. Either a psm_utils
        filetype key (e.g. ``"maxquant"``) or a reader class name (e.g.
        ``"MSMSReader"``). When ``None``, auto-detection from filename is
        attempted.
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
    # Default (digest=False): use only LC-MS/MS identified peptides as candidates.
    # With digest=True: also digest the provided FASTA for additional candidates.
    lcms_ids = None  # set below if lcms_peptides_path is provided
    if lcms_peptides_path is not None:
        from ms1rescore.lcms_ids import parse_lcms_ids

        logger.info("Step 1: Parsing LC-MS/MS identifications...")
        lcms_ids = parse_lcms_ids(
            proteins_path=lcms_proteins_path,
            peptides_path=lcms_peptides_path,
            psms_path=lcms_psms_path,
            protein_fdr=protein_fdr,
            peptide_fdr=peptide_fdr,
            format=lcms_id_format,
            psm_utils_reader=psm_utils_reader,
        )
        if verbose:
            logger.debug("Writing parsed LC-MS/MS IDs to debug_lcms_ids.tsv")
            lcms_ids.peptides.to_csv(
                f"{output_dir}/5_debug_lcms_ids.tsv", sep="\t", index=False
            )

        # Pass fasta_path only when --digest is active; None = LC-MS/MS peptides only.
        _digest_fasta_arg = fasta_path if digest else None
        if digest:
            logger.info(
                f"  --digest active: digesting identified proteins from {fasta_path}"
            )
        else:
            logger.info("  Using LC-MS/MS identified peptides only (pass --digest to also digest FASTA).")

        peptide_db = digest_identified_proteins(
            _digest_fasta_arg,
            lcms_ids,
            missed_cleavages=missed_cleavages,
            min_length=min_length,
            max_length=max_length,
        )
        if verbose:
            logger.debug(
                f"Writing peptide database to {output_dir}/6_debug_peptide_db.tsv"
            )
            pd.DataFrame(peptide_db).to_csv(
                f"{output_dir}/6_debug_peptide_db.tsv", sep="\t", index=False
            )
        if len(peptide_db) == 0 and digest and fasta_path:
            logger.warning(
                "  No candidates from identified proteins — falling back to full FASTA digest."
            )
            peptide_db = digest_fasta(
                fasta_path,
                missed_cleavages=missed_cleavages,
                min_length=min_length,
                max_length=max_length,
                generate_decoys=True,
            )
    elif digest and fasta_path:
        logger.info("Step 1: --digest active, no LC-MS/MS IDs — digesting full FASTA...")
        peptide_db = digest_fasta(
            fasta_path,
            missed_cleavages=missed_cleavages,
            min_length=min_length,
            max_length=max_length,
            generate_decoys=True,
        )
        if verbose:
            pd.DataFrame(peptide_db).to_csv(
                f"{output_dir}/7_debug_peptide_db_full.tsv", sep="\t", index=False
            )
    else:
        raise ValueError(
            "No candidate source available. Provide --lcms-peptides (or --msf) "
            "to use LC-MS/MS identified peptides as candidates, or add --digest "
            "with --fasta to perform an in-silico digest."
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

    _maldi_intensities_arr = None
    maldi_intensities_p90 = None
    maldi_intensities_sum = None
    if spatial_features is not None:
        if "intensity_p90" in spatial_features.columns:
            maldi_intensities_p90 = spatial_features["intensity_p90"].to_numpy(dtype=np.float32)
        if "intensity_sum" in spatial_features.columns:
            maldi_intensities_sum = spatial_features["intensity_sum"].to_numpy(dtype=np.float32)
        if "mean_intensity" in spatial_features.columns:
            _maldi_intensities_arr = spatial_features["mean_intensity"].to_numpy(dtype=np.float32)
    elif ion_images is not None:
        _maldi_intensities_arr = np.array(
            [img[img > 0].mean() if (img > 0).any() else 0.0 for img in ion_images]
        )
    # SCiLS CSV 'Intensity [Regions]' takes priority over raw-derived mean intensity.
    if maldi_intensities is not None:
        _maldi_intensities_arr = np.asarray(maldi_intensities, dtype=np.float32)
        logger.info("  Using SCiLS per-feature intensities for log_maldi_intensity")
    candidates = match_to_maldi_features(
        maldi_mzs,
        peptide_db,
        ppm_tolerance,
        maldi_intensities=_maldi_intensities_arr,
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

    # --- Steps 2–5: LC-MS/MS data (skipped entirely when no mzML is provided) ---
    lcms_evidence = {}

    if mzml_paths:
        # --- Step 2: Load LC-MS/MS data ---
        logger.info("Step 2: Loading LC-MS/MS data...")
        lcms_data = load_lcms_data(mzml_paths, cache_path=_cache("lcms_data.pkl"))
        if verbose:
            logger.debug(f"Writing LC-MS/MS data to {output_dir}/10_debug_lcms_data.pkl")
            with open(f"{output_dir}/10_debug_lcms_data.pkl", "wb") as f:
                pickle.dump(lcms_data, f)

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
        logger.info(
            f"  {len(peptide_charge_pairs)} unique (peptide, charge) pairs for MS2PIP"
        )

        ms2pip_cache = get_ms2pip_predictions(
            list(peptide_charge_pairs),
            model="timsTOF2024",
            cache_path=_cache("ms2pip_predictions.pkl"),
        )

        # --- Step 4: DeepLC predictions ---
        logger.info("Step 4: Computing DeepLC predictions...")
        unique_peptides = candidates["peptide"].unique().tolist()
        deeplc_model = None
        _model_cache = _cache("deeplc_model.pt")
        _model_was_cached = _model_cache is not None and os.path.exists(_model_cache)
        if msf_path:
            deeplc_model = finetune_deeplc(msf_path, cache_path=_model_cache)
        elif lcms_ids is not None:
            # rt_mean is in minutes after unit auto-detection in _join_psm_rt_intensity.
            deeplc_model = finetune_deeplc_from_df(
                lcms_ids.peptides, cache_path=_model_cache
            )
        _pred_cache = _cache("deeplc_predictions.pkl")
        # If model was freshly trained (not loaded from cache), the existing
        # predictions cache is stale — delete it so predictions are recomputed.
        if not _model_was_cached and _pred_cache is not None and os.path.exists(_pred_cache):
            os.remove(_pred_cache)
            logger.info("  Deleted stale DeepLC predictions cache (model was retrained)")
        deeplc_cache = get_deeplc_predictions(
            unique_peptides,
            model=deeplc_model,
            cache_path=_pred_cache,
        )

        # Estimate MS1 RT window from fine-tuning calibration error.
        # Window = 2 × 95th-percentile |predicted - observed| RT for calibration peptides.
        # This adapts to dataset-specific DeepLC calibration quality.
        rt_window_min = 0.0
        if deeplc_cache and lcms_ids is not None:
            ft_df = lcms_ids.peptides[["sequence", "rt_mean"]].dropna(subset=["rt_mean"])
            pred_rts_cal = np.array([deeplc_cache.get(seq, np.nan) for seq in ft_df["sequence"]])
            obs_rts_cal = ft_df["rt_mean"].values.astype(float)
            valid = ~np.isnan(pred_rts_cal) & ~np.isnan(obs_rts_cal)
            if valid.sum() > 10:
                p95_mae = float(np.percentile(np.abs(pred_rts_cal[valid] - obs_rts_cal[valid]), 95))
                rt_window_min = 2.0 * p95_mae
                logger.info(
                    f"  DeepLC RT window: 2 × p95 MAE = {rt_window_min:.3f} min "
                    f"({valid.sum()} calibration peptides, p95={p95_mae:.3f} min)"
                )

        # --- Step 5: Compute LC-MS/MS evidence ---
        # MS1 features are anchored by DeepLC predicted RT (fully symmetric:
        # targets and decoys receive identical treatment).
        logger.info("Step 5: Computing LC-MS/MS evidence features...")
        lcms_evidence = compute_all_lcms_evidence(
            candidates,
            lcms_data,
            ms2pip_cache,
            deeplc_cache,
            maldi_envelopes=maldi_envelopes,
            ppm_tolerance=ppm_tolerance,
            rt_window_min=rt_window_min,
        )

        if verbose:
            logger.debug(
                f"Writing LC-MS/MS evidence to {output_dir}/11_debug_lcms_evidence.tsv"
            )
            pd.DataFrame(lcms_evidence).T.to_csv(
                f"{output_dir}/11_debug_lcms_evidence.tsv", sep="\t", index=True
            )

    else:
        logger.info("Steps 2–5: No mzML files provided — skipping LC-MS/MS evidence.")

    # --- Step 6: Compute all features ---
    logger.info("Step 6: Computing all features...")
    features_df = compute_all_features(
        candidates,
        lcms_evidence=lcms_evidence,
        spatial_features=spatial_features,
        ion_images=ion_images,
        ion_image_mzs=ion_image_mzs,
        extra_ion_images=extra_ion_images,
        maldi_envelopes=maldi_envelopes,
        observed_ccs_per_feature=observed_ccs_per_feature,
        im2deep_calibration=im2deep_calibration,
    )
    if verbose:
        logger.debug(f"Writing computed features to {output_dir}/13_debug_features.tsv")
        features_df.to_csv(f"{output_dir}/13_debug_features.tsv", sep="\t", index=False)

    # --- Step 6b: Generative scoring (optional pre-step) ---
    has_generative = False
    if (model in ("svm", "catboost") and compute_generative) or model == "generative":
        from ms1rescore.probabilistic_scorer import run_generative_scoring

        logger.info("Step 6b: Running generative scorer...")
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
        has_generative=has_generative,
        has_ccs=observed_ccs_per_feature is not None,
    )
    logger.debug(f"Selected feature names: {feature_names}")

    # Intrinsic features that are actually present in the DataFrame.
    # Protein-level features are excluded by default (TDC correctness); opt-in
    # only when use_protein_level_features=True (--use-protein-level-feats).
    _intrinsic_pool = MALDI_INTRINSIC_FEATURES + (
        PROTEIN_LEVEL_FEATURES if use_protein_level_features else []
    )
    intrinsic_present = [f for f in _intrinsic_pool if f in features_df.columns]
    lcms_present = [f for f in LCMS_PRIOR_FEATURES if f in features_df.columns]
    spatial_present = [f for f in SPATIAL_PRIOR_FEATURES if f in features_df.columns]

    # Log-transform heavy-tail features in place.  These features span 4+
    # orders of magnitude on real data; after StandardScaler the few extreme
    # values dominate and suppress discrimination from well-behaved features.
    _HEAVY_TAIL_FEATURES = (
        "chca_cluster_distance_ppm",
        "theo_isotope_chi2",
        "ppm_best_ratio",
        "theo_m1_ratio_diff",
        "theo_m2_ratio_diff",
    )
    for _f in _HEAVY_TAIL_FEATURES:
        if _f in features_df.columns:
            features_df[_f] = np.log1p(
                np.clip(features_df[_f].values.astype(float), 0.0, None)
            )

    # Drop constant / near-constant features from the ranker input.  A column
    # with one unique value contributes zero variance and only consumes a slot
    # in the SVM grid search; in pathological CV folds it can also produce
    # warnings.
    _constant = [f for f in intrinsic_present if features_df[f].nunique(dropna=True) <= 1]
    if _constant:
        logger.info(f"  Dropping {len(_constant)} constant features from ranker: {_constant}")
        intrinsic_present = [f for f in intrinsic_present if f not in _constant]

    # --- Generative-only backend: two-pass ---
    if model == "generative":
        from ms1rescore.probabilistic_scorer import (
            compute_generative_scores,
            estimate_noise_params,
        )

        logger.info("Step 8: Generative backend — two-pass scoring...")
        feature_col = "feature_idx" if "feature_idx" in features_df.columns else "feature_mz"

        # Round-1 scores already computed by run_generative_scoring (step 7b)
        scores1 = features_df["generative_score"].values

        # Per-feature winner selection
        winner_pos, winners_df = _select_feature_winners(features_df, scores1, feature_col)
        logger.info(
            f"  Round-1 winner selection: {len(winners_df)} candidates retained "
            f"({int(winners_df['is_decoy'].sum())} decoys)"
        )

        # Round-2: re-estimate noise on winner subset, recompute log-likelihoods
        noise_params_r2 = estimate_noise_params(winners_df)
        scores2_series, _ = compute_generative_scores(winners_df, noise_params_r2)
        scores2 = scores2_series.values

        # Standard TDC FDR on winners
        is_decoy_w = winners_df["is_decoy"].values.astype(bool)
        q2 = _tdc_qvalues(scores2, is_decoy_w)
        lcms_prior_w = compute_lcms_prior(winners_df, lcms_present)
        spatial_prior_w = compute_spatial_prior(winners_df, spatial_present)
        # Additive log-prior: rank-correct for arbitrary-sign scores.
        # Multiplying a negative score by a prior in [0,1] would invert
        # the ranking (a bad candidate with low prior becomes less negative,
        # i.e. higher-ranked).  Adding log(prior) penalises low-prior
        # candidates uniformly regardless of score sign.
        _LOG_EPS = 1e-12
        reweighted2 = (
            scores2
            + np.log(np.clip(lcms_prior_w, _LOG_EPS, None))
            + np.log(np.clip(spatial_prior_w, _LOG_EPS, None))
        )
        rw_q2 = _tdc_qvalues(reweighted2, is_decoy_w)

        # Map back to full candidate table
        is_winner_full = np.zeros(len(features_df), dtype=bool)
        is_winner_full[winner_pos] = True
        scores2_full = np.full(len(features_df), np.nan)
        scores2_full[winner_pos] = scores2
        q_full = np.full(len(features_df), np.nan)
        q_full[winner_pos] = q2
        rw_full = np.full(len(features_df), np.nan)
        rw_full[winner_pos] = reweighted2
        rw_q_full = np.full(len(features_df), np.nan)
        rw_q_full[winner_pos] = rw_q2

        is_decoy = features_df["is_decoy"].values.astype(bool)
        result_df = pd.DataFrame(
            {
                "peptide": features_df["peptide"].values,
                "protein": features_df["protein"].values if "protein" in features_df.columns else "",
                "feature_idx": features_df.get(
                    "feature_idx", pd.Series(range(len(features_df)))
                ).values,
                "is_decoy": is_decoy,
                "generative_score_r1": scores1,
                "generative_score_r2": scores2_full,
                "Delta_m": features_df.get("Delta_m", pd.Series(np.nan, index=features_df.index)).values,
                "q_value": q_full,
                "is_tdc_winner": is_winner_full,
                "reweighted_score": rw_full,
                "reweighted_q_value": rw_q_full,
            }
        )

        for fdr_threshold in [0.01, 0.05, 0.10]:
            n = (is_winner_full & ~is_decoy & (q_full <= fdr_threshold)).sum()
            n_rw = (is_winner_full & ~is_decoy & (rw_q_full <= fdr_threshold)).sum()
            logger.info(
                f"  At {fdr_threshold*100:.0f}% FDR: {n} winners (base), "
                f"{n_rw} winners (reweighted)"
            )

        psm_list = candidates_to_psm_list(features_df)

        if debug_dir is not None:
            from ms1rescore.debug_viz import save_debug_figures
            save_debug_figures(
                features_df, result_df,
                ion_images=ion_images, ion_image_mzs=ion_image_mzs,
                maldi_envelopes=maldi_envelopes,
                feature_names=intrinsic_present, model_name="generative",
                debug_dir=debug_dir, n_subset=n_debug, seed=debug_seed,
                gt_peptides=gt_peptides,
            )

        return psm_list, result_df, feature_names

    # --- Step 7: Build PSMList ---
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
    logger.info(f"Step 8: Running rescoring (model='{model}')...")

    if model == "svm":
        feature_col = "feature_idx" if "feature_idx" in features_df.columns else "feature_mz"

        # --- Round 1: score all candidates ---
        populate_psm_features(psm_list, features_df, intrinsic_present)
        if verbose:
            logger.debug(
                f"Writing PSM list with intrinsic features to {output_dir}/14_debug_psm_list_after_intrinsic.tsv"
            )
            psm_list.to_dataframe().to_csv(
                f"{output_dir}/14_debug_psm_list_after_intrinsic.tsv", sep="\t", index=False
            )
        conf_obj_r1, scores1, _imp_r1_svm, _imp_names_r1_svm = _rescore_svm(
            psm_list, features_df, intrinsic_present, train_fdr
        )
        if verbose:
            with open(f"{output_dir}/15_debug_mokapot_conf_r1.pkl", "wb") as f:
                pickle.dump(conf_obj_r1, f)
            with open(f"{output_dir}/15_debug_svm_scores_r1.pkl", "wb") as f:
                pickle.dump(scores1 if scores1 is not None else np.array([]), f)

        if scores1 is None:
            logger.warning("SVM round-1 score extraction failed; using zeros.")
            scores1 = np.zeros(len(features_df))

        # --- Per-feature winner selection ---
        winner_pos, winners_df = _select_feature_winners(features_df, scores1, feature_col)
        logger.info(
            f"  Round-1 winner selection: {len(winners_df)} candidates retained "
            f"({int(winners_df['is_decoy'].sum())} decoys)"
        )

        # --- Round 2: retrain on winner subset ---
        psm_list_r2 = candidates_to_psm_list(winners_df)
        populate_psm_features(psm_list_r2, winners_df, intrinsic_present)
        conf_obj_r2, scores2, svm_imp_r2, svm_imp_names_r2 = _rescore_svm(
            psm_list_r2, winners_df, intrinsic_present, train_fdr
        )
        if verbose:
            with open(f"{output_dir}/15_debug_mokapot_conf_r2.pkl", "wb") as f:
                pickle.dump(conf_obj_r2, f)
            with open(f"{output_dir}/15_debug_svm_scores_r2.pkl", "wb") as f:
                pickle.dump(scores2 if scores2 is not None else np.array([]), f)

        if scores2 is None:
            logger.warning("SVM round-2 score extraction failed; using zeros.")
            scores2 = np.zeros(len(winners_df))

        # --- Standard TDC FDR on winners ---
        is_decoy_w = winners_df["is_decoy"].values.astype(bool)
        q2 = _tdc_qvalues(scores2, is_decoy_w)
        lcms_prior_w = compute_lcms_prior(winners_df, lcms_present)
        spatial_prior_w = compute_spatial_prior(winners_df, spatial_present)
        # Additive log-prior: rank-correct for arbitrary-sign scores.
        # Multiplying a negative score by a prior in [0,1] would invert
        # the ranking (a bad candidate with low prior becomes less negative,
        # i.e. higher-ranked).  Adding log(prior) penalises low-prior
        # candidates uniformly regardless of score sign.
        _LOG_EPS = 1e-12
        reweighted2 = (
            scores2
            + np.log(np.clip(lcms_prior_w, _LOG_EPS, None))
            + np.log(np.clip(spatial_prior_w, _LOG_EPS, None))
        )
        rw_q2 = _tdc_qvalues(reweighted2, is_decoy_w)

        # --- Map back to full candidate table ---
        is_winner_full = np.zeros(len(features_df), dtype=bool)
        is_winner_full[winner_pos] = True
        scores2_full = np.full(len(features_df), np.nan)
        scores2_full[winner_pos] = scores2
        q_full = np.full(len(features_df), np.nan)
        q_full[winner_pos] = q2
        rw_full = np.full(len(features_df), np.nan)
        rw_full[winner_pos] = reweighted2
        rw_q_full = np.full(len(features_df), np.nan)
        rw_q_full[winner_pos] = rw_q2

        is_decoy = features_df["is_decoy"].values.astype(bool)
        result_df = pd.DataFrame(
            {
                "peptide": features_df["peptide"].values,
                "protein": features_df["protein"].values if "protein" in features_df.columns else "",
                "feature_mz": features_df["feature_mz"].values if "feature_mz" in features_df.columns else np.nan,
                "feature_idx": features_df.get(
                    "feature_idx", pd.Series(range(len(features_df)))
                ).values,
                "is_decoy": is_decoy,
                "svm_score_r1": scores1,
                "svm_score_r2": scores2_full,
                "q_value": q_full,
                "is_tdc_winner": is_winner_full,
                "reweighted_score": rw_full,
                "reweighted_q_value": rw_q_full,
            }
        )

        for fdr_threshold in [0.01, 0.05, 0.10]:
            n = (is_winner_full & ~is_decoy & (q_full <= fdr_threshold)).sum()
            n_rw = (is_winner_full & ~is_decoy & (rw_q_full <= fdr_threshold)).sum()
            logger.info(
                f"  At {fdr_threshold*100:.0f}% FDR: {n} target features (base), "
                f"{n_rw} target features (reweighted)"
            )

        if debug_dir is not None:
            from ms1rescore.debug_viz import save_debug_figures
            save_debug_figures(
                features_df, result_df,
                ion_images=ion_images, ion_image_mzs=ion_image_mzs,
                maldi_envelopes=maldi_envelopes,
                feature_names=intrinsic_present, model_name="svm",
                importances_r1=_imp_r1_svm, importances_r2=svm_imp_r2,
                importance_names=svm_imp_names_r2 or _imp_names_r1_svm,
                debug_dir=debug_dir, n_subset=n_debug, seed=debug_seed,
                gt_peptides=gt_peptides,
            )

        return psm_list, result_df, feature_names

    elif model == "catboost":
        feature_col = "feature_idx" if "feature_idx" in features_df.columns else "feature_mz"

        # --- Round 1: score all candidates ---
        scores1, _imp_r1_cb, _imp_names_cb = _rescore_catboost(
            features_df,
            intrinsic_present,
            train_fdr=train_fdr,
            init_ppm_threshold=init_ppm_threshold,
            init_isotope_threshold=init_isotope_threshold,
        )
        if verbose:
            with open(f"{output_dir}/16_debug_catboost_scores_r1.pkl", "wb") as f:
                pickle.dump(scores1, f)

        # --- Per-feature winner selection ---
        winner_pos, winners_df = _select_feature_winners(features_df, scores1, feature_col)
        logger.info(
            f"  Round-1 winner selection: {len(winners_df)} candidates retained "
            f"({int(winners_df['is_decoy'].sum())} decoys)"
        )

        # --- Round 2: retrain on winner subset ---
        scores2, cb_imp_r2, cb_imp_names_r2 = _rescore_catboost(
            winners_df,
            intrinsic_present,
            train_fdr=train_fdr,
            init_ppm_threshold=init_ppm_threshold,
            init_isotope_threshold=init_isotope_threshold,
        )
        if verbose:
            with open(f"{output_dir}/16_debug_catboost_scores_r2.pkl", "wb") as f:
                pickle.dump(scores2, f)

        # --- Standard TDC FDR on winners ---
        is_decoy_w = winners_df["is_decoy"].values.astype(bool)
        q2 = _tdc_qvalues(scores2, is_decoy_w)
        lcms_prior_w = compute_lcms_prior(winners_df, lcms_present)
        spatial_prior_w = compute_spatial_prior(winners_df, spatial_present)
        # Additive log-prior: rank-correct for arbitrary-sign scores.
        # Multiplying a negative score by a prior in [0,1] would invert
        # the ranking (a bad candidate with low prior becomes less negative,
        # i.e. higher-ranked).  Adding log(prior) penalises low-prior
        # candidates uniformly regardless of score sign.
        _LOG_EPS = 1e-12
        reweighted2 = (
            scores2
            + np.log(np.clip(lcms_prior_w, _LOG_EPS, None))
            + np.log(np.clip(spatial_prior_w, _LOG_EPS, None))
        )
        rw_q2 = _tdc_qvalues(reweighted2, is_decoy_w)

        # --- Map back to full candidate table ---
        is_winner_full = np.zeros(len(features_df), dtype=bool)
        is_winner_full[winner_pos] = True
        scores2_full = np.full(len(features_df), np.nan)
        scores2_full[winner_pos] = scores2
        q_full = np.full(len(features_df), np.nan)
        q_full[winner_pos] = q2
        rw_full = np.full(len(features_df), np.nan)
        rw_full[winner_pos] = reweighted2
        rw_q_full = np.full(len(features_df), np.nan)
        rw_q_full[winner_pos] = rw_q2

        is_decoy = features_df["is_decoy"].values.astype(bool)
        result_df = pd.DataFrame(
            {
                "peptide": features_df["peptide"].values,
                "protein": features_df["protein"].values if "protein" in features_df.columns else "",
                "feature_mz": features_df["feature_mz"].values if "feature_mz" in features_df.columns else np.nan,
                "feature_idx": features_df.get(
                    "feature_idx", pd.Series(range(len(features_df)))
                ).values,
                "is_decoy": is_decoy,
                "catboost_score_r1": scores1,
                "catboost_score_r2": scores2_full,
                "q_value": q_full,
                "is_tdc_winner": is_winner_full,
                "reweighted_score": rw_full,
                "reweighted_q_value": rw_q_full,
            }
        )

        for fdr_threshold in [0.01, 0.05, 0.10]:
            n = (is_winner_full & ~is_decoy & (q_full <= fdr_threshold)).sum()
            n_rw = (is_winner_full & ~is_decoy & (rw_q_full <= fdr_threshold)).sum()
            logger.info(
                f"  At {fdr_threshold*100:.0f}% FDR: {n} target features (base), "
                f"{n_rw} target features (reweighted)"
            )

        if debug_dir is not None:
            from ms1rescore.debug_viz import save_debug_figures
            save_debug_figures(
                features_df, result_df,
                ion_images=ion_images, ion_image_mzs=ion_image_mzs,
                maldi_envelopes=maldi_envelopes,
                feature_names=intrinsic_present, model_name="catboost",
                importances_r1=_imp_r1_cb, importances_r2=cb_imp_r2,
                importance_names=cb_imp_names_r2 or _imp_names_cb,
                debug_dir=debug_dir, n_subset=n_debug, seed=debug_seed,
                gt_peptides=gt_peptides,
            )

        return psm_list, result_df, feature_names

    elif model == "lda":
        feature_col = "feature_idx" if "feature_idx" in features_df.columns else "feature_mz"

        # --- Round 1: score all candidates ---
        scores1, _imp_r1_lda, _imp_names_lda = _rescore_lda(
            features_df,
            intrinsic_present,
            init_ppm_threshold=init_ppm_threshold,
        )
        # Output importances
        if verbose:
            with open(f"{output_dir}/17_debug_lda_scores_r1.pkl", "wb") as f:
                pickle.dump(scores1, f)
            lda_importances_df = pd.DataFrame({
                "feature": _imp_names_lda,
                "importance": _imp_r1_lda,
            }).sort_values("importance", ascending=False)
            lda_importances_df.to_csv(
                f"{output_dir}/17_debug_lda_importances_r1.tsv", sep="\t", index=False
            )

        # --- Per-feature winner selection ---
        winner_pos, winners_df = _select_feature_winners(features_df, scores1, feature_col)
        logger.info(
            f"  Round-1 winner selection: {len(winners_df)} candidates retained "
            f"({int(winners_df['is_decoy'].sum())} decoys)"
        )

        # --- Round 2: retrain on winner subset ---
        # Seed R2 from the top-20% of target winners by R1 score.  After winner
        # selection the remaining target winners may have ppm > init_ppm_threshold
        # (R1 lifted them on other features), so re-seeding from ppm alone would
        # leave only a handful of seeds, causing the LDA to degenerate.  Using
        # q-values from R1 has the same problem when R1 itself identified very few
        # pseudo-positives.  A percentile cut on raw R1 scores is guaranteed to
        # produce a reasonably sized seed regardless of how well R1 converged.
        is_decoy_w = winners_df["is_decoy"].values.astype(bool)
        target_scores_w = scores1[winner_pos][~is_decoy_w]
        score_threshold = np.percentile(target_scores_w, 80)  # top 20% of targets
        r2_seed_mask = (~is_decoy_w) & (scores1[winner_pos] >= score_threshold)
        logger.info(
            f"  LDA R2: seeding from top-20% R1 target scores "
            f"(score ≥ {score_threshold:.3f}) → {r2_seed_mask.sum()} positives"
        )

        scores2, lda_imp_r2, lda_imp_names_r2 = _rescore_lda(
            winners_df,
            intrinsic_present,
            init_ppm_threshold=init_ppm_threshold,
            seed_mask=r2_seed_mask,
        )
        # Output importances
        if verbose:
            with open(f"{output_dir}/17_debug_lda_scores_r2.pkl", "wb") as f:
                pickle.dump(scores2, f)
            lda_importances_df = pd.DataFrame({
                "feature": lda_imp_names_r2 or _imp_names_lda,
                "importance": lda_imp_r2,
            }).sort_values("importance", ascending=False)
            lda_importances_df.to_csv(
                f"{output_dir}/17_debug_lda_importances_r2.tsv", sep="\t", index=False
            )

        # --- Standard TDC FDR on winners ---
        is_decoy_w = winners_df["is_decoy"].values.astype(bool)
        q2 = _tdc_qvalues(scores2, is_decoy_w)
        lcms_prior_w = compute_lcms_prior(winners_df, lcms_present)
        spatial_prior_w = compute_spatial_prior(winners_df, spatial_present)
        # Additive log-prior: rank-correct for arbitrary-sign scores.
        # Multiplying a negative score by a prior in [0,1] would invert
        # the ranking (a bad candidate with low prior becomes less negative,
        # i.e. higher-ranked).  Adding log(prior) penalises low-prior
        # candidates uniformly regardless of score sign.
        _LOG_EPS = 1e-12
        reweighted2 = (
            scores2
            + np.log(np.clip(lcms_prior_w, _LOG_EPS, None))
            + np.log(np.clip(spatial_prior_w, _LOG_EPS, None))
        )
        rw_q2 = _tdc_qvalues(reweighted2, is_decoy_w)

        # --- Map back to full candidate table ---
        is_winner_full = np.zeros(len(features_df), dtype=bool)
        is_winner_full[winner_pos] = True
        scores2_full = np.full(len(features_df), np.nan)
        scores2_full[winner_pos] = scores2
        q_full = np.full(len(features_df), np.nan)
        q_full[winner_pos] = q2
        rw_full = np.full(len(features_df), np.nan)
        rw_full[winner_pos] = reweighted2
        rw_q_full = np.full(len(features_df), np.nan)
        rw_q_full[winner_pos] = rw_q2

        is_decoy = features_df["is_decoy"].values.astype(bool)
        result_df = pd.DataFrame(
            {
                "peptide": features_df["peptide"].values,
                "protein": features_df["protein"].values if "protein" in features_df.columns else "",
                "feature_mz": features_df["feature_mz"].values if "feature_mz" in features_df.columns else np.nan,
                "feature_idx": features_df.get(
                    "feature_idx", pd.Series(range(len(features_df)))
                ).values,
                "is_decoy": is_decoy,
                "lda_score_r1": scores1,
                "lda_score_r2": scores2_full,
                "q_value": q_full,
                "is_tdc_winner": is_winner_full,
                "reweighted_score": rw_full,
                "reweighted_q_value": rw_q_full,
            }
        )

        for fdr_threshold in [0.01, 0.05, 0.10]:
            n = (is_winner_full & ~is_decoy & (q_full <= fdr_threshold)).sum()
            n_rw = (is_winner_full & ~is_decoy & (rw_q_full <= fdr_threshold)).sum()
            logger.info(
                f"  At {fdr_threshold*100:.0f}% FDR: {n} target features (base), "
                f"{n_rw} target features (reweighted)"
            )

        if debug_dir is not None:
            from ms1rescore.debug_viz import save_debug_figures
            save_debug_figures(
                features_df, result_df,
                ion_images=ion_images, ion_image_mzs=ion_image_mzs,
                maldi_envelopes=maldi_envelopes,
                feature_names=intrinsic_present, model_name="lda",
                importances_r1=_imp_r1_lda, importances_r2=lda_imp_r2,
                importance_names=lda_imp_names_r2 or _imp_names_lda,
                debug_dir=debug_dir, n_subset=n_debug, seed=debug_seed,
                gt_peptides=gt_peptides,
            )

        return psm_list, result_df, feature_names

    else:
        raise ValueError(f"Unknown model '{model}'. Choose 'svm', 'catboost', 'lda', or 'generative'.")
