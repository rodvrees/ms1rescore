"""End-to-end symmetric MALDI-MSI rescoring pipeline."""

import logging
import os
import pickle

import numpy as np
import pandas as pd

from MSI-PICASSO.candidates import (
    digest_fasta,
    digest_identified_proteins,
    generate_balanced_shuffle_candidates,
    match_to_maldi_features,
)
from MSI-PICASSO.feature_generator import (
    FEATURE_NAN_FILL,
    LCMS_PRIOR_FEATURES,
    MAIN_FEATURES,
    MALDI_INTRINSIC_FEATURES,
    PROTEIN_LEVEL_FEATURES,
    SPATIAL_PRIOR_FEATURES,
    candidates_to_psm_list,
    compute_all_features,
    get_feature_names,
    populate_psm_features,
)
from MSI-PICASSO.lcms_evidence import (
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

    Each feature is min-max normalized to [0, 1] using the non-NaN range.
    NaN values (e.g. lcms_ms2_spectral_angle when no MS2 prediction is
    available) are excluded from that candidate's mean so they don't
    suppress the prior for candidates that lack MS2 evidence.
    Features where all non-NaN values are identical are skipped.
    Returns the nanmean of normalized features, or 1.0 where all features
    are NaN or no informative features are present.
    """
    normed: list[np.ndarray] = []

    for feat in present_lcms_features:
        if feat not in candidates_df.columns:
            continue
        col = candidates_df[feat].values.astype(float)
        col_min = np.nanmin(col)
        col_max = np.nanmax(col)
        if not np.isfinite(col_min) or not np.isfinite(col_max) or col_max - col_min < 1e-12:
            continue
        normed.append((col - col_min) / (col_max - col_min))

    if not normed:
        return np.ones(len(candidates_df))

    stacked = np.stack(normed, axis=0)
    result = np.nanmean(stacked, axis=0)
    # candidates where every feature is NaN get a neutral weight
    return np.where(np.isnan(result), 1.0, result)


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
    mokapot_max_iter: int = 10,
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
    model = PercolatorModel(train_fdr=train_fdr, max_iter=mokapot_max_iter)
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
            X_all = features_df[present_raw].values.astype(float)
            _apply_nan_fill(X_all, present_raw, FEATURE_NAN_FILL)
            X_all = np.where(np.isfinite(X_all), X_all, 0.0)
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
    pseudo_label_max_iter: int = 5,
    pseudo_label_fdr: float = 0.10,
    r1_seed_percentile: float = 0.10,
    catboost_iterations: int = 500,
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
    X = df[present].values.astype(np.float64)
    _apply_nan_fill(X, present, FEATURE_NAN_FILL)
    X = np.where(np.isfinite(X), X, 0.0).astype(np.float32)
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
        seed_mask = is_target & (ppm_col < ppm_col[is_target].quantile(r1_seed_percentile)).values

    catboost_params = dict(
        iterations=catboost_iterations,
        learning_rate=0.05,
        depth=6,
        loss_function="YetiRank",
        verbose=False,
        random_seed=42,
    )

    scores = np.zeros(len(df))
    prev_pos_size = -1
    model_cb = None

    for iteration in range(pseudo_label_max_iter):
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

        # Expand pseudo-positives: all targets with q <= pseudo_label_fdr
        new_seed = is_target & (q_values <= pseudo_label_fdr)
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


def _apply_nan_fill(
    X: np.ndarray,
    feature_names: list[str],
    fill_spec: dict[str, "float | str"],
) -> np.ndarray:
    """Apply feature-specific NaN fills before the generic median imputer.

    Operates in-place on *X* (caller should pass a copy if the original must
    be preserved).  Only columns present in both *feature_names* and
    *fill_spec* are touched; remaining NaN values are left for the downstream
    ``SimpleImputer`` (or ``fillna``) to handle.
    """
    for fname, fill_val in fill_spec.items():
        if fname not in feature_names:
            continue
        j = feature_names.index(fname)
        col = X[:, j]
        nan_mask = np.isnan(col)
        n_nan = int(nan_mask.sum())
        if n_nan == 0:
            continue
        if isinstance(fill_val, str):
            if fill_val == "col_max":
                value = float(np.nanmax(col)) if np.isfinite(col[~nan_mask]).any() else 0.0
            elif fill_val == "col_min":
                value = float(np.nanmin(col)) if np.isfinite(col[~nan_mask]).any() else 0.0
            else:
                raise ValueError(f"Unknown fill_spec value {fill_val!r} for feature {fname!r}")
        else:
            value = float(fill_val)
        col[nan_mask] = value
        logger.debug(
            "  _apply_nan_fill: %s — filled %d NaN with %.4f (%s)",
            fname, n_nan, value, fill_val if isinstance(fill_val, str) else "constant",
        )
    return X


def _log_imputation_debug(
    label: str,
    X_fit: np.ndarray,
    fit_names: list[str],
    is_target: np.ndarray,
    is_decoy: np.ndarray,
    pipe,
) -> None:
    """Log per-feature NaN counts and imputation values, split by target/decoy.

    Gated on DEBUG level so it is a no-op in normal runs.

    Columns:
      nan_tgt / nan_dec  — how many rows of each group had NaN and were imputed
      tgt% / dec%        — NaN rate per group
      imputed            — value filled in (train-set median from SimpleImputer)
      tgt_med            — median of real (non-NaN) target values
      dec_med            — median of real (non-NaN) decoy values
      bias               — imputed − dec_med  (positive = decoys with NaN got
                           pulled above their natural median toward target territory)

    A large positive bias on a feature with high dec% is a sign that imputation
    may be inflating decoy scores for that feature.
    """
    if not logger.isEnabledFor(logging.DEBUG):
        return

    nan_mask = np.isnan(X_fit)
    has_nan = nan_mask.any(axis=0)
    if not has_nan.any():
        logger.debug("  %s imputation: no NaN values in any feature", label)
        return

    n_tgt = int(is_target.sum())
    n_dec = int(is_decoy.sum())
    imp_vals = pipe["imputer"].statistics_

    header = (
        f"  {'feature':<42s}  {'nan_tgt':>7}  {'nan_dec':>7}"
        f"  {'tgt%':>6}  {'dec%':>6}  {'imputed':>9}  {'tgt_med':>9}  {'dec_med':>9}  {'bias':>9}"
    )
    rows = [header]
    for j, fname in enumerate(fit_names):
        if not has_nan[j]:
            continue
        n_nan_tgt = int(nan_mask[is_target, j].sum())
        n_nan_dec = int(nan_mask[is_decoy, j].sum())
        pct_tgt = 100.0 * n_nan_tgt / n_tgt if n_tgt else 0.0
        pct_dec = 100.0 * n_nan_dec / n_dec if n_dec else 0.0

        tgt_real = X_fit[is_target & ~nan_mask[:, j], j]
        dec_real = X_fit[is_decoy & ~nan_mask[:, j], j]
        tgt_med = float(np.median(tgt_real)) if len(tgt_real) else float("nan")
        dec_med = float(np.median(dec_real)) if len(dec_real) else float("nan")
        bias = imp_vals[j] - dec_med if np.isfinite(dec_med) else float("nan")

        rows.append(
            f"  {fname:<42s}  {n_nan_tgt:>7d}  {n_nan_dec:>7d}"
            f"  {pct_tgt:>5.1f}%  {pct_dec:>5.1f}%  {imp_vals[j]:>9.3f}"
            f"  {tgt_med:>9.3f}  {dec_med:>9.3f}  {bias:>+9.3f}"
        )

    logger.debug(
        "  %s imputation stats (n_tgt=%d, n_dec=%d, imputed=train-set median):\n%s",
        label, n_tgt, n_dec, "\n".join(rows),
    )


# Features excluded from best-feature initialization.  These measure amino acid
# composition rather than spectral quality.  Since decoys are K/R-preserving
# shuffles from the same protein pool, composition features can have arbitrary
# systematic differences that produce spurious pseudo-positives.
_BEST_FEAT_SKIP: frozenset[str] = frozenset({
    # Basic sequence properties
    "peptide_length", "n_missed_cleavages",
    # Peptide composition (C-group)
    "has_oxidized_met", "has_cys", "n_proline", "acidic_residue_density",
    # Ionization / physicochemistry
    "n_arginine", "n_basic_residues", "n_aromatic", "gravy_score", "charge_proxy",
})


def _find_best_feature_labels(
    X: np.ndarray,
    is_decoy: np.ndarray,
    feature_names: list[str],
    init_fdr: float = 0.2,
    min_seed_positives: int = 50,
) -> tuple[np.ndarray, str, int] | None:
    """
    Mokapot-style best-feature seed initialization.

    For each column in X and each ranking direction (ascending/descending),
    run TDC q-value computation and count target candidates at q <= init_fdr.
    Select the (feature, direction) pair that yields the most such targets.

    When the best single-feature result has fewer than ``min_seed_positives``
    targets, all unique pairwise sums and differences of eligible features are
    tried. The composite score that yields the most targets is used if it beats
    the single-feature result.

    Columns whose names appear in _BEST_FEAT_SKIP are excluded — they measure
    amino acid composition rather than spectral quality and can yield spurious
    pseudo-positives due to composition differences between shuffled decoys and
    targets.

    NaN values in X are filled with the column median before ranking.

    Returns
    -------
    (labels, best_feature_name, n_passing) or None when n_passing == 0.
        labels: int8 array aligned to X rows.
            +1  — pseudo-positive: target at q <= init_fdr under best feature
            -1  — pseudo-negative: decoy
             0  — excluded: target at q > init_fdr
    """
    is_decoy = np.asarray(is_decoy, dtype=bool)

    X_imp = X.copy()
    for j in range(X_imp.shape[1]):
        col = X_imp[:, j]
        finite_vals = col[np.isfinite(col)]
        fill = float(np.median(finite_vals)) if len(finite_vals) > 0 else 0.0
        col[~np.isfinite(col)] = fill

    # Sub-ULP random noise to break ties in argsort.  A stable sort on a feature
    # with many tied values preserves the original DataFrame row order, which
    # places targets before decoys (digest_fasta output order) and assigns them
    # artificially low q-values.
    rng = np.random.default_rng(0)
    tiebreak = rng.uniform(-1e-9, 1e-9, X_imp.shape[0])

    best_n = 0
    best_j = -1
    best_asc = True
    best_q: np.ndarray | None = None

    for j, fname in enumerate(feature_names):
        if fname in _BEST_FEAT_SKIP:
            continue
        col = X_imp[:, j]
        if col.std() == 0.0:
            continue
        for ascending in (True, False):
            scores = (col if ascending else -col) + tiebreak
            q = _tdc_qvalues(scores, is_decoy)
            n_pass = int(((~is_decoy) & (q <= init_fdr)).sum())
            if n_pass > best_n:
                best_n = n_pass
                best_j = j
                best_asc = ascending
                best_q = q.copy()

    # --- Pairwise sweep when single-feature result is weak ---
    if best_n < min_seed_positives:
        eligible = [
            j for j, fname in enumerate(feature_names)
            if fname not in _BEST_FEAT_SKIP and X_imp[:, j].std() > 0
        ]
        # Scale to zero mean, unit variance so both features contribute equally.
        col_means = X_imp[:, eligible].mean(axis=0)
        col_stds = X_imp[:, eligible].std(axis=0)
        col_stds[col_stds == 0] = 1.0
        X_sc = np.empty((X_imp.shape[0], len(eligible)), dtype=np.float64)
        for k, j in enumerate(eligible):
            X_sc[:, k] = (X_imp[:, j] - col_means[k]) / col_stds[k]

        pair_best_n = best_n
        pair_best_labels: np.ndarray | None = None
        pair_best_name: str | None = None

        for ii in range(len(eligible)):
            for jj in range(ii + 1, len(eligible)):
                gi, gj = eligible[ii], eligible[jj]
                for sign in (+1, -1):
                    composite = X_sc[:, ii] + sign * X_sc[:, jj]
                    for ascending in (True, False):
                        scores = (composite if ascending else -composite) + tiebreak
                        q = _tdc_qvalues(scores, is_decoy)
                        n_pass = int(((~is_decoy) & (q <= init_fdr)).sum())
                        if n_pass > pair_best_n:
                            pair_best_n = n_pass
                            pair_best_labels = np.where(
                                is_decoy, np.int8(-1),
                                np.where(q <= init_fdr, np.int8(1), np.int8(0)),
                            ).astype(np.int8)
                            sign_str = "+" if sign == +1 else "-"
                            pair_best_name = (
                                f"{feature_names[gi]} {sign_str} {feature_names[gj]}"
                            )

        if pair_best_labels is not None:
            logger.info(
                "  Selected pair (%s) with %d PSMs at q<=%g",
                pair_best_name, pair_best_n, init_fdr,
            )
            return pair_best_labels, pair_best_name, pair_best_n

    if best_n == 0 or best_j < 0:
        return None

    assert best_q is not None
    labels = np.where(
        is_decoy, np.int8(-1),
        np.where(best_q <= init_fdr, np.int8(1), np.int8(0)),
    ).astype(np.int8)
    return labels, feature_names[best_j], best_n


def _rescore_lda(
    features_df: pd.DataFrame,
    intrinsic_feature_names: list[str],
    init_ppm_threshold: float,
    seed_mask: np.ndarray | None = None,
    n_interaction_features: int = 0,
    r1_importances: np.ndarray | None = None,
    r1_feature_names: list[str] | None = None,
    init_fdr: float = 0.2,
    train_fdr: float = 0.05,
    max_iter: int = 5,
    r1_seed_percentile: float = 0.10,
    min_seed_positives: int = 50,
) -> np.ndarray:
    """
    Semi-supervised LDA on MALDI-intrinsic features.

    Pre-processing: ±inf replaced with NaN, then median imputation and
    StandardScaler inside a sklearn Pipeline.

    Round-1 seed (when ``seed_mask`` is None): calls ``_find_best_feature_labels``
    to pick the single feature that yields the most targets at q <= ``train_fdr``.
    Falls back to ppm_error_abs < ``init_ppm_threshold`` OR n_candidates == 1
    (with a top-percentile fallback) if no feature yields any targets.

    Round-2 seed (``seed_mask`` provided): the boolean mask is converted to the
    same three-valued label scheme (+1 / -1 / 0) for the iteration loop.

    Pseudo-label iteration (up to ``max_iter`` rounds): trains on +1 vs -1 rows
    only (label-0 targets are excluded from training but still scored); updates
    labels by running TDC q-values on the new scores and marking targets with
    q <= ``train_fdr`` as +1. Stops when the positive count changes by < 1%.

    When ``n_interaction_features > 0`` and R1 importances are supplied, the
    top-k features are expanded with pairwise interaction terms before LDA.

    Returns ``(scores, importances, feature_names_used)``.
    """
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    df = features_df.reset_index(drop=True)
    present = [f for f in intrinsic_feature_names if f in df.columns]
    X_raw = df[present].values.astype(np.float64)
    X = np.where(np.isfinite(X_raw), X_raw, np.nan)  # ±inf → nan for imputer
    _apply_nan_fill(X, present, FEATURE_NAN_FILL)

    is_decoy = df["is_decoy"].values.astype(bool)
    is_target = ~is_decoy

    # --- Polynomial interaction setup (R2 only when importances supplied) ---
    use_poly = (
        n_interaction_features > 0
        and r1_importances is not None
        and r1_feature_names is not None
        and len(r1_importances) == len(r1_feature_names)
    )
    if use_poly:
        from sklearn.preprocessing import PolynomialFeatures
        present_set = set(present)
        r1_order = np.argsort(np.abs(r1_importances))[::-1]
        top_names = [
            r1_feature_names[i] for i in r1_order
            if r1_feature_names[i] in present_set
        ][:n_interaction_features]
        top_col_idx = [present.index(n) for n in top_names]
        X_fit = X[:, top_col_idx]
        _poly_probe = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
        _poly_probe.fit(np.zeros((1, len(top_names))))
        expanded_names = list(_poly_probe.get_feature_names_out(top_names))
        n_cross = len(expanded_names) - len(top_names)
        logger.info(
            f"  LDA R2 interactions: top-{len(top_names)} features "
            f"({', '.join(top_names)}) → {len(expanded_names)} total "
            f"({len(top_names)} original + {n_cross} cross-terms)"
        )
    else:
        X_fit = X
        expanded_names = None

    # --- Initial label assignment ---
    n_init_positives: int | None = None  # for post-loop comparison (R1 only)

    if seed_mask is None:
        bf_result = _find_best_feature_labels(
            X, is_decoy, present, init_fdr, min_seed_positives=min_seed_positives
        )
        if bf_result is not None:
            labels, _best_feat, _n_init = bf_result
            n_init_positives = _n_init
            logger.info(
                f"  LDA: best-feature init on '{_best_feat}', "
                f"{_n_init} targets at q≤{init_fdr:.3g}"
            )
        else:
            logger.warning(
                f"  LDA: best-feature init yielded 0 targets at q≤{init_fdr:.3g} "
                "— falling back to ppm-based seeding"
            )
            ppm_col = df.get("ppm_error_abs", pd.Series(np.inf, index=df.index))
            n_cand_col = df.get("n_candidates", pd.Series(np.inf, index=df.index))
            init_mask = (
                is_target & ((ppm_col < init_ppm_threshold) | (n_cand_col == 1))
            ).values
            if not init_mask.any():
                logger.warning("  LDA: no ppm-based seed positives — falling back to top-ppm init")
                init_mask = (
                    is_target & (ppm_col < ppm_col[is_target].quantile(r1_seed_percentile))
                ).values
            labels = np.where(
                is_decoy, np.int8(-1), np.where(init_mask, np.int8(1), np.int8(0))
            ).astype(np.int8)
    else:
        seed_arr = seed_mask.values if hasattr(seed_mask, "values") else np.asarray(seed_mask)
        labels = np.where(
            is_decoy, np.int8(-1), np.where(seed_arr, np.int8(1), np.int8(0))
        ).astype(np.int8)

    n_seed = int((labels == 1).sum())
    logger.info(f"  LDA: seed positives = {n_seed}, decoys = {is_decoy.sum()}")

    scores = np.zeros(len(df))
    prev_pos_size = -1
    pipe = None

    for iteration in range(max_iter):
        pos_idx = np.where(labels == 1)[0]
        neg_idx = np.where(labels == -1)[0]  # all decoys

        if len(pos_idx) == 0:
            logger.warning(f"  LDA iter {iteration + 1}: no positives — stopping early")
            break

        if len(neg_idx) == 0:
            logger.warning(f"  LDA iter {iteration + 1}: no negatives (decoys) — cannot train LDA, stopping early")
            break

        train_idx = np.concatenate([pos_idx, neg_idx])
        y_train = np.concatenate([np.ones(len(pos_idx)), np.zeros(len(neg_idx))])

        if use_poly:
            pipe = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("poly", PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
                ("lda", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto", priors=[0.5, 0.5])),
            ])
        else:
            pipe = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("lda", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto", priors=[0.5, 0.5])),
            ])
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=1, user_api="blas"):
            pipe.fit(X_fit[train_idx], y_train)
            scores = pipe.decision_function(X_fit).ravel()

        q_values = _tdc_qvalues(scores, is_decoy)
        new_labels = np.where(
            is_decoy, np.int8(-1),
            np.where(q_values <= train_fdr, np.int8(1), np.int8(0)),
        ).astype(np.int8)
        n_new = int((new_labels == 1).sum())

        logger.info(
            f"  LDA iter {iteration + 1}: pseudo-positives = {n_new} (prev = {prev_pos_size})"
        )

        if n_new == 0:
            logger.warning(f"  LDA: no pseudo-positives at q≤{train_fdr:.3g} — stopping early")
            break

        change = abs(n_new - prev_pos_size) / max(prev_pos_size, 1)
        if prev_pos_size >= 0 and change < 0.01:
            logger.info("  LDA: converged")
            break

        prev_pos_size = n_new
        labels = new_labels

    if n_init_positives is not None and 0 <= prev_pos_size < n_init_positives:
        logger.warning(
            f"  LDA: final iteration positives ({prev_pos_size}) < "
            f"best-feature init ({n_init_positives}). Using model anyway."
        )

    if pipe is not None:
        _log_imputation_debug(
            "LDA",
            X_fit,
            top_names if use_poly else present,
            is_target,
            is_decoy,
            pipe,
        )

    importances = None
    struct_coefs: np.ndarray | None = None
    struct_names_out: list[str] = top_names if use_poly else present
    feature_names_out = present
    if pipe is not None:
        try:
            importances = pipe["lda"].coef_[0]
            if use_poly:
                feature_names_out = expanded_names
        except Exception:
            pass
        try:
            # Structure coefficients: correlation of each (imputed+scaled) original
            # feature with the discriminant score.  Unlike raw LDA coefficients,
            # these are unaffected by collinearity between features.
            X_imp = pipe["imputer"].transform(X_fit)
            X_sc = pipe["scaler"].transform(X_imp)
            struct_coefs = np.array([
                float(np.corrcoef(X_sc[:, j], scores)[0, 1])
                for j in range(X_sc.shape[1])
            ])
            struct_coefs = np.nan_to_num(struct_coefs, nan=0.0)
        except Exception:
            pass
    return scores, importances, struct_coefs, struct_names_out, feature_names_out


def _rescore_qda(
    features_df: pd.DataFrame,
    intrinsic_feature_names: list[str],
    init_ppm_threshold: float,
    seed_mask: np.ndarray | None = None,
    init_fdr: float = 0.2,
    train_fdr: float = 0.05,
    max_iter: int = 5,
    r1_seed_percentile: float = 0.10,
    min_seed_positives: int = 50,
) -> np.ndarray:
    """
    Semi-supervised QDA on MALDI-intrinsic features.

    Same seed and iteration logic as _rescore_lda (three-valued labels,
    best-feature initialization, ppm fallback), but uses
    QuadraticDiscriminantAnalysis(reg_param=0.5).

    Importances are estimated as (mu_pos - mu_neg) / pooled_std in the
    standardized feature space (a t-statistic proxy).

    Returns ``(scores, pep_proba, importances, feature_names_used)``.
    """
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    df = features_df.reset_index(drop=True)
    present = [f for f in intrinsic_feature_names if f in df.columns]
    X_raw = df[present].values.astype(np.float64)
    X = np.where(np.isfinite(X_raw), X_raw, np.nan)
    _apply_nan_fill(X, present, FEATURE_NAN_FILL)

    is_decoy = df["is_decoy"].values.astype(bool)
    is_target = ~is_decoy

    # --- Initial label assignment ---
    n_init_positives: int | None = None

    if seed_mask is None:
        bf_result = _find_best_feature_labels(
            X, is_decoy, present, init_fdr, min_seed_positives=min_seed_positives
        )
        if bf_result is not None:
            labels, _best_feat, _n_init = bf_result
            n_init_positives = _n_init
            logger.info(
                f"  QDA: best-feature init on '{_best_feat}', "
                f"{_n_init} targets at q≤{init_fdr:.3g}"
            )
        else:
            logger.warning(
                f"  QDA: best-feature init yielded 0 targets at q≤{init_fdr:.3g} "
                "— falling back to ppm-based seeding"
            )
            ppm_col = df.get("ppm_error_abs", pd.Series(np.inf, index=df.index))
            n_cand_col = df.get("n_candidates", pd.Series(np.inf, index=df.index))
            init_mask = (
                is_target & ((ppm_col < init_ppm_threshold) | (n_cand_col == 1))
            ).values
            if not init_mask.any():
                logger.warning("  QDA: no ppm-based seed positives — falling back to top-ppm init")
                init_mask = (
                    is_target & (ppm_col < ppm_col[is_target].quantile(r1_seed_percentile))
                ).values
            labels = np.where(
                is_decoy, np.int8(-1), np.where(init_mask, np.int8(1), np.int8(0))
            ).astype(np.int8)
    else:
        seed_arr = seed_mask.values if hasattr(seed_mask, "values") else np.asarray(seed_mask)
        labels = np.where(
            is_decoy, np.int8(-1), np.where(seed_arr, np.int8(1), np.int8(0))
        ).astype(np.int8)

    n_seed = int((labels == 1).sum())
    logger.info(f"  QDA: seed positives = {n_seed}, decoys = {is_decoy.sum()}")

    scores = np.zeros(len(df))
    prev_pos_size = -1
    pipe = None

    for iteration in range(max_iter):
        pos_idx = np.where(labels == 1)[0]
        neg_idx = np.where(labels == -1)[0]  # all decoys

        if len(pos_idx) == 0:
            logger.warning(f"  QDA iter {iteration + 1}: no positives — stopping early")
            break

        if len(neg_idx) == 0:
            logger.warning(f"  QDA iter {iteration + 1}: no negatives (decoys) — cannot train QDA, stopping early")
            break

        train_idx = np.concatenate([pos_idx, neg_idx])
        y_train = np.concatenate([np.ones(len(pos_idx)), np.zeros(len(neg_idx))])

        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("qda", QuadraticDiscriminantAnalysis(reg_param=0.5)),
        ])
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=1, user_api="blas"):
            pipe.fit(X[train_idx], y_train)
            scores = pipe.decision_function(X).ravel()

        q_values = _tdc_qvalues(scores, is_decoy)
        new_labels = np.where(
            is_decoy, np.int8(-1),
            np.where(q_values <= train_fdr, np.int8(1), np.int8(0)),
        ).astype(np.int8)
        n_new = int((new_labels == 1).sum())

        logger.info(
            f"  QDA iter {iteration + 1}: pseudo-positives = {n_new} (prev = {prev_pos_size})"
        )

        if n_new == 0:
            logger.warning(f"  QDA: no pseudo-positives at q≤{train_fdr:.3g} — stopping early")
            break

        change = abs(n_new - prev_pos_size) / max(prev_pos_size, 1)
        if prev_pos_size >= 0 and change < 0.01:
            logger.info("  QDA: converged")
            break

        prev_pos_size = n_new
        labels = new_labels

    if n_init_positives is not None and 0 <= prev_pos_size < n_init_positives:
        logger.warning(
            f"  QDA: final iteration positives ({prev_pos_size}) < "
            f"best-feature init ({n_init_positives}). Using model anyway."
        )

    if pipe is not None:
        _log_imputation_debug("QDA", X, present, is_target, is_decoy, pipe)

    pep_proba = np.full(len(df), np.nan)
    importances = None
    if pipe is not None:
        try:
            classes = list(pipe["qda"].classes_)
            decoy_col = classes.index(0)
            pep_proba = np.clip(pipe.predict_proba(X)[:, decoy_col], 0.0, 1.0)
        except Exception:
            pass
        try:
            X_t = pipe[:-1].transform(X)
            pos_mask = (labels == 1)
            neg_mask = is_decoy
            mu_pos = np.nanmean(X_t[pos_mask], axis=0)
            mu_neg = np.nanmean(X_t[neg_mask], axis=0)
            std_pos = np.nanstd(X_t[pos_mask], axis=0)
            std_neg = np.nanstd(X_t[neg_mask], axis=0)
            n_pos = int(pos_mask.sum())
            n_neg = int(neg_mask.sum())
            pooled = np.sqrt(
                ((n_pos - 1) * std_pos**2 + (n_neg - 1) * std_neg**2)
                / max(n_pos + n_neg - 2, 1)
            )
            pooled = np.where(pooled > 0, pooled, 1.0)
            importances = (mu_pos - mu_neg) / pooled
        except Exception:
            pass
    return scores, pep_proba, importances, present


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


def _tdc_qvalues(scores: np.ndarray, is_decoy: np.ndarray, pi0: float = 1.0) -> np.ndarray:
    """
    Compute per-candidate target-decoy q-values (Storey/Käll TDC).

    Sort by descending score with a stable sort, compute cumulative
    FDR = (1 + n_decoy) / max(n_target, 1) at each position (the +1
    correction is the standard Storey/Käll adjustment for small-N), then
    take the minimum FDR seen at or below each score (rolling min from
    the tail).  When ``pi0 < 1``, the raw FDR is multiplied by ``pi0``
    before the monotone accumulation (Storey 2002 correction).
    """
    scores = np.asarray(scores)
    is_decoy = np.asarray(is_decoy).astype(bool)
    order = np.argsort(-scores, kind="stable")
    n_target_cum = np.cumsum(~is_decoy[order]).astype(float)
    n_decoy_cum = np.cumsum(is_decoy[order]).astype(float)

    fdr = pi0 * (n_decoy_cum + 1.0) / np.maximum(n_target_cum, 1.0)

    # q-value: minimum FDR at or below this score (monotone from the tail)
    qval_ordered = np.minimum.accumulate(fdr[::-1])[::-1]

    # Map back to original order
    q_values = np.empty_like(qval_ordered)
    q_values[order] = qval_ordered
    return np.clip(q_values, 0.0, 1.0)


def _estimate_pi0_storey(
    scores: np.ndarray,
    is_decoy: np.ndarray,
    lambda_range: tuple[float, float] = (0.05, 0.95),
    n_lambda: int = 20,
) -> float:
    """
    Estimate the null fraction pi0 among targets using a Storey-style sweep.

    Under the TDC null model, targets and decoys are drawn from the same
    score distribution, so their empirical CDFs should be identical when
    pi0 = 1.  True positives shift target scores upward, causing the target
    CDF to fall below the decoy CDF at any threshold.  At each score
    quantile lambda:

        pi0(lambda) = (#{targets <= lambda} / n_targets)
                    / (#{decoys  <= lambda} / n_decoys)

    The minimum over all lambdas is returned, capped at 1.0.
    Lambdas are quantiles of the combined target+decoy score distribution
    spanning ``lambda_range``.
    """
    scores = np.asarray(scores)
    is_decoy = np.asarray(is_decoy).astype(bool)
    t_scores = scores[~is_decoy]
    d_scores = scores[is_decoy]

    if len(t_scores) == 0 or len(d_scores) == 0:
        return 1.0

    combined = np.concatenate([t_scores, d_scores])
    lambda_vals = np.quantile(combined, np.linspace(lambda_range[0], lambda_range[1], n_lambda))

    pi0_estimates = []
    for lam in lambda_vals:
        d_cdf = (d_scores <= lam).mean()
        if d_cdf <= 0:
            continue
        t_cdf = (t_scores <= lam).mean()
        pi0_estimates.append(t_cdf / d_cdf)

    if not pi0_estimates:
        return 1.0
    return float(min(min(pi0_estimates), 1.0))


def estimate_pep(
    scores: np.ndarray,
    is_decoy: np.ndarray,
    method: str = "gaussian",
) -> np.ndarray:
    """
    Estimate posterior error probability (PEP) via a two-component mixture.

    Model
    -----
    f0 : null distribution fitted to decoy scores.
    f1 : signal distribution fitted to target scores above the target median.
         Using only the right tail avoids contamination from incorrect target
         matches that overlap with the null.
    pi0 : n_decoy / n_total.

    PEP(s) = pi0 * f0(s) / (pi0 * f0(s) + (1 - pi0) * f1(s)), clipped to [0, 1].

    Parameters
    ----------
    method : "gaussian" (default) or "kde".
        "gaussian" fits parametric Gaussians to f0 and f1 — appropriate when
        score distributions are approximately normal (LDA, SVM).
        "kde" uses scipy.stats.gaussian_kde — appropriate when score
        distributions are skewed or heavy-tailed (QDA).  Falls back to
        "gaussian" if the KDE covariance is singular.

    Returns NaN for all entries when fewer than 2 decoys or fewer than 2 targets
    are present (mixture is unidentifiable).
    """
    scores = np.asarray(scores, dtype=float)
    is_decoy = np.asarray(is_decoy, dtype=bool)

    n_decoy = int(is_decoy.sum())
    n_target = int((~is_decoy).sum())
    if n_decoy < 2 or n_target < 2:
        return np.full(len(scores), np.nan)

    pi0 = n_decoy / len(scores)

    decoy_scores = scores[is_decoy]
    target_scores = scores[~is_decoy]

    median_t = float(np.median(target_scores))
    high_t = target_scores[target_scores > median_t]
    if len(high_t) < 2:
        high_t = target_scores

    if method == "kde":
        from scipy.stats import gaussian_kde
        try:
            kde0 = gaussian_kde(decoy_scores)
            kde1 = gaussian_kde(high_t)
            f0 = kde0(scores)
            f1 = kde1(scores)
        except np.linalg.LinAlgError:
            method = "gaussian"

    if method == "gaussian":
        from scipy.stats import norm
        mu0 = float(np.mean(decoy_scores))
        sigma0 = max(float(np.std(decoy_scores)), 1e-6)
        mu1 = float(np.mean(high_t))
        sigma1 = max(float(np.std(high_t)), 1e-6)
        f0 = norm.pdf(scores, mu0, sigma0)
        f1 = norm.pdf(scores, mu1, sigma1)

    numer = pi0 * f0
    denom = numer + (1.0 - pi0) * f1
    with np.errstate(invalid="ignore", divide="ignore"):
        pep = np.where(denom > 0.0, numer / denom, 1.0)

    return np.clip(pep, 0.0, 1.0)


def _pep_qvalues(pep: np.ndarray) -> np.ndarray:
    """
    Convert PEP values to q-values using the cumulative-mean estimator.

    PSMs are sorted by ascending PEP; q(k) = mean(PEP_1 ... PEP_k).  This is
    the BH-style q-value interpretation of PEP.  NaN entries (non-winners) are
    propagated as NaN.
    """
    pep = np.asarray(pep, dtype=float)
    q = np.full(len(pep), np.nan)
    finite = np.isfinite(pep)
    if not finite.any():
        return q
    idx = np.where(finite)[0]
    order = np.argsort(pep[idx])
    sorted_pep = pep[idx][order]
    cumavg = np.cumsum(sorted_pep) / (np.arange(len(sorted_pep)) + 1.0)
    result = np.empty(len(idx))
    result[order] = cumavg
    q[idx] = result
    return q


def _select_calibration_peptides(
    candidates: pd.DataFrame,
    percentile: float = 0.10,
) -> np.ndarray:
    """
    Select the best target candidates to finetune/calibrate DeepLC and IM2Deep on.

    Returns a boolean array aligned with ``candidates`` rows: True for the top
    ``percentile`` fraction of TARGET rows ranked by spectral quality (low
    ``ppm_error_abs`` and high ``theo_isotope_cosine``).  Decoys are never
    eligible — a decoy has no observed RT and assigning a feature's observed CCS
    to a decoy is a false (peptide, label) pair, so decoys cannot supply a valid
    calibration anchor.

    The ranking is computed only over targets and uses features that are blind to
    ``is_decoy``, so the selection introduces no target/decoy asymmetry into the
    downstream ranker.  No target/decoy competition or q-value is used here: this
    is a quality filter, not an FDR estimate.  Unlike the previous
    ``n_candidates == 1`` heuristic, the size of this set does not shrink when
    decoys are paired onto target features (e.g. under ``paired_shuffle``), and it
    selects likely-correct peptides rather than merely mass-unambiguous ones.
    """
    n = len(candidates)
    keep = np.zeros(n, dtype=bool)
    if n == 0 or percentile <= 0:
        return keep

    is_target = ~candidates["is_decoy"].to_numpy(dtype=bool)
    if not is_target.any():
        return keep

    def _quality_z(values: np.ndarray, higher_is_better: bool) -> np.ndarray:
        """Standardised quality contribution; NaN/constant columns contribute 0."""
        x = np.asarray(values, dtype=float)
        finite = np.isfinite(x)
        if finite.sum() < 2:
            return np.zeros_like(x)
        mu = float(np.mean(x[finite]))
        sd = float(np.std(x[finite]))
        if sd < 1e-12:
            return np.zeros_like(x)
        z = (x - mu) / sd
        z = np.where(np.isfinite(z), z, 0.0)
        return z if higher_is_better else -z

    tgt = candidates.loc[is_target]
    ppm = tgt["ppm_error_abs"].to_numpy(dtype=float) if "ppm_error_abs" in tgt else np.full(len(tgt), np.nan)
    iso = (
        tgt["theo_isotope_cosine"].to_numpy(dtype=float)
        if "theo_isotope_cosine" in tgt
        else np.full(len(tgt), np.nan)
    )
    # Low ppm error is good; high theoretical isotope cosine is good.
    score = _quality_z(ppm, higher_is_better=False) + _quality_z(iso, higher_is_better=True)

    thr = float(np.quantile(score, 1.0 - percentile))
    tgt_keep = score >= thr

    target_pos = np.flatnonzero(is_target)
    keep[target_pos[tgt_keep]] = True
    return keep


def _select_feature_winners(
    features_df: pd.DataFrame,
    scores: np.ndarray,
    feature_col: str,
    winner_percentile: float = 0.02,
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    For each MALDI feature select the candidate with the highest round-1 score,
    then drop features whose winner score falls below the 1st quartile of all
    winner scores.  Filtered features receive ``is_tdc_winner=False`` and NaN
    round-2 scores in the final result.

    Returns
    -------
    winner_pos : np.ndarray[int]
        Integer positions (iloc-style) in ``features_df`` of the retained winners.
    winners_df : pd.DataFrame
        Subset of ``features_df`` with one row per retained feature, reset index.
    """
    score_series = pd.Series(scores, index=features_df.index)
    winner_idx = score_series.groupby(features_df[feature_col].values).idxmax().values
    winner_pos = features_df.index.get_indexer(winner_idx)
    winners_df = features_df.loc[winner_idx].copy().reset_index(drop=True)

    winner_scores = scores[winner_pos]
    q1 = np.quantile(winner_scores, winner_percentile)
    keep = winner_scores >= q1
    n_dropped = int((~keep).sum())
    if n_dropped:
        logger.info(
            f"  R1 winner filter: dropped {n_dropped} features with score < Quantile({winner_percentile}) ({q1:.4f})"
        )
    winner_pos = winner_pos[keep]
    winners_df = winners_df[keep].reset_index(drop=True)
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
    init_fdr: float = 0.2,
    train_fdr: float = 0.05,
    missed_cleavages: int = 2,
    min_length: int = 7,
    max_length: int = 30,
    model: str = "svm",
    init_ppm_threshold: float = 5.0,
    init_isotope_threshold: float = 0.7,
    n_interaction_features: int = 5,
    lda_r2_median_filter: bool = False,
    storey_pi0: bool = False,
    only_main_features: bool = False,
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
    decoy_method: str = "shuffle",
    max_shuffle_rounds: int = 50,
    target_ratio: float = 1.0,
    features_preset: str = "all",
    features_exclude: list[str] | None = None,
    pseudo_label_max_iter: int = 5,
    pseudo_label_fdr: float = 0.10,
    r1_seed_percentile: float = 0.10,
    r2_seed_percentile: float = 0.20,
    catboost_iterations: int = 500,
    mokapot_max_iter: int = 10,
    max_iter: int = 5,
    min_seed_positives: int = 50,
    im2deep_kwargs: dict | None = None,
    deeplc_finetune_epochs: int = 40,
    deeplc_finetune_lr: float = 0.001,
    deeplc_finetune_patience: int = 10,
    calibration_percentile: float = 0.10,
    matching_ppm: float = 20.0,
    winner_percentile: float = 0.02,
    rt_window_multiplier: float = 2.0,
    lcms_prior_weight: float = 1.0,
    spatial_prior_weight: float = 1.0,
    fragment_tol_da: float = 0.02,
    match_ccs: bool = False,
    ccs_window_multiplier: float = 2.0,
    tdf_path: str | None = None,
    mob_coloc: bool = False,
    mob_window_multiplier: float = 2.0,
):
    """
    End-to-end symmetric MALDI-MSI rescoring pipeline.

    Parameters
    ----------
    fasta_path
        Path to protein FASTA file (forward sequences only; decoys are generated
        internally via K/R-preserving protein-level shuffle).
    maldi_mzs
        Array of MALDI feature m/z values.
    mzml_paths
        Paths to LC-MS/MS mzML or Bruker .d files. Pass an empty list to skip
        all LC-MS/MS evidence steps.
    spatial_features
        Pre-computed spatial features DataFrame aligned with ``maldi_mzs``
        (optional; produced by ``compute_spatial_features`` or loaded from NPZ).
    ion_images
        MALDI ion images, shape ``(n_features, H, W)`` float32 (optional).
        Required for colocalization and spatial autocorrelation features.
    ion_image_mzs
        m/z values aligned with ``ion_images`` rows (optional).
    extra_ion_images
        Dict of ion images extracted at shifted m/z positions for direct
        colocalization without requiring those peaks in the feature list
        (optional). Keys: ``"m1"``, ``"m2"`` (M+1/M+2 isotopologues) and
        ``"na"``, ``"k"``, ``"chca"`` (adducts). Each value is
        ``(n_features, H, W)`` float32. Populated by the Bruker .d extraction
        path; ``None`` when loading from imzML or pre-computed NPZ.
    maldi_envelopes
        MALDI isotope envelopes: ``feature_mz → normalized envelope array``
        (optional). When provided, MALDI vs LC-MS/MS envelope comparison
        features are computed (``isotope_envelope_cosine`` etc.).
    msf_path
        Path to ProteomeDiscoverer .msf SQLite file for DeepLC fine-tuning
        (optional). When ``None``, DeepLC is used without fine-tuning unless
        ``lcms_peptides_path`` is provided with RT information.
    ppm_tolerance
        Mass accuracy window for matching in-silico peptides to MALDI features
        (ppm). Also used as the extraction window for LC-MS/MS MS1 signal.
    train_fdr
        FDR threshold for: (1) pseudo-label iteration in LDA/QDA (best-feature
        init and per-iteration label update); (2) mokapot SVM training target.
    max_iter
        Maximum pseudo-label iterations for LDA/QDA backends.
    missed_cleavages
        Maximum missed cleavages for tryptic in-silico digest.
    min_length
        Minimum peptide length (residues) after digest filtering.
    max_length
        Maximum peptide length (residues) after digest filtering.
    model
        Rescoring backend: ``"lda"`` (default, LinearDiscriminantAnalysis),
        ``"qda"`` (QuadraticDiscriminantAnalysis, reg_param=0.1), ``"svm"``
        (mokapot PercolatorModel), or ``"catboost"`` (semi-supervised
        CatBoostRanker). All backends train on ``MALDI_INTRINSIC_FEATURES``
        only; LC-MS/MS evidence is applied as an additive log-prior after
        scoring.
    init_ppm_threshold
        ppm_error_abs threshold for the initial positive seed in the LDA/QDA
        ppm-fallback path and in the CatBoost backend. Targets below this
        threshold (or with ``n_candidates == 1``) are used as the initial
        pseudo-positive set when best-feature initialization yields no passing
        targets.
    init_isotope_threshold
        CatBoost only: ``theo_isotope_cosine`` threshold for the initial seed
        (used in conjunction with ``init_ppm_threshold``).
    n_interaction_features
        Number of top LDA R1 features to expand into pairwise polynomial
        interactions for R2. Set to 0 to disable interaction features.
    lda_r2_median_filter
        If True, apply a per-feature median filter to LDA R2 scores before
        FDR computation (experimental).
    storey_pi0
        If True, estimate the null fraction pi0 via the Storey method and apply
        it as a correction factor in ``_tdc_qvalues``.
    only_main_features
        If True, restrict the feature set to ``MAIN_FEATURES`` (one
        representative per collinear group, ~19 features) instead of the full
        ``MALDI_INTRINSIC_FEATURES`` (~46 features).
    lcms_peptides_path
        Path to LC-MS/MS peptide-level identification results. When provided,
        activates Strategy C: candidates are generated by digesting only
        identified proteins plus directly identified peptides, instead of the
        full FASTA (Strategy A). Falls back to Strategy A with a warning if no
        identified proteins are found in the FASTA.
    lcms_proteins_path
        Path to LC-MS/MS protein-level results (optional; proteins are derived
        from the peptide table when omitted).
    lcms_psms_path
        Path to LC-MS/MS PSM-level file for RT and intensity aggregation
        (optional; improves DeepLC calibration when available).
    lcms_id_format
        Format of the LC-MS/MS ID files: ``"percolator"`` (default),
        ``"mzidentml"``, ``"psm_utils"``, or ``"msf"``.
    psm_utils_reader
        Reader hint for ``lcms_id_format="psm_utils"``. A psm_utils filetype
        key (e.g. ``"maxquant"``) or reader class name. When ``None``,
        auto-detection from the filename is attempted.
    protein_fdr
        Protein-level FDR threshold for Strategy C protein filtering.
    peptide_fdr
        Peptide-level FDR threshold for Strategy C candidate inclusion.
    extra_fasta_path
        Additional FASTA file (e.g. contaminants database). All proteins in
        this file are always included regardless of LC-MS/MS identification
        status. Peptides already present in the primary digest are not
        duplicated; when the same sequence appears in both, the primary entry
        (with LC-MS/MS evidence) is kept.
    use_protein_level_features
        If True, include ``PROTEIN_LEVEL_FEATURES`` (protein consistency and
        colocalization) in the ranker feature set. Disabled by default because
        decoys inherit inflated protein-level counts from co-occurring target
        proteins, breaking TDC null-model symmetry.
    verbose
        If True, write per-step debug files to ``output_dir`` and enable DEBUG
        logging.
    output_dir
        Directory for all output files (TSV results, debug files, config dump).
    debug_dir
        Directory for additional debug figures (optional; set by CLI when
        ``--verbose`` is active).
    n_debug
        Number of MALDI features to include in debug visualizations.
    debug_seed
        Random seed for debug feature sampling.
    observed_ccs_per_feature
        Dict mapping ``feature_idx → observed CCS`` value from MALDI ion
        mobility data (optional). When provided, enables IM2Deep CCS
        comparison features (B-group).
    im2deep_calibration
        IM2Deep CCS calibration mode: ``"linear"``, ``"spline"``, or
        ``"finetune"`` (transfer-learning fine-tune on single-candidate
        MALDI observations).
    im2deep_kwargs
        Dict of keyword arguments forwarded to ``im2deep.core.finetune`` when
        ``im2deep_calibration="finetune"``. Recognized keys:
        ``finetune_epochs`` (default 10), ``finetune_batch_size`` (default 64),
        ``finetune_lr`` (default 0.001).
    digest
        If True (and ``lcms_peptides_path`` is set), also digest identified
        proteins from ``fasta_path`` for additional candidates (Strategy C
        digest sub-mode). Ignored when ``lcms_peptides_path`` is ``None``.
    gt_peptides
        List of ground-truth peptide sequences for diagnostic FDR reporting
        (optional; not used in scoring).
    maldi_intensities
        Raw MALDI intensity array aligned with ``maldi_mzs`` (optional).
        Used when spatial features are not pre-computed.
    decoy_method
        Decoy generation strategy: ``"shuffle"`` (K/R-preserving protein
        shuffle), ``"balanced_shuffle"`` (iterative shuffle with MALDI-match
        filtering, length-stratified subsample to ~1:1 target:decoy ratio), or
        ``"paired_shuffle"`` (same iterative shuffle pool, but decoys are
        selected to occupy the same MALDI features as targets — feature-paired
        selection — to maximise per-feature target-decoy competition while
        preserving the same global ~1:1 ratio).
    max_shuffle_rounds
        Maximum shuffle rounds for ``decoy_method`` in
        {``"balanced_shuffle"``, ``"paired_shuffle"``}.
    target_ratio
        Target decoy:target ratio for ``decoy_method`` in
        {``"balanced_shuffle"``, ``"paired_shuffle"``} (default 1.0 = 1:1).
    calibration_percentile
        Fraction (0–1, default 0.10) of TARGET candidates used to finetune /
        calibrate DeepLC and IM2Deep. The top fraction is selected by spectral
        quality (low ``ppm_error_abs`` and high ``theo_isotope_cosine``), blind
        to ``is_decoy``. Replaces the old ``n_candidates == 1`` heuristic, whose
        size collapsed when decoys were paired onto target features (e.g. under
        ``paired_shuffle``).
    features_preset
        Feature set preset: ``"all"`` (full ``MALDI_INTRINSIC_FEATURES``) or
        ``"main"`` (``MAIN_FEATURES``, one representative per collinear group).
    features_exclude
        List of feature names to remove from the ranker feature set after
        applying the preset. Applied before optional protein-level feature
        inclusion.
    pseudo_label_max_iter
        Legacy parameter (SVM/CatBoost): maximum pseudo-label iterations.
        Use ``max_iter`` for LDA/QDA.
    pseudo_label_fdr
        Legacy parameter (SVM/CatBoost): FDR threshold for pseudo-label
        iteration. Use ``train_fdr`` for LDA/QDA.
    r1_seed_percentile
        Percentile of target R1 scores used as the seed threshold for R2
        (LDA/QDA). The top ``(1 - r1_seed_percentile)`` fraction of target
        winners by R1 score are used as pseudo-positives for R2 training.
    r2_seed_percentile
        Fraction of target R1 winners used as seeds for R2 training (LDA/QDA).
        Seeds are selected as the top ``r2_seed_percentile`` fraction by R1
        score, i.e. scores >= ``np.percentile(target_scores, 100*(1-r2_seed_percentile))``.
        Default 0.20 (top 20%).
    catboost_iterations
        Number of boosting iterations for ``model="catboost"``.
    mokapot_max_iter
        Maximum mokapot training iterations for ``model="svm"``.

    Returns
    -------
    tuple of (psm_list, result_df, feature_names)
        ``psm_list`` is a ``PSMList`` of all candidates with rescoring features
        populated. ``result_df`` is a DataFrame with per-candidate columns
        including ``peptide``, ``protein``, ``feature_mz``, ``feature_idx``,
        ``is_decoy``, backend-specific round-1 and round-2 scores
        (e.g. ``lda_score_r1``, ``lda_score_r2``), ``q_value``,
        ``is_tdc_winner``, ``reweighted_score``, and ``reweighted_q_value``.
        Round-2 scores and q-values are ``NaN`` for non-winners. The reweighted
        scores and q-values incorporate the LC-MS/MS and spatial log-priors.
        ``feature_names`` is the list of ranker features used.
    """
    # --- Step 1: Candidate generation ---
    # Default (digest=False): use only LC-MS/MS identified peptides as candidates.
    # With digest=True: also digest the provided FASTA for additional candidates.
    lcms_ids = None  # set below if lcms_peptides_path is provided
    if lcms_peptides_path is not None:
        from MSI-PICASSO.lcms_ids import parse_lcms_ids

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
        logger.info("  Using SCiLS per-feature intensities for log_maldi_intensity_p90")

    # --- Step 1c: Candidate generation (own if/elif/else; independent of the
    # intensity-source selection above) ---
    if decoy_method in ("balanced_shuffle", "paired_shuffle"):
        # Pass fasta_path only when --digest is active (Strategy A or C with digest).
        # LC-only mode (digest=False, lcms_ids set): fasta_path=None triggers pseudo-protein.
        _bshuffle_fasta = fasta_path if digest else None
        _selection_mode = "feature" if decoy_method == "paired_shuffle" else "length"
        logger.info(
            f"Step 1c: Generating {decoy_method} observation-space decoys "
            f"(max {max_shuffle_rounds} rounds, target_ratio={target_ratio}, "
            f"selection={_selection_mode})..."
        )
        candidates = generate_balanced_shuffle_candidates(
            fasta_path=_bshuffle_fasta,
            lcms_ids=lcms_ids,
            feature_mzs=maldi_mzs,
            matching_ppm=matching_ppm,
            max_shuffle_rounds=max_shuffle_rounds,
            target_ratio=target_ratio,
            maldi_intensities=_maldi_intensities_arr,
            maldi_intensities_p90=maldi_intensities_p90,
            maldi_intensities_sum=maldi_intensities_sum,
            missed_cleavages=missed_cleavages,
            min_length=min_length,
            max_length=max_length,
            selection_mode=_selection_mode,
        )
    else:
        candidates = match_to_maldi_features(
            maldi_mzs,
            peptide_db,
            matching_ppm,
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

    # --- Select calibration peptides (DeepLC / IM2Deep finetuning anchors) ---
    # theo_isotope_cosine is needed for the quality ranking and is computed in
    # full at Step 6; compute it here (cheap, lru_cache'd, idempotent — Step 6
    # overwrites with identical values) so the calibration set is available before
    # DeepLC finetuning at Step 4.  is_calibration_peptide is carried on the
    # candidates DataFrame and reused by the DeepLC, IM2Deep CCS, mobility, and CCS
    # filter steps instead of the decoy-sensitive n_candidates == 1 heuristic.
    from MSI-PICASSO.maldi_features import compute_theoretical_isotope_features
    candidates = compute_theoretical_isotope_features(candidates, maldi_envelopes=maldi_envelopes)
    candidates["is_calibration_peptide"] = _select_calibration_peptides(
        candidates, calibration_percentile
    )
    logger.info(
        f"  Calibration set: {int(candidates['is_calibration_peptide'].sum())} target "
        f"candidates (top {calibration_percentile:.0%} by low ppm + high isotope cosine)"
    )

    # --- Steps 2–5: LC-MS/MS data (skipped entirely when no mzML is provided) ---
    lcms_evidence = {}

    if mzml_paths:
        # --- Step 2: Load LC-MS/MS data ---
        logger.info("Step 2: Loading LC-MS/MS data...")
        lcms_data = load_lcms_data(mzml_paths)
        if verbose:
            logger.debug(f"Writing LC-MS/MS data to {output_dir}/10_debug_lcms_data.pkl")
            with open(f"{output_dir}/10_debug_lcms_data.pkl", "wb") as f:
                pickle.dump(lcms_data, f)

        # --- Step 3: MS2PIP predictions ---
        logger.info("Step 3: Finding MS2 matches and running MS2PIP...")
        from MSI-PICASSO.lcms_evidence import _find_matching_ms2_scans
        from MSI-PICASSO.utils import mz_to_mass

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
            sorted(peptide_charge_pairs),
            model="timsTOF2024",
        )

        # --- Step 4: DeepLC predictions ---
        logger.info("Step 4: Computing DeepLC predictions...")
        unique_peptides = candidates["peptide"].unique().tolist()
        _dlc_kw = dict(
            epochs=deeplc_finetune_epochs,
            lr=deeplc_finetune_lr,
            patience=deeplc_finetune_patience,
        )

        # Build finetuning calibration set: the high-quality target peptides
        # (is_calibration_peptide — top calibration_percentile by low ppm + high
        # isotope cosine) that are also observed in LC-MS/MS with RT.  Selection is
        # blind to is_decoy and unaffected by paired decoys, unlike the previous
        # n_candidates == 1 heuristic.
        _deeplc_cal_df: "pd.DataFrame | None" = None
        if lcms_ids is not None and "is_calibration_peptide" in candidates.columns:
            _cal_peps = set(
                candidates.loc[candidates["is_calibration_peptide"], "peptide"].unique()
            )
            _cal = (
                lcms_ids.peptides
                .loc[lcms_ids.peptides["sequence"].isin(_cal_peps), ["sequence", "rt_mean"]]
                .dropna(subset=["rt_mean"])
                .reset_index(drop=True)
            )
            if len(_cal) >= 20:
                _deeplc_cal_df = _cal
                logger.info(
                    f"  DeepLC calibration set: {len(_cal)} high-quality target peptides "
                    f"with observed RT (of {len(_cal_peps)} calibration candidates)"
                )
            else:
                logger.warning(
                    f"  Only {len(_cal)} calibration peptides with observed RT "
                    f"— skipping DeepLC finetuning (using default model)"
                )

        deeplc_model = None
        if _deeplc_cal_df is not None:
            deeplc_model = finetune_deeplc_from_df(_deeplc_cal_df, **_dlc_kw)
        elif msf_path:
            # lcms_ids unavailable: fall back to reading the MSF directly
            deeplc_model = finetune_deeplc(msf_path, **_dlc_kw)
        deeplc_cache = get_deeplc_predictions(unique_peptides, model=deeplc_model)

        # Estimate RT window from calibration error on the same set used for finetuning.
        # Window = rt_window_multiplier × p95(|pred_RT − obs_RT|).
        rt_window_min = 0.0
        _cal_for_window = _deeplc_cal_df
        if deeplc_cache and _cal_for_window is not None and len(_cal_for_window) > 10:
            pred_rts_cal = np.array(
                [deeplc_cache.get(seq, np.nan) for seq in _cal_for_window["sequence"]]
            )
            obs_rts_cal = _cal_for_window["rt_mean"].values.astype(float)
            valid = ~np.isnan(pred_rts_cal) & ~np.isnan(obs_rts_cal)
            if valid.sum() > 10:
                p95_mae = float(
                    np.percentile(np.abs(pred_rts_cal[valid] - obs_rts_cal[valid]), 95)
                )
                rt_window_min = rt_window_multiplier * p95_mae
                logger.info(
                    f"  DeepLC RT window: {rt_window_multiplier} × p95 MAE "
                    f"= {rt_window_min:.3f} min "
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
            fragment_tol_da=fragment_tol_da,
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
        im2deep_kwargs=im2deep_kwargs,
    )
    if verbose:
        logger.debug(f"Writing computed features to {output_dir}/13_debug_features.tsv")
        _debug_cols = [c for c in features_df.columns if c not in set(features_exclude or [])]
        features_df[_debug_cols].to_csv(f"{output_dir}/13_debug_features.tsv", sep="\t", index=False)

    # --- CCS-based candidate filtering (optional) ---
    # IM2Deep finetuning (inside compute_all_features) uses the calibration-peptide
    # set as its CCS reference.  After finetuning, im2deep_abs_delta_ccs_pct is
    # available for all candidates.  We derive a data-driven threshold from the p95
    # calibration residual on that same set (analogous to
    # rt_window_min = rt_window_multiplier * p95_mae).
    _ccs_tol_pct: float | None = None
    if match_ccs:
        if observed_ccs_per_feature is not None and "im2deep_abs_delta_ccs_pct" in features_df.columns:
            _cal_mask = (
                features_df["is_calibration_peptide"]
                if "is_calibration_peptide" in features_df.columns
                else (features_df["n_candidates"] == 1)
            )
            _single_ccs = features_df.loc[_cal_mask, "im2deep_abs_delta_ccs_pct"].dropna()
            if len(_single_ccs) >= 10:
                _p95_ccs = float(np.percentile(_single_ccs, 95))
                _ccs_tol_pct = float(ccs_window_multiplier * _p95_ccs)
                logger.info(
                    f"CCS filter: p95 |delta_CCS%| on {len(_single_ccs)} calibration "
                    f"peptides = {_p95_ccs:.2f}%. Threshold = {ccs_window_multiplier}× = "
                    f"{_ccs_tol_pct:.2f}%."
                )
                n_before = len(features_df)
                _ccs_fail = (
                    features_df["im2deep_abs_delta_ccs_pct"].notna()
                    & (features_df["im2deep_abs_delta_ccs_pct"] > _ccs_tol_pct)
                )
                features_df = features_df[~_ccs_fail].reset_index(drop=True)
                n_after = len(features_df)
                logger.info(
                    f"  Removed {n_before - n_after} of {n_before} candidates "
                    f"({100*(n_before-n_after)/max(n_before,1):.1f}%). {n_after} remain."
                )
                # n_candidates / log_n_candidates are now stale; recompute per feature.
                _feat_col = "feature_idx" if "feature_idx" in features_df.columns else "feature_mz"
                features_df["n_candidates"] = (
                    features_df.groupby(_feat_col)[_feat_col].transform("count")
                )
                features_df["log_n_candidates"] = np.log1p(features_df["n_candidates"])
            else:
                logger.warning(
                    f"CCS filter: only {len(_single_ccs)} single-candidate matches with "
                    "observed CCS — too few for a reliable data-driven threshold. "
                    "Skipping CCS filter."
                )
        else:
            logger.warning(
                "match_ccs=True but CCS filter cannot be applied: no observed CCS values "
                "were provided or IM2Deep features were not computed. Skipping CCS filter."
            )

    # --- Step 6c: per-candidate mobility-filtered colocalization (optional) ---
    _has_mob_coloc = False
    if mob_coloc and tdf_path is not None and "im2deep_predicted_ccs" in features_df.columns:
        try:
            from MSI-PICASSO.maldi_features import compute_mobility_colocalization_features
            logger.info("Computing per-candidate mobility colocalization features (step 6c)…")
            features_df = compute_mobility_colocalization_features(
                features_df,
                extraction_ppm=ppm_tolerance,
            )
            _has_mob_coloc = True
        except Exception as exc:
            logger.warning(f"Per-candidate mobility colocalization failed: {exc}. Skipping.")

    feature_names = get_feature_names(
        has_spatial=spatial_features is not None,
        has_ion_images=ion_images is not None,
        has_ccs=observed_ccs_per_feature is not None,
        has_mob_coloc=_has_mob_coloc,
    )
    logger.debug(f"Selected feature names: {feature_names}")

    # Intrinsic features that are actually present in the DataFrame.
    # Protein-level features are excluded by default (TDC correctness); opt-in
    # only when use_protein_level_features=True (--use-protein-level-feats).
    # Feature list assembly: preset → exclude → optional protein-level
    _use_main = only_main_features or features_preset == "main"
    _base_features = MAIN_FEATURES if _use_main else MALDI_INTRINSIC_FEATURES
    if _use_main:
        logger.info(
            f"  Features preset 'main': using {len(MAIN_FEATURES)} representative features "
            f"(vs {len(MALDI_INTRINSIC_FEATURES)} in the full set)"
        )
    _exclude_set = set(features_exclude or [])
    if _exclude_set:
        logger.info(f"  Excluding {len(_exclude_set)} features: {sorted(_exclude_set)}")
    _intrinsic_pool = [
        f for f in _base_features + (PROTEIN_LEVEL_FEATURES if use_protein_level_features else [])
        if f not in _exclude_set
    ]
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

    # --- Step 7: Build PSMList ---
    logger.info("Step 8: Building PSMList...")
    psm_list = candidates_to_psm_list(features_df)
    if verbose:
        logger.debug(
            f"Writing PSM list to {output_dir}/14_debug_psm_list.pkl for scorer input"
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
            psm_list, features_df, intrinsic_present, train_fdr,
            mokapot_max_iter=mokapot_max_iter,
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
        winner_pos, winners_df = _select_feature_winners(features_df, scores1, feature_col, winner_percentile)
        logger.info(
            f"  Round-1 winner selection: {len(winners_df)} candidates retained "
            f"({int(winners_df['is_decoy'].sum())} decoys)"
        )

        # --- Round 2: retrain on winner subset ---
        psm_list_r2 = candidates_to_psm_list(winners_df)
        populate_psm_features(psm_list_r2, winners_df, intrinsic_present)
        conf_obj_r2, scores2, svm_imp_r2, svm_imp_names_r2 = _rescore_svm(
            psm_list_r2, winners_df, intrinsic_present, train_fdr,
            mokapot_max_iter=mokapot_max_iter,
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
        pep_w = estimate_pep(scores2, is_decoy_w)
        pep_q_w = _pep_qvalues(pep_w)
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
            + lcms_prior_weight * np.log(np.clip(lcms_prior_w, _LOG_EPS, None))
            + spatial_prior_weight * np.log(np.clip(spatial_prior_w, _LOG_EPS, None))
        )
        rw_q2 = _tdc_qvalues(reweighted2, is_decoy_w)

        # --- Optional Storey pi0 correction ---
        if storey_pi0:
            _pi0 = _estimate_pi0_storey(scores2, is_decoy_w)
            logger.info(f"  Storey pi0 estimate: {_pi0:.4f}")
            storey_q2 = _tdc_qvalues(scores2, is_decoy_w, pi0=_pi0)
            storey_rw_q2 = _tdc_qvalues(reweighted2, is_decoy_w, pi0=_pi0)
        else:
            _pi0 = None

        # --- Map back to full candidate table ---
        is_winner_full = np.zeros(len(features_df), dtype=bool)
        is_winner_full[winner_pos] = True
        scores2_full = np.full(len(features_df), np.nan)
        scores2_full[winner_pos] = scores2
        q_full = np.full(len(features_df), np.nan)
        q_full[winner_pos] = q2
        pep_full = np.full(len(features_df), np.nan)
        pep_full[winner_pos] = pep_w
        pep_q_full = np.full(len(features_df), np.nan)
        pep_q_full[winner_pos] = pep_q_w
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
                "pep": pep_full,
                "pep_q_value": pep_q_full,
                "is_tdc_winner": is_winner_full,
                "reweighted_score": rw_full,
                "reweighted_q_value": rw_q_full,
            }
        )
        _ccs_map = observed_ccs_per_feature or {}
        result_df["feature_ccs"] = [
            _ccs_map.get(int(i), np.nan) for i in result_df["feature_idx"]
        ]
        if storey_pi0 and _pi0 is not None:
            storey_q_full = np.full(len(features_df), np.nan)
            storey_q_full[winner_pos] = storey_q2
            storey_rw_q_full = np.full(len(features_df), np.nan)
            storey_rw_q_full[winner_pos] = storey_rw_q2
            result_df["storey_q_value"] = storey_q_full
            result_df["storey_reweighted_q_value"] = storey_rw_q_full

        for fdr_threshold in [0.01, 0.05, 0.10]:
            n = (is_winner_full & ~is_decoy & (q_full <= fdr_threshold)).sum()
            n_rw = (is_winner_full & ~is_decoy & (rw_q_full <= fdr_threshold)).sum()
            extra = ""
            if storey_pi0 and _pi0 is not None:
                n_st = (is_winner_full & ~is_decoy & (storey_q_full <= fdr_threshold)).sum()
                extra = f", {n_st} (Storey π₀={_pi0:.3f})"
            logger.info(
                f"  At {fdr_threshold*100:.0f}% FDR: {n} target features (base), "
                f"{n_rw} target features (reweighted){extra}"
            )

        if debug_dir is not None:
            from MSI-PICASSO.debug_viz import save_debug_figures
            save_debug_figures(
                features_df, result_df,
                ion_images=ion_images, ion_image_mzs=ion_image_mzs,
                maldi_envelopes=maldi_envelopes,
                feature_names=intrinsic_present, model_name="svm",
                importances_r1=_imp_r1_svm, importances_r2=svm_imp_r2,
                importance_names=svm_imp_names_r2 or _imp_names_r1_svm,
                debug_dir=debug_dir, n_subset=n_debug, seed=debug_seed,
                gt_peptides=gt_peptides, storey_pi0_val=_pi0,
                ccs_tol_pct=_ccs_tol_pct,
            )

        return psm_list, result_df, feature_names, features_df

    elif model == "catboost":
        feature_col = "feature_idx" if "feature_idx" in features_df.columns else "feature_mz"

        # --- Round 1: score all candidates ---
        scores1, _imp_r1_cb, _imp_names_cb = _rescore_catboost(
            features_df,
            intrinsic_present,
            train_fdr=train_fdr,
            init_ppm_threshold=init_ppm_threshold,
            init_isotope_threshold=init_isotope_threshold,
            pseudo_label_max_iter=pseudo_label_max_iter,
            pseudo_label_fdr=pseudo_label_fdr,
            r1_seed_percentile=r1_seed_percentile,
            catboost_iterations=catboost_iterations,
        )
        if verbose:
            with open(f"{output_dir}/16_debug_catboost_scores_r1.pkl", "wb") as f:
                pickle.dump(scores1, f)

        # --- Per-feature winner selection ---
        winner_pos, winners_df = _select_feature_winners(features_df, scores1, feature_col, winner_percentile)
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
            pseudo_label_max_iter=pseudo_label_max_iter,
            pseudo_label_fdr=pseudo_label_fdr,
            r1_seed_percentile=r1_seed_percentile,
            catboost_iterations=catboost_iterations,
        )
        if verbose:
            with open(f"{output_dir}/16_debug_catboost_scores_r2.pkl", "wb") as f:
                pickle.dump(scores2, f)

        # --- Standard TDC FDR on winners ---
        is_decoy_w = winners_df["is_decoy"].values.astype(bool)
        q2 = _tdc_qvalues(scores2, is_decoy_w)
        pep_w = estimate_pep(scores2, is_decoy_w)
        pep_q_w = _pep_qvalues(pep_w)
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
            + lcms_prior_weight * np.log(np.clip(lcms_prior_w, _LOG_EPS, None))
            + spatial_prior_weight * np.log(np.clip(spatial_prior_w, _LOG_EPS, None))
        )
        rw_q2 = _tdc_qvalues(reweighted2, is_decoy_w)

        # --- Optional Storey pi0 correction ---
        if storey_pi0:
            _pi0 = _estimate_pi0_storey(scores2, is_decoy_w)
            logger.info(f"  Storey pi0 estimate: {_pi0:.4f}")
            storey_q2 = _tdc_qvalues(scores2, is_decoy_w, pi0=_pi0)
            storey_rw_q2 = _tdc_qvalues(reweighted2, is_decoy_w, pi0=_pi0)
        else:
            _pi0 = None

        # --- Map back to full candidate table ---
        is_winner_full = np.zeros(len(features_df), dtype=bool)
        is_winner_full[winner_pos] = True
        scores2_full = np.full(len(features_df), np.nan)
        scores2_full[winner_pos] = scores2
        q_full = np.full(len(features_df), np.nan)
        q_full[winner_pos] = q2
        pep_full = np.full(len(features_df), np.nan)
        pep_full[winner_pos] = pep_w
        pep_q_full = np.full(len(features_df), np.nan)
        pep_q_full[winner_pos] = pep_q_w
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
                "pep": pep_full,
                "pep_q_value": pep_q_full,
                "is_tdc_winner": is_winner_full,
                "reweighted_score": rw_full,
                "reweighted_q_value": rw_q_full,
            }
        )
        _ccs_map = observed_ccs_per_feature or {}
        result_df["feature_ccs"] = [
            _ccs_map.get(int(i), np.nan) for i in result_df["feature_idx"]
        ]
        if storey_pi0 and _pi0 is not None:
            storey_q_full = np.full(len(features_df), np.nan)
            storey_q_full[winner_pos] = storey_q2
            storey_rw_q_full = np.full(len(features_df), np.nan)
            storey_rw_q_full[winner_pos] = storey_rw_q2
            result_df["storey_q_value"] = storey_q_full
            result_df["storey_reweighted_q_value"] = storey_rw_q_full

        for fdr_threshold in [0.01, 0.05, 0.10]:
            n = (is_winner_full & ~is_decoy & (q_full <= fdr_threshold)).sum()
            n_rw = (is_winner_full & ~is_decoy & (rw_q_full <= fdr_threshold)).sum()
            extra = ""
            if storey_pi0 and _pi0 is not None:
                n_st = (is_winner_full & ~is_decoy & (storey_q_full <= fdr_threshold)).sum()
                extra = f", {n_st} (Storey π₀={_pi0:.3f})"
            logger.info(
                f"  At {fdr_threshold*100:.0f}% FDR: {n} target features (base), "
                f"{n_rw} target features (reweighted){extra}"
            )

        if debug_dir is not None:
            from MSI-PICASSO.debug_viz import save_debug_figures
            save_debug_figures(
                features_df, result_df,
                ion_images=ion_images, ion_image_mzs=ion_image_mzs,
                maldi_envelopes=maldi_envelopes,
                feature_names=intrinsic_present, model_name="catboost",
                importances_r1=_imp_r1_cb, importances_r2=cb_imp_r2,
                importance_names=cb_imp_names_r2 or _imp_names_cb,
                debug_dir=debug_dir, n_subset=n_debug, seed=debug_seed,
                gt_peptides=gt_peptides, storey_pi0_val=_pi0,
                ccs_tol_pct=_ccs_tol_pct,
            )

        return psm_list, result_df, feature_names, features_df

    elif model == "lda":
        feature_col = "feature_idx" if "feature_idx" in features_df.columns else "feature_mz"

        # --- Round 1: score all candidates ---
        scores1, _imp_r1_lda, _struct_coefs_r1_lda, _struct_names_r1_lda, _imp_names_lda = _rescore_lda(
            features_df,
            intrinsic_present,
            init_ppm_threshold=init_ppm_threshold,
            init_fdr=init_fdr,
            train_fdr=train_fdr,
            max_iter=max_iter,
            r1_seed_percentile=r1_seed_percentile,
            min_seed_positives=min_seed_positives,
        )
        # Output importances
        if verbose:
            with open(f"{output_dir}/17_debug_lda_scores_r1.pkl", "wb") as f:
                pickle.dump(scores1, f)
            _imp_df_r1 = pd.DataFrame({
                "feature": _imp_names_lda,
                "importance": _imp_r1_lda,
            })
            if _struct_coefs_r1_lda is not None and _struct_names_r1_lda:
                _imp_df_r1 = _imp_df_r1.merge(
                    pd.DataFrame({"feature": _struct_names_r1_lda, "structure_coef": _struct_coefs_r1_lda}),
                    on="feature", how="left",
                )
            _imp_df_r1.sort_values("importance", ascending=False).to_csv(
                f"{output_dir}/17_debug_lda_importances_r1.tsv", sep="\t", index=False
            )

        # --- Per-feature winner selection ---
        winner_pos, winners_df = _select_feature_winners(features_df, scores1, feature_col, winner_percentile)
        logger.info(
            f"  Round-1 winner selection: {len(winners_df)} candidates retained "
            f"({int(winners_df['is_decoy'].sum())} decoys)"
        )

        # --- Round 2: retrain on winner subset ---
        # Seed R2 from the top-r2_seed_percentile of target winners by R1 score.
        # After winner selection the remaining targets may have ppm > init_ppm_threshold
        # (R1 lifted them on other features), so re-seeding from ppm alone would
        # leave only a handful of seeds, causing the LDA to degenerate.  Using
        # q-values from R1 has the same problem when R1 itself identified very few
        # pseudo-positives.  A percentile cut on raw R1 scores is guaranteed to
        # produce a reasonably sized seed regardless of how well R1 converged.
        is_decoy_w = winners_df["is_decoy"].values.astype(bool)
        target_scores_w = scores1[winner_pos][~is_decoy_w]
        score_threshold = np.percentile(target_scores_w, 100.0 * (1.0 - r2_seed_percentile))
        r2_seed_mask = (~is_decoy_w) & (scores1[winner_pos] >= score_threshold)
        logger.info(
            f"  LDA R2: seeding from top-{r2_seed_percentile*100:.0f}% R1 target scores "
            f"(score ≥ {score_threshold:.3f}) → {r2_seed_mask.sum()} positives"
        )

        # --- Feature selection for R2: drop below-median importance from R1 ---
        if lda_r2_median_filter and _imp_r1_lda is not None and _imp_names_lda:
            imp_abs = np.abs(_imp_r1_lda)
            imp_median = np.median(imp_abs)
            lda_r2_features = [f for f, a in zip(_imp_names_lda, imp_abs) if a >= imp_median]
            logger.info(
                f"  LDA R2: using {len(lda_r2_features)}/{len(_imp_names_lda)} features "
                f"with |importance| ≥ median ({imp_median:.4f})"
            )
        else:
            lda_r2_features = intrinsic_present

        scores2, lda_imp_r2, _struct_coefs_r2_lda, _struct_names_r2_lda, lda_imp_names_r2 = _rescore_lda(
            winners_df,
            lda_r2_features,
            init_ppm_threshold=init_ppm_threshold,
            seed_mask=r2_seed_mask,
            n_interaction_features=n_interaction_features,
            r1_importances=_imp_r1_lda,
            r1_feature_names=_imp_names_lda,
            init_fdr=init_fdr,
            train_fdr=train_fdr,
            max_iter=max_iter,
            r1_seed_percentile=r1_seed_percentile,
            min_seed_positives=min_seed_positives,
        )
        # Output importances
        if verbose:
            with open(f"{output_dir}/17_debug_lda_scores_r2.pkl", "wb") as f:
                pickle.dump(scores2, f)
            _imp_df_r2 = pd.DataFrame({
                "feature": lda_imp_names_r2 or _imp_names_lda,
                "importance": lda_imp_r2,
            })
            if _struct_coefs_r2_lda is not None and _struct_names_r2_lda:
                _imp_df_r2 = _imp_df_r2.merge(
                    pd.DataFrame({"feature": _struct_names_r2_lda, "structure_coef": _struct_coefs_r2_lda}),
                    on="feature", how="left",
                )
            _imp_df_r2.sort_values("importance", ascending=False).to_csv(
                f"{output_dir}/17_debug_lda_importances_r2.tsv", sep="\t", index=False
            )

        # --- Standard TDC FDR on winners ---
        is_decoy_w = winners_df["is_decoy"].values.astype(bool)
        q2 = _tdc_qvalues(scores2, is_decoy_w)
        pep_w = estimate_pep(scores2, is_decoy_w)
        pep_q_w = _pep_qvalues(pep_w)
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
            + lcms_prior_weight * np.log(np.clip(lcms_prior_w, _LOG_EPS, None))
            + spatial_prior_weight * np.log(np.clip(spatial_prior_w, _LOG_EPS, None))
        )
        rw_q2 = _tdc_qvalues(reweighted2, is_decoy_w)

        # --- Optional Storey pi0 correction ---
        if storey_pi0:
            _pi0 = _estimate_pi0_storey(scores2, is_decoy_w)
            logger.info(f"  Storey pi0 estimate: {_pi0:.4f}")
            storey_q2 = _tdc_qvalues(scores2, is_decoy_w, pi0=_pi0)
            storey_rw_q2 = _tdc_qvalues(reweighted2, is_decoy_w, pi0=_pi0)
        else:
            _pi0 = None

        # --- Map back to full candidate table ---
        is_winner_full = np.zeros(len(features_df), dtype=bool)
        is_winner_full[winner_pos] = True
        scores2_full = np.full(len(features_df), np.nan)
        scores2_full[winner_pos] = scores2
        q_full = np.full(len(features_df), np.nan)
        q_full[winner_pos] = q2
        pep_full = np.full(len(features_df), np.nan)
        pep_full[winner_pos] = pep_w
        pep_q_full = np.full(len(features_df), np.nan)
        pep_q_full[winner_pos] = pep_q_w
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
                "pep": pep_full,
                "pep_q_value": pep_q_full,
                "is_tdc_winner": is_winner_full,
                "reweighted_score": rw_full,
                "reweighted_q_value": rw_q_full,
            }
        )
        _ccs_map = observed_ccs_per_feature or {}
        result_df["feature_ccs"] = [
            _ccs_map.get(int(i), np.nan) for i in result_df["feature_idx"]
        ]
        if storey_pi0 and _pi0 is not None:
            storey_q_full = np.full(len(features_df), np.nan)
            storey_q_full[winner_pos] = storey_q2
            storey_rw_q_full = np.full(len(features_df), np.nan)
            storey_rw_q_full[winner_pos] = storey_rw_q2
            result_df["storey_q_value"] = storey_q_full
            result_df["storey_reweighted_q_value"] = storey_rw_q_full

        for fdr_threshold in [0.01, 0.05, 0.10]:
            n = (is_winner_full & ~is_decoy & (q_full <= fdr_threshold)).sum()
            n_rw = (is_winner_full & ~is_decoy & (rw_q_full <= fdr_threshold)).sum()
            extra = ""
            if storey_pi0 and _pi0 is not None:
                n_st = (is_winner_full & ~is_decoy & (storey_q_full <= fdr_threshold)).sum()
                extra = f", {n_st} (Storey π₀={_pi0:.3f})"
            logger.info(
                f"  At {fdr_threshold*100:.0f}% FDR: {n} target features (base), "
                f"{n_rw} target features (reweighted){extra}"
            )

        if debug_dir is not None:
            from MSI-PICASSO.debug_viz import save_debug_figures
            save_debug_figures(
                features_df, result_df,
                ion_images=ion_images, ion_image_mzs=ion_image_mzs,
                maldi_envelopes=maldi_envelopes,
                feature_names=intrinsic_present, model_name="lda",
                importances_r1=_imp_r1_lda, importances_r2=lda_imp_r2,
                importance_names=_imp_names_lda,
                importance_names_r2=lda_imp_names_r2 or _imp_names_lda,
                structure_coefs_r1=_struct_coefs_r1_lda,
                structure_names_r1=_struct_names_r1_lda,
                structure_coefs_r2=_struct_coefs_r2_lda,
                structure_names_r2=_struct_names_r2_lda,
                debug_dir=debug_dir, n_subset=n_debug, seed=debug_seed,
                gt_peptides=gt_peptides, storey_pi0_val=_pi0,
                ccs_tol_pct=_ccs_tol_pct,
            )

        return psm_list, result_df, feature_names, features_df

    elif model == "qda":
        feature_col = "feature_idx" if "feature_idx" in features_df.columns else "feature_mz"

        # --- Round 1: score all candidates ---
        scores1, _, _imp_r1_qda, _imp_names_qda = _rescore_qda(
            features_df,
            intrinsic_present,
            init_ppm_threshold=init_ppm_threshold,
            init_fdr=init_fdr,
            train_fdr=train_fdr,
            max_iter=max_iter,
            r1_seed_percentile=r1_seed_percentile,
            min_seed_positives=min_seed_positives,
        )
        if verbose:
            with open(f"{output_dir}/17_debug_qda_scores_r1.pkl", "wb") as f:
                pickle.dump(scores1, f)
            qda_importances_df = pd.DataFrame({
                "feature": _imp_names_qda,
                "importance": _imp_r1_qda if _imp_r1_qda is not None else np.zeros(len(_imp_names_qda)),
            }).sort_values("importance", key=np.abs, ascending=False)
            qda_importances_df.to_csv(
                f"{output_dir}/17_debug_qda_importances_r1.tsv", sep="\t", index=False
            )

        # --- Per-feature winner selection ---
        winner_pos, winners_df = _select_feature_winners(features_df, scores1, feature_col, winner_percentile)
        logger.info(
            f"  Round-1 winner selection: {len(winners_df)} candidates retained "
            f"({int(winners_df['is_decoy'].sum())} decoys)"
        )

        # --- Round 2: retrain on winner subset ---
        is_decoy_w = winners_df["is_decoy"].values.astype(bool)
        target_scores_w = scores1[winner_pos][~is_decoy_w]
        score_threshold = np.percentile(target_scores_w, 100.0 * (1.0 - r2_seed_percentile))
        r2_seed_mask = (~is_decoy_w) & (scores1[winner_pos] >= score_threshold)
        logger.info(
            f"  QDA R2: seeding from top-{r2_seed_percentile*100:.0f}% R1 target scores "
            f"(score ≥ {score_threshold:.3f}) → {r2_seed_mask.sum()} positives"
        )

        if _imp_r1_qda is not None and _imp_names_qda:
            imp_abs = np.abs(_imp_r1_qda)
            imp_median = np.median(imp_abs)
            r2_features = [f for f, a in zip(_imp_names_qda, imp_abs) if a >= imp_median]
            logger.info(
                f"  QDA R2: using {len(r2_features)}/{len(_imp_names_qda)} features "
                f"with |importance| ≥ median ({imp_median:.4f})"
            )
        else:
            r2_features = intrinsic_present

        scores2, pep_proba2, qda_imp_r2, qda_imp_names_r2 = _rescore_qda(
            winners_df,
            r2_features,
            init_ppm_threshold=init_ppm_threshold,
            seed_mask=r2_seed_mask,
            init_fdr=init_fdr,
            train_fdr=train_fdr,
            max_iter=max_iter,
            r1_seed_percentile=r1_seed_percentile,
            min_seed_positives=min_seed_positives,
        )
        if verbose:
            with open(f"{output_dir}/17_debug_qda_scores_r2.pkl", "wb") as f:
                pickle.dump(scores2, f)
            qda_importances_df = pd.DataFrame({
                "feature": qda_imp_names_r2 or _imp_names_qda,
                "importance": qda_imp_r2 if qda_imp_r2 is not None else np.zeros(len(qda_imp_names_r2 or _imp_names_qda)),
            }).sort_values("importance", key=np.abs, ascending=False)
            qda_importances_df.to_csv(
                f"{output_dir}/17_debug_qda_importances_r2.tsv", sep="\t", index=False
            )

        # --- Standard TDC FDR on winners ---
        is_decoy_w = winners_df["is_decoy"].values.astype(bool)
        q2 = _tdc_qvalues(scores2, is_decoy_w)
        # Use QDA's own posterior probabilities: predict_proba(X)[:, decoy_class]
        # is exactly P(false | features) — no separate mixture model needed.
        pep_w = pep_proba2
        pep_q_w = _pep_qvalues(pep_w)
        lcms_prior_w = compute_lcms_prior(winners_df, lcms_present)
        spatial_prior_w = compute_spatial_prior(winners_df, spatial_present)
        _LOG_EPS = 1e-12
        reweighted2 = (
            scores2
            + lcms_prior_weight * np.log(np.clip(lcms_prior_w, _LOG_EPS, None))
            + spatial_prior_weight * np.log(np.clip(spatial_prior_w, _LOG_EPS, None))
        )
        rw_q2 = _tdc_qvalues(reweighted2, is_decoy_w)

        # --- Optional Storey pi0 correction ---
        if storey_pi0:
            _pi0 = _estimate_pi0_storey(scores2, is_decoy_w)
            logger.info(f"  Storey pi0 estimate: {_pi0:.4f}")
            storey_q2 = _tdc_qvalues(scores2, is_decoy_w, pi0=_pi0)
            storey_rw_q2 = _tdc_qvalues(reweighted2, is_decoy_w, pi0=_pi0)
        else:
            _pi0 = None

        # --- Map back to full candidate table ---
        is_winner_full = np.zeros(len(features_df), dtype=bool)
        is_winner_full[winner_pos] = True
        scores2_full = np.full(len(features_df), np.nan)
        scores2_full[winner_pos] = scores2
        q_full = np.full(len(features_df), np.nan)
        q_full[winner_pos] = q2
        pep_full = np.full(len(features_df), np.nan)
        pep_full[winner_pos] = pep_w
        pep_q_full = np.full(len(features_df), np.nan)
        pep_q_full[winner_pos] = pep_q_w
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
                "qda_score_r1": scores1,
                "qda_score_r2": scores2_full,
                "q_value": q_full,
                "pep": pep_full,
                "pep_q_value": pep_q_full,
                "is_tdc_winner": is_winner_full,
                "reweighted_score": rw_full,
                "reweighted_q_value": rw_q_full,
            }
        )
        _ccs_map = observed_ccs_per_feature or {}
        result_df["feature_ccs"] = [
            _ccs_map.get(int(i), np.nan) for i in result_df["feature_idx"]
        ]
        if storey_pi0 and _pi0 is not None:
            storey_q_full = np.full(len(features_df), np.nan)
            storey_q_full[winner_pos] = storey_q2
            storey_rw_q_full = np.full(len(features_df), np.nan)
            storey_rw_q_full[winner_pos] = storey_rw_q2
            result_df["storey_q_value"] = storey_q_full
            result_df["storey_reweighted_q_value"] = storey_rw_q_full

        for fdr_threshold in [0.01, 0.05, 0.10]:
            n = (is_winner_full & ~is_decoy & (q_full <= fdr_threshold)).sum()
            n_rw = (is_winner_full & ~is_decoy & (rw_q_full <= fdr_threshold)).sum()
            extra = ""
            if storey_pi0 and _pi0 is not None:
                n_st = (is_winner_full & ~is_decoy & (storey_q_full <= fdr_threshold)).sum()
                extra = f", {n_st} (Storey π₀={_pi0:.3f})"
            logger.info(
                f"  At {fdr_threshold*100:.0f}% FDR: {n} target features (base), "
                f"{n_rw} target features (reweighted){extra}"
            )

        if debug_dir is not None:
            from MSI-PICASSO.debug_viz import save_debug_figures
            save_debug_figures(
                features_df, result_df,
                ion_images=ion_images, ion_image_mzs=ion_image_mzs,
                maldi_envelopes=maldi_envelopes,
                feature_names=intrinsic_present, model_name="qda",
                pep_method="kde",
                importances_r1=_imp_r1_qda, importances_r2=qda_imp_r2,
                importance_names=_imp_names_qda,
                importance_names_r2=qda_imp_names_r2 or _imp_names_qda,
                debug_dir=debug_dir, n_subset=n_debug, seed=debug_seed,
                gt_peptides=gt_peptides, storey_pi0_val=_pi0,
                ccs_tol_pct=_ccs_tol_pct,
            )

        return psm_list, result_df, feature_names, features_df

    else:
        raise ValueError(f"Unknown model '{model}'. Choose 'svm', 'catboost', 'lda', or 'qda'.")
