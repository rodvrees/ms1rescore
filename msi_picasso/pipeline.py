"""End-to-end symmetric MALDI-MSI rescoring pipeline."""

import logging
import os
import pickle
import warnings

import numpy as np
import pandas as pd

from msi_picasso.candidates import (
    generate_substitution_candidates,
    digest_fasta,
    digest_identified_proteins,
    generate_balanced_shuffle_candidates,
    generate_mz_shift_candidates,
    generate_mz_shuffle_candidates,
    load_entrapment_candidates,
    match_to_maldi_features,
)
from msi_picasso.feature_generator import (
    FEATURE_NAN_FILL,
    LCMS_PRIOR_FEATURES,
    MAIN_FEATURES,
    MALDI_INTRINSIC_FEATURES,
    MOB_QUALITY_FEATURES,
    REGION_COLOCALIZATION_FEATURES,
    WITHIN_REGION_COLOCALIZATION_FEATURES,
    PROTEIN_LEVEL_FEATURES,
    SPATIAL_PRIOR_FEATURES,
    SPATIAL_RANKER_FEATURES,
    candidates_to_psm_list,
    compute_all_features,
    get_feature_names,
    populate_psm_features,
)
from msi_picasso.lcms_evidence import (
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

# Decoy methods whose decoys land on real MALDI features, giving spatial ranker
# features a symmetric null.  shuffle/balanced_shuffle/paired_shuffle do not.
_SPATIAL_RANKER_OK_DECOYS = frozenset(["entrapment", "mz_shift", "mz_shuffle", "substitution"])

# Decoy methods for which the intrinsic 2D peak-quality features (MOB_QUALITY_FEATURES)
# enter the ranker by default. For substitution the decoy sits at a distinct (often
# empty/noisy) m/z, so feature-level peak quality discriminates. For mz_shuffle the
# co-located target+decoy share the feature, so these columns are exactly symmetric
# (AUC ~= 0.5) — a harmless default and a built-in leak-safety check. No m/z-baseline
# leak (they use only the observed peak, no prediction), so unlike the im2deep_* CCS
# scalars they are NOT in _MZ_SHUFFLE_CCS_LEAK_FEATURES.
_MOB_QUALITY_DEFAULT_DECOYS = frozenset(["substitution", "mz_shuffle"])

# Features that use the candidate's PREDICTED CCS/mobility to gate or compare against
# the observed feature. For mz_shuffle (peptide relocated far in mass; CCS/1-K0 ∝ m/z)
# these leak the m/z baseline rather than testing identity, so they are dropped from
# the ranker — the m/z-detrended *_resid CCS features replace them.
_MZ_SHUFFLE_CCS_LEAK_FEATURES = frozenset([
    "im2deep_delta_ccs", "im2deep_abs_delta_ccs_pct",
    "im2deep_ccs_zscore", "im2deep_ccs_rank",
    "isotope_colocalization_m1_mob", "isotope_colocalization_m2_mob",
    "isotope_colocalization_mean_mob",
    "adduct_colocalization_na_mob", "adduct_colocalization_k_mob",
    "adduct_colocalization_chca_mob",
])


def _resolve_spatial_ranker_features(
    use_spatial_ranker_features: bool, decoy_method: str
) -> bool:
    """Return whether spatial ranker features may be used with this decoy method.

    Emits a ``UserWarning`` and returns ``False`` when the flag is requested with
    a decoy method that lacks a consistent spatial anchor (any shuffle variant).
    """
    if use_spatial_ranker_features and decoy_method not in _SPATIAL_RANKER_OK_DECOYS:
        warnings.warn(
            f"--use-spatial-ranker-features is only valid with --decoy-method entrapment, "
            f"mz_shift, or mz_shuffle. With '{decoy_method}' decoys, spatial features are "
            f"asymmetric: shuffle/balanced_shuffle/paired_shuffle decoys have no consistent "
            f"spatial anchor and their ion images reflect arbitrary m/z space rather than a "
            f"real decoy identity. Disabling --use-spatial-ranker-features.",
            UserWarning,
            stacklevel=2,
        )
        return False
    return use_spatial_ranker_features


def _observed_ccs_by_feature_idx(
    candidates: pd.DataFrame,
    maldi_mzs: np.ndarray,
    ccs_arr: np.ndarray,
) -> dict | None:
    """Build an ``observed_ccs_per_feature`` dict keyed by candidate ``feature_idx``.

    ``ccs_arr`` is aligned with ``maldi_mzs`` (the queried m/z grid).  In raw-query
    mode a candidate's ``feature_idx`` indexes the digest grid, not ``maldi_mzs``,
    so the mapping is bridged via ``feature_mz`` (1:1 with ``feature_idx``).  This
    matches how ``compute_im2deep_features`` consumes the dict
    (``df["feature_idx"].map(observed_ccs_per_feature)``).  Non-finite CCS values
    are dropped.  Returns ``None`` when no feature has a finite CCS, so downstream
    ``is not None`` guards behave like the no-mobility path.
    """
    mz_to_ccs = {
        float(m): float(c)
        for m, c in zip(np.asarray(maldi_mzs), np.asarray(ccs_arr))
        if np.isfinite(c)
    }
    ccs_map = {
        int(fi): mz_to_ccs[float(mz)]
        for fi, mz in candidates[["feature_idx", "feature_mz"]]
        .drop_duplicates()
        .itertuples(index=False)
        if float(mz) in mz_to_ccs
    }
    return ccs_map or None


def _mob_hist_by_feature_idx(
    candidates: pd.DataFrame,
    maldi_mzs: np.ndarray,
    mob_k0_hist: np.ndarray | None,
) -> dict | None:
    """Build a ``mob_k0_hist_by_feature`` dict keyed by candidate ``feature_idx``.

    ``mob_k0_hist`` is a ``(len(maldi_mzs), n_bins)`` array aligned with the queried
    m/z grid.  Bridged to candidate ``feature_idx`` via ``feature_mz`` exactly like
    :func:`_observed_ccs_by_feature_idx` (each value is the candidate's 1/K0
    histogram row).  Returns ``None`` when ``mob_k0_hist`` is absent.
    """
    if mob_k0_hist is None:
        return None
    maldi_mzs = np.asarray(maldi_mzs)
    mz_to_row = {float(m): i for i, m in enumerate(maldi_mzs)}
    hist_map = {
        int(fi): mob_k0_hist[mz_to_row[float(mz)]]
        for fi, mz in candidates[["feature_idx", "feature_mz"]]
        .drop_duplicates()
        .itertuples(index=False)
        if float(mz) in mz_to_row
    }
    return hist_map or None


def _recompute_ppm_from_centroids(
    feature_mz: np.ndarray,
    maldi_mzs: np.ndarray,
    centroid_mz: np.ndarray,
    worst_case_ppm: float | None = None,
) -> np.ndarray:
    """Symmetric raw-query ``ppm_error``: observed peak centroid vs the candidate anchor.

    In raw-query mode every candidate (target or decoy) is matched against the
    *theoretical* digest grid, so the usual ``(feature_mz - mh_mz)`` ppm is 0 by
    construction and decoys inherit 0.  Instead, for each candidate row compute the
    mass accuracy of the observed peak in its own extraction window:
    ``(observed_centroid - feature_mz) / feature_mz * 1e6``, where ``feature_mz`` is
    the candidate's queried anchor (the peptide's [M+H]+ for a target, the shifted
    m/z for an ``mz_shift`` decoy).  Identical treatment for targets and decoys (no
    inheritance), bounded by ±extraction_ppm, and non-leaking (it never references
    the peptide mass for a decoy).

    A window with no observed peak has unmeasurable mass accuracy.  When
    ``worst_case_ppm`` is given, such rows are set to that worst-case value (the
    extraction window edge — a real peak's centroid is always within
    ±extraction_ppm of the anchor, so this is the worst in-distribution value) so
    empty-signal candidates (e.g. ``mz_shift`` decoys shifted into empty m/z space)
    are penalised on ppm rather than median-imputed to an average value.  When
    ``worst_case_ppm`` is ``None`` those rows are left ``NaN``.

    ``centroid_mz`` is aligned with ``maldi_mzs``; the result is aligned with the
    per-row ``feature_mz`` input.
    """
    mz_to_centroid = {
        float(m): float(c)
        for m, c in zip(np.asarray(maldi_mzs), np.asarray(centroid_mz))
        if np.isfinite(c)
    }
    fmz = np.asarray(feature_mz, dtype=np.float64)
    obs = np.array([mz_to_centroid.get(float(m), np.nan) for m in fmz], dtype=np.float64)
    with np.errstate(invalid="ignore"):
        ppm = (obs - fmz) / fmz * 1e6
    if worst_case_ppm is not None:
        ppm = np.where(np.isfinite(ppm), ppm, float(worst_case_ppm))
    return ppm


# Protein colocalization features are Pearson r aggregates (higher = more target-like
# protein co-distribution). A candidate whose MALDI feature has *no signal* (constant
# ion image) has an undefined correlation -> NaN, which the scoring imputer would fill
# with the column median, i.e. an *average* coloc value, silently rewarding a
# zero-evidence candidate. Mirror the ppm worst-case fill in _recompute_ppm_from_centroids:
# set those NaNs to the worst in-distribution value (the pooled finite minimum) so a
# no-signal candidate is penalised on coloc rather than imputed up to average.
_PROTEIN_COLOC_WORST_PREFIXES = (
    "protein_colocalization",
    "protein_region_colocalization",
    "protein_within_region_colocalization",
    "protein_dominant_region_colocalization",
)


def _fill_nosignal_coloc_worst_case(features_df: pd.DataFrame) -> pd.DataFrame:
    """Worst-case fill of protein-colocalization NaNs for zero-signal candidates only.

    Symmetry: the no-signal mask is read from ``feature_intensity_sum`` (the ion image
    alone, an ``is_decoy``-blind quantity that a co-located target/decoy pair share under
    mz_shuffle), and the fill constant is the pooled finite minimum over all candidates,
    so both the mask and the value are label-blind and no target/decoy asymmetry is
    introduced. Only no-signal rows are touched; NaNs from a single-feature protein (no
    within-protein partner) are left for the downstream median imputer, since those are
    "coloc undefined", not "no evidence".
    """
    if "feature_intensity_sum" not in features_df.columns:
        return features_df
    no_signal = ~(features_df["feature_intensity_sum"] > 0)  # True for 0, NaN, negative
    if not no_signal.any():
        return features_df
    cols = [
        c
        for c in features_df.columns
        if c.startswith(_PROTEIN_COLOC_WORST_PREFIXES) and not c.endswith("_n_partners")
    ]
    n_filled = 0
    for c in cols:
        finite = features_df[c][np.isfinite(features_df[c])]
        if finite.empty:
            continue
        worst = float(finite.min())
        fill_mask = no_signal & ~np.isfinite(features_df[c])
        n = int(fill_mask.sum())
        if n:
            features_df.loc[fill_mask, c] = worst
            n_filled += n
    if cols:
        logger.info(
            f"  No-signal coloc fill: set {n_filled} NaN entries across {len(cols)} "
            f"protein-colocalization columns to the worst-case (pooled min) for "
            f"{int(no_signal.sum())} zero-signal candidates (symmetric, is_decoy-blind)."
        )
    return features_df


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


def _encode_labels(is_decoy, positive_mask):
    """Three-valued semi-supervised labels: -1 for decoys, +1 for positive
    targets (``positive_mask`` True), 0 for unlabelled targets. int8."""
    return np.where(
        is_decoy, np.int8(-1),
        np.where(positive_mask, np.int8(1), np.int8(0)),
    ).astype(np.int8)


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

    When the pairwise result is *still* below ``min_seed_positives``, a shallow
    decision tree (depth 3, target vs. decoy) is fit on all eligible features at
    once, so weakly-correlated evidence can combine beyond what raw sums/pairs
    reach. Its leaf-probability score is used if it beats the pairwise result.

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

    result: tuple[np.ndarray, str, int] | None = None
    if best_j >= 0:
        assert best_q is not None
        result = (_encode_labels(is_decoy, best_q <= init_fdr), feature_names[best_j], best_n)

    eligible = [
        j for j, fname in enumerate(feature_names)
        if fname not in _BEST_FEAT_SKIP and X_imp[:, j].std() > 0
    ]

    # --- Pairwise sweep when single-feature result is weak ---
    if best_n < min_seed_positives and eligible:
        # Scale to zero mean, unit variance so both features contribute equally.
        col_means = X_imp[:, eligible].mean(axis=0)
        col_stds = X_imp[:, eligible].std(axis=0)
        col_stds[col_stds == 0] = 1.0
        X_sc = np.empty((X_imp.shape[0], len(eligible)), dtype=np.float64)
        for k, j in enumerate(eligible):
            X_sc[:, k] = (X_imp[:, j] - col_means[k]) / col_stds[k]

        pair_best_n = best_n
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
                            sign_str = "+" if sign == +1 else "-"
                            pair_best_name = (
                                f"{feature_names[gi]} {sign_str} {feature_names[gj]}"
                            )
                            result = (_encode_labels(is_decoy, q <= init_fdr), pair_best_name, n_pass)

        if pair_best_n > best_n:
            logger.info(
                "  Selected pair (%s) with %d PSMs at q<=%g",
                result[1], pair_best_n, init_fdr,
            )
            best_n = pair_best_n

    # --- Shallow-tree sweep when the pairwise result is still weak ---
    # A depth-3 tree on all eligible spectral-quality features at once lets
    # weakly-correlated evidence combine beyond what raw sums/pairs reach.
    is_target = ~is_decoy
    if best_n < min_seed_positives and eligible and is_decoy.any() and is_target.any():
        from sklearn.tree import DecisionTreeClassifier

        tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=20, random_state=0)
        tree.fit(X_imp[:, eligible], is_target.astype(int))
        scores = tree.predict_proba(X_imp[:, eligible])[:, 1] + tiebreak
        q = _tdc_qvalues(scores, is_decoy)
        n_pass = int((is_target & (q <= init_fdr)).sum())
        if n_pass > best_n:
            tree_name = f"tree(depth=3, n_features={len(eligible)})"
            logger.info(
                "  Selected shallow-tree seed (%s) with %d PSMs at q<=%g",
                tree_name, n_pass, init_fdr,
            )
            result = (_encode_labels(is_decoy, q <= init_fdr), tree_name, n_pass)
            best_n = n_pass
        else:
            logger.info(
                "  Shallow-tree seed (depth=3, %d features) gave %d PSMs at q<=%g — "
                "did not beat pairwise result (%d), keeping pairwise",
                len(eligible), n_pass, init_fdr, best_n,
            )

    if result is None or result[2] == 0:
        return None
    return result


def _make_fold_ids(is_decoy: np.ndarray, cv_folds: int) -> np.ndarray | None:
    """Fixed, is_decoy-stratified fold assignment for out-of-fold scoring.

    Returns an int array (row → fold) or ``None`` when there are too few targets
    or decoys for ``cv_folds``-fold CV (caller then scores in-sample).
    """
    is_decoy = np.asarray(is_decoy, dtype=bool)
    n = len(is_decoy)
    if int(is_decoy.sum()) < cv_folds * 2 or int((~is_decoy).sum()) < cv_folds * 2:
        return None
    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=0)
    fold_ids = np.empty(n, dtype=np.int64)
    for k, (_, test) in enumerate(skf.split(np.zeros(n), is_decoy.astype(int))):
        fold_ids[test] = k
    return fold_ids


def _cv_semisup_scores(X_fit, labels, fold_ids, make_pipe):
    """Semi-supervised scores with **out-of-fold** cross-validation.

    Trains a discriminant on ``label==1`` (positives) vs ``label==-1`` (decoys);
    ``label==0`` rows (unlabelled targets) are scored but never trained on.

    Returns ``(scores, pipe_full)``:
    - ``scores`` — when ``fold_ids`` is given, each row is scored by a model trained
      on the *other* folds' pos/neg rows (no row is scored by a model that trained
      on it), so the discriminant cannot manufacture target/decoy separation by
      overfitting.  ``None`` fold_ids (too few pos/neg) → in-sample scores.
    - ``pipe_full`` — a model fit on ALL current pos/neg rows, used only for
      reporting feature importances / structure coefficients (never for FDR).
    """
    labels = np.asarray(labels)
    pos = labels == 1
    neg = labels == -1
    train = pos | neg
    pipe_full = make_pipe()
    pipe_full.fit(X_fit[train], pos[train].astype(float))
    if fold_ids is None:
        return pipe_full.decision_function(X_fit).ravel(), pipe_full

    oof = np.full(len(labels), np.nan)
    for k in np.unique(fold_ids):
        test = fold_ids == k
        tr = train & ~test
        ytr = pos[tr].astype(float)
        # Need both classes in the training partition; otherwise fall back in-sample.
        if ytr.sum() < 1 or (len(ytr) - ytr.sum()) < 1:
            return pipe_full.decision_function(X_fit).ravel(), pipe_full
        p = make_pipe()
        p.fit(X_fit[tr], ytr)
        oof[test] = p.decision_function(X_fit[test]).ravel()
    if not np.isfinite(oof).all():
        return pipe_full.decision_function(X_fit).ravel(), pipe_full
    return oof, pipe_full


def _rescore_linear(
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
    cv_folds: int = 3,
    make_clf=None,
    clf_name: str = "lda",
    fitted_out: dict | None = None,
) -> np.ndarray:
    """
    Semi-supervised rescoring on MALDI-intrinsic features with a linear,
    ``decision_function``-based classifier (LDA by default; LinearSVC for
    ``clf_name="svm"`` — see ``_rescore_lda`` / ``_rescore_svm`` wrappers).
    ``make_clf`` is a zero-arg factory returning the final pipeline estimator;
    ``clf_name`` is its pipeline step key and the user-facing log/importance tag.

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

    Scoring is **cross-validated** (``cv_folds`` out-of-fold splits, stratified by
    ``is_decoy``): at each iteration every candidate is scored by a model trained
    on the other folds, so the LDA cannot manufacture target/decoy separation by
    overfitting (which would make the TDC FDR anti-conservative).  Falls back to
    in-sample scoring only when there are too few targets/decoys for CV.  The
    returned importances/structure coefficients come from a model fit on all
    pos/neg rows (reporting only — never used for the FDR scores).

    When ``n_interaction_features > 0`` and R1 importances are supplied, the
    top-k features are expanded with pairwise interaction terms before LDA.

    Returns ``(scores, importances, feature_names_used)``.
    """
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    _tag = clf_name.upper()
    if make_clf is None:
        def make_clf():
            return LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto", priors=[0.5, 0.5])

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
            f"  {_tag} R2 interactions: top-{len(top_names)} features "
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
                f"  {_tag}: best-feature init on '{_best_feat}', "
                f"{_n_init} targets at q≤{init_fdr:.3g}"
            )
        else:
            logger.warning(
                f"  {_tag}: best-feature init yielded 0 targets at q≤{init_fdr:.3g} "
                "— falling back to ppm-based seeding"
            )
            ppm_col = df.get("ppm_error_abs", pd.Series(np.inf, index=df.index))
            n_cand_col = df.get("n_candidates", pd.Series(np.inf, index=df.index))
            init_mask = (
                is_target & ((ppm_col < init_ppm_threshold) | (n_cand_col == 1))
            ).values
            if not init_mask.any():
                logger.warning(f"  {_tag}: no ppm-based seed positives — falling back to top-ppm init")
                init_mask = (
                    is_target & (ppm_col < ppm_col[is_target].quantile(r1_seed_percentile))
                ).values
            labels = _encode_labels(is_decoy, init_mask)
    else:
        seed_arr = seed_mask.values if hasattr(seed_mask, "values") else np.asarray(seed_mask)
        labels = _encode_labels(is_decoy, seed_arr)

    n_seed = int((labels == 1).sum())
    logger.info(f"  {_tag}: seed positives = {n_seed}, decoys = {is_decoy.sum()}")

    def _make_pipe():
        steps = [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
        if use_poly:
            steps.append(
                ("poly", PolynomialFeatures(degree=2, interaction_only=True, include_bias=False))
            )
        steps.append((clf_name, make_clf()))
        return Pipeline(steps)

    # Out-of-fold cross-validation prevents the semi-supervised LDA from
    # manufacturing target/decoy separation by overfitting (each candidate is
    # scored by a model trained on other folds).  Folds are fixed and stratified
    # by is_decoy; falls back to in-sample scoring if there are too few pos/neg.
    fold_ids = _make_fold_ids(is_decoy, cv_folds)
    if fold_ids is None:
        logger.warning(
            f"  {_tag}: too few targets/decoys for {cv_folds}-fold CV — scoring "
            "in-sample (overfitting risk)"
        )
    else:
        logger.info(f"  {_tag}: {cv_folds}-fold cross-validated (out-of-fold) scoring")

    scores = np.zeros(len(df))
    prev_pos_size = -1
    pipe = None
    from threadpoolctl import threadpool_limits

    for iteration in range(max_iter):
        pos_idx = np.where(labels == 1)[0]
        neg_idx = np.where(labels == -1)[0]  # all decoys

        if len(pos_idx) == 0:
            logger.warning(f"  {_tag} iter {iteration + 1}: no positives — stopping early")
            break

        if len(neg_idx) == 0:
            logger.warning(f"  {_tag} iter {iteration + 1}: no negatives (decoys) — cannot train, stopping early")
            break

        with threadpool_limits(limits=1, user_api="blas"):
            scores, pipe = _cv_semisup_scores(X_fit, labels, fold_ids, _make_pipe)

        q_values = _tdc_qvalues(scores, is_decoy)
        new_labels = _encode_labels(is_decoy, q_values <= train_fdr)
        n_new = int((new_labels == 1).sum())

        logger.info(
            f"  {_tag} iter {iteration + 1}: pseudo-positives = {n_new} (prev = {prev_pos_size})"
        )

        if n_new == 0:
            logger.warning(f"  {_tag}: no pseudo-positives at q≤{train_fdr:.3g} — stopping early")
            break

        change = abs(n_new - prev_pos_size) / max(prev_pos_size, 1)
        if prev_pos_size >= 0 and change < 0.01:
            logger.info(f"  {_tag}: converged")
            break

        prev_pos_size = n_new
        labels = new_labels

    if n_init_positives is not None and 0 <= prev_pos_size < n_init_positives:
        logger.warning(
            f"  {_tag}: final iteration positives ({prev_pos_size}) < "
            f"best-feature init ({n_init_positives}). Using model anyway."
        )

    if pipe is not None:
        _log_imputation_debug(
            _tag,
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
            est = pipe[clf_name]
            if hasattr(est, "coef_"):
                importances = est.coef_[0]
            elif hasattr(est, "feature_importances_"):
                # Tree ensembles (e.g. the gbt backend) expose impurity-based
                # feature_importances_ instead of coef_. Same reporting slot; the
                # FDR scores are always the decision_function values, never these.
                importances = est.feature_importances_
            if importances is not None and use_poly:
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
        # Kernel models (e.g. the rbf_svm backend) expose neither coef_ nor
        # feature_importances_ — the decision function lives in kernel space, not
        # per-feature. Fall back to |structure coefficient| so the importance TSV
        # stays populated and sortable. Only valid when not using poly expansion
        # (struct_coefs align with `present`, the un-expanded feature list).
        if importances is None and struct_coefs is not None and not use_poly:
            importances = np.abs(struct_coefs)
    # Expose the fitted pipeline + raw feature matrix for downstream SHAP debug
    # explanations (populated only when the caller passes a mutable dict).
    if fitted_out is not None and pipe is not None:
        fitted_out["pipe"] = pipe
        fitted_out["X"] = X_fit
        fitted_out["feature_names"] = top_names if use_poly else present
    return scores, importances, struct_coefs, struct_names_out, feature_names_out


def _rescore_lda(features_df, intrinsic_feature_names, init_ppm_threshold, **kwargs):
    """Semi-supervised LDA backend (thin wrapper over ``_rescore_linear``)."""
    return _rescore_linear(
        features_df, intrinsic_feature_names, init_ppm_threshold,
        clf_name="lda", make_clf=None, **kwargs,
    )


def _rescore_svm(features_df, intrinsic_feature_names, init_ppm_threshold, svm_c: float = 1.0, **kwargs):
    """Semi-supervised Linear SVM backend.

    Uses ``sklearn.svm.LinearSVC`` (penalty="l2", loss="squared_hinge",
    ``C=svm_c``, dual="auto", max_iter=2000) as the final pipeline step. LinearSVC
    exposes ``decision_function()`` and ``coef_[0]`` exactly like LDA, so the
    out-of-fold CV, pseudo-label iteration, and importance reporting are reused
    unchanged. The median imputer is kept (LinearSVC rejects NaN)."""
    from sklearn.svm import LinearSVC

    def make_clf():
        return LinearSVC(penalty="l2", loss="squared_hinge", C=svm_c, dual="auto", max_iter=2000)

    return _rescore_linear(
        features_df, intrinsic_feature_names, init_ppm_threshold,
        clf_name="svm", make_clf=make_clf, **kwargs,
    )


def _rescore_gbt(
    features_df, intrinsic_feature_names, init_ppm_threshold,
    gbt_n_estimators: int = 200,
    gbt_max_depth: int = 3,
    gbt_learning_rate: float = 0.1,
    gbt_subsample: float = 0.7,
    **kwargs,
):
    """Semi-supervised gradient-boosted-tree backend (nonlinear).

    Uses ``sklearn.ensemble.GradientBoostingClassifier`` as the final pipeline
    step. It exposes ``decision_function()`` (consumed by the out-of-fold CV
    scorer) and ``feature_importances_`` (impurity-based, written to the
    importance TSV), so it slots into the exact same ``_rescore_linear`` machinery
    as LDA/SVM — seed selection, pseudo-label iteration, per-feature winner
    selection, TDC, PEP-from-scores, and reweighting are all unchanged; only the
    final estimator differs. Score columns are tagged ``gbt_score_r1/r2`` and the
    importance TSV ``17_debug_gbt_importances_r{1,2}.tsv``.

    Unlike the linear backends, gbt can fit nonlinear feature interactions (e.g.
    colocalization only discriminating where MALDI signal is present), which is
    where it materially outperforms LDA/SVM on heterogeneous samples.
    ``subsample<1`` (stochastic gradient boosting) regularises against overfitting
    the target/decoy null, complementing the out-of-fold CV. The median imputer
    is kept (GradientBoostingClassifier rejects NaN); the StandardScaler is a
    harmless no-op for trees (monotone transforms do not change split points)."""
    from sklearn.ensemble import GradientBoostingClassifier

    def make_clf():
        return GradientBoostingClassifier(
            n_estimators=gbt_n_estimators,
            max_depth=gbt_max_depth,
            learning_rate=gbt_learning_rate,
            subsample=gbt_subsample,
            random_state=0,
        )

    return _rescore_linear(
        features_df, intrinsic_feature_names, init_ppm_threshold,
        clf_name="gbt", make_clf=make_clf, **kwargs,
    )


def _rescore_rbf_svm(
    features_df, intrinsic_feature_names, init_ppm_threshold,
    rbf_svm_c: float = 1.0,
    rbf_svm_gamma="scale",
    **kwargs,
):
    """Semi-supervised RBF-kernel SVM backend (nonlinear, continuous scores).

    Uses ``sklearn.svm.SVC(kernel="rbf")`` as the final pipeline step. Unlike the
    ``gbt`` (tree) backend, a kernel SVM's ``decision_function`` is a smooth,
    continuous function of the inputs, so the score distribution stays continuous
    and bimodal-looking (no discrete leaf-value spikes) even with a small
    semi-supervised seed set. This is the nonlinear analog of the linear ``svm``
    backend and reuses the exact same ``_rescore_linear`` machinery — seed,
    pseudo-label iteration, out-of-fold CV, winner selection, TDC, PEP,
    reweighting — via ``decision_function``. Score columns are tagged
    ``rbf_svm_score_r1/r2``.

    A kernel SVM has no ``coef_``/``feature_importances_`` (its weights live in
    kernel space), so ``_rescore_linear`` reports per-feature importances as
    ``|structure coefficient|`` (correlation of each feature with the score).

    ``rbf_svm_gamma`` accepts ``"scale"``/``"auto"`` or a float. On standardized
    features, gamma ≈ 0.01-0.03 with ``rbf_svm_c`` ≈ 5-10 typically outperforms
    the ``"scale"`` default; tune via ``--rbf-svm-gamma`` / ``--rbf-svm-c``. The
    median imputer + StandardScaler are kept (SVC rejects NaN and is scale-
    sensitive). Note SVC training is ~O(N^2); fine at MALDI candidate scale
    (~5-10 K rows) but slower than the linear backends."""
    from sklearn.svm import SVC

    gamma = rbf_svm_gamma
    if isinstance(gamma, str) and gamma not in ("scale", "auto"):
        gamma = float(gamma)

    def make_clf():
        return SVC(kernel="rbf", C=rbf_svm_c, gamma=gamma)

    return _rescore_linear(
        features_df, intrinsic_feature_names, init_ppm_threshold,
        clf_name="rbf_svm", make_clf=make_clf, **kwargs,
    )


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
    cv_folds: int = 3,
) -> np.ndarray:
    """
    Semi-supervised QDA on MALDI-intrinsic features.

    Same seed and iteration logic as _rescore_lda (three-valued labels,
    best-feature initialization, ppm fallback), but uses
    QuadraticDiscriminantAnalysis(reg_param=0.5).  Scoring is cross-validated
    (out-of-fold) like _rescore_lda to avoid overfitting the target/decoy
    separation.

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
            labels = _encode_labels(is_decoy, init_mask)
    else:
        seed_arr = seed_mask.values if hasattr(seed_mask, "values") else np.asarray(seed_mask)
        labels = _encode_labels(is_decoy, seed_arr)

    n_seed = int((labels == 1).sum())
    logger.info(f"  QDA: seed positives = {n_seed}, decoys = {is_decoy.sum()}")

    def _make_pipe():
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("qda", QuadraticDiscriminantAnalysis(reg_param=0.5)),
        ])

    fold_ids = _make_fold_ids(is_decoy, cv_folds)
    if fold_ids is None:
        logger.warning(
            f"  QDA: too few targets/decoys for {cv_folds}-fold CV — scoring "
            "in-sample (overfitting risk)"
        )
    else:
        logger.info(f"  QDA: {cv_folds}-fold cross-validated (out-of-fold) scoring")

    scores = np.zeros(len(df))
    prev_pos_size = -1
    pipe = None
    from threadpoolctl import threadpool_limits

    for iteration in range(max_iter):
        pos_idx = np.where(labels == 1)[0]
        neg_idx = np.where(labels == -1)[0]  # all decoys

        if len(pos_idx) == 0:
            logger.warning(f"  QDA iter {iteration + 1}: no positives — stopping early")
            break

        if len(neg_idx) == 0:
            logger.warning(f"  QDA iter {iteration + 1}: no negatives (decoys) — cannot train QDA, stopping early")
            break

        with threadpool_limits(limits=1, user_api="blas"):
            scores, pipe = _cv_semisup_scores(X, labels, fold_ids, _make_pipe)

        q_values = _tdc_qvalues(scores, is_decoy)
        new_labels = _encode_labels(is_decoy, q_values <= train_fdr)
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


def _report_entrapment(result_df: "pd.DataFrame", features_df: "pd.DataFrame", output_dir: str) -> None:
    """Count entrapment pseudo-target survivals and write entrapment_result.tsv."""
    if "source" not in features_df.columns:
        return
    result_df = result_df.copy()
    result_df["source"] = features_df["source"].values
    ent = result_df["source"] == "entrapment_shuffled"
    if not ent.any():
        return
    winner = result_df["is_tdc_winner"].fillna(False).astype(bool)
    q = result_df["q_value"].fillna(np.inf)
    n_sub = int(ent.sum())
    is_decoy = result_df["is_decoy"].astype(bool)
    # Denominator = real targets only (exclude entrapment pseudo-targets and all decoys).
    real_target = (~is_decoy) & (~ent)
    lines = ["\nEntrapment validation results:"]
    lines.append(f"  Entrapment peptides submitted: {n_sub}")
    for fdr in [0.01, 0.05, 0.10]:
        mask = winner & (q <= fdr)
        n_ent = int((ent & mask).sum())
        n_all = int((real_target & mask).sum())
        frac = n_ent / n_all if n_all else 0.0
        lines.append(
            f"  Entrapment IDs at {fdr*100:.0f}% FDR: {n_ent} "
            f"({frac*100:.1f}% of {n_all} total IDs; expected ≤{fdr*100:.0f}%)"
        )
    print("\n".join(lines))
    out = os.path.join(output_dir, "entrapment_result.tsv")
    result_df[ent].to_csv(out, sep="\t", index=False)
    logger.info("entrapment: results written to %s", out)


def rescore(
    fasta_path: str,
    maldi_mzs: np.ndarray,
    mzml_paths: list[str],
    spatial_features: pd.DataFrame | None = None,
    ion_images: np.ndarray | None = None,
    ion_image_mzs: np.ndarray | None = None,
    extra_ion_images: dict | None = None,
    maldi_envelopes: dict | None = None,
    maldi_query_raw: bool = False,
    maldi_d_path: str | None = None,
    raw_query_cache: dict | None = None,
    extraction_ppm: float = 25.0,
    mob_quality_mz_window_ppm: float = 25.0,
    mob_quality_k0_tol: float = 0.02,
    msf_path: str | None = None,
    ppm_tolerance: float = 20.0,
    init_fdr: float = 0.2,
    train_fdr: float = 0.05,
    missed_cleavages: int = 2,
    min_length: int = 7,
    max_length: int = 30,
    model: str = "lda",
    svm_c: float = 1.0,
    gbt_n_estimators: int = 200,
    gbt_max_depth: int = 3,
    gbt_learning_rate: float = 0.1,
    rbf_svm_c: float = 1.0,
    rbf_svm_gamma: str = "scale",
    single_round: bool = False,
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
    use_spatial_ranker_features: bool = False,
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
    entrapment_fasta: str | None = None,
    mz_shift_delta_min: float = 5.0,
    mz_shift_delta_max: float = 20.0,
    mz_shift_snap_tolerance_ppm: float = 50.0,
    max_shuffle_rounds: int = 50,
    target_ratio: float = 1.0,
    features_preset: str = "all",
    features_exclude: list[str] | None = None,
    pseudo_label_max_iter: int = 5,
    pseudo_label_fdr: float = 0.10,
    r1_seed_percentile: float = 0.10,
    r2_seed_percentile: float = 0.20,
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
    mob_protein_coloc: bool = False,
    mob_window_multiplier: float = 2.0,
    coloc_tic_quantile: float = 0.0,
    coloc_measured_pixel_mask: "np.ndarray | None" = None,
    coloc_tic_normalize: bool = False,
    coloc_common_mode: bool = False,
    region_coloc: bool = False,
    region_coloc_k: int = 20,
    within_region_coloc: bool = False,
    drop_zero_signal: bool = False,
    entrapment: bool = False,
    substitution_n_residues: int = 1,
    substitution_seed: int = 42,
    substitution_collision_filter: bool = True,
    substitution_mass_shift_min_da: float | None = None,
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
        (LinearSVC, C=``svm_c``), ``"gbt"`` (GradientBoostingClassifier,
        nonlinear; tuned by ``gbt_n_estimators`` / ``gbt_max_depth`` /
        ``gbt_learning_rate``), or ``"rbf_svm"`` (RBF-kernel SVC, nonlinear with
        continuous scores; tuned by ``rbf_svm_c`` / ``rbf_svm_gamma``). All train
        on ``MALDI_INTRINSIC_FEATURES`` only; LC-MS/MS evidence is applied as an
        additive log-prior after scoring.
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
    use_spatial_ranker_features
        If True, include ``SPATIAL_RANKER_FEATURES`` (feature-level spatial
        quality and protein colocalization) in the ranker feature set.  Only
        valid with ``decoy_method`` in {``"entrapment"``, ``"mz_shift"``}; with
        any shuffle variant it is force-disabled with a warning because those
        decoys lack a consistent spatial anchor.  Deduplicated against
        ``PROTEIN_LEVEL_FEATURES`` when ``use_protein_level_features`` is also set.
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
        shuffle), ``"mz_shift"`` (observation-space m/z-shift decoys: each
        target peptide is shifted by a random delta and snapped to a foreign
        MALDI feature), ``"mz_shuffle"`` (derangement of the peptide→feature
        assignment: each real target peptide is relocated onto another peptide's
        real feature, co-located 1 target + 1 decoy per feature so feature-quality
        features are identical between them and the ranker must discriminate on the
        peptide-specific predicted-vs-observed match such as CCS/isotope),
        ``"balanced_shuffle"`` (iterative shuffle with MALDI-match filtering,
        length-stratified subsample to ~1:1 target:decoy ratio), or
        ``"paired_shuffle"`` (same iterative shuffle pool, but decoys are selected
        to occupy the same MALDI features as targets — feature-paired selection —
        to maximise per-feature target-decoy competition while preserving the same
        global ~1:1 ratio).
    entrapment_fasta
        Path to a foreign-organism FASTA used as the null when
        ``decoy_method="entrapment"``.  Required for that method; ignored
        otherwise.
    maldi_query_raw
        When ``True``, ion images are queried directly from the raw ``.d`` data
        at candidate-derived m/z values instead of from a pre-picked feature
        list.  Inverts the pipeline ordering: candidates are generated first
        (against the digest m/z grid), then ``query_raw_maldi`` extracts ion
        images at ``candidates_df["feature_mz"]``.  Requires ``maldi_d_path``.
    maldi_d_path
        Path to the raw Bruker ``.d`` directory.  Required when
        ``maldi_query_raw=True``.
    extraction_ppm
        Ion image extraction half-window (ppm) used by raw-query mode.
    mz_shift_delta_min
        Minimum absolute mass shift (Da) for ``decoy_method="mz_shift"``.
    mz_shift_delta_max
        Maximum absolute mass shift (Da) for ``decoy_method="mz_shift"``.
    mz_shift_snap_tolerance_ppm
        Maximum ppm distance between the shifted query and the nearest MALDI
        feature for the snap to be accepted (``decoy_method="mz_shift"``).
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
        from msi_picasso.lcms_ids import parse_lcms_ids

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

    # --- True full-digest peptide count per protein (for protein_coverage) ---
    # peptide_db is the complete in-silico tryptic digest (length-filtered),
    # BEFORE m/z matching, so its per-protein unique-peptide count is the true
    # number of theoretically observable peptides. This is the correct
    # denominator for protein_coverage. Computed over target peptides only and
    # keyed by the base accession; decoys (DECOY_/ENTRAPMENT_ namespaces) inherit
    # the count of their source protein, keeping coverage symmetric. (The
    # per-candidate count produced downstream is the *observed* peptide pool, not
    # the full digest, and would make coverage degenerate — see Step 6.)
    _pdb_decoy = (
        peptide_db["is_decoy"].astype(bool)
        if "is_decoy" in peptide_db.columns
        else pd.Series(False, index=peptide_db.index)
    )
    protein_full_tryptic_count = (
        peptide_db.loc[~_pdb_decoy].groupby("protein")["peptide"].nunique().to_dict()
    )

    # --- Pre-generate entrapment DB (before maldi_mzs is fixed for raw-query mode) ---
    # In raw-query mode maldi_mzs is derived from peptide_db mzs; entrapment
    # peptides are not in that grid.  Building _entrapment_db here lets us expand
    # the grid before Step 1c so entrapment mzs are included in the extraction.
    _entrapment_db = None
    if entrapment:
        if lcms_ids is None:
            raise ValueError(
                "--entrapment requires LC-MS/MS IDs (--lcms-peptides or --msf); "
                "entrapment candidates are generated from confirmed sequences."
            )
        from msi_picasso.candidates import generate_entrapment_from_lcms_ids
        _entrapment_db = generate_entrapment_from_lcms_ids(
            lcms_ids,
            matching_ppm=matching_ppm,
            missed_cleavages=missed_cleavages,
            min_length=min_length,
            max_length=max_length,
        )

    # --- Raw-query mode: invert ordering (candidates drive MALDI extraction) ---
    # The candidate digest m/z become the matching grid; the actual ion images
    # are queried from the raw .d AFTER candidate generation (see below).
    if maldi_query_raw:
        if maldi_d_path is None:
            raise ValueError(
                "maldi_query_raw=True requires maldi_d_path (the raw Bruker .d directory)"
            )
        if decoy_method == "mz_shift" and mz_shift_delta_min < 10.0:
            warnings.warn(
                "mz_shift with delta_min < 10 Da in raw-query mode may produce "
                "zero-signal decoy ion images if shifts land in empty m/z space. "
                "Consider increasing --mz-shift-delta-min or validating decoy signal "
                "in the mean spectrum.",
                UserWarning,
                stacklevel=2,
            )
        _target_mzs = np.sort(np.unique(
            peptide_db["mh_mz"].dropna().to_numpy(dtype=np.float64)
        ))
        if _entrapment_db is not None and len(_entrapment_db) > 0:
            _ent_extra = np.sort(np.unique(
                _entrapment_db["mh_mz"].dropna().to_numpy(dtype=np.float64)
            ))
            maldi_mzs = np.sort(np.unique(np.concatenate([_target_mzs, _ent_extra])))
            logger.info(
                "Raw-query mode + entrapment: expanded matching grid from %d to %d "
                "unique m/z (added %d entrapment mzs).",
                len(_target_mzs), len(maldi_mzs), len(_ent_extra),
            )
        else:
            maldi_mzs = _target_mzs
        logger.info(
            "Raw-query mode: using %d unique candidate m/z as the matching grid; "
            "ion images will be extracted from %s after candidate generation.",
            len(maldi_mzs), maldi_d_path,
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
    if decoy_method == "mz_shift":
        # Strip any shuffle decoys that may have been added (e.g. from extra_fasta).
        # generate_mz_shift_candidates() works exclusively with target peptides and
        # generates its own decoys via shifted m/z queries.
        target_db = peptide_db[~peptide_db["is_decoy"].astype(bool)].reset_index(drop=True)
        logger.info(
            "Step 1c: Generating m/z-shift observation-space decoys "
            f"(delta {mz_shift_delta_min}–{mz_shift_delta_max} Da)..."
        )
        candidates = generate_mz_shift_candidates(
            target_db,
            maldi_mzs,
            matching_ppm=matching_ppm,
            delta_min=mz_shift_delta_min,
            delta_max=mz_shift_delta_max,
            snap_tolerance_ppm=mz_shift_snap_tolerance_ppm,
            # Raw-query images any m/z on demand, so place decoys at the exact shifted
            # m/z (no snap) — distinct feature per decoy, avoiding the clustering that
            # collapses decoys onto few grid points and skews the winner T:D ratio.
            snap_to_features=not maldi_query_raw,
            maldi_intensities=_maldi_intensities_arr,
            maldi_intensities_p90=maldi_intensities_p90,
            maldi_intensities_sum=maldi_intensities_sum,
        )
    elif decoy_method == "mz_shuffle":
        # Derangement of the peptide->feature assignment: each target peptide is
        # relocated onto another peptide's real feature (co-located 1 target + 1
        # decoy per feature). Feature-quality features are then identical between a
        # feature's target and decoy, so the ranker must discriminate on the
        # peptide-specific predicted-vs-observed match (CCS, isotope).
        target_db = peptide_db[~peptide_db["is_decoy"].astype(bool)].reset_index(drop=True)
        logger.info("Step 1c: Generating m/z-assignment-shuffle (derangement) decoys...")
        # In raw-query mode with entrapment the grid is expanded; restrict the
        # shuffle destinations to target mzs so decoys don't land on entrapment
        # features (_target_mzs is set in the raw-query block above, else maldi_mzs).
        _shuffle_grid = (
            _target_mzs
            if maldi_query_raw and _entrapment_db is not None and len(_entrapment_db) > 0
            else maldi_mzs
        )
        candidates = generate_mz_shuffle_candidates(
            target_db,
            _shuffle_grid,
            matching_ppm=matching_ppm,
            maldi_intensities=_maldi_intensities_arr,
            maldi_intensities_p90=maldi_intensities_p90,
            maldi_intensities_sum=maldi_intensities_sum,
        )
    elif decoy_method == "entrapment":
        if entrapment_fasta is None:
            raise ValueError(
                "entrapment_fasta is required when decoy_method='entrapment' "
                "(pass --entrapment-fasta)"
            )
        # Targets are matched normally; entrapment decoys come from a foreign FASTA.
        target_db = peptide_db[~peptide_db["is_decoy"].astype(bool)].reset_index(drop=True)
        logger.info(
            "Step 1c: Generating entrapment decoys from %s ...", entrapment_fasta
        )
        target_candidates = match_to_maldi_features(
            maldi_mzs, target_db, matching_ppm,
            maldi_intensities=_maldi_intensities_arr,
            maldi_intensities_p90=maldi_intensities_p90,
            maldi_intensities_sum=maldi_intensities_sum,
        )
        if "source" not in target_candidates.columns:
            target_candidates["source"] = "target"
        decoy_df = load_entrapment_candidates(
            entrapment_fasta,
            target_candidates,
            maldi_mzs,
            matching_ppm=matching_ppm,
            missed_cleavages=missed_cleavages,
            min_length=min_length,
            max_length=max_length,
            maldi_intensities=_maldi_intensities_arr,
            maldi_intensities_p90=maldi_intensities_p90,
            maldi_intensities_sum=maldi_intensities_sum,
        )
        candidates = pd.concat([target_candidates, decoy_df], ignore_index=True)
        candidates["is_decoy"] = candidates["is_decoy"].astype(bool)
        # Recompute per-feature / per-protein occupancy stats over the combined set.
        candidates["n_candidates"] = candidates.groupby("feature_mz")["feature_mz"].transform("count")
        prot_feat_count = candidates.groupby("protein")["feature_mz"].nunique()
        candidates["protein_n_features"] = (
            candidates["protein"].map(prot_feat_count).fillna(0).astype(int)
        )
    elif decoy_method in ("balanced_shuffle", "paired_shuffle"):
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
    elif decoy_method == "substitution":
        target_db = peptide_db[~peptide_db["is_decoy"].astype(bool)].reset_index(drop=True)
        logger.info(
            "Step 1c: Generating substitution decoys "
            "(n_residues=%d, seed=%d, collision_filter=%s)...",
            substitution_n_residues, substitution_seed, substitution_collision_filter,
        )
        candidates = generate_substitution_candidates(
            target_db,
            maldi_mzs,
            matching_ppm=matching_ppm,
            n_residues=substitution_n_residues,
            random_seed=substitution_seed,
            mass_shift_min_da=substitution_mass_shift_min_da,
            collision_filter=substitution_collision_filter,
            snap_to_features=not maldi_query_raw,
            maldi_intensities=_maldi_intensities_arr,
            maldi_intensities_p90=maldi_intensities_p90,
            maldi_intensities_sum=maldi_intensities_sum,
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
    # --- Entrapment: inject shuffled pseudo-target candidates ---
    if _entrapment_db is not None and len(_entrapment_db) > 0:
        from msi_picasso.candidates import match_to_maldi_features as _mtf
        _ent_cands = _mtf(
            maldi_mzs, _entrapment_db, matching_ppm,
            maldi_intensities=_maldi_intensities_arr,
            maldi_intensities_p90=maldi_intensities_p90,
            maldi_intensities_sum=maldi_intensities_sum,
        )
        if len(_ent_cands) > 0:
            candidates = pd.concat([candidates, _ent_cands], ignore_index=True)
            candidates["is_decoy"] = candidates["is_decoy"].astype(bool)
            _fc = "feature_idx" if "feature_idx" in candidates.columns else "feature_mz"
            candidates["n_candidates"] = (
                candidates.groupby(_fc)[_fc].transform("count")
            )
            logger.info(
                "entrapment: added %d shuffled pseudo-target candidates "
                "(%d ENTRAPMENT proteins, %d features).",
                len(_ent_cands), _ent_cands["protein"].nunique(), _ent_cands[_fc].nunique(),
            )
        else:
            logger.warning(
                "entrapment: no shuffled candidates matched any MALDI feature "
                "(matching_ppm=%.1f). Validation will report 0 entrapment IDs.",
                matching_ppm,
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

    # --- Raw-query mode: extract ion images at the candidate-derived m/z ---
    # candidates["feature_mz"] now holds every queried m/z (for mz_shift decoys
    # this is the shifted anchor).  Extract directly from the raw .d, then attach
    # the freshly computed per-feature intensities back onto the candidate rows.
    if maldi_query_raw:
        from msi_picasso.maldi_query import (
            extract_observed_feature_stats_raw,
            query_raw_maldi,
        )

        query_mzs = np.sort(
            candidates["feature_mz"].dropna().to_numpy(dtype=np.float64)
        )
        query_mzs = np.unique(query_mzs)

        # Bidirectional extraction cache. ``raw_query_cache`` lets a caller (e.g. a
        # grid search) extract the candidate-grid ion images / observed centroids /
        # CCS once and reuse them across many rescore() runs that vary only the
        # scoring parameters: the candidate m/z set is fixed by the digest + decoy
        # method (constant across such runs), so the cached full-grid arrays are a
        # superset of any run's query_mzs and are reused as-is — extra ion images
        # are ignored by the feature_mz → image lookups. Semantics:
        #   None                       → always extract (default).
        #   {} (or no "ion_images")    → extract, then populate the dict for reuse.
        #   {"ion_images": ...}         → reuse without touching the .d.
        if raw_query_cache is not None and raw_query_cache.get("ion_images") is not None:
            maldi_mzs = raw_query_cache["maldi_mzs"]
            ion_images = raw_query_cache["ion_images"]
            extra_ion_images = raw_query_cache["extra_ion_images"]
            spatial_features = raw_query_cache["spatial_features"]
            maldi_envelopes = raw_query_cache["maldi_envelopes"]
            _ccs_arr = raw_query_cache["ccs_arr"]
            _centroid_arr = raw_query_cache["centroid_arr"]
            _peak_quality = raw_query_cache.get("peak_quality")
            ion_image_mzs = maldi_mzs
            logger.info(
                "Raw-query mode: reusing cached extraction (%d grid ion images) "
                "for %d candidate m/z.", len(maldi_mzs), len(query_mzs),
            )
        else:
            (
                maldi_mzs,
                ion_images,
                extra_ion_images,
                spatial_features,
                maldi_envelopes,
            ) = query_raw_maldi(maldi_d_path, query_mzs, extraction_ppm=extraction_ppm)
            ion_image_mzs = maldi_mzs
            # Observed peak centroids + CCS from the raw .d (alphatims). imzy exposes
            # neither, so the .d is opened a second time here.
            _ccs_arr, _centroid_arr, _peak_quality = extract_observed_feature_stats_raw(
                maldi_d_path, maldi_mzs, extraction_ppm=extraction_ppm,
                mob_quality_window_ppm=mob_quality_mz_window_ppm,
                mob_quality_k0_tol=mob_quality_k0_tol,
            )
            logger.info(
                "Raw-query mode: extracted %d ion images; %d features with M0 envelope signal.",
                len(maldi_mzs), len(maldi_envelopes),
            )
            if raw_query_cache is not None:
                raw_query_cache.update(
                    maldi_mzs=maldi_mzs, ion_images=ion_images,
                    extra_ion_images=extra_ion_images, spatial_features=spatial_features,
                    maldi_envelopes=maldi_envelopes, ccs_arr=_ccs_arr, centroid_arr=_centroid_arr,
                    peak_quality=_peak_quality,
                )

        # Attach per-feature intensities (mapped by m/z) onto the candidate rows.
        _p90 = dict(zip(spatial_features["feature_mz"], spatial_features["intensity_p90"]))
        _sum = dict(zip(spatial_features["feature_mz"], spatial_features["intensity_sum"]))
        _mean = dict(zip(spatial_features["feature_mz"], spatial_features["mean_intensity"]))
        candidates["feature_intensity_p90"] = candidates["feature_mz"].map(_p90)
        candidates["feature_intensity_sum"] = candidates["feature_mz"].map(_sum)
        candidates["feature_intensity"] = candidates["feature_mz"].map(_mean)

        # Attach intrinsic 2D peak-quality columns (feature-level, mapped by m/z) when
        # ion mobility was extracted.  Aligned with maldi_mzs; bridged via feature_mz
        # exactly like the intensity maps above.  Absent for TSF/no-mobility data.
        if _peak_quality is not None:
            for _col, _vals in _peak_quality.items():
                _qmap = {
                    float(m): float(v)
                    for m, v in zip(np.asarray(maldi_mzs), np.asarray(_vals))
                    if np.isfinite(v)
                }
                candidates[_col] = candidates["feature_mz"].map(_qmap)

        # Recompute ppm_error symmetrically from the observed peak centroid in each
        # candidate's own window. In raw-query, candidates are matched against the
        # theoretical digest grid, so the default (feature_mz - mh_mz) ppm is 0 for
        # every self-match and decoys inherit 0. Replacing it with
        # (observed_centroid - feature_mz)/feature_mz * 1e6 gives a real, symmetric
        # mass-accuracy feature: targets and decoys are measured identically against
        # their own anchor, with no inheritance and no label leak.  Candidates whose
        # window has no observed peak (e.g. mz_shift decoys shifted into empty m/z)
        # get the worst-case ppm (the extraction window edge) rather than NaN, so
        # they are penalised on ppm instead of median-imputed to an average value.
        if np.isfinite(_centroid_arr).any():
            _fmz = candidates["feature_mz"].to_numpy()
            _ppm = _recompute_ppm_from_centroids(
                _fmz, maldi_mzs, _centroid_arr, worst_case_ppm=extraction_ppm
            )
            candidates["ppm_error"] = _ppm
            candidates["ppm_error_abs"] = np.abs(_ppm)
            # Count rows whose own window had a real observed peak (vs worst-case fill).
            _sig_mz = {
                float(m) for m, c in zip(np.asarray(maldi_mzs), _centroid_arr)
                if np.isfinite(c)
            }
            _n_signal = int(sum(float(m) in _sig_mz for m in _fmz))
            logger.info(
                "Raw-query mode: recomputed ppm_error from observed peak centroids "
                "for %d/%d candidate rows; %d empty-window rows set to worst-case "
                "%.1f ppm.",
                _n_signal, len(_ppm), len(_ppm) - _n_signal, extraction_ppm,
            )
        else:
            logger.warning(
                "Raw-query mode: no observed peak centroids available (alphatims "
                "missing or no in-window signal); ppm_error left as matched-grid value."
            )

        # observed_ccs_per_feature unlocks the IM2Deep CCS features, the match_ccs
        # filter, and (with --mob-coloc) mobility-filtered colocalization. Keyed by
        # the candidates' own feature_idx (which indexes the digest grid in raw-query
        # mode, not maldi_mzs), bridged via feature_mz — matching how
        # compute_im2deep_features consumes it (df["feature_idx"].map(...)).
        observed_ccs_per_feature = _observed_ccs_by_feature_idx(
            candidates, maldi_mzs, _ccs_arr
        )
        logger.info(
            "Raw-query mode: observed CCS available for %d features.",
            0 if observed_ccs_per_feature is None else len(observed_ccs_per_feature),
        )

    # --- Select calibration peptides (DeepLC / IM2Deep finetuning anchors) ---
    # theo_isotope_cosine is needed for the quality ranking and is computed in
    # full at Step 6; compute it here (cheap, lru_cache'd, idempotent — Step 6
    # overwrites with identical values) so the calibration set is available before
    # DeepLC finetuning at Step 4.  is_calibration_peptide is carried on the
    # candidates DataFrame and reused by the DeepLC, IM2Deep CCS, mobility, and CCS
    # filter steps instead of the decoy-sensitive n_candidates == 1 heuristic.
    from msi_picasso.maldi_features import compute_theoretical_isotope_features
    candidates = compute_theoretical_isotope_features(candidates, maldi_envelopes=maldi_envelopes)
    candidates["is_calibration_peptide"] = _select_calibration_peptides(
        candidates, calibration_percentile
    )
    logger.info(
        f"  Calibration set: {int(candidates['is_calibration_peptide'].sum())} target "
        f"candidates (top {calibration_percentile:.0%} by low ppm + high isotope cosine)"
    )

    # --- Steps 2–5: LC-MS/MS data (skipped when no mzML or prior weight is 0) ---
    lcms_evidence = {}

    if mzml_paths and lcms_prior_weight > 0:
        # --- Step 2: Load LC-MS/MS data ---
        logger.info("Step 2: Loading LC-MS/MS data...")
        lcms_data = load_lcms_data(mzml_paths)
        if verbose:
            logger.debug(f"Writing LC-MS/MS data to {output_dir}/10_debug_lcms_data.pkl")
            with open(f"{output_dir}/10_debug_lcms_data.pkl", "wb") as f:
                pickle.dump(lcms_data, f)

        # --- Step 3: MS2PIP predictions ---
        logger.info("Step 3: Finding MS2 matches and running MS2PIP...")
        from msi_picasso.lcms_evidence import _find_matching_ms2_scans
        from msi_picasso.utils import mz_to_mass

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
        if not mzml_paths:
            logger.info("Steps 2–5: No mzML files provided — skipping LC-MS/MS evidence.")
        else:
            logger.info("Steps 2–5: lcms_prior_weight=0 — skipping DeepLC / MS2PIP / LC-MS/MS evidence.")

    # --- Override protein_tryptic_count with the true full-digest count ---
    # Replaces the candidate-pool count (= observed peptides, which makes
    # protein_coverage degenerate/leaky) with the true full tryptic digest count
    # per protein. Decoys strip their namespace prefix to inherit the source
    # protein's count, so protein_coverage is symmetric between a protein and its
    # decoy. See compute_protein_consistency_features for the matching numerator.
    if protein_full_tryptic_count:
        _base_prot = (
            candidates["protein"].astype(str)
            .str.replace(r"^DECOY_", "", regex=True)
            .str.replace(r"^ENTRAPMENT_", "", regex=True)
        )
        _mapped = _base_prot.map(protein_full_tryptic_count)
        # Keep any existing (candidate-pool) count only where the protein is not
        # in the digest map (e.g. LC-only novel peptides with no FASTA protein).
        if "protein_tryptic_count" in candidates.columns:
            _mapped = _mapped.fillna(candidates["protein_tryptic_count"])
        candidates["protein_tryptic_count"] = _mapped.fillna(0).astype(int)

    # --- Step 6: Compute all features ---
    logger.info("Step 6: Computing all features...")
    # When region colocalization is on and debug figures are requested, collect the
    # segmentation map + region fingerprints so save_debug_figures can visualize them.
    region_coloc_debug = {} if (region_coloc and debug_dir is not None) else None
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
        coloc_tic_quantile=coloc_tic_quantile,
        coloc_measured_pixel_mask=coloc_measured_pixel_mask,
        coloc_tic_normalize=coloc_tic_normalize,
        coloc_common_mode=coloc_common_mode,
        region_coloc=region_coloc,
        region_coloc_k=region_coloc_k,
        region_coloc_debug=region_coloc_debug,
        within_region_coloc=within_region_coloc,
    )
    # Worst-case fill of protein-colocalization NaNs for zero-signal candidates, so a
    # feature with no MALDI signal is penalised rather than median-imputed to an average
    # coloc value (see _fill_nosignal_coloc_worst_case for the symmetry argument).
    features_df = _fill_nosignal_coloc_worst_case(features_df)
    # --- Optional zero-signal candidate removal (drop_zero_signal) ---
    # Under raw-query mode every candidate gets a genuine extraction attempt; a zero
    # feature_intensity_sum means no signal was detected at that m/z across all pixels.
    # Such candidates carry no MALDI evidence and, under mz_shuffle, their co-located
    # target/decoy pair diverge only on peptide_length (AUC 0.91 in the zero-signal
    # subpopulation), which leaks the mass-sorted derangement into the FDR.  Dropping
    # them is symmetric: the mask is is_decoy-blind (feature_intensity_sum is shared
    # between co-located target+decoy under mz_shuffle, and drawn from the same ion
    # image for all other decoy methods), so target and decoy counts drop in lock-step.
    if drop_zero_signal and "feature_intensity_sum" in features_df.columns:
        _no_signal = ~(features_df["feature_intensity_sum"] > 0)
        n_drop = int(_no_signal.sum())
        if n_drop:
            _n_t = int(_no_signal[~features_df["is_decoy"]].sum())
            _n_d = int(_no_signal[features_df["is_decoy"]].sum())
            features_df = features_df[~_no_signal].reset_index(drop=True)
            logger.info(
                f"  drop_zero_signal: removed {n_drop} zero-signal candidates "
                f"({_n_t} targets + {_n_d} decoys)."
            )
        else:
            logger.info("  drop_zero_signal: no zero-signal candidates found.")
    # Resolve the set of features explicitly excluded from the ranker: the
    # user-supplied features_exclude plus, for mz_shuffle, the raw CCS + mobility-
    # gated colocalization features that leak the m/z baseline (see the ranker
    # feature-pool assembly below for the rationale). Computed here so the
    # 13_debug_features.tsv table reflects exactly the same exclusions the ranker
    # applies, and reused (not recomputed) when assembling the pool.
    _exclude_set = set(features_exclude or [])
    if decoy_method == "mz_shuffle":
        _ccs_mz_leak_feats = set(_MZ_SHUFFLE_CCS_LEAK_FEATURES)
        if _ccs_mz_leak_feats - _exclude_set:
            logger.info(
                "  decoy_method='mz_shuffle': excluding raw CCS + mobility-gated "
                "colocalization features from the ranker (they leak the m/z baseline); "
                "keeping only the m/z-detrended *_resid CCS features."
            )
        _exclude_set |= _ccs_mz_leak_feats
    if _exclude_set:
        logger.info(f"  Excluding {len(_exclude_set)} features: {sorted(_exclude_set)}")

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
            from msi_picasso.maldi_features import compute_mobility_colocalization_features
            logger.info("Computing per-candidate mobility colocalization features (step 6c)…")
            features_df = compute_mobility_colocalization_features(
                features_df,
                tdf_path,
                mob_window_multiplier=mob_window_multiplier,
                extraction_ppm=ppm_tolerance,
                protein_coloc=mob_protein_coloc,
                tic_normalize=coloc_tic_normalize,
            )
            _has_mob_coloc = True
        except Exception as exc:
            logger.warning(f"Per-candidate mobility colocalization failed: {exc}. Skipping.")

    if verbose:
        logger.debug(f"Writing computed features to {output_dir}/13_debug_features.tsv")
        _debug_cols = [c for c in features_df.columns if c not in _exclude_set]
        features_df[_debug_cols].to_csv(f"{output_dir}/13_debug_features.tsv", sep="\t", index=False)

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

    # --- Spatial ranker features (opt-in) — gated on decoy method ---
    # Decoys must land on real MALDI features for spatial features to form a
    # symmetric null.  entrapment and mz_shift decoys do; shuffle variants do not.
    use_spatial_ranker_features = _resolve_spatial_ranker_features(
        use_spatial_ranker_features, decoy_method
    )

    # _exclude_set (features_exclude + the mz_shuffle CCS/mobility leak features) was
    # resolved earlier (and 13_debug_features.tsv written after mob_coloc), so the debug
    # table and the ranker apply identical exclusions. See that block for the mz_shuffle rationale.
    # Assemble the intrinsic feature pool: base + optional protein-level + optional
    # spatial-ranker.  protein_colocalization_* appear in both PROTEIN_LEVEL_FEATURES
    # and SPATIAL_RANKER_FEATURES; the order-preserving dedup below prevents
    # double-inclusion when both flags are active.
    _pool = list(_base_features)
    if use_protein_level_features:
        _pool += PROTEIN_LEVEL_FEATURES
    if use_spatial_ranker_features:
        _pool += SPATIAL_RANKER_FEATURES
        logger.info(
            f"  Spatial ranker features enabled ({len(SPATIAL_RANKER_FEATURES)} features) "
            f"with decoy_method='{decoy_method}'"
        )
    if region_coloc:
        _pool += REGION_COLOCALIZATION_FEATURES
        logger.info(
            f"  Region colocalization features enabled ({len(REGION_COLOCALIZATION_FEATURES)} features)"
        )
    if within_region_coloc:
        _pool += WITHIN_REGION_COLOCALIZATION_FEATURES
        logger.info(
            f"  Within-region colocalization features enabled "
            f"({len(WITHIN_REGION_COLOCALIZATION_FEATURES)} features, experimental — O3)"
        )
    # Intrinsic 2D peak-quality features: default-on for the decoy methods where they
    # are valid/safe (see _MOB_QUALITY_DEFAULT_DECOYS).  Only present when extracted
    # (raw-query + ion mobility); the intrinsic_present intersection below drops them
    # otherwise.  Drop explicitly via features_exclude if unwanted.
    if decoy_method in _MOB_QUALITY_DEFAULT_DECOYS and any(
        f in features_df.columns for f in MOB_QUALITY_FEATURES
    ):
        _pool += MOB_QUALITY_FEATURES
        logger.info(
            f"  2D peak-quality features enabled ({len(MOB_QUALITY_FEATURES)} features) "
            f"with decoy_method='{decoy_method}'"
        )
    _seen: set[str] = set()
    _intrinsic_pool = [
        f for f in _pool
        if f not in _exclude_set and not (f in _seen or _seen.add(f))
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

    if model in ("lda", "svm", "gbt", "rbf_svm"):
        # LDA, SVM, GBT, and RBF-SVM all score via decision_function and share the
        # entire dispatch (seed, pseudo-label iteration, winner selection, TDC, PEP,
        # reweighting). LDA/SVM are linear; GBT is a nonlinear gradient-boosted-tree
        # backend; RBF-SVM is a nonlinear kernel backend with continuous scores.
        # Select the routine and tag the score columns / importance files by name.
        _linear = {"svm": _rescore_svm, "gbt": _rescore_gbt,
                   "rbf_svm": _rescore_rbf_svm}.get(model, _rescore_lda)
        if model == "svm":
            _svm_kwargs = {"svm_c": svm_c}
        elif model == "gbt":
            _svm_kwargs = {
                "gbt_n_estimators": gbt_n_estimators,
                "gbt_max_depth": gbt_max_depth,
                "gbt_learning_rate": gbt_learning_rate,
            }
        elif model == "rbf_svm":
            _svm_kwargs = {"rbf_svm_c": rbf_svm_c, "rbf_svm_gamma": rbf_svm_gamma}
        else:
            _svm_kwargs = {}
        feature_col = "feature_idx" if "feature_idx" in features_df.columns else "feature_mz"

        # Fitted-pipeline capture for SHAP debug explanations (see debug_pfm_explanations).
        _r1_fitted: dict = {}
        _r2_fitted: dict = {}

        # --- Round 1: score all candidates ---
        scores1, _imp_r1_lda, _struct_coefs_r1_lda, _struct_names_r1_lda, _imp_names_lda = _linear(
            features_df,
            intrinsic_present,
            init_ppm_threshold=init_ppm_threshold,
            init_fdr=init_fdr,
            train_fdr=train_fdr,
            max_iter=max_iter,
            r1_seed_percentile=r1_seed_percentile,
            min_seed_positives=min_seed_positives,
            fitted_out=_r1_fitted,
            **_svm_kwargs,
        )
        # Output importances
        if verbose:
            with open(f"{output_dir}/17_debug_{model}_scores_r1.pkl", "wb") as f:
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
                f"{output_dir}/17_debug_{model}_importances_r1.tsv", sep="\t", index=False
            )

        # --- Per-feature winner selection ---
        winner_pos, winners_df = _select_feature_winners(features_df, scores1, feature_col, winner_percentile)
        logger.info(
            f"  Round-1 winner selection: {len(winners_df)} candidates retained "
            f"({int(winners_df['is_decoy'].sum())} decoys)"
        )

        # --- Round 2: retrain on winner subset (skipped when single_round) ---
        if single_round:
            # Single-round mode: use the R1 winner scores directly for TDC. The
            # per-feature winner selection above (the target-vs-decoy competition)
            # and thus the TDC FDR are unchanged — only the final discriminant
            # refit is skipped. Motivated by raw-query, where R1 already trains on
            # a clean ~1:1 target:decoy set, so R2 typically adds little. R2
            # importance/struct outputs reuse the R1 model (reporting only).
            scores2 = scores1[winner_pos]
            lda_imp_r2 = _imp_r1_lda
            _struct_coefs_r2_lda = _struct_coefs_r1_lda
            _struct_names_r2_lda = _struct_names_r1_lda
            lda_imp_names_r2 = _imp_names_lda
            logger.info("  Single-round mode: skipping R2 retrain; TDC on R1 winner scores")
        else:
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
                f"  {model.upper()} R2: seeding from top-{r2_seed_percentile*100:.0f}% R1 target scores "
                f"(score ≥ {score_threshold:.3f}) → {r2_seed_mask.sum()} positives"
            )

            # --- Feature selection for R2: drop below-median importance from R1 ---
            if lda_r2_median_filter and _imp_r1_lda is not None and _imp_names_lda:
                imp_abs = np.abs(_imp_r1_lda)
                imp_median = np.median(imp_abs)
                lda_r2_features = [f for f, a in zip(_imp_names_lda, imp_abs) if a >= imp_median]
                logger.info(
                    f"  {model.upper()} R2: using {len(lda_r2_features)}/{len(_imp_names_lda)} features "
                    f"with |importance| ≥ median ({imp_median:.4f})"
                )
            else:
                lda_r2_features = intrinsic_present

            scores2, lda_imp_r2, _struct_coefs_r2_lda, _struct_names_r2_lda, lda_imp_names_r2 = _linear(
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
                fitted_out=_r2_fitted,
                **_svm_kwargs,
            )
            # Output importances
            if verbose:
                with open(f"{output_dir}/17_debug_{model}_scores_r2.pkl", "wb") as f:
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
                    f"{output_dir}/17_debug_{model}_importances_r2.tsv", sep="\t", index=False
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
                f"{model}_score_r1": scores1,
                f"{model}_score_r2": scores2_full,
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
            from msi_picasso.debug_viz import save_debug_figures
            save_debug_figures(
                features_df, result_df,
                ion_images=ion_images, ion_image_mzs=ion_image_mzs,
                maldi_envelopes=maldi_envelopes,
                feature_names=intrinsic_present, model_name=model,
                importances_r1=_imp_r1_lda,
                importances_r2=None if single_round else lda_imp_r2,
                importance_names=_imp_names_lda,
                importance_names_r2=lda_imp_names_r2 or _imp_names_lda,
                structure_coefs_r1=_struct_coefs_r1_lda,
                structure_names_r1=_struct_names_r1_lda,
                structure_coefs_r2=None if single_round else _struct_coefs_r2_lda,
                structure_names_r2=_struct_names_r2_lda,
                debug_dir=debug_dir, n_subset=n_debug, seed=debug_seed,
                gt_peptides=gt_peptides, storey_pi0_val=_pi0,
                ccs_tol_pct=_ccs_tol_pct, single_round=single_round,
                region_debug=region_coloc_debug,
            )

            # Per-PFM SHAP explanations on the fitted linear model. Use the R2
            # fitted pipeline + winner feature matrix; for single_round (no R2
            # retrain) fall back to the R1 model restricted to winner rows.
            if verbose:
                try:
                    from msi_picasso.debug_viz import debug_pfm_explanations

                    if not single_round and _r2_fitted.get("pipe") is not None:
                        _pfm_pipe = _r2_fitted["pipe"]
                        _pfm_X = _r2_fitted["X"]
                        _pfm_names = _r2_fitted["feature_names"]
                    elif _r1_fitted.get("pipe") is not None:
                        _pfm_pipe = _r1_fitted["pipe"]
                        _pfm_X = _r1_fitted["X"][winner_pos]
                        _pfm_names = _r1_fitted["feature_names"]
                    else:
                        _pfm_pipe = None
                    if _pfm_pipe is not None:
                        debug_pfm_explanations(
                            result_df.iloc[winner_pos].reset_index(drop=True),
                            _pfm_X, _pfm_pipe, _pfm_names,
                            ion_images=ion_images, feature_mzs=ion_image_mzs,
                            spatial_df=spatial_features, output_dir=debug_dir,
                        )
                except Exception as _pfm_exc:
                    logger.warning("debug_pfm_explanations failed: %s", _pfm_exc)

        if entrapment:
            _report_entrapment(result_df, features_df, output_dir)
        return psm_list, result_df, feature_names, features_df

    elif model == "qda":
        feature_col = "feature_idx" if "feature_idx" in features_df.columns else "feature_mz"

        # --- Round 1: score all candidates ---
        scores1, pep_proba1, _imp_r1_qda, _imp_names_qda = _rescore_qda(
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

        # --- Round 2: retrain on winner subset (skipped when single_round) ---
        if single_round:
            # Single-round: TDC on R1 winner scores; reuse R1 posteriors for PEP.
            scores2 = scores1[winner_pos]
            pep_proba2 = pep_proba1[winner_pos]
            qda_imp_r2, qda_imp_names_r2 = _imp_r1_qda, _imp_names_qda
            logger.info("  Single-round mode: skipping R2 retrain; TDC on R1 winner scores")
        else:
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
            from msi_picasso.debug_viz import save_debug_figures
            save_debug_figures(
                features_df, result_df,
                ion_images=ion_images, ion_image_mzs=ion_image_mzs,
                maldi_envelopes=maldi_envelopes,
                feature_names=intrinsic_present, model_name="qda",
                pep_method="kde",
                importances_r1=_imp_r1_qda,
                importances_r2=None if single_round else qda_imp_r2,
                importance_names=_imp_names_qda,
                importance_names_r2=qda_imp_names_r2 or _imp_names_qda,
                debug_dir=debug_dir, n_subset=n_debug, seed=debug_seed,
                gt_peptides=gt_peptides, storey_pi0_val=_pi0,
                ccs_tol_pct=_ccs_tol_pct, single_round=single_round,
                region_debug=region_coloc_debug,
            )

        if entrapment:
            _report_entrapment(result_df, features_df, output_dir)
        return psm_list, result_df, feature_names, features_df

    else:
        raise ValueError(f"Unknown model '{model}'. Choose 'lda', 'qda', 'svm', 'gbt', or 'rbf_svm'.")
