"""
MALDI-side feature computation: mass accuracy, protein consistency, spatial,
isotope envelope, co-localization, ionization properties.

All functions are symmetric — no is_decoy branching in feature computation.
"""

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from msi_picasso.utils import (
    AVERAGINE_C, AVERAGINE_H, AVERAGINE_N, AVERAGINE_O,
    NEUTRON, theoretical_isotope_distribution,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Kyte-Doolittle hydropathy scale
_KD_SCALE = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5,
    "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
    "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}

# CHCA matrix cluster ions [M+H]+ m/z values (A12)
CHCA_CLUSTER_MZS = np.array([
    190.0499,  # [M+H]+
    212.0318,  # [M+Na]+
    228.0058,  # [M+K]+
    379.0925,  # [2M+H]+
    401.0744,  # [2M+Na]+
    568.1351,  # [3M+H]+
    757.1777,  # [4M+H]+
    172.0393,  # [M+H-H2O]+
])

# Adduct mass differences vs [M+H]+ (Da) for adduct colocalization (E2)
_ADDUCT_DELTAS = {
    "na":   21.9819,   # [M+Na]+  — Na replaces H
    "k":    37.9559,   # [M+K]+   — K replaces H
    "chca": 171.0320,  # [M+CHCA+H-H2O]+ CHCA matrix adduct
}


def compute_mass_accuracy_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute ppm_error_abs, ppm_rank, ppm_best_ratio, ppm_error_pct, ppm_error_squared."""
    df["ppm_rank"] = df.groupby("feature_mz")["ppm_error_abs"].rank(method="min")
    best_ppm = df.groupby("feature_mz")["ppm_error_abs"].transform("min")
    df["ppm_best_ratio"] = df["ppm_error_abs"] / best_ppm.clip(lower=1e-6)
    df['log_ppm_best_ratio'] = np.log1p(df['ppm_best_ratio'])
    df["ppm_error_pct"] = df["ppm_error_abs"] / df["feature_mz"].clip(lower=1.0) * 100
    df["ppm_error_squared"] = df["ppm_error_abs"] ** 2
    return df


def compute_candidate_ambiguity_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute n_candidates and log_n_candidates."""
    df["log_n_candidates"] = np.log1p(df["n_candidates"])
    return df


def compute_protein_consistency_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute protein consistency features from the candidate table.

    protein_n_features should already be computed in match_to_maldi_features().
    This adds derived features.
    """
    df["log_protein_n_features"] = np.log1p(df["protein_n_features"])

    if "protein_tryptic_count" in df.columns:
        total_peps = df["protein_tryptic_count"].clip(lower=1)
    else:
        protein_total_peptides = df.groupby("protein")["peptide"].nunique()
        total_peps = df["protein"].map(protein_total_peptides).clip(lower=1)
    # protein_coverage = fraction of the protein's tryptic peptides that are
    # observed. The numerator is the count of distinct *observed peptides*, NOT
    # distinct features: a target peptide can match several near-isobaric MALDI
    # features (inflating a feature count), whereas every decoy peptide is placed
    # on exactly one feature by construction (mz_shift/mz_shuffle). Counting
    # features made coverage a perfect target/decoy separator (decoys pinned to
    # n_features == tryptic_count → coverage 1.0). Counting peptides is the
    # correct sequence-coverage definition and is symmetric: a protein and its
    # DECOY_/ENTRAPMENT_ namespace share the same peptide set, so they get equal
    # coverage. The denominator is the true full-digest count (set in pipeline.py).
    protein_n_peptides = df.groupby("protein")["peptide"].nunique()
    df["protein_n_peptides"] = df["protein"].map(protein_n_peptides).fillna(0).astype(int)
    df["protein_coverage"] = (df["protein_n_peptides"] / total_peps).clip(upper=1.0)

    # ``is_single_peptide_protein``: 1.0 when the candidate's protein has exactly one
    # observed peptide, 0.0 otherwise.  A structural indicator (independent of MALDI
    # signal / image validity, unlike ``has_coloc``): single-peptide proteins have no
    # within-protein colocalization partner, so all protein-colocalization features are
    # undefined for them and get median-imputed by the ranker — an average coloc value
    # they did not earn, while ``log_protein_n_features`` simultaneously buries them.
    # Exposing the indicator lets a (nonlinear) ranker apply a separate offset to this
    # group and lean on the peptide's own intrinsic evidence (ppm / CCS / isotope)
    # instead of the imputed coloc.  Symmetric under all decoy methods (a protein and
    # its DECOY_/ENTRAPMENT_ namespace share the same peptide set), so it cannot leak.
    df["is_single_peptide_protein"] = (df["protein_n_peptides"] == 1).astype(np.float32)

    df["protein_rank"] = df.groupby("feature_mz")["protein_n_features"].rank(
        ascending=False, method="min"
    )
    best_prot = df.groupby("feature_mz")["protein_n_features"].transform("max")
    df["protein_best_ratio"] = df["protein_n_features"] / best_prot.clip(lower=1)

    return df


def compute_peptide_properties(df: pd.DataFrame) -> pd.DataFrame:
    """Compute peptide_length and n_missed_cleavages."""
    df["peptide_length"] = df["peptide"].str.len()
    try:
        from ms1rescore_rs import count_missed_cleavages_batch
        df["n_missed_cleavages"] = count_missed_cleavages_batch(df["peptide"].tolist())
    except ImportError:
        # K/R not followed by P, excluding the C-terminal cleavage site.
        df["n_missed_cleavages"] = df["peptide"].str[:-1].str.count(r"[KR](?!P)")
    return df


def compute_maldi_signal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute log-intensity features from feature_intensity columns."""
    if "feature_intensity_p90" in df.columns:
        df["log_maldi_intensity_p90"] = np.log1p(df["feature_intensity_p90"])
    elif "feature_intensity" in df.columns:
        df["log_maldi_intensity_p90"] = np.log1p(df["feature_intensity"])
    else:
        df["log_maldi_intensity_p90"] = 0.0

    if "feature_intensity_sum" in df.columns:
        df["log_maldi_intensity_sum"] = np.log1p(df["feature_intensity_sum"])
    else:
        df["log_maldi_intensity_sum"] = 0.0

    return df


def _residue_counts_batch(unique: np.ndarray, residues: str) -> dict[str, np.ndarray]:
    """
    Count occurrences of each residue in ``residues`` for all unique sequences.

    Encodes all sequences into a single concatenated byte array once, then
    uses ``np.add.reduceat`` on per-residue indicator arrays — one numpy call
    per residue instead of ``len(unique)`` Python ``str.count`` calls.

    Returns a dict mapping each character in ``residues`` to a per-sequence
    int32 count array.
    """
    seq_lens = np.array([len(s) for s in unique], dtype=np.int64)
    concat_bytes = np.frombuffer("".join(unique).encode("ascii"), dtype=np.uint8)
    split_pts = np.concatenate([[0], np.cumsum(seq_lens[:-1])]).astype(np.intp)

    counts = {}
    for aa in residues:
        indicator = (concat_bytes == ord(aa)).view(np.uint8)
        counts[aa] = np.add.reduceat(indicator, split_pts).astype(np.int32)
    return counts


def compute_maldi_ionization_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute MALDI-specific ionization features from peptide sequence.

    Uses the Rust extension (rayon parallel) when available; falls back to
    pd.factorize + np.add.reduceat otherwise.
    """
    codes, uniques = pd.factorize(df["peptide"])
    uniques_list = uniques.tolist()

    try:
        from ms1rescore_rs import compute_ionization_features
        n_R, n_K, n_H, n_F, n_W, n_Y, gravy_u, _ = compute_ionization_features(uniques_list)
        n_R = np.asarray(n_R); n_K = np.asarray(n_K); n_H = np.asarray(n_H)
        n_F = np.asarray(n_F); n_W = np.asarray(n_W); n_Y = np.asarray(n_Y)
        gravy_u = np.asarray(gravy_u)
    except ImportError:
        cnt = _residue_counts_batch(uniques, "RKHFWY")
        n_R = cnt["R"]; n_K = cnt["K"]; n_H = cnt["H"]
        n_F = cnt["F"]; n_W = cnt["W"]; n_Y = cnt["Y"]
        kd_arr = np.zeros(256, dtype=np.float64)
        for aa, val in _KD_SCALE.items():
            kd_arr[ord(aa)] = val
        seq_lens = np.array([len(s) for s in uniques], dtype=np.int64)
        concat_bytes = np.frombuffer("".join(uniques).encode("ascii"), dtype=np.uint8)
        split_pts = np.concatenate([[0], np.cumsum(seq_lens[:-1])]).astype(np.intp)
        gravy_u = np.add.reduceat(kd_arr[concat_bytes], split_pts) / np.maximum(seq_lens, 1)

    n_basic_u = n_R.astype(np.float64) + n_K + n_H
    acidic_cnt = _residue_counts_batch(uniques, "DE")
    n_D_u = acidic_cnt["D"].astype(np.float64)
    n_E_u = acidic_cnt["E"].astype(np.float64)

    df["n_arginine"]       = n_R[codes].astype(np.float32)
    df["n_basic_residues"] = n_basic_u[codes].astype(np.float32)
    df["n_aromatic"]       = (n_F + n_W + n_Y)[codes].astype(np.float32)
    df["gravy_score"]      = gravy_u[codes]
    # Net basic minus acidic count: signed charge proxy at near-neutral pH.
    df["charge_proxy"]     = (n_basic_u - (n_D_u + n_E_u))[codes].astype(np.float32)

    logger.info("Computed MALDI ionization features (basic residue counts, GRAVY, charge proxy)")
    return df


def compute_spatial_features(
    df: pd.DataFrame,
    spatial_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge pre-computed spatial features (per MALDI feature) into candidate table.

    Expected columns in spatial_df: feature_mz, spatial_autocorrelation,
    fraction_detected, intensity_cv, log_mean_intensity, spatial_entropy.
    """
    spatial_cols = [
        "spatial_autocorrelation",
        "fraction_detected",
        "intensity_cv",
        "log_mean_intensity",
        "spatial_entropy",
    ]
    available = [c for c in spatial_cols if c in spatial_df.columns]
    if not available:
        for c in spatial_cols:
            df[c] = 0.0
        return df

    merge_cols = ["feature_mz"] + available
    spatial_subset = spatial_df[merge_cols].drop_duplicates(subset=["feature_mz"])

    df = df.merge(spatial_subset, on="feature_mz", how="left")
    for c in spatial_cols:
        if c not in df.columns:
            df[c] = 0.0
        df[c] = df[c].fillna(0.0)

    return df


def compute_tissue_mask(
    ion_images: np.ndarray,
    tic_quantile: float = 0.0,
) -> np.ndarray:
    """
    Build an on-tissue pixel mask from a total-ion-current (TIC) proxy.

    Every MALDI ion image is ~0 in the unmeasured padding that surrounds the
    acquired pixel grid and follows the tissue footprint within it.  A raw
    Pearson r between two ion images is therefore dominated by this shared
    on/off-tissue structure, inflating the correlation between *any* pair of
    images (real or decoy) toward the tissue outline rather than measuring
    co-distribution within the tissue.  Restricting the correlation to
    on-tissue pixels removes that common component.

    The TIC proxy is the per-pixel sum over all supplied ion images.  Pixels
    with TIC == 0 are unmeasured padding and are always excluded.  When
    ``tic_quantile > 0`` the threshold is raised to that quantile of the
    measured-pixel TIC, additionally trimming low-signal tissue edges.

    Parameters
    ----------
    ion_images : (n_feat, H, W) array
    tic_quantile : float in [0, 1)
        0.0 keeps every measured pixel (drops only structural padding).
        e.g. 0.25 keeps pixels above the 25th percentile of measured TIC.

    Returns
    -------
    (H*W,) boolean mask over flattened pixels (True = on-tissue / keep).
    """
    n_feat = ion_images.shape[0]
    tic = ion_images.reshape(n_feat, -1).sum(axis=0)
    measured = tic > 0
    if tic_quantile and tic_quantile > 0.0 and measured.any():
        thr = float(np.quantile(tic[measured], tic_quantile))
        return tic >= thr  # >= keeps pixels at the threshold; > can leave an empty mask when many pixels tie
    return measured


def _pearson_r_matrix(
    ion_images: np.ndarray,
    ion_image_mzs: np.ndarray,
    pixel_mask: np.ndarray | None = None,
    tic_normalize: bool = False,
    common_mode_removal: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Compute the full (n_valid × n_valid) Pearson correlation matrix.

    Uses a manual float32 normalise + BLAS sgemm instead of ``np.corrcoef``
    which unconditionally upcasts to float64, allocating ~2× as much memory.
    For 1398 features × 49 K pixels the saving is ~550 MB at peak.

    When ``pixel_mask`` (a flattened (H*W,) boolean) is supplied the correlation
    is computed over the selected (on-tissue) pixels only; see
    ``compute_tissue_mask``.  This is the recommended MALDI default because raw
    images share a dominant on/off-tissue component that inflates every pairwise
    r.  Image validity (non-constant) is then assessed on the masked pixels too.

    ``tic_normalize`` and ``common_mode_removal`` are optional pre-correlation
    transforms that strip the shared tissue/ionization envelope which otherwise
    dominates the per-image Pearson centring and inflates every pairwise r toward
    a common value (see the per-pixel masking note above — masking removes only
    the off-tissue component, not the on-tissue abundance gradient shared across
    images).  Both operate on the (n_feat, n_pix) matrix *before* the usual
    per-image centring:
      * ``tic_normalize``: divide each pixel (column) by its total signal across
        images, removing the multiplicative "more tissue = more of everything"
        brightness envelope.
      * ``common_mode_removal``: subtract the per-pixel mean image (the additive
        shared spatial component) so only the protein-specific residual remains.

    Returns
    -------
    corr_matrix : (n_valid, n_valid) float32
    valid_mz_arr : (n_valid,) float64 — m/z values of non-constant images
    mz_to_idx : dict mapping float(mz) → row/col index into corr_matrix
    """
    mz_arr = np.asarray(ion_image_mzs, dtype=np.float64)
    n_feat = len(mz_arr)
    n_pix = ion_images.shape[1] * ion_images.shape[2]

    # Reshape to (n_feat, n_pix); view if already C-contiguous, else copy.
    flat_all = ion_images.reshape(n_feat, n_pix)
    if pixel_mask is not None:
        # Restrict to on-tissue pixels (copy: fancy-index breaks the view).
        flat_all = flat_all[:, np.asarray(pixel_mask, dtype=bool)]

    if tic_normalize or common_mode_removal:
        # Own a writeable float32 buffer (the no-mask path above is a view onto
        # ion_images; the in-place transforms below must not mutate the caller's
        # array).
        flat_all = flat_all.astype(np.float32, copy=True)
        if tic_normalize:
            col_tic = flat_all.sum(axis=0, keepdims=True)
            np.divide(flat_all, col_tic, out=flat_all, where=col_tic > 0)
        if common_mode_removal:
            flat_all -= flat_all.mean(axis=0, keepdims=True)

    stds = flat_all.std(axis=1)
    valid_mask = stds > 1e-10
    valid_mz_arr = mz_arr[valid_mask]

    # float32 copy of valid rows only — explicitly freed below.
    X = flat_all[valid_mask].astype(np.float32, copy=True)
    del flat_all  # release reshape view

    # Centre and L2-normalise in-place (no extra allocations).
    X -= X.mean(axis=1, keepdims=True)
    norms = np.sqrt((X * X).sum(axis=1, keepdims=True))
    X /= np.where(norms > 1e-10, norms, 1.0)
    del norms

    # BLAS sgemm: (n_valid, n_pix) @ (n_pix, n_valid) → (n_valid, n_valid) float32
    corr_matrix = X @ X.T
    del X  # free ~(n_valid × n_pix × 4) bytes immediately

    mz_to_idx = {float(mz): i for i, mz in enumerate(valid_mz_arr)}
    return corr_matrix, valid_mz_arr, mz_to_idx


def _find_partner_indices(
    source_mzs: np.ndarray,
    targets: np.ndarray,
    ppm_tol: float,
) -> np.ndarray:
    """
    Vectorized nearest-neighbour lookup on a sorted m/z array.

    Returns an integer array of the same length as *targets*.  Each entry is
    the index into *source_mzs* of the closest value within *ppm_tol* ppm,
    or -1 if no such value exists.
    """
    idx = np.searchsorted(source_mzs, targets)
    n = len(source_mzs)
    idx_lo = np.clip(idx - 1, 0, n - 1)
    idx_hi = np.clip(idx,     0, n - 1)
    ppm_lo = np.abs(source_mzs[idx_lo] - targets) / targets * 1e6
    ppm_hi = np.abs(source_mzs[idx_hi] - targets) / targets * 1e6
    best_idx = np.where(ppm_lo <= ppm_hi, idx_lo, idx_hi)
    within = np.minimum(ppm_lo, ppm_hi) <= ppm_tol
    return np.where(within, best_idx, -1).astype(np.intp)


_COLOC_FEATURE_COLS = [
    "protein_colocalization",
    "protein_colocalization_max",
    "protein_colocalization_median",
    "protein_colocalization_n_partners",
    "protein_colocalization_weighted",
    "protein_colocalization_weighted_max",
    "protein_colocalization_top2",
    "protein_colocalization_top3",
    "protein_colocalization_top5",
]


def compute_colocalization_features(
    df: pd.DataFrame,
    ion_images: np.ndarray,
    ion_image_mzs: np.ndarray,
    _corr_cache: tuple | None = None,
) -> pd.DataFrame:
    """Compute protein co-localization features from MALDI ion images.

    Builds the full (n_valid_features × n_valid_features) Pearson correlation
    matrix in a single BLAS dgemm call, then uses a pandas self-join on protein
    to enumerate all within-protein feature pairs and aggregates with groupby —
    no Python loop over features or candidates.

    Pass ``_corr_cache`` (the return value of ``_pearson_r_matrix``) to reuse a
    correlation matrix already computed for the same ion images, avoiding 3×
    redundant BLAS calls when all three colocalization functions are called
    together (see ``feature_generator.compute_all_features``).
    """
    if _corr_cache is not None:
        corr_matrix, valid_mz_arr, mz_to_idx = _corr_cache
    else:
        corr_matrix, valid_mz_arr, mz_to_idx = _pearson_r_matrix(ion_images, ion_image_mzs)
    valid_mzs = set(mz_to_idx.keys())

    # Unique (feature_mz, protein) pairs that have a valid ion image
    base = (
        df[["feature_mz", "protein"]]
        .drop_duplicates()
        .assign(corr_idx=lambda d: d["feature_mz"].map(lambda m: mz_to_idx.get(float(m))))
    )
    base = base[base["corr_idx"].notna()].copy()
    base["corr_idx"] = base["corr_idx"].astype(int)

    # Self-join on protein to get all within-protein feature-feature pairs
    pairs = base.merge(
        base.rename(columns={"feature_mz": "partner_mz", "corr_idx": "partner_idx"}),
        on="protein",
    )
    pairs = pairs[pairs["feature_mz"] != pairs["partner_mz"]]

    if len(pairs) > 0:
        # Vectorized correlation lookup
        pairs = pairs.copy()
        pairs["r"] = corr_matrix[
            pairs["corr_idx"].to_numpy(dtype=int),
            pairs["partner_idx"].to_numpy(dtype=int),
        ]

        # Per-pair intensity weight w = sqrt(I_a * I_b), using the LINEAR p90
        # intensity (not the log) of each feature. Higher-abundance partner pairs
        # carry more weight. Falls back to feature_intensity, then to uniform
        # weights (w=1, so weighted == unweighted mean) when no intensity column
        # is present — blind to is_decoy throughout.
        _int_col = next(
            (c for c in ("feature_intensity_p90", "feature_intensity") if c in df.columns),
            None,
        )
        if _int_col is not None:
            _mz_int = df.drop_duplicates("feature_mz").set_index("feature_mz")[_int_col]
            _Ia = pairs["feature_mz"].map(_mz_int).to_numpy(dtype=float)
            _Ib = pairs["partner_mz"].map(_mz_int).to_numpy(dtype=float)
            _Ia = np.where(np.isfinite(_Ia) & (_Ia > 0), _Ia, 0.0)
            _Ib = np.where(np.isfinite(_Ib) & (_Ib > 0), _Ib, 0.0)
            pairs["w"] = np.sqrt(_Ia * _Ib)
        else:
            pairs["w"] = 1.0
        pairs["wr"] = pairs["w"] * pairs["r"]

        g = pairs.groupby(["feature_mz", "protein"])
        agg = g["r"].agg(
            protein_colocalization="mean",
            protein_colocalization_max="max",
            protein_colocalization_median="median",
            protein_colocalization_n_partners="count",
        )
        # Intensity-weighted mean r (Σ w·r / Σ w) and max of w·r.
        _wsum = g["w"].sum()
        agg["protein_colocalization_weighted"] = g["wr"].sum() / _wsum.where(_wsum > 0, np.nan)
        agg["protein_colocalization_weighted_max"] = g["wr"].max()
        # Rank-weighted: mean r over the top-k highest-weight partner pairs.
        # head(k) on the weight-sorted frame returns all rows when < k exist.
        _pairs_by_w = pairs.sort_values("w", ascending=False)
        for k in (2, 3, 5):
            topk = (
                _pairs_by_w.groupby(["feature_mz", "protein"]).head(k)
                .groupby(["feature_mz", "protein"])["r"].mean()
            )
            agg[f"protein_colocalization_top{k}"] = topk
        agg = agg.reset_index()
        df = df.merge(agg, on=["feature_mz", "protein"], how="left")
    else:
        for col in _COLOC_FEATURE_COLS:
            df[col] = np.nan

    # ``has_coloc``: 1.0 where within-protein colocalization is *defined* (the
    # candidate has >=1 same-protein partner with a valid ion image), 0.0
    # otherwise.  Computed before any imputation so the ranker can distinguish
    # "not colocalizable" (a singleton / small protein) from "colocalizes
    # poorly".  Blind to ``is_decoy``.
    df["has_coloc"] = df["protein_colocalization"].notna().astype(np.float32)

    # ``protein_colocalization_n_partners`` is a genuine count: 0 (not
    # undefined) when a candidate has no partners.  Keep it as an honest 0.
    df["protein_colocalization_n_partners"] = (
        df["protein_colocalization_n_partners"].fillna(0.0)
    )

    # The correlation summaries (mean / median / max / weighted / top-k) are
    # *undefined* without partners.  Leave them as NaN so the ranker's median
    # imputer fills them at the train-set median (the colocalizable plateau),
    # rather than 0.0 which would impose a protein-size-correlated floor that
    # lifts decoys of abundant proteins into the high-score mode (see
    # notebooks/check_good_decoys analysis).  ``has_coloc`` carries the
    # "was it measurable" bit separately.

    n_scored = int(df["protein_colocalization"].notna().sum())
    logger.info(
        f"Co-localization: {len(valid_mzs)} valid features, "
        f"{len(pairs) if len(pairs) > 0 else 0} within-protein pairs, "
        f"{n_scored}/{len(df)} candidates scored"
    )
    return df


_REGION_COLOC_COLS = [
    "protein_region_colocalization",
    "protein_region_colocalization_max",
    "protein_region_colocalization_median",
]

# O3 : within-region and dominant-region Pearson-r
# colocalization column families — see compute_within_region_colocalization_features.
_WITHIN_REGION_COLOC_COLS = [
    "protein_within_region_colocalization",
    "protein_within_region_colocalization_max",
    "protein_within_region_colocalization_median",
]
_DOMINANT_REGION_COLOC_COLS = [
    "protein_dominant_region_colocalization",
    "protein_dominant_region_colocalization_max",
    "protein_dominant_region_colocalization_median",
]


def _aggregate_protein_pairwise_corr(
    df: pd.DataFrame,
    corr_matrix: np.ndarray,
    valid_mz_arr: np.ndarray,
    mz_to_idx: dict,
    col_prefix: str,
    fill_value: float = 0.0,
) -> pd.DataFrame:
    """Within-protein self-join + mean/max/median aggregation of a precomputed
    pairwise correlation matrix.

    Shared by ``compute_region_colocalization_features`` and the within-region /
    dominant-region variants (``compute_within_region_colocalization_features``)
    — only the correlation matrix and the resulting column names differ.
    Produces ``col_prefix``, ``{col_prefix}_max``, ``{col_prefix}_median``,
    filled with ``fill_value`` where undefined (no same-protein partner with a
    valid image, or no pair cleared whatever floor produced ``corr_matrix``).
    Blind to ``is_decoy``.
    """
    cols = [col_prefix, f"{col_prefix}_max", f"{col_prefix}_median"]
    base = (
        df[["feature_mz", "protein"]]
        .drop_duplicates()
        .assign(corr_idx=lambda d: d["feature_mz"].map(lambda m: mz_to_idx.get(float(m))))
    )
    base = base[base["corr_idx"].notna()].copy()
    base["corr_idx"] = base["corr_idx"].astype(int)

    pairs = base.merge(
        base.rename(columns={"feature_mz": "partner_mz", "corr_idx": "partner_idx"}),
        on="protein",
    )
    pairs = pairs[pairs["feature_mz"] != pairs["partner_mz"]]

    if len(pairs) > 0:
        pairs = pairs.copy()
        pairs["c"] = corr_matrix[
            pairs["corr_idx"].to_numpy(dtype=int),
            pairs["partner_idx"].to_numpy(dtype=int),
        ]
        agg = (
            pairs.groupby(["feature_mz", "protein"])["c"]
            .agg(["mean", "max", "median"])
            .rename(columns={"mean": cols[0], "max": cols[1], "median": cols[2]})
            .reset_index()
        )
        df = df.merge(agg, on=["feature_mz", "protein"], how="left")
    else:
        for col in cols:
            df[col] = fill_value

    for col in cols:
        df[col] = df[col].fillna(fill_value)
    return df


def _region_profile_corr_matrix(
    ion_images: np.ndarray,
    ion_image_mzs: np.ndarray,
    pixel_mask: np.ndarray | None = None,
    n_regions: int = 20,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, dict, np.ndarray, np.ndarray]:
    """
    Segment the tissue into ``n_regions`` regions and return the full Pearson
    correlation matrix of the per-image per-region composition profiles.

    Rationale: a raw Pearson r over all on-tissue pixels is dominated by the
    shared tissue/ionization envelope every image carries, so it measures "is
    this on tissue" rather than protein-specific co-distribution.  This instead
    asks "do two peptides light up the *same regions*?":

      1. Per-pixel TIC-normalise the on-tissue pixels (divide out the shared
         abundance envelope), ``log1p``, and segment the pixels with
         MiniBatchKMeans into ``n_regions`` regions (pixels = samples, images =
         features).
      2. Reduce each image to its per-region mean of the *per-image*
         TIC-normalised image — an ``n_regions``-vector "region fingerprint".
      3. Correlate the fingerprints with signed Pearson r (centre + L2-normalise
         + gemm).  Unlike a cosine of non-negative loadings, this can go negative
         when two peptides occupy *different* regions, which is exactly the
         discriminative signal a decoy at a foreign m/z should produce.

    Returns ``(corr_matrix, valid_mz_arr, mz_to_idx, region_labels, profiles)``.
    The first three match ``_pearson_r_matrix``'s contract so the within-protein
    aggregation is shared.  ``region_labels`` is an ``(H*W,)`` int32 array of the
    per-pixel region id (off-mask pixels = -1) and ``profiles`` is the
    ``(n_valid, n_regions)`` fingerprint matrix; both are returned only for the
    debug visualization.  Only images with any on-tissue signal are valid.
    """
    from sklearn.cluster import MiniBatchKMeans

    mz_arr = np.asarray(ion_image_mzs, dtype=np.float64)
    n_feat = len(mz_arr)
    H, W = ion_images.shape[1], ion_images.shape[2]
    flat = ion_images.reshape(n_feat, H * W)
    mask = (
        np.asarray(pixel_mask, dtype=bool) if pixel_mask is not None
        else np.ones(H * W, dtype=bool)
    )
    on = flat[:, mask].astype(np.float32, copy=True)  # (n_feat, n_on)
    n_on = on.shape[1]

    # Segmentation feature matrix: per-pixel TIC-normalised composition, log1p.
    col_tic = on.sum(axis=0, keepdims=True)
    comp = np.divide(on, col_tic, out=np.zeros_like(on), where=col_tic > 0)
    seg_feat = np.log1p(comp).T  # (n_on, n_feat) — pixels are samples

    if n_on < 2:
        logger.warning("Region colocalization: %d on-tissue pixels — skipping (need ≥ 2).", n_on)
        empty_mz = np.empty(0, dtype=np.float64)
        region_labels = np.full(H * W, -1, dtype=np.int32)
        return (
            np.empty((0, 0), dtype=np.float32),
            empty_mz,
            {},
            region_labels,
            np.empty((0, 0), dtype=np.float32),
        )

    k = int(min(n_regions, max(2, n_on - 1)))
    km = MiniBatchKMeans(n_clusters=k, random_state=random_state, n_init=3)
    labels = km.fit_predict(seg_feat)  # (n_on,)

    # Per-image, per-region mean of the per-IMAGE TIC-normalised image.
    img_sum = on.sum(axis=1, keepdims=True)
    img_norm = np.divide(on, img_sum, out=np.zeros_like(on), where=img_sum > 0)
    onehot = np.zeros((n_on, k), dtype=np.float32)
    onehot[np.arange(n_on), labels] = 1.0
    counts = onehot.sum(axis=0) + 1e-9
    profiles = (img_norm @ onehot) / counts  # (n_feat, k) region fingerprints

    # Valid = images with any on-tissue signal.
    valid_mask = img_sum.ravel() > 0
    valid_mz_arr = mz_arr[valid_mask]
    P = profiles[valid_mask]  # (n_valid, k)

    # Signed Pearson r between region fingerprints.
    Pc = P - P.mean(axis=1, keepdims=True)
    norms = np.sqrt((Pc * Pc).sum(axis=1, keepdims=True))
    Pn = Pc / np.where(norms > 1e-12, norms, 1.0)
    corr_matrix = (Pn @ Pn.T).astype(np.float32)

    mz_to_idx = {float(mz): i for i, mz in enumerate(valid_mz_arr)}
    region_labels = np.full(H * W, -1, dtype=np.int32)
    region_labels[mask] = labels
    logger.info(
        f"Region colocalization: K={k}, {len(valid_mz_arr)}/{n_feat} images with "
        f"on-tissue signal, {n_on} on-tissue pixels"
    )
    return corr_matrix, valid_mz_arr, mz_to_idx, region_labels, P


def compute_region_colocalization_features(
    df: pd.DataFrame,
    ion_images: np.ndarray,
    ion_image_mzs: np.ndarray,
    pixel_mask: np.ndarray | None = None,
    n_regions: int = 20,
    random_state: int = 42,
    debug: dict | None = None,
) -> pd.DataFrame:
    """Within-protein region-profile colocalization features (opt-in, ``--region-coloc``).

    Mirrors ``compute_colocalization_features`` but the pairwise quantity is the
    Pearson r of the per-region composition fingerprints (see
    ``_region_profile_corr_matrix``) instead of the raw pixel Pearson r.  This
    asks whether same-protein peptides occupy the *same tissue regions*, a sharper
    question than global ion-image correlation (which is dominated by overall
    tissue morphology).  Decoys occupy a separate protein namespace (every decoy
    method gives them a ``DECOY_`` / ``ENTRAPMENT_`` label), so the within-protein
    aggregation never pools a decoy with its source target — the feature stays
    symmetric (blind to ``is_decoy``).

    Features added: ``protein_region_colocalization`` (mean), ``_max``, ``_median``
    — the within-protein mean/max/median pairwise region-profile r.

    When ``debug`` is a dict it is populated with ``region_labels`` (per-pixel
    region map), ``region_profiles`` ((n_valid, n_regions) fingerprints) and
    ``region_profile_mzs`` for the debug visualization.
    """
    corr_matrix, valid_mz_arr, mz_to_idx, region_labels, profiles = _region_profile_corr_matrix(
        ion_images, ion_image_mzs, pixel_mask=pixel_mask,
        n_regions=n_regions, random_state=random_state,
    )
    if debug is not None:
        debug["region_labels"] = region_labels
        debug["region_profiles"] = profiles
        debug["region_profile_mzs"] = valid_mz_arr

    return _aggregate_protein_pairwise_corr(
        df, corr_matrix, valid_mz_arr, mz_to_idx, "protein_region_colocalization",
    )


def _within_region_corr_matrices(
    ion_images: np.ndarray,
    ion_image_mzs: np.ndarray,
    pixel_mask: np.ndarray | None = None,
    n_regions: int = 20,
    random_state: int = 42,
    min_region_pixels: int = 30,
    tic_normalize: bool = False,
    common_mode_removal: bool = False,
    _global_corr_cache: tuple | None = None,
    debug: dict | None = None,
) -> dict:
    """
    Segment on-tissue pixels into ``n_regions`` regions using the identical
    recipe as ``_region_profile_corr_matrix`` (per-pixel TIC-normalised log1p
    composition, MiniBatchKMeans), then compute the RAW pixel Pearson r (via
    ``_pearson_r_matrix``, restricted to each region's own pixels) instead of
    the per-region MEAN fingerprint that ``_region_profile_corr_matrix`` uses.

    This asks whether same-protein peptides co-vary pixel-to-pixel *inside* a
    shared tissue region — a question the fingerprint metric cannot answer,
    since it only compares per-region averages and is blind to within-region
    pixel covariance (experimental).

    Returns a dict with two ``(corr_matrix, valid_mz_arr, mz_to_idx)`` triples
    — the same contract as ``_pearson_r_matrix`` — so both plug directly into
    ``_aggregate_protein_pairwise_corr``:

    - ``"weighted"``: pixel-count-weighted mean of each region's own Pearson r,
      over regions with >= ``min_region_pixels`` on-tissue pixels, expressed in
      the canonical (full on-tissue) valid-image index space. Any image that
      is non-constant within some region subset is necessarily non-constant
      over the full on-tissue mask too, so every region's valid image set is
      guaranteed to be a subset of the canonical one.
    - ``"dominant"``: the single largest region's own Pearson r, verbatim (its
      own, possibly smaller, index space) — not subject to ``min_region_pixels``.

    ``tic_normalize``/``common_mode_removal`` are forwarded to every
    ``_pearson_r_matrix`` call (region-level and canonical) so this uses the
    same preprocessing recipe as the best-performing global colocalization
    metric  rather than raw pixels.

    When ``debug`` is a dict it is populated with ``region_labels`` (per-pixel
    region id, ``(H*W,)`` int32, -1 off-mask), ``region_pixel_counts``, and
    ``dominant_region_id``.
    """
    from sklearn.cluster import MiniBatchKMeans

    mz_arr = np.asarray(ion_image_mzs, dtype=np.float64)
    n_feat = len(mz_arr)
    H, W = ion_images.shape[1], ion_images.shape[2]
    mask = (
        np.asarray(pixel_mask, dtype=bool) if pixel_mask is not None
        else np.ones(H * W, dtype=bool)
    )
    flat = ion_images.reshape(n_feat, H * W)
    on = flat[:, mask].astype(np.float32, copy=True)  # (n_feat, n_on)
    n_on = on.shape[1]

    empty = (np.empty((0, 0), dtype=np.float32), np.empty(0, dtype=np.float64), {})
    if n_on < 2:
        logger.warning("Within-region colocalization: %d on-tissue pixels — skipping (need >= 2).", n_on)
        if debug is not None:
            debug["region_labels"] = np.full(H * W, -1, dtype=np.int32)
            debug["region_pixel_counts"] = np.empty(0, dtype=np.int64)
            debug["dominant_region_id"] = -1
        return {"weighted": empty, "dominant": empty}

    # Segmentation feature matrix — identical recipe to _region_profile_corr_matrix.
    # Duplicated rather than shared: this is an experimental, likely-to-be-reverted
    # feature (see O3), so the tested/shipping function is
    # left untouched.
    col_tic = on.sum(axis=0, keepdims=True)
    comp = np.divide(on, col_tic, out=np.zeros_like(on), where=col_tic > 0)
    seg_feat = np.log1p(comp).T  # (n_on, n_feat) — pixels are samples

    k = int(min(n_regions, max(2, n_on - 1)))
    km = MiniBatchKMeans(n_clusters=k, random_state=random_state, n_init=3)
    labels = km.fit_predict(seg_feat)  # (n_on,)

    region_labels_full = np.full(H * W, -1, dtype=np.int32)
    region_labels_full[mask] = labels
    region_pixel_counts = np.bincount(labels, minlength=k)
    dominant_region_id = int(np.argmax(region_pixel_counts))

    if debug is not None:
        debug["region_labels"] = region_labels_full
        debug["region_pixel_counts"] = region_pixel_counts
        debug["dominant_region_id"] = dominant_region_id

    if _global_corr_cache is not None:
        _canon_corr, canon_mz, canon_idx = _global_corr_cache
    else:
        _canon_corr, canon_mz, canon_idx = _pearson_r_matrix(
            ion_images, ion_image_mzs, pixel_mask=mask,
            tic_normalize=tic_normalize, common_mode_removal=common_mode_removal,
        )
    n_canon = len(canon_mz)
    sum_wr = np.zeros((n_canon, n_canon), dtype=np.float64)
    sum_w = np.zeros((n_canon, n_canon), dtype=np.float64)
    dominant_triple = empty

    for r in range(k):
        n_r = int(region_pixel_counts[r])
        if n_r < 2:
            continue
        region_mask_r = region_labels_full == r
        corr_r, mz_r, idx_r = _pearson_r_matrix(
            ion_images, ion_image_mzs, pixel_mask=region_mask_r,
            tic_normalize=tic_normalize, common_mode_removal=common_mode_removal,
        )
        if r == dominant_region_id:
            dominant_triple = (corr_r, mz_r, idx_r)
        if n_r >= min_region_pixels and n_canon > 0 and len(mz_r) > 0:
            pos = np.fromiter((canon_idx[float(mz)] for mz in mz_r), dtype=int, count=len(mz_r))
            ix = np.ix_(pos, pos)
            sum_wr[ix] += n_r * corr_r
            sum_w[ix] += n_r

    weighted_matrix = np.divide(
        sum_wr, sum_w, out=np.full((n_canon, n_canon), np.nan), where=sum_w > 0
    ).astype(np.float32)

    logger.info(
        f"Within-region colocalization: K={k}, dominant region has "
        f"{region_pixel_counts[dominant_region_id]}/{n_on} on-tissue pixels, "
        f"{int((region_pixel_counts >= min_region_pixels).sum())}/{k} regions "
        f"clear min_region_pixels={min_region_pixels}"
    )
    return {
        "weighted": (weighted_matrix, canon_mz, canon_idx),
        "dominant": dominant_triple,
    }


def compute_within_region_colocalization_features(
    df: pd.DataFrame,
    ion_images: np.ndarray,
    ion_image_mzs: np.ndarray,
    pixel_mask: np.ndarray | None = None,
    n_regions: int = 20,
    random_state: int = 42,
    min_region_pixels: int = 30,
    tic_normalize: bool = False,
    common_mode_removal: bool = False,
    _global_corr_cache: tuple | None = None,
    debug: dict | None = None,
) -> pd.DataFrame:
    """O3 : within-region and dominant-region
    Pearson-r colocalization (opt-in, ``--within-region-coloc``; experimental /
    unvalidated — see PROGRESS.md for the validation protocol before trusting this in
    the ranker).

    Segments on-tissue pixels the same way as
    ``compute_region_colocalization_features`` but correlates RAW pixel
    intensities restricted to each region (not the per-region mean
    fingerprint), asking whether same-protein peptides co-vary pixel-to-pixel
    *inside* a shared tissue region rather than merely sharing a region
    average — a question ``compute_region_colocalization_features`` is blind
    to. Adds:

    - ``protein_within_region_colocalization`` (+ ``_max``, ``_median``): the
      pixel-count-weighted mean of each region's own within-protein Pearson r
      across regions with >= ``min_region_pixels`` on-tissue pixels.
    - ``protein_dominant_region_colocalization`` (+ ``_max``, ``_median``):
      within-protein Pearson r restricted to just the single largest region.

    Both blind to ``is_decoy``; decoys occupy a separate protein namespace
    exactly as for ``compute_region_colocalization_features``.
    """
    mats = _within_region_corr_matrices(
        ion_images, ion_image_mzs, pixel_mask=pixel_mask, n_regions=n_regions,
        random_state=random_state, min_region_pixels=min_region_pixels,
        tic_normalize=tic_normalize, common_mode_removal=common_mode_removal,
        _global_corr_cache=_global_corr_cache, debug=debug,
    )
    corr_w, mz_w, idx_w = mats["weighted"]
    corr_d, mz_d, idx_d = mats["dominant"]
    df = _aggregate_protein_pairwise_corr(
        df, corr_w, mz_w, idx_w, "protein_within_region_colocalization",
    )
    df = _aggregate_protein_pairwise_corr(
        df, corr_d, mz_d, idx_d, "protein_dominant_region_colocalization",
    )
    return df


def compute_theoretical_isotope_features(
    df: pd.DataFrame,
    maldi_envelopes: dict | None = None,
) -> pd.DataFrame:
    """
    Compute sequence-specific theoretical isotope features.

    Vectorized: uses pre-computed n_C/n_H/n_N/n_O/n_S and mass columns
    from the digest DataFrame. No pyteomics calls in the hot path.
    """
    n = len(df)

    # --- sequence-specific theoretical isotope distribution (brainpy, cached) ---
    comp_cols = ["n_C", "n_H", "n_N", "n_O", "n_S"]
    comp_arr = df[comp_cols].astype(int).values
    unique_comps = {tuple(row) for row in comp_arr}
    iso_cache = {k: theoretical_isotope_distribution(*k, n_peaks=3) for k in unique_comps}
    dist = np.array([iso_cache[tuple(row)] for row in comp_arr])  # (n, 3)
    theo_m0, theo_m1, theo_m2 = dist[:, 0], dist[:, 1], dist[:, 2]

    df["theo_has_sulfur"] = (comp_arr[:, 4] > 0).astype(float)

    # A8 — monoisotopic confidence: M0/(M0+M1). Invariant to normalization scheme.
    denom_conf = theo_m0 + theo_m1
    mono_conf = np.where(denom_conf > 0, theo_m0 / denom_conf, 1.0)

    # --- averagine theoretical (brainpy, cached; S=0 consistent with prior behaviour) ---
    pep_mass = df["mass"].values.astype(float)
    nc_avg = np.round(pep_mass * AVERAGINE_C).astype(int)
    nh_avg = np.round(pep_mass * AVERAGINE_H).astype(int)
    nn_avg = np.round(pep_mass * AVERAGINE_N).astype(int)
    no_avg = np.round(pep_mass * AVERAGINE_O).astype(int)
    avg_comps = list(zip(
        nc_avg.tolist(), nh_avg.tolist(), nn_avg.tolist(), no_avg.tolist(),
        [0] * n,
    ))
    unique_avg = set(avg_comps)
    avg_cache = {k: theoretical_isotope_distribution(*k, n_peaks=3) for k in unique_avg}
    avg_dist = np.array([avg_cache[k] for k in avg_comps])  # (n, 3)
    avg_m0, avg_m1, avg_m2 = avg_dist[:, 0], avg_dist[:, 1], avg_dist[:, 2]

    # Averagine deviation (vectorized)
    dot_ta = theo_m0 * avg_m0 + theo_m1 * avg_m1 + theo_m2 * avg_m2
    norm_t = np.sqrt(theo_m0**2 + theo_m1**2 + theo_m2**2)
    norm_a = np.sqrt(avg_m0**2 + avg_m1**2 + avg_m2**2)
    denom = norm_t * norm_a
    denom = np.where(denom > 0, denom, 1.0)
    df["averagine_deviation"] = 1.0 - dot_ta / denom

    # Sulfur-specific: |M2/M0 candidate - M2/M0 averagine|
    safe_theo_m0 = np.where(theo_m0 > 0, theo_m0, 1.0)
    safe_avg_m0 = np.where(avg_m0 > 0, avg_m0, 1.0)
    df["averagine_deviation_sulfur"] = np.abs(
        theo_m2 / safe_theo_m0 - avg_m2 / safe_avg_m0
    )

    # --- MALDI envelope comparison (requires per-feature lookup, but isotope math is done) ---
    theo_cosine = np.zeros(n)
    theo_chi2 = np.zeros(n)
    theo_kl = np.zeros(n)
    theo_m1_diff = np.full(n, np.nan)
    theo_m2_diff = np.full(n, np.nan)

    if maldi_envelopes:
        feature_mzs = df["feature_mz"].values
        for i in range(n):
            t = np.array([theo_m0[i], theo_m1[i], theo_m2[i]])
            nt = norm_t[i]

            maldi_env = maldi_envelopes.get(feature_mzs[i])
            if maldi_env is not None and len(maldi_env) >= 3:
                obs = np.array(maldi_env[:3], dtype=np.float64)
                no_val = np.linalg.norm(obs)
                if no_val > 0 and nt > 0:
                    theo_cosine[i] = np.dot(obs, t) / (no_val * nt)
                expected = t * obs.sum()
                mask = expected > 0
                if mask.any():
                    theo_chi2[i] = np.sum((obs[mask] - expected[mask]) ** 2 / expected[mask])
                obs_s = obs.sum()
                if obs_s > 0:
                    obs_norm = obs / obs_s
                    theo_safe = np.clip(t, 1e-10, None)
                    obs_safe = np.clip(obs_norm, 1e-10, None)
                    theo_kl[i] = np.sum(obs_safe * np.log(obs_safe / theo_safe))
                if obs[0] > 0 and t[0] > 0:
                    theo_m1_diff[i] = abs(obs[1] / obs[0] - t[1] / t[0])
                    theo_m2_diff[i] = abs(obs[2] / obs[0] - t[2] / t[0])

    df["theo_isotope_cosine"] = theo_cosine
    df["theo_isotope_chi2"] = theo_chi2
    df["theo_isotope_kl"] = theo_kl
    df["theo_m1_ratio_diff"] = theo_m1_diff
    df["theo_m2_ratio_diff"] = theo_m2_diff
    df["monoisotopic_confidence"] = mono_conf

    logger.info(f"Theoretical isotope features: {(theo_cosine > 0).sum()}/{n} scored")
    return df


# ---------------------------------------------------------------------------
# A11 — Peptide mass defect residual
# ---------------------------------------------------------------------------

def compute_mass_defect_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute peptide mass defect residual vs the averagine prediction (A11).

    mass_defect_residual = (theoretical_mass - floor(theoretical_mass))
                         - (averagine_mono_mass - floor(averagine_mono_mass))

    A non-zero residual indicates the peptide's elemental composition deviates
    from the averagine model — useful for discriminating sulfur-rich or
    aromatic-rich peptides.
    """
    mass = df["mass"].values.astype(float)

    peptide_md = mass - np.floor(mass)

    # Averagine monoisotopic mass (same atom counts as compute_theoretical_isotope_features)
    nc_avg = np.round(mass * AVERAGINE_C)
    nh_avg = np.round(mass * AVERAGINE_H)
    nn_avg = np.round(mass * AVERAGINE_N)
    no_avg = np.round(mass * AVERAGINE_O)
    avg_mono_mass = (
        nc_avg * 12.0
        + nh_avg * 1.00782503207
        + nn_avg * 14.00307400480
        + no_avg * 15.99491461956
    )
    avg_md = avg_mono_mass - np.floor(avg_mono_mass)

    df["mass_defect_residual"] = peptide_md - avg_md
    return df


# ---------------------------------------------------------------------------
# A12 — CHCA matrix cluster distance
# ---------------------------------------------------------------------------

def compute_chca_cluster_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute minimum ppm distance from each MALDI feature to CHCA cluster ions (A12).

    Low values indicate potential matrix interference. Feature-level scalar
    broadcast to all candidates at that feature.
    """
    mzs = df["feature_mz"].values[:, None]           # (n, 1)
    refs = CHCA_CLUSTER_MZS[None, :]                 # (1, k)
    dists_ppm = np.abs(mzs - refs) / refs * 1e6
    df["chca_cluster_distance_ppm"] = dists_ppm.min(axis=1)
    return df


# ---------------------------------------------------------------------------
# A3 — LOWESS-calibrated mass-error z-score
# ---------------------------------------------------------------------------

def compute_calibrated_ppm_features(
    df: pd.DataFrame,
    maldi_mzs: np.ndarray | None = None,
    pixel_coords: np.ndarray | None = None,
) -> pd.DataFrame:
    """
    LOWESS-calibrated ppm error z-score (A3).

    Fits a LOWESS curve of (per-feature median) ppm error vs pixel scan-order
    position to model the instrument's local mass offset as a function of
    spatial position. The z-score of each feature's residual is stored as
    ``ppm_error_calibrated_z`` and broadcast to all candidates at that feature.

    Requires statsmodels. Silently skips if not installed or pixel_coords is None.
    """
    if pixel_coords is None or maldi_mzs is None:
        return df

    pixel_coords = np.asarray(pixel_coords)
    maldi_mzs = np.asarray(maldi_mzs)

    # Convert 2-D (row, col) coordinates to 1-D scan index
    if pixel_coords.ndim == 2:
        ncols = int(pixel_coords[:, 1].max()) + 1
        coords_1d = (pixel_coords[:, 0] * ncols + pixel_coords[:, 1]).astype(float)
    else:
        coords_1d = pixel_coords.astype(float)

    feat_idx_to_coord = {i: float(coords_1d[i]) for i in range(len(coords_1d))}

    # Per-feature representative ppm: median across all candidates
    feat_ppm = (
        df.groupby("feature_idx")["ppm_error"]
        .median()
        .reset_index()
    )
    feat_ppm["pixel_coord"] = feat_ppm["feature_idx"].map(feat_idx_to_coord)
    feat_ppm = feat_ppm.dropna(subset=["pixel_coord"]).sort_values("pixel_coord")

    if len(feat_ppm) < 5:
        return df

    x = feat_ppm["pixel_coord"].values.astype(float)
    y = feat_ppm["ppm_error"].values.astype(float)

    # Cubic smoothing spline: O(n) vs statsmodels LOWESS O(n² × frac).
    # Equivalent smoothness for this application; ~100× faster on 1400 features.
    logger.debug(f"Running smoothing-spline ppm calibration (A3) on {len(feat_ppm)} features")
    try:
        from scipy.interpolate import make_smoothing_spline
        smoothed = make_smoothing_spline(x, y)(x)
    except ImportError:
        # scipy < 1.10 fallback: UnivariateSpline with automatic knot selection
        from scipy.interpolate import UnivariateSpline
        smoothed = UnivariateSpline(x, y, s=len(x))(x)
    residuals = y - smoothed

    mean_r, std_r = float(residuals.mean()), float(residuals.std())
    if std_r < 1e-10:
        return df

    feat_ppm["ppm_error_calibrated_z"] = (residuals - mean_r) / std_r
    idx_to_z = dict(zip(feat_ppm["feature_idx"], feat_ppm["ppm_error_calibrated_z"]))
    df["ppm_error_calibrated_z"] = df["feature_idx"].map(idx_to_z).fillna(0.0)

    logger.info("LOWESS ppm calibration (A3): computed ppm_error_calibrated_z")
    return df


# ---------------------------------------------------------------------------
# B — IM2Deep CCS features
# ---------------------------------------------------------------------------

def _finetune_and_predict(
    unique_seqs: list,
    cal_dedup: pd.DataFrame,
    pred_dict: dict,
    df: pd.DataFrame,
    n_cal: int,
    n_cal_rows: int,
    im2deep_kwargs: dict | None = None,
) -> np.ndarray:
    """
    Fine-tune the IM2Deep model on single-candidate MALDI CCS observations
    (transfer learning) and return re-predicted CCS for all rows in *df*.

    Requires ≥ 5 calibration peptides; logs a warning and returns the
    uncalibrated predictions if fewer are available or finetuning fails.
    """
    import logging
    import tempfile
    logging.getLogger("onnx2torch").setLevel(logging.WARNING)
    from im2deep.core import finetune as _im2deep_finetune
    from im2deep.core import predict as _predict
    from psm_utils.psm import PSM
    from psm_utils.psm_list import PSMList
    from psm_utils.peptidoform import Peptidoform

    # Build calibration PSMList with observed CCS in metadata
    cal_psms = []
    for _, row in cal_dedup.iterrows():
        seq = row["peptide"]
        ccs_obs = float(row["ccs_obs"])
        psm = PSM(
            peptidoform=Peptidoform(f"{seq}/1"),
            precursor_charge=1,
            spectrum_id=seq,
            metadata={"CCS": ccs_obs},
        )
        cal_psms.append(psm)
    cal_psm_list = PSMList(psm_list=cal_psms)

    with tempfile.NamedTemporaryFile(suffix=".ckpt", delete=False) as tmp:
        model_path = tmp.name

    _ft_kwargs = im2deep_kwargs or {}
    _ft_epochs = _ft_kwargs.get("finetune_epochs", 10)
    _ft_batch_size = _ft_kwargs.get("finetune_batch_size", 64)
    _ft_lr = _ft_kwargs.get("finetune_lr", 0.001)
    _ft_patience = _ft_kwargs.get("finetune_patience", 10)
    import random as _random
    import torch as _torch
    from threadpoolctl import threadpool_limits
    _torch.manual_seed(42)
    np.random.seed(42)
    _random.seed(42)
    _orig_threads = _torch.get_num_threads()
    _torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        finetuned_model = _im2deep_finetune(
            cal_psm_list,
            model_save_path=model_path,
            epochs=_ft_epochs,
            batch_size=_ft_batch_size,
            learning_rate=_ft_lr,
            patience=_ft_patience,
        )
    _torch.set_num_threads(_orig_threads)
    logger.info(
        f"  IM2Deep transfer learning on {n_cal} unique peptides "
        f"({n_cal_rows} calibration features): model saved to {model_path}"
    )

    # Re-predict all unique sequences with the finetuned model
    all_psms = [
        PSM(peptidoform=Peptidoform(f"{seq}/1"), precursor_charge=1, spectrum_id=seq)
        for seq in unique_seqs
    ]
    preds = _predict(PSMList(psm_list=all_psms), model=model_path)
    preds_arr = np.asarray(preds).flatten()
    ft_pred_dict = {seq: float(preds_arr[i]) for i, seq in enumerate(unique_seqs)}
    return df["peptide"].map(ft_pred_dict).values.astype(float)


def _ccs_mz_baseline(
    feature_mz: np.ndarray,
    mh_mz: np.ndarray,
    fit_mz: np.ndarray,
    fit_ccs: np.ndarray,
    min_points: int = 10,
) -> np.ndarray | None:
    """Expected CCS difference ``g(feature_mz) − g(mh_mz)`` from a power-law CCS↔m/z
    trend ``g(mz) = A·mz^B`` fit on ``(fit_mz, fit_ccs)`` (the calibration peptides).

    Subtracting this baseline from the raw CCS delta removes the m/z-gap component,
    leaving the conformational residual.  For a candidate whose peptide m/z equals
    its feature m/z (targets) the baseline is 0.  Returns ``None`` when fewer than
    ``min_points`` finite positive calibration pairs are available (caller then
    falls back to the raw delta).  Pure / unit-testable (no im2deep needed).
    """
    fit_mz = np.asarray(fit_mz, dtype=float)
    fit_ccs = np.asarray(fit_ccs, dtype=float)
    ok = np.isfinite(fit_mz) & (fit_mz > 0) & np.isfinite(fit_ccs) & (fit_ccs > 0)
    if int(ok.sum()) < min_points:
        return None
    B, logA = np.polyfit(np.log(fit_mz[ok]), np.log(fit_ccs[ok]), 1)
    logger.info(
        "  IM2Deep CCS m/z-trend fit: CCS = %.3g·mz^%.3f (on %d calibration features)",
        float(np.exp(logA)), float(B), int(ok.sum()),
    )

    def _g(mz):
        mz = np.asarray(mz, dtype=float)
        out = np.full(len(mz), np.nan)
        m = np.isfinite(mz) & (mz > 0)
        out[m] = np.exp(logA) * mz[m] ** B
        return out

    base = _g(feature_mz) - _g(mh_mz)
    return np.where(np.isfinite(base), base, 0.0)


def compute_im2deep_features(
    df: pd.DataFrame,
    observed_ccs_per_feature: dict | None = None,
    calibration_method: str = "linear",
    im2deep_kwargs: dict | None = None,
) -> pd.DataFrame:
    """
    Compute IM2Deep CCS prediction features (B1–B6).

    ``observed_ccs_per_feature`` maps feature_idx → observed CCS (Å²). If None
    or empty, the function returns the DataFrame unchanged.

    Requires im2deep. Silently skips if not installed.

    Features added
    --------------
    im2deep_delta_ccs           signed CCS residual (observed − predicted)
    im2deep_abs_delta_ccs_pct   |residual| / predicted × 100
    im2deep_ccs_zscore          z-score of delta_ccs within each feature group
    im2deep_ccs_rank            rank of abs_delta_ccs_pct within feature group
    im2deep_delta_ccs_resid           m/z-detrended (conformational) signed residual
    im2deep_abs_delta_ccs_pct_resid   |detrended residual| / predicted × 100
    im2deep_ccs_zscore_resid          z-score of the detrended residual per group
    im2deep_ccs_rank_resid            rank of the detrended |residual| per group
    # im2deep_mahalanobis         Mahalanobis distance in (ppm_error_abs, delta_ccs) space

    The ``*_resid`` features subtract the fitted CCS↔m/z trend so they measure
    conformational mismatch rather than the m/z baseline (see below).  For
    targets they equal the raw features; for decoys whose peptide m/z differs
    from the feature m/z they remove the trivial m/z-gap separation.
    """
    if not observed_ccs_per_feature:
        return df

    try:
        from im2deep.core import predict as _im2deep_predict
        from psm_utils.psm import PSM
        from psm_utils.psm_list import PSMList as _PSMList
        from psm_utils.peptidoform import Peptidoform
    except ImportError:
        logger.debug("im2deep not installed — skipping CCS features (B)")
        return df

    unique_seqs = df["peptide"].unique().tolist()
    pred_dict: dict[str, float] = {}

    try:
        psms = [PSM(peptidoform=Peptidoform(f"{seq}/1"), precursor_charge=1, spectrum_id=seq) for seq in unique_seqs]
        psm_list = _PSMList(psm_list=psms)
        preds = _im2deep_predict(psm_list)
        preds_arr = np.asarray(preds).flatten()
        for i, seq in enumerate(unique_seqs):
            pred_dict[seq] = float(preds_arr[i])
    except Exception as exc:
        logger.warning(f"IM2Deep prediction failed: {exc} — skipping CCS features")
        return df

    predicted = df["peptide"].map(pred_dict).values.astype(float)
    observed = (
        df["feature_idx"]
        .map(observed_ccs_per_feature)
        .values.astype(float)
    )

    # Calibrate predicted CCS using the calibration-peptide set (high-quality
    # targets: top percentile by low ppm + high isotope cosine) as reference
    # points.  Falls back to single-candidate features (n_candidates == 1) when the
    # is_calibration_peptide column is absent (e.g. direct callers).  Three methods:
    #   linear  — global additive shift (LinearCCSCalibration, default; symmetric
    #             across targets/decoys, so safe for the ranker CCS features)
    #   raw     — no calibration (use raw IM2Deep predictions)
    #   spline  — piecewise spline mapping (SplineCCSCalibration)
    #   finetune — transfer-learning re-training of the neural network weights
    predicted_cal = predicted.copy()
    if calibration_method != "raw" and ("is_calibration_peptide" in df.columns or "n_candidates" in df.columns):
        try:
            if "is_calibration_peptide" in df.columns:
                single_mask = df["is_calibration_peptide"].to_numpy(dtype=bool)
            else:
                single_mask = df["n_candidates"].values == 1
            valid_cal = single_mask & np.isfinite(observed) & np.isfinite(predicted)

            if valid_cal.sum() >= 5:
                cal_df = df.loc[valid_cal, ["peptide", "feature_idx"]].copy()
                cal_df["ccs_obs"] = cal_df["feature_idx"].map(observed_ccs_per_feature)
                cal_df = cal_df.dropna(subset=["ccs_obs"])

                # Deduplicate: one row per unique peptide sequence (mean observed CCS)
                cal_dedup = cal_df.groupby("peptide")["ccs_obs"].mean().reset_index()
                n_cal = len(cal_dedup)

                if calibration_method == "finetune":
                    predicted_cal = _finetune_and_predict(
                        unique_seqs, cal_dedup, pred_dict, df,
                        n_cal, int(valid_cal.sum()),
                        im2deep_kwargs=im2deep_kwargs,
                    )
                else:
                    from im2deep.calibration import (
                        LinearCCSCalibration as _LinearCal,
                        SplineCCSCalibration as _SplineCal,
                    )

                    psm_df_target = pd.DataFrame({
                        "peptidoform": cal_dedup["peptide"].apply(lambda s: f"{s}/1"),
                        "CCS": cal_dedup["ccs_obs"].values,
                    })
                    psm_df_source = pd.DataFrame({
                        "peptidoform": cal_dedup["peptide"].apply(lambda s: f"{s}/1"),
                        "CCS": cal_dedup["peptide"].map(pred_dict).values,
                    })
                    df_transform = pd.DataFrame({
                        "peptidoform": df["peptide"].apply(lambda s: f"{s}/1"),
                        "predicted_CCS_uncalibrated": predicted,
                    })

                    if calibration_method == "spline":
                        cal = _SplineCal(n_knots=3, degree=2)
                        cal.fit(psm_df_target, psm_df_source)
                        predicted_cal = cal.transform(df_transform).astype(float)
                        logger.info(
                            f"  IM2Deep spline calibration on {n_cal} unique peptides "
                            f"({valid_cal.sum()} calibration features)"
                        )
                    else:  # linear (default)
                        cal = _LinearCal(per_charge=False, use_charge_state=1)
                        cal.fit(psm_df_target, psm_df_source)
                        predicted_cal = np.array(
                            list(cal.transform(df_transform)), dtype=float
                        )
                        logger.info(
                            f"  IM2Deep linear calibration on {n_cal} unique peptides "
                            f"({valid_cal.sum()} calibration features): "
                            f"shift={cal.general_shift:.2f} Å²"
                        )
        except Exception as exc:
            logger.warning(f"IM2Deep calibration failed ({exc}) — using uncalibrated predictions")

    delta = observed - predicted_cal
    abs_pct = np.abs(delta) / np.where(np.abs(predicted_cal) > 1e-6, np.abs(predicted_cal), 1.0) * 100.0

    df["_im2deep_delta"] = delta
    df["_im2deep_abspct"] = abs_pct

    df["im2deep_delta_ccs"] = delta
    df["im2deep_abs_delta_ccs_pct"] = abs_pct
    df["im2deep_observed_ccs"] = observed
    df["im2deep_predicted_ccs"] = predicted_cal

    # Z-score and rank within each MALDI feature
    df["im2deep_ccs_zscore"] = df.groupby("feature_mz")["_im2deep_delta"].transform(
        lambda x: pd.Series(0.0, index=x.index)
        if len(x) < 2 or x.std() < 1e-12
        else (x - x.mean()) / x.std()
    )
    df["im2deep_ccs_rank"] = df.groupby("feature_mz")["_im2deep_abspct"].rank(method="min")

    # --- m/z-detrended (conformational) CCS residual ---------------------------
    # The raw CCS delta leaks the m/z baseline when a candidate's peptide m/z
    # differs from its feature m/z (decoys: mz_shift / mz_shuffle / entrapment).
    # CCS rises monotonically with m/z, so |observed_CCS(feature) −
    # predicted_CCS(peptide)| is dominated by the m/z gap, not conformation —
    # e.g. mz_shuffle deliberately relocates peptides far in mass, giving decoys a
    # huge, trivially-separable CCS delta that does NOT reflect a real (isobaric)
    # false positive.  Fit the population CCS↔m/z trend g(mz) = A·mz^B on the
    # calibration peptides and subtract it from both sides: the residual measures
    # the conformational mismatch only.  For targets (feature_mz == mh_mz) the two
    # g terms cancel → identical to the raw delta; for decoys the baseline cancels,
    # leaving conf(observed) − conf(predicted), exchangeable with an isobaric FP.
    if "is_calibration_peptide" in df.columns:
        cal_m = df["is_calibration_peptide"].to_numpy(dtype=bool)
    elif "n_candidates" in df.columns:
        cal_m = df["n_candidates"].values == 1
    else:
        cal_m = ~np.asarray(
            df.get("is_decoy", pd.Series(False, index=df.index)), dtype=bool
        )
    fmz = df["feature_mz"].to_numpy(dtype=float)
    baseline = _ccs_mz_baseline(
        fmz, df["mh_mz"].to_numpy(dtype=float),
        fit_mz=fmz[cal_m], fit_ccs=observed[cal_m],
    )
    if baseline is not None:
        resid_delta = delta - baseline  # remove the expected m/z-gap CCS difference
    else:
        logger.warning("IM2Deep CCS m/z-detrend: trend fit unavailable — residual = raw delta")
        resid_delta = delta.copy()
    resid_abspct = (
        np.abs(resid_delta)
        / np.where(np.abs(predicted_cal) > 1e-6, np.abs(predicted_cal), 1.0)
        * 100.0
    )

    df["_resid_delta"] = resid_delta
    df["_resid_abspct"] = resid_abspct
    df["im2deep_delta_ccs_resid"] = resid_delta
    df["im2deep_abs_delta_ccs_pct_resid"] = resid_abspct
    df["im2deep_ccs_zscore_resid"] = df.groupby("feature_mz")["_resid_delta"].transform(
        lambda x: pd.Series(0.0, index=x.index)
        if len(x) < 2 or x.std() < 1e-12
        else (x - x.mean()) / x.std()
    )
    df["im2deep_ccs_rank_resid"] = df.groupby("feature_mz")["_resid_abspct"].rank(method="min")

    # Diagnostic: the detrended decoy residual should NOT track the m/z relocation
    # distance (decoy_delta_da), unlike the raw delta — confirms the leak is gone.
    if "is_decoy" in df.columns and "decoy_delta_da" in df.columns:
        dmask = df["is_decoy"].astype(bool).to_numpy()
        dd = np.abs(df.loc[dmask, "decoy_delta_da"].to_numpy(dtype=float))

        def _abscorr(vals) -> tuple[float, int]:
            # Correlate only over decoys with a FINITE CCS delta AND a finite,
            # non-zero relocation distance. Masking per-array (not just on dd) is
            # essential: a single NaN CCS delta makes np.std/np.corrcoef return NaN,
            # which previously produced an all-NaN leak check even when most decoys
            # had a valid CCS delta.
            a = np.abs(np.asarray(vals, dtype=float))
            m = np.isfinite(a) & np.isfinite(dd) & (dd > 0)
            n = int(m.sum())
            if n < 10 or np.std(a[m]) < 1e-12 or np.std(dd[m]) < 1e-12:
                return float("nan"), n
            return float(abs(np.corrcoef(a[m], dd[m])[0, 1])), n

        raw_c, n_raw = _abscorr(df.loc[dmask, "im2deep_delta_ccs"].to_numpy())
        res_c, _ = _abscorr(resid_delta[dmask])
        _fmt = lambda x: "n/a" if not np.isfinite(x) else f"{x:.2f}"
        logger.info(
            "  CCS m/z-leak check (decoys, n=%d): |corr(raw Δ, decoy_delta_da)|=%s → "
            "|corr(residual Δ, decoy_delta_da)|=%s",
            n_raw, _fmt(raw_c), _fmt(res_c),
        )

    df.drop(
        columns=["_im2deep_delta", "_im2deep_abspct", "_resid_delta", "_resid_abspct"],
        inplace=True, errors="ignore",
    )

    # Fill NaN with median (symmetric fill — no target/decoy information)
    for col in [
        "im2deep_delta_ccs", "im2deep_abs_delta_ccs_pct",
        "im2deep_ccs_zscore", "im2deep_ccs_rank",
        "im2deep_delta_ccs_resid", "im2deep_abs_delta_ccs_pct_resid",
        "im2deep_ccs_zscore_resid", "im2deep_ccs_rank_resid",
    ]:
        valid = df[col].dropna()
        fill = float(valid.median()) if len(valid) > 0 else 0.0
        df[col] = df[col].fillna(fill)

    logger.info(
        f"IM2Deep CCS features (B): scored {(~np.isnan(observed)).sum()}/{len(df)} candidates"
    )
    return df


def compute_lcms_ccs_features(
    df: pd.DataFrame,
    observed_ccs_per_feature: dict | None = None,
) -> pd.DataFrame:
    """
    Compare MALDI-observed CCS to LC-MS/MS-observed CCS per candidate.

    Requires ``lcms_ccs`` column in *df* (populated from LC-MS/MS ion mobility
    data via ``lcms_ids._parse_psm_utils``) and ``observed_ccs_per_feature``
    (MALDI CCS from the SCiLS Lab CSV).  Skips silently if either is absent.

    Note: MALDI CCS is measured at charge 1 ([M+H]+); LC-MS/MS CCS is measured
    at the search charge (typically 2–4) and converted via the Mason-Schamp
    equation.  The delta is therefore not zero for the correct identification,
    but it is consistent across correct assignments and inconsistent for
    misassignments.

    Features added
    --------------
    lcms_ccs_delta      MALDI CCS − LC-MS/MS CCS  (Å²)
    lcms_ccs_abs_pct    |delta| / LC-MS/MS CCS × 100  (%)
    """
    if not observed_ccs_per_feature:
        return df
    if "lcms_ccs" not in df.columns or df["lcms_ccs"].isna().all():
        return df

    maldi_ccs = df["feature_idx"].map(observed_ccs_per_feature).values.astype(float)
    lcms_ccs = df["lcms_ccs"].values.astype(float)

    delta = maldi_ccs - lcms_ccs
    abs_pct = np.abs(delta) / np.where(np.abs(lcms_ccs) > 1e-6, np.abs(lcms_ccs), 1.0) * 100.0

    df["lcms_ccs_delta"] = delta
    df["lcms_ccs_abs_pct"] = abs_pct

    for col in ["lcms_ccs_delta", "lcms_ccs_abs_pct"]:
        valid = df[col].dropna()
        fill = float(valid.median()) if len(valid) > 0 else 0.0
        df[col] = df[col].fillna(fill)

    n_valid = int(np.isfinite(delta).sum())
    logger.info(
        f"MALDI vs LC-MS/MS CCS features: {n_valid}/{len(df)} candidates with both CCS values"
    )
    return df


# ---------------------------------------------------------------------------
# C — Additional peptide property features
# ---------------------------------------------------------------------------

def compute_peptide_property_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute additional peptide property features.

    Features added
    --------------
    has_oxidized_met         1 if M present (oxidation susceptibility proxy)
    has_cys                  1 if C present (carbamidomethylation proxy)
    n_proline                count of P residues
    acidic_residue_density   (count(D)+count(E)) / length
    """
    codes, uniques = pd.factorize(df["peptide"])
    uniques_list = uniques.tolist()

    try:
        from ms1rescore_rs import compute_property_features
        n_D, n_E, n_C, n_P, n_M, _, _, seq_lens_u, _, _ = \
            compute_property_features(uniques_list)
        n_D = np.asarray(n_D); n_E = np.asarray(n_E); n_C = np.asarray(n_C)
        n_P = np.asarray(n_P); n_M = np.asarray(n_M)
        seq_lens_u = np.asarray(seq_lens_u)
    except ImportError:
        cnt = _residue_counts_batch(uniques, "DECPM")
        n_D = cnt["D"]; n_E = cnt["E"]; n_C = cnt["C"]
        n_P = cnt["P"]; n_M = cnt["M"]
        seq_lens_u = np.array([len(s) for s in uniques], dtype=np.int32)

    df["has_oxidized_met"]       = (n_M > 0)[codes].astype(float)
    df["has_cys"]                = (n_C > 0)[codes].astype(float)
    df["n_proline"]              = n_P[codes].astype(float)
    df["acidic_residue_density"] = ((n_D + n_E).astype(float) / np.maximum(seq_lens_u, 1))[codes]

    return df


# ---------------------------------------------------------------------------
# E1 — Isotopologue spatial co-localization
# ---------------------------------------------------------------------------

def _pearson_r_pairwise(
    images_a: np.ndarray,
    images_b: np.ndarray,
    pixel_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Per-feature Pearson r between corresponding images in two (N, H, W) arrays.

    When ``pixel_mask`` (flattened (H*W,) boolean) is given, r is computed over
    the selected on-tissue pixels only (see ``compute_tissue_mask``).
    """
    n = len(images_a)
    if n == 0:
        return np.empty(0, dtype=np.float32)
    a = images_a.reshape(n, -1).astype(np.float32)
    b = images_b.reshape(n, -1).astype(np.float32)
    if pixel_mask is not None:
        m = np.asarray(pixel_mask, dtype=bool)
        a = a[:, m]
        b = b[:, m]
    a -= a.mean(axis=1, keepdims=True)
    b -= b.mean(axis=1, keepdims=True)
    num = (a * b).sum(axis=1)
    denom = np.sqrt((a * a).sum(axis=1)) * np.sqrt((b * b).sum(axis=1))
    return np.where(denom > 1e-10, num / denom, np.nan).astype(np.float32)


def _m0_valid_mask(ion_images, ion_image_mzs, pixel_mask=None):
    """Boolean (N,) mask of M0 images with non-constant signal over the
    (optionally pixel-masked) pixels. Shared by the direct-image colocalization
    paths."""
    flat_m0 = ion_images.reshape(len(np.asarray(ion_image_mzs, dtype=np.float64)), -1)
    if pixel_mask is not None:
        flat_m0 = flat_m0[:, np.asarray(pixel_mask, dtype=bool)]
    return flat_m0.std(axis=1) > 1e-10


def _assign_coloc_column(df, col, mapping):
    """Map ``feature_mz`` → Pearson r and fill missing values with the column
    median (0.0 when no finite value exists). Shared colocalization tail."""
    df[col] = df["feature_mz"].map(mapping)
    valid = df[col].dropna()
    df[col] = df[col].fillna(float(valid.median()) if len(valid) > 0 else 0.0)


def compute_isotopologue_colocalization(
    df: pd.DataFrame,
    ion_images: np.ndarray,
    ion_image_mzs: np.ndarray,
    ppm_tolerance: float = 10.0,
    _corr_cache: tuple | None = None,
    extra_ion_images: dict | None = None,
    pixel_mask: np.ndarray | None = None,
) -> pd.DataFrame:
    """
    Pearson r between the M0 ion image and the M+1 / M+2 ion images (E1).

    When ``extra_ion_images`` contains keys ``"m1"`` and ``"m2"``, Pearson r is
    computed directly per feature without needing M+1/M+2 entries in the feature
    list.  This is the normal path: MALDI feature lists contain only monoisotopic
    M0 peaks, so M+1/M+2 images are pre-extracted at ``feature_mzs + NEUTRON``.

    Without pre-extracted images the function falls back to searching for M+1/M+2
    partners inside the feature list (works only when both M0 and M+1 are listed).

    Features added
    --------------
    isotope_image_colocalization_m1     Pearson r(M0, M+1)
    isotope_image_colocalization_m2     Pearson r(M0, M+2)
    isotope_image_colocalization_mean   mean of available correlations
    """
    if _corr_cache is not None:
        corr_matrix, valid_mz_arr, mz_to_idx = _corr_cache
    else:
        corr_matrix, valid_mz_arr, mz_to_idx = _pearson_r_matrix(ion_images, ion_image_mzs)
    n_valid = len(valid_mz_arr)

    m1_images = extra_ion_images.get("m1") if extra_ion_images else None
    m2_images = extra_ion_images.get("m2") if extra_ion_images else None

    if m1_images is not None and m2_images is not None:
        # Direct per-feature Pearson r: no partner lookup needed.
        valid_mask = _m0_valid_mask(ion_images, ion_image_mzs, pixel_mask)
        r_m1_arr = _pearson_r_pairwise(ion_images[valid_mask], m1_images[valid_mask], pixel_mask=pixel_mask)
        r_m2_arr = _pearson_r_pairwise(ion_images[valid_mask], m2_images[valid_mask], pixel_mask=pixel_mask)
        n_m1_found = int(np.isfinite(r_m1_arr).sum())
        logger.info(
            f"Isotopologue colocalization (E1): direct images — "
            f"{n_m1_found}/{n_valid} features with finite r(M0, M+1)"
        )
    else:
        # Fallback: partner lookup inside the feature list.
        self_idx = np.arange(n_valid, dtype=np.intp)
        m1_idx = _find_partner_indices(valid_mz_arr, valid_mz_arr + NEUTRON, ppm_tolerance)
        m2_idx = _find_partner_indices(valid_mz_arr, valid_mz_arr + 2.0 * NEUTRON, ppm_tolerance)
        r_m1_arr = np.where(m1_idx >= 0, corr_matrix[self_idx, np.clip(m1_idx, 0, n_valid - 1)], np.nan)
        r_m2_arr = np.where(m2_idx >= 0, corr_matrix[self_idx, np.clip(m2_idx, 0, n_valid - 1)], np.nan)
        n_m1_found = int((m1_idx >= 0).sum())
        logger.info(
            f"Isotopologue colocalization (E1): {n_m1_found}/{n_valid} features with M+1 in feature list"
        )

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        r_mean_arr = np.nanmean(np.stack([r_m1_arr, r_m2_arr], axis=1), axis=1)

    r_m1_map   = {float(mz): float(r) for mz, r in zip(valid_mz_arr, r_m1_arr)}
    r_m2_map   = {float(mz): float(r) for mz, r in zip(valid_mz_arr, r_m2_arr)}
    r_mean_map = {float(mz): float(r) for mz, r in zip(valid_mz_arr, r_mean_arr)}

    for col, mapping in [
        ("isotope_image_colocalization_m1",   r_m1_map),
        ("isotope_image_colocalization_m2",   r_m2_map),
        ("isotope_image_colocalization_mean", r_mean_map),
    ]:
        _assign_coloc_column(df, col, mapping)

    return df


# ---------------------------------------------------------------------------
# E2 — Adduct co-localization
# ---------------------------------------------------------------------------

def compute_adduct_colocalization(
    df: pd.DataFrame,
    ion_images: np.ndarray,
    ion_image_mzs: np.ndarray,
    ppm_tolerance: float = 10.0,
    _corr_cache: tuple | None = None,
    extra_ion_images: dict | None = None,
    pixel_mask: np.ndarray | None = None,
) -> pd.DataFrame:
    """
    Pearson r between the M0 ion image and Na/K/CHCA adduct ion images (E2).

    When ``extra_ion_images`` contains keys ``"na"``, ``"k"``, ``"chca"``, Pearson
    r is computed directly per feature without needing adduct peaks in the feature
    list.  This is the normal path: adduct peaks are typically absent from
    monoisotopic-only feature lists.

    Without pre-extracted images the function falls back to searching for adduct
    partners inside the feature list.

    Features added
    --------------
    adduct_colocalization_na    Pearson r with [M+Na]+ image
    adduct_colocalization_k     Pearson r with [M+K]+ image
    adduct_colocalization_chca  Pearson r with [M+CHCA+H-H2O]+ image
    """
    if _corr_cache is not None:
        corr_matrix, valid_mz_arr, mz_to_idx = _corr_cache
    else:
        corr_matrix, valid_mz_arr, mz_to_idx = _pearson_r_matrix(ion_images, ion_image_mzs)
    n_valid = len(valid_mz_arr)

    if extra_ion_images and any(k in extra_ion_images for k in _ADDUCT_DELTAS):
        # Direct per-feature Pearson r using pre-extracted adduct images.
        valid_mask = _m0_valid_mask(ion_images, ion_image_mzs, pixel_mask)
        m0_valid = ion_images[valid_mask]
        for adduct_name in _ADDUCT_DELTAS:
            adduct_imgs = extra_ion_images.get(adduct_name)
            if adduct_imgs is not None:
                r_arr = _pearson_r_pairwise(m0_valid, adduct_imgs[valid_mask], pixel_mask=pixel_mask)
            else:
                r_arr = np.full(n_valid, np.nan, dtype=np.float32)
            mapping = {float(mz): float(r) for mz, r in zip(valid_mz_arr, r_arr)}
            _assign_coloc_column(df, f"adduct_colocalization_{adduct_name}", mapping)
        logger.info(
            f"Adduct colocalization (E2): direct images used for "
            f"{[k for k in _ADDUCT_DELTAS if extra_ion_images.get(k) is not None]}"
        )
    else:
        # Fallback: partner lookup inside the feature list.
        self_idx = np.arange(n_valid, dtype=np.intp)
        for adduct_name, delta in _ADDUCT_DELTAS.items():
            partner_idx = _find_partner_indices(valid_mz_arr, valid_mz_arr + delta, ppm_tolerance)
            r_arr = np.where(
                partner_idx >= 0,
                corr_matrix[self_idx, np.clip(partner_idx, 0, n_valid - 1)],
                np.nan,
            )
            mapping = {float(mz): float(r) for mz, r in zip(valid_mz_arr, r_arr)}
            _assign_coloc_column(df, f"adduct_colocalization_{adduct_name}", mapping)
        n_found = sum(
            int((_find_partner_indices(valid_mz_arr, valid_mz_arr + d, ppm_tolerance) >= 0).sum())
            for d in _ADDUCT_DELTAS.values()
        )
        logger.info(
            f"Adduct colocalization (E2): {n_found} total adduct partners found in feature list"
        )

    return df


# ---------------------------------------------------------------------------
# E5/E6 — Moran's I and Geary's C (full spatial autocorrelation)
# ---------------------------------------------------------------------------

def _neighbor_sum_batch(imgs: np.ndarray) -> np.ndarray:
    """
    Queen's-contiguity (8-neighbour) sum for a batch of images.

    Parameters
    ----------
    imgs : (n, H, W) float64 — zero-boundary padding assumed.

    Returns
    -------
    (n, H, W) float64 — for each pixel, sum of its up-to-8 neighbours.
    """
    _, H, W = imgs.shape
    pad = np.pad(imgs, ((0, 0), (1, 1), (1, 1)), mode="constant", constant_values=0.0)
    result = (
        pad[:, 0:H,   0:W  ]
        + pad[:, 0:H,   1:W+1]
        + pad[:, 0:H,   2:W+2]
        + pad[:, 1:H+1, 0:W  ]
        + pad[:, 1:H+1, 2:W+2]
        + pad[:, 2:H+2, 0:W  ]
        + pad[:, 2:H+2, 1:W+1]
        + pad[:, 2:H+2, 2:W+2]
    )
    return result


def _morans_gearys_chunk(
    chunk_images: np.ndarray,
    neighbor_counts: np.ndarray,
    N: int,
    W_sum: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Moran's I and Geary's C for a batch of images.

    Module-level so ``ThreadPoolExecutor`` can pickle it.  Uses float32
    throughout (except final sums, which accumulate in float64) to halve
    memory bandwidth vs float64 arrays.

    Parameters
    ----------
    chunk_images : (c, H, W) float32
    neighbor_counts : (H, W) float32 — precomputed queen-contiguity degree map
    N : int — total number of pixels (H * W)
    W_sum : float — sum of all edge weights

    Returns
    -------
    morans : (c,) float64
    gearys : (c,) float64
    """
    imgs = chunk_images.astype(np.float32)
    x_bar = imgs.mean(axis=(1, 2), keepdims=True)
    z = imgs - x_bar
    z_sq = (z * z).sum(axis=(1, 2), dtype=np.float64)

    ok = z_sq > 1e-12
    morans = np.zeros(len(imgs), dtype=np.float64)
    gearys = np.ones(len(imgs), dtype=np.float64)

    if not ok.any():
        return morans, gearys

    Wz = _neighbor_sum_batch(z)
    numerators_m = (z * Wz).sum(axis=(1, 2), dtype=np.float64)
    morans = np.where(ok, (N / W_sum) * numerators_m / np.where(ok, z_sq, 1.0), 0.0)

    Wx = _neighbor_sum_batch(imgs)
    xTDx = (imgs * imgs * neighbor_counts).sum(axis=(1, 2), dtype=np.float64)
    xTWx = (imgs * Wx).sum(axis=(1, 2), dtype=np.float64)
    gearys = np.where(
        ok,
        ((N - 1) / W_sum) * (xTDx - xTWx) / np.where(ok, z_sq, 1.0),
        1.0,
    )
    return morans, gearys


def _pearson_r_images(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson r between two same-shape images. Returns nan for constant images."""
    af = a.ravel().astype(np.float32)
    bf = b.ravel().astype(np.float32)
    af -= af.mean()
    bf -= bf.mean()
    denom = np.sqrt((af * af).sum()) * np.sqrt((bf * bf).sum())
    if denom < 1e-10:
        return np.nan
    return float((af * bf).sum() / denom)


def _log_progress(prefix: str, n_done: int, n_total: int, t_start: float, every: int) -> None:
    """Emit a throttled '<prefix>: k/n (p%) elapsed As, ETA Bs' progress line.

    Logs on every ``every``-th item (and always on the last), so long loops with
    no other per-item logging (mobility image rebuild, protein-pairwise coloc,
    the pure-Python mob-coloc fallback) show where time is actually going instead
    of a single silent multi-hour gap between phase-boundary log lines.
    """
    if n_total <= 0 or (n_done % every != 0 and n_done != n_total):
        return
    elapsed = time.perf_counter() - t_start
    rate = n_done / elapsed if elapsed > 0 else 0.0
    eta = (n_total - n_done) / rate if rate > 0 else float("nan")
    logger.debug(
        f"{prefix}: {n_done}/{n_total} ({100.0 * n_done / n_total:.0f}%) "
        f"elapsed {elapsed:.0f}s, ETA {eta:.0f}s"
    )


def compute_mobility_colocalization_features(
    df: pd.DataFrame,
    tdf_path: str,
    mob_window_multiplier: float = 2.0,
    extraction_ppm: float = 25.0,
    protein_coloc: bool = False,
    tic_normalize: bool = False,
) -> pd.DataFrame:
    """
    Per-candidate isotopologue and adduct colocalization using mobility-filtered images.

    Requires im2deep_predicted_ccs and im2deep_observed_ccs to be present in df.
    The 1/K0 window half-width is derived from p95(|Δ1/K0|) on single-candidate
    (n_candidates==1) features.

    Adds ten new columns:
      isotope_colocalization_m1_mob, isotope_colocalization_m2_mob,
      isotope_colocalization_mean_mob,
      adduct_colocalization_na_mob, adduct_colocalization_k_mob,
      adduct_colocalization_chca_mob,
      fraction_detected_mob, intensity_cv_mob,
      log_mean_intensity_mob, spatial_morans_i_mob.

    When ms1rescore_rs is available, builds a flat CSR once (one alphatims call
    per pixel) and dispatches to a rayon-parallel Rust kernel that processes all
    feature groups simultaneously.  Falls back to the per-feature Python loop
    if the Rust extension is unavailable.
    """
    import sqlite3
    from pathlib import Path

    _t_func_start = time.perf_counter()

    NEW_COLS = [
        "isotope_colocalization_m1_mob", "isotope_colocalization_m2_mob",
        "isotope_colocalization_mean_mob",
        "adduct_colocalization_na_mob", "adduct_colocalization_k_mob",
        "adduct_colocalization_chca_mob",
        "fraction_detected_mob", "intensity_cv_mob",
        "log_mean_intensity_mob", "spatial_morans_i_mob",
    ]

    if "im2deep_predicted_ccs" not in df.columns:
        logger.warning("im2deep_predicted_ccs not found — skipping mobility colocalization.")
        for col in NEW_COLS:
            df[col] = np.nan
        return df

    from im2deep.utils import ccs2im

    # Convert predicted CCS → predicted 1/K0; MALDI is always [M+H]+ (charge=1)
    df = df.copy()
    df["_pred_inv_k0"] = ccs2im(
        df["im2deep_predicted_ccs"].values, mz=df["feature_mz"].values, charge=1
    )

    # Data-driven 1/K0 window from the calibration-peptide set (falls back to
    # single-candidate features when is_calibration_peptide is absent).
    k0_half_win = 0.05  # fallback
    if "im2deep_observed_ccs" in df.columns:
        _cal_mask = (
            df["is_calibration_peptide"]
            if "is_calibration_peptide" in df.columns
            else (df["n_candidates"] == 1)
        )
        single = df[_cal_mask].dropna(
            subset=["im2deep_observed_ccs", "_pred_inv_k0"]
        )
        if len(single) >= 10:
            obs_inv_k0 = ccs2im(
                single["im2deep_observed_ccs"].values,
                mz=single["feature_mz"].values,
                charge=1,
            )
            delta = np.abs(single["_pred_inv_k0"].values - obs_inv_k0)
            p95 = float(np.percentile(delta, 95))
            k0_half_win = mob_window_multiplier * p95
            logger.info(
                f"Mobility colocalization: p95 |Δ1/K0| = {p95:.4f} V·s/cm² on "
                f"{len(single)} single-candidate features → window = "
                f"{mob_window_multiplier}× = {k0_half_win:.4f} V·s/cm²"
            )
        else:
            logger.warning(
                f"Mobility colocalization: only {len(single)} single-candidate matches "
                f"with observed CCS — using fallback window {k0_half_win} V·s/cm²."
            )
    else:
        logger.warning(
            "im2deep_observed_ccs not found — using fallback 1/K0 window "
            f"{k0_half_win} V·s/cm²."
        )

    # Load alphatims TimsTOF object and pixel coordinate map
    tdf_path = Path(tdf_path)
    try:
        import alphatims.bruker as atb
    except ImportError:
        logger.warning("alphatims not installed — skipping mobility colocalization.")
        for col in NEW_COLS:
            df[col] = np.nan
        return df

    logger.info(f"Mobility colocalization: loading TDF from {tdf_path}")
    _t0 = time.perf_counter()
    tims = atb.TimsTOF(str(tdf_path))
    logger.info(f"Mobility colocalization: TDF loaded in {time.perf_counter() - _t0:.0f}s")

    with sqlite3.connect(str(tdf_path / "analysis.tdf")) as conn:
        frames_meta = pd.read_sql(
            "SELECT Frame, XIndexPos, YIndexPos FROM MaldiFrameInfo ORDER BY Frame",
            conn,
        ).dropna(subset=["XIndexPos", "YIndexPos"])

    coord_map = {
        int(r.Frame): (int(r.XIndexPos), int(r.YIndexPos))
        for r in frames_meta.itertuples()
    }
    max_x = int(frames_meta["XIndexPos"].max()) + 1
    max_y = int(frames_meta["YIndexPos"].max()) + 1

    mob_arr = tims.mobility_values  # (n_scans,) float64

    # m/z offsets: m0, m1, m2, na, k, chca (same order assumed by Rust)
    _MZ_OFFSET_VALUES = [
        0.0,
        NEUTRON,
        2.0 * NEUTRON,
        _ADDUCT_DELTAS["na"],
        _ADDUCT_DELTAS["k"],
        _ADDUCT_DELTAS["chca"],
    ]

    feat_col = "feature_idx" if "feature_idx" in df.columns else "feature_mz"
    groups = list(df.groupby(feat_col))

    try:
        from ms1rescore_rs import mob_coloc_features as _rs_mob_coloc
        _use_rust = True
    except ImportError:
        _use_rust = False
    logger.info(
        f"Mobility colocalization: {len(groups)} feature groups, "
        f"{len(df)} candidates; backend={'Rust' if _use_rust else 'pure-Python fallback (slow)'}"
    )

    # Per-candidate M0 images collected by whichever path runs; used below for
    # protein-level pairwise r.
    _cand_m0_images: dict = {}

    if _use_rust:
        # ------------------------------------------------------------------ #
        # Rust path: direct push_indptr access + m/z pre-filtering.         #
        #                                                                    #
        # The old approach (one alphatims batch read) collected ALL peaks    #
        # for all pixels into a ~1-5B element array, then lexsorted it —    #
        # typically 5-40 GB of intermediate data and 25+ minutes.           #
        #                                                                    #
        # This approach:                                                     #
        #  1. Reads peak ranges directly from push_indptr (no filter call)  #
        #  2. Builds a TOF-bin boolean mask for the feature windows (~1-3%) #
        #  3. Loads and filters peaks in pixel batches (5 000 pixels each)  #
        #  4. Keeps only the ~1-3% of peaks that fall in a window           #
        #  5. Sorts only the filtered peaks (50-100× fewer)                 #
        # ------------------------------------------------------------------ #
        scan_max     = int(tims.scan_max_index)
        frame_offset = int(tims.zeroth_frame)   # 1 if zeroth frame exists (default), else 0
        push_indptr  = np.asarray(tims.push_indptr, dtype=np.int64)
        tof_max_idx  = int(tims.tof_max_index)
        tof_idx_np   = np.asarray(tims.tof_indices, dtype=np.int32)
        intensity_np = np.asarray(tims.intensity_values, dtype=np.float32)
        mz_arr_np    = np.asarray(tims.mz_values, dtype=np.float32)  # per-TOF-bin
        mob_arr      = tims.mobility_values                           # per-scan 1/K0

        sorted_fids = sorted(coord_map.keys())
        n_pixels    = len(sorted_fids)

        # Peak range per pixel: derived from push_indptr, no alphatims call
        frame_positions = np.array(sorted_fids, dtype=np.int64) - 1 + frame_offset
        push_starts_pp  = frame_positions * scan_max
        push_ends_pp    = (frame_positions + 1) * scan_max
        peak_starts_pp  = push_indptr[push_starts_pp]
        peak_ends_pp    = push_indptr[push_ends_pp]
        peak_counts_pp  = peak_ends_pp - peak_starts_pp

        # Build TOF-bin filter mask: mark every bin that falls inside at least
        # one of the 6 extraction windows for any feature.
        _t0 = time.perf_counter()
        ppm_factor   = extraction_ppm * 1e-6
        relevant_tof = np.zeros(tof_max_idx, dtype=np.bool_)
        for _, grp in groups:
            feat_mz = float(grp["feature_mz"].iloc[0])
            for offset in _MZ_OFFSET_VALUES:
                qmz = feat_mz + offset
                lo = int(np.searchsorted(mz_arr_np, float(qmz * (1.0 - ppm_factor)), "left"))
                hi = int(np.searchsorted(mz_arr_np, float(qmz * (1.0 + ppm_factor)), "right"))
                if lo < hi:
                    relevant_tof[lo:hi] = True

        n_rel = int(relevant_tof.sum())
        logger.debug(
            f"Mobility colocalization: TOF-bin mask built in "
            f"{time.perf_counter() - _t0:.0f}s — {n_pixels} pixels, "
            f"{n_rel}/{tof_max_idx} TOF bins in extraction windows "
            f"({100.0 * n_rel / max(tof_max_idx, 1):.1f}%); loading filtered peaks…"
        )

        # Process pixels in batches of _BATCH to bound peak memory per iteration
        _t0 = time.perf_counter()
        _BATCH = 5000
        _n_batches = (n_pixels + _BATCH - 1) // _BATCH
        all_pix:   list[np.ndarray] = []
        all_mzs_l: list[np.ndarray] = []
        all_scn_l: list[np.ndarray] = []
        all_int_l: list[np.ndarray] = []

        for _bi, b0 in enumerate(range(0, n_pixels, _BATCH), start=1):
            _log_progress("Mobility colocalization: loading filtered peaks", _bi, _n_batches, _t0, every=max(1, _n_batches // 20))
            b1        = min(b0 + _BATCH, n_pixels)
            b_starts  = peak_starts_pp[b0:b1]
            b_counts  = peak_counts_pp[b0:b1]
            b_total   = int(b_counts.sum())
            if b_total == 0:
                continue

            # Vectorised global raw peak indices for this batch
            b_out_starts = np.concatenate([[np.int64(0)], np.cumsum(b_counts[:-1])])
            b_base       = np.repeat(b_starts - b_out_starts, b_counts)
            b_raw        = np.arange(b_total, dtype=np.int64) + b_base

            # Filter to relevant m/z windows via TOF-bin index
            b_tof  = tof_idx_np[b_raw]
            b_mask = relevant_tof[b_tof.astype(np.int64)]
            if not b_mask.any():
                continue

            b_raw_f = b_raw[b_mask]
            b_pix   = np.repeat(np.arange(b0, b1, dtype=np.int32), b_counts)[b_mask]
            b_tof_f = b_tof[b_mask].astype(np.int64)

            b_mzs  = mz_arr_np[b_tof_f]
            b_ints = intensity_np[b_raw_f]
            b_push = np.searchsorted(push_indptr, b_raw_f, side="right") - 1
            b_scns = (b_push % scan_max).astype(np.uint32)

            all_pix.append(b_pix)
            all_mzs_l.append(b_mzs)
            all_scn_l.append(b_scns)
            all_int_l.append(b_ints)

        if not all_pix:
            logger.warning("No peaks found in any extraction window — skipping mob coloc.")
            for col in NEW_COLS:
                df[col] = np.nan
            return df

        pix_ids  = np.concatenate(all_pix)
        mzs_all  = np.concatenate(all_mzs_l)
        scan_ids = np.concatenate(all_scn_l)
        ints_all = np.concatenate(all_int_l)

        total_filtered = len(mzs_all)
        total_all      = int(peak_counts_pp.sum())
        n_workers = min(os.cpu_count() or 1, 16)
        logger.info(
            f"Mobility colocalization: peak loading done in "
            f"{time.perf_counter() - _t0:.0f}s — {total_filtered:,} relevant peaks "
            f"({100.0 * total_filtered / max(total_all, 1):.1f}% of {total_all:,} total); "
            f"sorting per-pixel m/z across {n_workers} threads…"
        )
        _t0 = time.perf_counter()

        # Rust binary-searches per-pixel m/z windows, so peaks must be ordered by
        # (pixel_id, mz). pix_ids is ALREADY globally non-decreasing here: the batch
        # loop above emits pixels in ascending order and b_pix is ascending within
        # each batch, so a global lexsort over ~2.5e9 rows is wasted work — only the
        # m/z axis needs sorting *within* each pixel. Compute the CSR pixel offsets
        # from the already-sorted pix_ids, then sort each pixel's disjoint m/z slice
        # in parallel (numpy argsort releases the GIL; disjoint slice writes are
        # race-free). This replaces a single-threaded ~11 min lexsort with a parallel
        # segmented sort that also does strictly less total work (Σ k·log k over
        # pixels, not N·log N).
        # ponytail: relies on pix_ids being pixel-sorted by construction above; if
        # that batch loop changes, restore `np.lexsort((mzs_all, pix_ids))`.
        pixel_offsets_arr = np.searchsorted(pix_ids, np.arange(n_pixels + 1), side="left")

        def _sort_pixel_slices(pxs):
            for px in pxs:
                s = int(pixel_offsets_arr[px])
                e = int(pixel_offsets_arr[px + 1])
                if e - s > 1:
                    loc = np.argsort(mzs_all[s:e], kind="stable")
                    mzs_all[s:e]  = mzs_all[s:e][loc]
                    scan_ids[s:e] = scan_ids[s:e][loc]
                    ints_all[s:e] = ints_all[s:e][loc]

        if n_pixels > 1:
            with ThreadPoolExecutor(max_workers=n_workers) as _ex:
                list(_ex.map(_sort_pixel_slices,
                             np.array_split(np.arange(n_pixels), n_workers * 4)))
        logger.info(f"Mobility colocalization: per-pixel sort done in {time.perf_counter() - _t0:.0f}s")

        # CSR offsets: pixel_offsets[px] … pixel_offsets[px+1] = peak range for pixel px
        pixel_offsets = pixel_offsets_arr.astype(np.uint64).tolist()

        # Pixel coordinates in the same order as sorted_fids
        pixel_xi_list = [coord_map[fid][0] for fid in sorted_fids]
        pixel_yi_list = [coord_map[fid][1] for fid in sorted_fids]

        flat_mzs   = mzs_all
        flat_scans = scan_ids
        flat_ints  = ints_all

        # Build feature m/z windows: flat array (n_features × 6 × 2)
        # ppm_factor already defined above
        n_features = len(groups)
        feature_mz_windows = np.empty(n_features * 6 * 2, dtype=np.float32)

        # Build candidate CSR arrays
        _t0 = time.perf_counter()
        cand_ptr: list[int] = [0]
        cand_k0_lo_list: list[float] = []
        cand_k0_hi_list: list[float] = []
        cand_df_indices: list = []  # df label indices in order

        for fi, (_, grp) in enumerate(groups):
            feat_mz = float(grp["feature_mz"].iloc[0])
            win_base = fi * 12
            for oi, offset in enumerate(_MZ_OFFSET_VALUES):
                qmz = feat_mz + offset
                feature_mz_windows[win_base + oi * 2]     = float(qmz * (1.0 - ppm_factor))
                feature_mz_windows[win_base + oi * 2 + 1] = float(qmz * (1.0 + ppm_factor))

            cand_indices = grp.index.tolist()
            cand_df_indices.extend(cand_indices)
            cand_ptr.append(cand_ptr[-1] + len(cand_indices))
            for idx in cand_indices:
                pred_k0 = df.at[idx, "_pred_inv_k0"]
                if pd.isna(pred_k0):
                    cand_k0_lo_list.append(float("nan"))
                    cand_k0_hi_list.append(float("nan"))
                else:
                    cand_k0_lo_list.append(float(pred_k0) - k0_half_win)
                    cand_k0_hi_list.append(float(pred_k0) + k0_half_win)

        logger.info(
            f"Mobility colocalization: candidate CSR built in "
            f"{time.perf_counter() - _t0:.0f}s — processing {n_features} feature groups "
            f"({len(cand_df_indices)} candidates) via Rust…"
        )
        _t0 = time.perf_counter()
        raw = _rs_mob_coloc(
            flat_mzs,
            flat_scans,
            flat_ints,
            pixel_offsets,
            pixel_xi_list,
            pixel_yi_list,
            mob_arr.astype(np.float64).tolist(),
            feature_mz_windows,
            cand_ptr,
            cand_k0_lo_list,
            cand_k0_hi_list,
            max_x,
            max_y,
        )
        logger.info(f"Mobility colocalization: Rust kernel done in {time.perf_counter() - _t0:.0f}s")
        raw_mat = np.asarray(raw, dtype=np.float64).reshape(len(cand_df_indices), 10)

        for col_i, col in enumerate(NEW_COLS):
            df[col] = pd.Series(raw_mat[:, col_i], index=cand_df_indices, dtype=np.float64)

        # Rebuild per-candidate M0 images from the already-loaded flat arrays so
        # protein-level pairwise r can be computed without a second TDF read.
        # Only build images for candidates in proteins with ≥2 detected members
        # (singletons cannot contribute to within-protein r).
        if protein_coloc and "protein" in df.columns:
            _prot_sizes = df.groupby("protein").size()
            _multi_prots = set(_prot_sizes[_prot_sizes >= 2].index)
            _needs_img = frozenset(
                df.index[
                    df["protein"].isin(_multi_prots)
                    & df["_pred_inv_k0"].notna()
                ]
            )
        else:
            _needs_img = frozenset()

        if _needs_img:
            pixel_xi_arr = np.array(pixel_xi_list, dtype=np.int32)
            pixel_yi_arr = np.array(pixel_yi_list, dtype=np.int32)
            _n_pix_total = max_y * max_x
            _pred_inv_k0_map = df["_pred_inv_k0"].to_dict()

            def _build_group_images(grp):
                cands_needed = [i for i in grp.index if i in _needs_img]
                if not cands_needed:
                    return []
                feat_mz = float(grp["feature_mz"].iloc[0])
                mz_lo = feat_mz * (1.0 - ppm_factor)
                mz_hi = feat_mz * (1.0 + ppm_factor)
                # NB: this is a full O(len(mzs_all)) scan (mzs_all is sorted
                # per-pixel, not globally, so no binary search shortcut here) —
                # ~2-3 billion elements on a whole-tissue section. Dispatched
                # across threads below because numpy's comparison/logical-and
                # ufuncs release the GIL for arrays past a small internal
                # threshold, so concurrent calls genuinely run on separate
                # cores. This was previously the dominant single-threaded cost
                # of mobility colocalization (hours on a dense whole section).
                # ponytail: still O(N) per feature; a global mz-sort + binary
                # search per feature would cut this further, at the cost of a
                # full extra sorted copy of mzs_all/pix_ids/scan_ids/ints_all.
                # Revisit if threading alone isn't enough headroom.
                mz_mask = (mzs_all >= mz_lo) & (mzs_all <= mz_hi)
                if not mz_mask.any():
                    return []
                sub_pix  = pix_ids[mz_mask]
                sub_mob  = mob_arr[scan_ids[mz_mask]]
                # bincount requires float64 weights
                sub_ints = ints_all[mz_mask].astype(np.float64)
                flat_xy  = (
                    pixel_yi_arr[sub_pix].astype(np.int64) * max_x
                    + pixel_xi_arr[sub_pix].astype(np.int64)
                )
                out = []
                for idx in cands_needed:
                    pred_k0 = _pred_inv_k0_map[idx]
                    k0_mask = (
                        (sub_mob >= pred_k0 - k0_half_win)
                        & (sub_mob <= pred_k0 + k0_half_win)
                    )
                    if k0_mask.any():
                        img_flat = np.bincount(
                            flat_xy[k0_mask],
                            weights=sub_ints[k0_mask],
                            minlength=_n_pix_total,
                        ).astype(np.float32)
                    else:
                        img_flat = np.zeros(_n_pix_total, dtype=np.float32)
                    out.append((idx, img_flat.reshape(max_y, max_x)))
                return out

            _needed_groups = [grp for _, grp in groups if any(i in _needs_img for i in grp.index)]
            _n_groups_needed = len(_needed_groups)
            _n_workers = min(os.cpu_count() or 1, 64)
            logger.info(
                f"Mobility colocalization: rebuilding per-candidate M0 images for "
                f"{_n_groups_needed} feature groups across {_n_workers} threads "
                f"(this is the O(N) full-array-scan step — watch the progress line below "
                f"for how it's tracking)…"
            )
            _t0 = time.perf_counter()
            with ThreadPoolExecutor(max_workers=_n_workers) as _ex:
                for _gi, _group_results in enumerate(_ex.map(_build_group_images, _needed_groups), start=1):
                    _log_progress(
                        "Mobility colocalization: M0 image rebuild",
                        _gi, _n_groups_needed, _t0, every=max(1, _n_groups_needed // 20),
                    )
                    for idx, img in _group_results:
                        _cand_m0_images[idx] = img
            logger.info(
                f"Mobility colocalization: M0 image rebuild done in "
                f"{time.perf_counter() - _t0:.0f}s ({len(_cand_m0_images)} images)"
            )

    else:
        # ------------------------------------------------------------------ #
        # Python fallback: per-feature read with pre-read optimisation        #
        # ------------------------------------------------------------------ #
        _ones = np.ones((1, max_y, max_x), dtype=np.float32)
        _neighbor_counts = _neighbor_sum_batch(_ones)[0]
        _N = max_y * max_x
        _W_sum = float(_neighbor_counts.sum())

        _MZ_OFFSETS_DICT = {
            "m0": _MZ_OFFSET_VALUES[0],
            "m1": _MZ_OFFSET_VALUES[1],
            "m2": _MZ_OFFSET_VALUES[2],
            "na": _MZ_OFFSET_VALUES[3],
            "k":  _MZ_OFFSET_VALUES[4],
            "chca": _MZ_OFFSET_VALUES[5],
        }
        ppm_factor = extraction_ppm * 1e-6
        result_rows: dict = {}

        _n_groups_fallback = len(groups)
        logger.info(
            f"Mobility colocalization: processing {_n_groups_fallback} feature groups "
            f"via the pure-Python fallback (no Rust extension found — this re-reads "
            f"the TDF per pixel per feature and is far slower; install/build "
            f"ms1rescore_rs for the Rust path)…"
        )
        _t0 = time.perf_counter()

        for _fi, (feat_key, grp) in enumerate(groups, start=1):
            _log_progress(
                "Mobility colocalization: fallback loop",
                _fi, _n_groups_fallback, _t0, every=max(1, _n_groups_fallback // 20),
            )
            feat_mz = float(grp["feature_mz"].iloc[0])
            cand_indices = grp.index.tolist()

            mz_ranges = {
                key: (
                    (feat_mz + offset) * (1.0 - ppm_factor),
                    (feat_mz + offset) * (1.0 + ppm_factor),
                )
                for key, offset in _MZ_OFFSETS_DICT.items()
            }

            pixel_data: dict[str, list] = {key: [] for key in _MZ_OFFSETS_DICT}
            for fid, (xi, yi) in coord_map.items():
                fd = tims[int(fid), :, :]
                if not len(fd):
                    continue
                mz_vals = fd["mz_values"].values
                scan_idx = fd["scan_indices"].values
                int_vals = fd["intensity_values"].values
                for key, (mz_lo, mz_hi) in mz_ranges.items():
                    mz_mask = (mz_vals >= mz_lo) & (mz_vals <= mz_hi)
                    if not mz_mask.any():
                        continue
                    pixel_data[key].append((xi, yi, scan_idx[mz_mask], int_vals[mz_mask]))

            for idx in cand_indices:
                pred_k0 = df.at[idx, "_pred_inv_k0"]
                if pd.isna(pred_k0):
                    result_rows[idx] = {c: np.nan for c in NEW_COLS}
                    continue

                scan_in = (mob_arr >= pred_k0 - k0_half_win) & (mob_arr <= pred_k0 + k0_half_win)

                imgs: dict[str, np.ndarray] = {}
                for key, hits in pixel_data.items():
                    img = np.zeros((max_y, max_x), dtype=np.float32)
                    for xi, yi, s_idx, i_vals in hits:
                        mask = scan_in[s_idx]
                        if mask.any():
                            img[yi, xi] += i_vals[mask].sum()
                    imgs[key] = img

                r_m1 = _pearson_r_images(imgs["m0"], imgs["m1"])
                r_m2 = _pearson_r_images(imgs["m0"], imgs["m2"])
                valid_r = [v for v in (r_m1, r_m2) if not np.isnan(v)]
                r_mean = float(np.mean(valid_r)) if valid_r else np.nan

                m0 = imgs["m0"]
                if protein_coloc:
                    _cand_m0_images[idx] = m0
                nonzero = m0[m0 > 0]
                cv = float(nonzero.std() / nonzero.mean()) if len(nonzero) > 1 else 0.0
                morans_val, _ = _morans_gearys_chunk(m0[np.newaxis], _neighbor_counts, _N, _W_sum)

                result_rows[idx] = {
                    "isotope_colocalization_m1_mob":   r_m1,
                    "isotope_colocalization_m2_mob":   r_m2,
                    "isotope_colocalization_mean_mob": r_mean,
                    "adduct_colocalization_na_mob":    _pearson_r_images(imgs["m0"], imgs["na"]),
                    "adduct_colocalization_k_mob":     _pearson_r_images(imgs["m0"], imgs["k"]),
                    "adduct_colocalization_chca_mob":  _pearson_r_images(imgs["m0"], imgs["chca"]),
                    "fraction_detected_mob":    float((m0 > 0).mean()),
                    "intensity_cv_mob":         cv,
                    "log_mean_intensity_mob":   float(np.log1p(m0.mean())),
                    "spatial_morans_i_mob":     float(morans_val[0]),
                }

        logger.info(f"Mobility colocalization: fallback loop done in {time.perf_counter() - _t0:.0f}s")
        result_df = pd.DataFrame.from_dict(result_rows, orient="index")
        for col in NEW_COLS:
            df[col] = result_df[col]

    # NaN fill with column median (same policy for both paths)
    for col in NEW_COLS:
        valid = df[col].dropna()
        df[col] = df[col].fillna(float(valid.median()) if len(valid) > 0 else 0.0)

    # Protein-level pairwise colocalization using mobility-gated M0 images.
    # Only added when protein_coloc=True; omitted entirely otherwise so the
    # columns are absent and the ranker's present-column filter ignores them.
    if protein_coloc:
        PROT_MOB_COLS = [
            "protein_colocalization_mob",
            "protein_colocalization_mob_max",
            "protein_colocalization_mob_n_partners",
        ]
        for col in PROT_MOB_COLS:
            df[col] = np.nan

    if protein_coloc and "protein" in df.columns and _cand_m0_images:
        _prot_groups = list(df.groupby("protein"))
        _n_prots = len(_prot_groups)
        _t0 = time.perf_counter()
        logger.info(f"Mobility colocalization: protein-pairwise coloc for {_n_prots} proteins…")
        for _pi, (_prot, _grp) in enumerate(_prot_groups, start=1):
            _log_progress(
                "Mobility colocalization: protein-pairwise coloc",
                _pi, _n_prots, _t0, every=max(1, _n_prots // 20),
            )
            valid_idx = [i for i in _grp.index if i in _cand_m0_images]
            n = len(valid_idx)
            if n == 0:
                continue
            if n == 1:
                df.at[valid_idx[0], "protein_colocalization_mob_n_partners"] = 0.0
                continue
            # Vectorised Pearson r: L2-normalise then sgemm for the full n×n matrix
            X = np.stack(
                [_cand_m0_images[i].ravel() for i in valid_idx], axis=0
            ).astype(np.float32)                       # (n, H*W)
            if tic_normalize:
                col_sum = X.sum(axis=0)
                X /= np.where(col_sum > 0, col_sum, 1.0)[np.newaxis, :]
            X -= X.mean(axis=1, keepdims=True)         # centre each image
            norms = np.linalg.norm(X, axis=1)          # (n,)
            good = norms > 1e-10
            X[~good] = 0.0
            norms = np.where(good, norms, 1.0)
            X /= norms[:, np.newaxis]
            C = X @ X.T                                # (n, n) Pearson r matrix
            np.fill_diagonal(C, np.nan)
            for ci, idx in enumerate(valid_idx):
                row = C[ci]
                partners = row[~np.isnan(row)]
                df.at[idx, "protein_colocalization_mob_n_partners"] = float(len(partners))
                if len(partners) > 0:
                    df.at[idx, "protein_colocalization_mob"] = float(partners.mean())
                    df.at[idx, "protein_colocalization_mob_max"] = float(partners.max())
        df["protein_colocalization_mob_n_partners"] = (
            df["protein_colocalization_mob_n_partners"].fillna(0.0)
        )
        logger.info(f"Mobility colocalization: protein-pairwise coloc done in {time.perf_counter() - _t0:.0f}s")

    df = df.drop(columns=["_pred_inv_k0"])
    logger.info(
        f"Mobility colocalization: computed 10 per-candidate + 3 protein-level "
        f"features for {len(df)} candidates across {len(groups)} features "
        f"in {time.perf_counter() - _t_func_start:.0f}s total."
    )
    return df


def compute_spatial_autocorrelation_full(
    df: pd.DataFrame,
    ion_images: np.ndarray,
    ion_image_mzs: np.ndarray,
    chunk_size: int = 200,
    n_workers: int | None = None,
) -> pd.DataFrame:
    """
    Compute Moran's I and Geary's C for each MALDI feature ion image (E5/E6).

    Replaces the per-feature scipy.signal.convolve2d loop with a batched
    numpy 8-neighbour sum over all features simultaneously.  Features are
    processed in chunks of ``chunk_size`` and chunks are dispatched to a
    thread pool (numpy releases the GIL for large array ops, so threads run
    in parallel without pickling overhead).

    Features added
    --------------
    spatial_morans_i    Moran's I (positive = clustered, negative = dispersed)
    spatial_gearys_c    Geary's C (< 1 clustered, ≈ 1 random, > 1 dispersed)
    """
    if ion_images is None or len(ion_images) == 0:
        return df

    _, height, width = ion_images.shape
    N = height * width

    # Precompute neighbour counts per pixel — same for all features.
    ones = np.ones((1, height, width), dtype=np.float32)
    neighbor_counts = _neighbor_sum_batch(ones)[0]  # (H, W) float32
    W_sum = float(neighbor_counts.sum())

    mz_arr = np.asarray(ion_image_mzs, dtype=np.float64)
    mz_to_img_idx = {float(mz): i for i, mz in enumerate(mz_arr)}

    unique_feat_mzs = df["feature_mz"].unique()
    feat_img_indices = np.array(
        [mz_to_img_idx[float(mz)] for mz in unique_feat_mzs if float(mz) in mz_to_img_idx],
        dtype=np.intp,
    )
    feat_mzs_used = np.array(
        [float(mz) for mz in unique_feat_mzs if float(mz) in mz_to_img_idx],
        dtype=np.float64,
    )
    n_feat = len(feat_img_indices)

    import os
    from concurrent.futures import ThreadPoolExecutor

    if n_workers is None:
        n_workers = os.cpu_count() or 1

    morans_vals = np.zeros(n_feat, dtype=np.float64)
    gearys_vals = np.ones(n_feat, dtype=np.float64)

    # Build chunk list
    chunks = [
        (start, min(start + chunk_size, n_feat))
        for start in range(0, n_feat, chunk_size)
    ]

    def _process_chunk(start_end):
        start, end = start_end
        chunk_imgs = ion_images[feat_img_indices[start:end]]
        return start, end, _morans_gearys_chunk(chunk_imgs, neighbor_counts, N, W_sum)

    with ThreadPoolExecutor(max_workers=min(n_workers, len(chunks))) as pool:
        for start, end, (m, g) in pool.map(_process_chunk, chunks):
            morans_vals[start:end] = m
            gearys_vals[start:end] = g

    morans_map = dict(zip(feat_mzs_used, morans_vals))
    gearys_map = dict(zip(feat_mzs_used, gearys_vals))

    df["spatial_morans_i"] = df["feature_mz"].map(morans_map).fillna(0.0)
    df["spatial_gearys_c"] = df["feature_mz"].map(gearys_map).fillna(1.0)

    logger.info(f"Spatial autocorrelation (E5/E6): computed for {n_feat} features")
    return df
