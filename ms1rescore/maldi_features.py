"""
MALDI-side feature computation: mass accuracy, protein consistency, spatial,
isotope envelope, co-localization, ionization properties.

All functions are symmetric — no is_decoy branching in feature computation.
"""

import logging

import numpy as np
import pandas as pd

from ms1rescore.utils import (
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
    df["protein_coverage"] = df["protein_n_features"] / total_peps

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


def _pearson_r_matrix(
    ion_images: np.ndarray,
    ion_image_mzs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Compute the full (n_valid × n_valid) Pearson correlation matrix.

    Uses a manual float32 normalise + BLAS sgemm instead of ``np.corrcoef``
    which unconditionally upcasts to float64, allocating ~2× as much memory.
    For 1398 features × 49 K pixels the saving is ~550 MB at peak.

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
        agg = (
            pairs.groupby(["feature_mz", "protein"])["r"]
            .agg(
                protein_colocalization="mean",
                protein_colocalization_max="max",
                protein_colocalization_median="median",
                protein_colocalization_n_partners="count",
            )
            .reset_index()
        )
        df = df.merge(agg, on=["feature_mz", "protein"], how="left")
    else:
        for col in ["protein_colocalization", "protein_colocalization_max",
                    "protein_colocalization_median", "protein_colocalization_n_partners"]:
            df[col] = 0.0

    for col in ["protein_colocalization", "protein_colocalization_max",
                "protein_colocalization_median", "protein_colocalization_n_partners"]:
        df[col] = df[col].fillna(0.0)

    n_scored = int((df["protein_colocalization"] != 0).sum())
    logger.info(
        f"Co-localization: {len(valid_mzs)} valid features, "
        f"{len(pairs) if len(pairs) > 0 else 0} within-protein pairs, "
        f"{n_scored}/{len(df)} candidates scored"
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

    finetuned_model = _im2deep_finetune(cal_psm_list, model_save_path=model_path, epochs=40)
    logger.info(
        f"  IM2Deep transfer learning on {n_cal} unique peptides "
        f"({n_cal_rows} single-candidate features): model saved to {model_path}"
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


def compute_im2deep_features(
    df: pd.DataFrame,
    observed_ccs_per_feature: dict | None = None,
    calibration_method: str = "linear",
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
    # im2deep_mahalanobis         Mahalanobis distance in (ppm_error_abs, delta_ccs) space
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

    # Calibrate predicted CCS using single-candidate features (n_candidates == 1)
    # as unambiguous reference points.  Three methods:
    #   linear  — global additive shift (LinearCCSCalibration, default)
    #   spline  — piecewise spline mapping (SplineCCSCalibration)
    #   finetune — transfer-learning re-training of the neural network weights
    predicted_cal = predicted.copy()
    if "n_candidates" in df.columns:
        try:
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
                            f"({valid_cal.sum()} single-candidate features)"
                        )
                    else:  # linear (default)
                        cal = _LinearCal(per_charge=False, use_charge_state=1)
                        cal.fit(psm_df_target, psm_df_source)
                        predicted_cal = np.array(
                            list(cal.transform(df_transform)), dtype=float
                        )
                        logger.info(
                            f"  IM2Deep linear calibration on {n_cal} unique peptides "
                            f"({valid_cal.sum()} single-candidate features): "
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

    # # Mahalanobis distance in (ppm_error_abs, delta_ccs) space per feature
    # mahal = np.full(len(df), np.nan)
    # df_reset = df.reset_index(drop=False)
    # for _, grp in df_reset.groupby("feature_mz"):
    #     if len(grp) < 3:
    #         continue
    #     X = grp[["ppm_error_abs", "_im2deep_delta"]].values.astype(float)
    #     valid_mask = ~np.isnan(X).any(axis=1)
    #     X_valid = X[valid_mask]
    #     if X_valid.shape[0] < 3:
    #         continue
    #     try:
    #         cov = np.cov(X_valid.T)
    #         if np.linalg.matrix_rank(cov) < 2:
    #             continue
    #         cov_inv = np.linalg.inv(cov)
    #         mean = X_valid.mean(axis=0)
    #         for row_pos, orig_idx in enumerate(grp.index):
    #             if valid_mask[row_pos]:
    #                 diff = X[row_pos] - mean
    #                 mahal[orig_idx] = float(np.sqrt(diff @ cov_inv @ diff))
    #     except np.linalg.LinAlgError:
    #         continue

    # df["im2deep_mahalanobis"] = mahal
    df.drop(columns=["_im2deep_delta", "_im2deep_abspct"], inplace=True, errors="ignore")

    # Fill NaN with median (symmetric fill — no target/decoy information)
    for col in [
        "im2deep_delta_ccs", "im2deep_abs_delta_ccs_pct",
        "im2deep_ccs_zscore", "im2deep_ccs_rank",
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

def _pearson_r_pairwise(images_a: np.ndarray, images_b: np.ndarray) -> np.ndarray:
    """Per-feature Pearson r between corresponding images in two (N, H, W) arrays."""
    n = len(images_a)
    a = images_a.reshape(n, -1).astype(np.float32)
    b = images_b.reshape(n, -1).astype(np.float32)
    a -= a.mean(axis=1, keepdims=True)
    b -= b.mean(axis=1, keepdims=True)
    num = (a * b).sum(axis=1)
    denom = np.sqrt((a * a).sum(axis=1)) * np.sqrt((b * b).sum(axis=1))
    return np.where(denom > 1e-10, num / denom, np.nan).astype(np.float32)


def compute_isotopologue_colocalization(
    df: pd.DataFrame,
    ion_images: np.ndarray,
    ion_image_mzs: np.ndarray,
    ppm_tolerance: float = 10.0,
    _corr_cache: tuple | None = None,
    extra_ion_images: dict | None = None,
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
        mz_arr = np.asarray(ion_image_mzs, dtype=np.float64)
        valid_mask = ion_images.reshape(len(mz_arr), -1).std(axis=1) > 1e-10
        r_m1_arr = _pearson_r_pairwise(ion_images[valid_mask], m1_images[valid_mask])
        r_m2_arr = _pearson_r_pairwise(ion_images[valid_mask], m2_images[valid_mask])
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
        df[col] = df["feature_mz"].map(mapping)
        valid = df[col].dropna()
        df[col] = df[col].fillna(float(valid.median()) if len(valid) > 0 else 0.0)

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
        mz_arr = np.asarray(ion_image_mzs, dtype=np.float64)
        valid_mask = ion_images.reshape(len(mz_arr), -1).std(axis=1) > 1e-10
        m0_valid = ion_images[valid_mask]
        for adduct_name in _ADDUCT_DELTAS:
            adduct_imgs = extra_ion_images.get(adduct_name)
            if adduct_imgs is not None:
                r_arr = _pearson_r_pairwise(m0_valid, adduct_imgs[valid_mask])
            else:
                r_arr = np.full(n_valid, np.nan, dtype=np.float32)
            mapping = {float(mz): float(r) for mz, r in zip(valid_mz_arr, r_arr)}
            col = f"adduct_colocalization_{adduct_name}"
            df[col] = df["feature_mz"].map(mapping)
            valid_col = df[col].dropna()
            df[col] = df[col].fillna(float(valid_col.median()) if len(valid_col) > 0 else 0.0)
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
            col = f"adduct_colocalization_{adduct_name}"
            df[col] = df["feature_mz"].map(mapping)
            valid_col = df[col].dropna()
            df[col] = df[col].fillna(float(valid_col.median()) if len(valid_col) > 0 else 0.0)
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
