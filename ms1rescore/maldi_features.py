"""
MALDI-side feature computation: mass accuracy, protein consistency, spatial,
isotope envelope, co-localization, ionization properties.

All functions are symmetric — no is_decoy branching in feature computation.
"""

import logging

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)

# Kyte-Doolittle hydropathy scale
_KD_SCALE = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5,
    "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
    "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}


def compute_mass_accuracy_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute ppm_error_abs, ppm_rank, ppm_best_ratio within feature groups."""
    df["ppm_rank"] = df.groupby("feature_mz")["ppm_error_abs"].rank(method="min")
    best_ppm = df.groupby("feature_mz")["ppm_error_abs"].transform("min")
    df["ppm_best_ratio"] = df["ppm_error_abs"] / best_ppm.clip(lower=1e-6)
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
    """Compute peptide_length, n_missed_cleavages, has_modifications."""
    df["peptide_length"] = df["peptide"].str.len()

    def _count_missed_cleavages(seq):
        count = 0
        for i, aa in enumerate(seq[:-1]):  # exclude last residue
            if aa in "KR" and (i + 1 >= len(seq) or seq[i + 1] != "P"):
                count += 1
        return count

    df["n_missed_cleavages"] = df["peptide"].apply(_count_missed_cleavages)
    df["has_modifications"] = 0  # plain sequences from digest have no mods
    return df


def compute_maldi_signal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute log_maldi_intensity from feature_intensity."""
    if "feature_intensity" in df.columns:
        df["log_maldi_intensity"] = np.log1p(df["feature_intensity"])
    else:
        df["log_maldi_intensity"] = 0.0
    return df


def compute_maldi_ionization_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute MALDI-specific ionization features from peptide sequence."""
    seqs = df["peptide"].values
    n = len(seqs)

    n_arg = np.zeros(n)
    n_basic = np.zeros(n)
    n_phe = np.zeros(n)
    n_aromatic = np.zeros(n)
    gravy = np.zeros(n)
    charge_proxy = np.zeros(n)

    for i, seq in enumerate(seqs):
        n_arg[i] = seq.count("R")
        n_basic[i] = seq.count("R") + seq.count("K") + seq.count("H")
        n_phe[i] = seq.count("F")
        n_aromatic[i] = seq.count("F") + seq.count("W") + seq.count("Y")
        if len(seq) > 0:
            gravy[i] = sum(_KD_SCALE.get(aa, 0) for aa in seq) / len(seq)
        charge_proxy[i] = seq.count("R") + seq.count("K") + seq.count("H") + 1

    df["n_arginine"] = n_arg
    df["n_basic_residues"] = n_basic
    df["n_phenylalanine"] = n_phe
    df["n_aromatic"] = n_aromatic
    df["gravy_score"] = gravy
    df["charge_proxy"] = charge_proxy
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


def compute_colocalization_features(
    df: pd.DataFrame,
    ion_images: np.ndarray,
    ion_image_mzs: np.ndarray,
) -> pd.DataFrame:
    """Compute protein co-localization features from MALDI ion images.

    Pre-computes a correlation cache keyed by (mz_a, mz_b) so that
    Pearson r is computed once per unique feature pair, not per candidate.
    """
    mz_to_idx = {mz: i for i, mz in enumerate(ion_image_mzs)}

    # Pre-flatten all ion images once
    flat_images = {}
    valid_mzs = set()
    for mz, idx in mz_to_idx.items():
        flat = ion_images[idx].flatten().astype(np.float64)
        if flat.std() > 0:
            flat_images[mz] = flat
            valid_mzs.add(mz)

    # Collect all (mz_a, mz_b) pairs that share a protein
    protein_to_mzs = (
        df.groupby("protein")["feature_mz"]
        .apply(lambda x: list(x.unique()))
        .to_dict()
    )
    pairs_needed = set()
    for mzs in protein_to_mzs.values():
        mzs_valid = [mz for mz in mzs if mz in valid_mzs]
        for i, a in enumerate(mzs_valid):
            for b in mzs_valid[i + 1 :]:
                pairs_needed.add((a, b))

    # Compute Pearson r once per unique pair
    corr_cache = {}
    for a, b in pairs_needed:
        r = float(np.corrcoef(flat_images[a], flat_images[b])[0, 1])
        corr_cache[(a, b)] = r
        corr_cache[(b, a)] = r

    logger.info(
        f"Co-localization: pre-computed {len(pairs_needed)} unique feature-pair correlations"
    )

    # Look up per candidate (fast dict lookups, no recomputation)
    n = len(df)
    mean_scores = np.zeros(n)
    max_scores = np.zeros(n)
    median_scores = np.zeros(n)
    n_partners = np.zeros(n)

    for i, (_, row) in enumerate(df.iterrows()):
        this_mz = row["feature_mz"]
        if this_mz not in valid_mzs:
            continue

        other_mzs = [
            mz
            for mz in protein_to_mzs.get(row["protein"], [])
            if mz != this_mz and mz in valid_mzs
        ]
        if not other_mzs:
            continue

        correlations = [corr_cache[(this_mz, mz)] for mz in other_mzs if (this_mz, mz) in corr_cache]
        if correlations:
            mean_scores[i] = float(np.mean(correlations))
            max_scores[i] = float(np.max(correlations))
            median_scores[i] = float(np.median(correlations))
            n_partners[i] = len(correlations)

    df["protein_colocalization"] = mean_scores
    df["protein_colocalization_max"] = max_scores
    df["protein_colocalization_median"] = median_scores
    df["protein_colocalization_n_partners"] = n_partners

    n_scored = np.count_nonzero(mean_scores)
    logger.info(f"Co-localization: {n_scored}/{n} candidates scored")
    return df


def compute_theoretical_isotope_features(
    df: pd.DataFrame,
    maldi_envelopes: dict | None = None,
    lcms_envelopes: dict | None = None,
) -> pd.DataFrame:
    """
    Compute sequence-specific theoretical isotope features.

    Vectorized: uses pre-computed n_C/n_H/n_N/n_O/n_S and mass columns
    from the digest DataFrame. No pyteomics calls in the hot path.
    """
    n = len(df)

    # --- Vectorized: sequence-specific theoretical isotope distribution ---
    nc = df["n_C"].values.astype(float)
    nh = df["n_H"].values.astype(float)
    nn = df["n_N"].values.astype(float)
    no = df["n_O"].values.astype(float)
    ns = df["n_S"].values.astype(float)

    lam = nc * 0.01109 + nh * 0.000115 + nn * 0.00364 + no * 0.00205 + ns * 0.04493
    # Poisson: P(k) = exp(-lam) * lam^k / k!
    theo_m0 = np.exp(-lam)
    theo_m1 = theo_m0 * lam
    theo_m2 = theo_m0 * lam**2 / 2.0
    theo_total = theo_m0 + theo_m1 + theo_m2
    theo_total = np.where(theo_total > 0, theo_total, 1.0)
    theo_m0 /= theo_total
    theo_m1 /= theo_total
    theo_m2 /= theo_total

    df["theo_has_sulfur"] = (ns > 0).astype(float)

    # --- Vectorized: averagine theoretical ---
    pep_mass = df["mass"].values.astype(float)
    nc_avg = np.round(pep_mass * 0.0444).astype(float)
    nh_avg = np.round(pep_mass * 0.0698).astype(float)
    nn_avg = np.round(pep_mass * 0.0123).astype(float)
    no_avg = np.round(pep_mass * 0.0133).astype(float)
    lam_avg = nc_avg * 0.01109 + nh_avg * 0.000115 + nn_avg * 0.00364 + no_avg * 0.00205
    avg_m0 = np.exp(-lam_avg)
    avg_m1 = avg_m0 * lam_avg
    avg_m2 = avg_m0 * lam_avg**2 / 2.0
    avg_total = avg_m0 + avg_m1 + avg_m2
    avg_total = np.where(avg_total > 0, avg_total, 1.0)
    avg_m0 /= avg_total
    avg_m1 /= avg_total
    avg_m2 /= avg_total

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
    theo_m1_diff = np.zeros(n)
    theo_m2_diff = np.zeros(n)
    theo_m1_diff_lcms = np.zeros(n)
    theo_m2_diff_lcms = np.zeros(n)

    if maldi_envelopes or lcms_envelopes:
        feature_mzs = df["feature_mz"].values
        for i in range(n):
            t = np.array([theo_m0[i], theo_m1[i], theo_m2[i]])
            nt = norm_t[i]

            if maldi_envelopes:
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

            if lcms_envelopes:
                lcms_env = lcms_envelopes.get(feature_mzs[i])
                if lcms_env is not None and len(lcms_env) >= 3 and t[0] > 0:
                    lobs = np.array(lcms_env[:3], dtype=np.float64)
                    if lobs[0] > 0:
                        theo_m1_diff_lcms[i] = abs(lobs[1] / lobs[0] - t[1] / t[0])
                        theo_m2_diff_lcms[i] = abs(lobs[2] / lobs[0] - t[2] / t[0])

    df["theo_isotope_cosine"] = theo_cosine
    df["theo_isotope_chi2"] = theo_chi2
    df["theo_isotope_kl"] = theo_kl
    df["theo_m1_ratio_diff"] = theo_m1_diff
    df["theo_m2_ratio_diff"] = theo_m2_diff
    df["theo_m1_ratio_diff_lcms"] = theo_m1_diff_lcms
    df["theo_m2_ratio_diff_lcms"] = theo_m2_diff_lcms

    logger.info(f"Theoretical isotope features: {(theo_cosine > 0).sum()}/{n} scored")
    return df


def compute_envelope_similarity(
    df: pd.DataFrame,
    maldi_envelopes: dict | None = None,
    lcms_envelopes: dict | None = None,
) -> pd.DataFrame:
    """
    Compute MALDI vs LC-MS/MS isotope envelope similarity.

    lcms_envelopes should be XIC-derived (per feature_mz), not PD-derived.
    Both targets and decoys with matching XIC peaks get real comparisons;
    those without get median fill.
    """
    from scipy.stats import pearsonr

    n = len(df)
    cosine = np.full(n, np.nan)
    pearson_arr = np.full(n, np.nan)
    mse_arr = np.full(n, np.nan)
    m1_diff = np.full(n, np.nan)
    m2_diff = np.full(n, np.nan)
    n_matched = np.zeros(n)

    for i, (_, row) in enumerate(df.iterrows()):
        maldi_env = maldi_envelopes.get(row["feature_mz"]) if maldi_envelopes else None
        lcms_env = lcms_envelopes.get(row["feature_mz"]) if lcms_envelopes else None

        if maldi_env is None or lcms_env is None:
            continue

        k = min(len(maldi_env), len(lcms_env))
        a = np.array(maldi_env[:k], dtype=np.float64)
        b = np.array(lcms_env[:k], dtype=np.float64)

        matched = int(np.sum((a > 0) & (b > 0)))
        n_matched[i] = matched

        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na > 0 and nb > 0:
            cosine[i] = float(np.dot(a, b) / (na * nb))
        if a.std() > 0 and b.std() > 0 and k >= 2:
            r, _ = pearsonr(a, b)
            pearson_arr[i] = float(r)
        mse_arr[i] = float(np.mean((a - b) ** 2))

        if a[0] > 0 and b[0] > 0 and k >= 2:
            m1_diff[i] = abs(a[1] / a[0] - b[1] / b[0])
        if a[0] > 0 and b[0] > 0 and k >= 3:
            m2_diff[i] = abs(a[2] / a[0] - b[2] / b[0])

    # Fill NaN with median
    for name, arr in [
        ("isotope_envelope_cosine", cosine),
        ("isotope_envelope_pearson", pearson_arr),
        ("isotope_envelope_mse", mse_arr),
        ("isotope_m1_ratio_diff", m1_diff),
        ("isotope_m2_ratio_diff", m2_diff),
    ]:
        valid = arr[~np.isnan(arr)]
        fill = float(np.median(valid)) if len(valid) > 0 else 0.0
        df[name] = np.where(np.isnan(arr), fill, arr)

    df["isotope_n_matched"] = n_matched

    n_scored = np.sum(~np.isnan(cosine))
    logger.info(f"Envelope similarity: {n_scored}/{n} candidates scored")
    return df
