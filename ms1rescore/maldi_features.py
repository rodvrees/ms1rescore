"""
MALDI-side feature computation: mass accuracy, protein consistency, spatial,
isotope envelope, co-localization, ionization properties.

All functions are symmetric — no is_decoy branching in feature computation.
"""

import logging

import numpy as np
import pandas as pd

from ms1rescore.utils import NEUTRON

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

# Lehninger pKa values for pI computation (C8)
_PKA = {
    "D": 3.9, "E": 4.1, "H": 6.0, "C": 8.3,
    "Y": 10.1, "K": 10.5, "R": 12.5,
    "nterm": 8.0, "cterm": 3.1,
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


def _compute_pi(sequence: str) -> float:
    """Compute peptide pI by bisection on Lehninger pKa values (C8)."""
    n_D = sequence.count("D"); n_E = sequence.count("E"); n_H = sequence.count("H")
    n_C = sequence.count("C"); n_Y = sequence.count("Y")
    n_K = sequence.count("K"); n_R = sequence.count("R")

    def net_charge(ph: float) -> float:
        q  =  1.0 / (1.0 + 10.0 ** (ph - _PKA["nterm"]))
        q -= 1.0 / (1.0 + 10.0 ** (_PKA["cterm"] - ph))
        q -= n_D / (1.0 + 10.0 ** (_PKA["D"] - ph))
        q -= n_E / (1.0 + 10.0 ** (_PKA["E"] - ph))
        q -= n_C / (1.0 + 10.0 ** (_PKA["C"] - ph))
        q -= n_Y / (1.0 + 10.0 ** (_PKA["Y"] - ph))
        q += n_H / (1.0 + 10.0 ** (ph - _PKA["H"]))
        q += n_K / (1.0 + 10.0 ** (ph - _PKA["K"]))
        q += n_R / (1.0 + 10.0 ** (ph - _PKA["R"]))
        return q

    lo, hi = 0.0, 14.0
    for _ in range(50):
        mid = (lo + hi) / 2.0
        if net_charge(mid) > 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def _compute_pi_batch(unique_seqs: np.ndarray,
                      n_D: np.ndarray, n_E: np.ndarray, n_H: np.ndarray,
                      n_C: np.ndarray, n_Y: np.ndarray,
                      n_K: np.ndarray, n_R: np.ndarray) -> np.ndarray:
    """
    Vectorized pI for all unique sequences simultaneously.

    Reformulates the bisection in terms of x = 10^ph so that only ONE
    np.power call is needed per iteration (vs 9 in the naive implementation).
    Scalar pKa terms are precomputed once outside the loop.

    50 × 1 np.power calls instead of 50 × 9 = ~9× reduction in the
    most expensive operation.
    """
    # Precompute 10^pKa scalars (computed once, not inside the loop).
    pk_nt = 10.0 ** _PKA["nterm"]
    pk_ct = 10.0 ** _PKA["cterm"]
    pk_D  = 10.0 ** _PKA["D"];  pk_E = 10.0 ** _PKA["E"]
    pk_C  = 10.0 ** _PKA["C"];  pk_Y = 10.0 ** _PKA["Y"]
    pk_H  = 10.0 ** _PKA["H"];  pk_K = 10.0 ** _PKA["K"]
    pk_R  = 10.0 ** _PKA["R"]

    lo = np.zeros(len(unique_seqs), dtype=np.float64)
    hi = np.full(len(unique_seqs), 14.0, dtype=np.float64)

    for _ in range(50):
        mid = (lo + hi) * 0.5
        x = np.power(10.0, mid)          # 1 power call per iteration
        # Rewrite 1/(1+10^(a-b)) = 10^b / (10^b + 10^a) = pk / (pk + x)
        q  =  pk_nt / (pk_nt + x)        # N-term basic
        q -= x / (x + pk_ct)             # C-term acidic
        q -= n_D * x / (x + pk_D)
        q -= n_E * x / (x + pk_E)
        q -= n_C * x / (x + pk_C)
        q -= n_Y * x / (x + pk_Y)
        q += n_H * pk_H / (pk_H + x)
        q += n_K * pk_K / (pk_K + x)
        q += n_R * pk_R / (pk_R + x)
        lo = np.where(q > 0, mid, lo)
        hi = np.where(q <= 0, mid, hi)

    return (lo + hi) * 0.5


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
    try:
        from ms1rescore_rs import count_missed_cleavages_batch
        df["n_missed_cleavages"] = count_missed_cleavages_batch(df["peptide"].tolist())
    except ImportError:
        # K/R not followed by P, excluding the C-terminal cleavage site.
        df["n_missed_cleavages"] = df["peptide"].str[:-1].str.count(r"[KR](?!P)")
    df["has_modifications"] = 0  # plain sequences from digest have no mods
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

    # Backwards-compatible alias — maps to p90 if not already present
    if "log_maldi_intensity" not in df.columns:
        df["log_maldi_intensity"] = df["log_maldi_intensity_p90"]
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
    df["n_arginine"]       = n_R[codes].astype(np.float32)
    df["n_phenylalanine"]  = n_F[codes].astype(np.float32)
    df["n_basic_residues"] = n_basic_u[codes].astype(np.float32)
    df["n_aromatic"]       = (n_F + n_W + n_Y)[codes].astype(np.float32)
    df["gravy_score"]      = gravy_u[codes]
    df["charge_proxy"]     = (n_basic_u + 1)[codes].astype(np.float32)

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

    # A8 — monoisotopic confidence: M0/(M0+M1) before normalisation.
    # Values near 0.5 or below indicate the monoisotopic peak may not be the
    # most abundant isotopologue, making the assignment less reliable.
    mono_conf = theo_m0 / np.where(theo_m0 + theo_m1 > 0, theo_m0 + theo_m1, 1.0)

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
    df["monoisotopic_confidence"] = mono_conf

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
    nc_avg = np.round(mass * 0.0444)
    nh_avg = np.round(mass * 0.0698)
    nn_avg = np.round(mass * 0.0123)
    no_avg = np.round(mass * 0.0133)
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

def compute_im2deep_features(
    df: pd.DataFrame,
    observed_ccs_per_feature: dict | None = None,
    calibration_slope: float = 1.0,
    calibration_intercept: float = 0.0,
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
    im2deep_mahalanobis         Mahalanobis distance in (ppm_error_abs, delta_ccs) space
    """
    if not observed_ccs_per_feature:
        return df

    try:
        from im2deep.predict import predict_ccs
    except ImportError:
        logger.debug("im2deep not installed — skipping CCS features (B)")
        return df

    unique_seqs = df["peptide"].unique().tolist()
    pred_dict: dict[str, float] = {}

    try:
        import pandas as _pd
        pred_input = _pd.DataFrame({"peptide": unique_seqs, "charge": [1] * len(unique_seqs)})
        preds = predict_ccs(pred_input)
        # Support array-like or Series output
        preds_arr = np.asarray(preds).flatten()
        for i, seq in enumerate(unique_seqs):
            pred_dict[seq] = float(preds_arr[i])
    except Exception as exc:
        logger.warning(f"IM2Deep prediction failed: {exc} — skipping CCS features")
        return df

    predicted = df["peptide"].map(pred_dict).values.astype(float)
    predicted_cal = calibration_slope * predicted + calibration_intercept
    observed = (
        df["feature_idx"]
        .map(observed_ccs_per_feature)
        .values.astype(float)
    )

    delta = observed - predicted_cal
    abs_pct = np.abs(delta) / np.where(np.abs(predicted_cal) > 1e-6, np.abs(predicted_cal), 1.0) * 100.0

    df["_im2deep_delta"] = delta
    df["_im2deep_abspct"] = abs_pct

    df["im2deep_delta_ccs"] = delta
    df["im2deep_abs_delta_ccs_pct"] = abs_pct

    # Z-score and rank within each MALDI feature
    df["im2deep_ccs_zscore"] = df.groupby("feature_mz")["_im2deep_delta"].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-10)
    )
    df["im2deep_ccs_rank"] = df.groupby("feature_mz")["_im2deep_abspct"].rank(method="min")

    # Mahalanobis distance in (ppm_error_abs, delta_ccs) space per feature
    mahal = np.full(len(df), np.nan)
    df_reset = df.reset_index(drop=False)
    for _, grp in df_reset.groupby("feature_mz"):
        if len(grp) < 3:
            continue
        X = grp[["ppm_error_abs", "_im2deep_delta"]].values.astype(float)
        valid_mask = ~np.isnan(X).any(axis=1)
        X_valid = X[valid_mask]
        if X_valid.shape[0] < 3:
            continue
        try:
            cov = np.cov(X_valid.T)
            if np.linalg.matrix_rank(cov) < 2:
                continue
            cov_inv = np.linalg.inv(cov)
            mean = X_valid.mean(axis=0)
            for row_pos, orig_idx in enumerate(grp.index):
                if valid_mask[row_pos]:
                    diff = X[row_pos] - mean
                    mahal[orig_idx] = float(np.sqrt(diff @ cov_inv @ diff))
        except np.linalg.LinAlgError:
            continue

    df["im2deep_mahalanobis"] = mahal
    df.drop(columns=["_im2deep_delta", "_im2deep_abspct"], inplace=True, errors="ignore")

    # Fill NaN with median (symmetric fill — no target/decoy information)
    for col in [
        "im2deep_delta_ccs", "im2deep_abs_delta_ccs_pct",
        "im2deep_ccs_zscore", "im2deep_ccs_rank", "im2deep_mahalanobis",
    ]:
        valid = df[col].dropna()
        fill = float(valid.median()) if len(valid) > 0 else 0.0
        df[col] = df[col].fillna(fill)

    logger.info(
        f"IM2Deep CCS features (B): scored {(~np.isnan(observed)).sum()}/{len(df)} candidates"
    )
    return df


# ---------------------------------------------------------------------------
# C — Additional peptide property features
# ---------------------------------------------------------------------------

def compute_peptide_property_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute additional peptide property features (C2, C8, C9, C12, C15).

    All features are computed from the peptide sequence column only — no
    external API calls or pre-computed lookup tables beyond _PKA and _KD_SCALE.

    Features added
    --------------
    nterm_basic              1 if N-terminal residue is R, K, or H (C2)
    peptide_pi               isoelectric point via bisection (C8)
    has_oxidized_met         1 if M present (oxidation susceptibility proxy, C9)
    has_cys                  1 if C present (carbamidomethylation proxy, C9)
    n_proline                count of P residues (C9)
    nterm_pyroglu_risk       1 if N-terminal residue is Q or E (C9)
    acidic_residue_density   (count(D)+count(E)) / length (C12)
    n_tryptophan             count of W (C15)
    n_tyrosine               count of Y (C15)
    """
    codes, uniques = pd.factorize(df["peptide"])
    uniques_list = uniques.tolist()

    try:
        from ms1rescore_rs import compute_property_features
        n_D, n_E, n_C, n_P, n_M, n_W, n_Y, seq_lens_u, nterm_code_u, pi_u = \
            compute_property_features(uniques_list)
        n_D = np.asarray(n_D); n_E = np.asarray(n_E); n_C = np.asarray(n_C)
        n_P = np.asarray(n_P); n_M = np.asarray(n_M)
        n_W = np.asarray(n_W); n_Y = np.asarray(n_Y)
        seq_lens_u = np.asarray(seq_lens_u); nterm_code_u = np.asarray(nterm_code_u)
        pi_u = np.asarray(pi_u)
    except ImportError:
        cnt = _residue_counts_batch(uniques, "DEHCKYRPMW")
        n_D = cnt["D"]; n_E = cnt["E"]; n_C = cnt["C"]
        n_P = cnt["P"]; n_M = cnt["M"]; n_W = cnt["W"]; n_Y = cnt["Y"]
        n_H = cnt["H"]; n_K = cnt["K"]; n_R = cnt["R"]
        seq_lens_u = np.array([len(s) for s in uniques], dtype=np.int32)
        nterm_strs = np.array([s[0] if s else "" for s in uniques], dtype="U1")
        nterm_code_u = np.array([ord(c) if c else 0 for c in nterm_strs], dtype=np.int32)
        pi_u = _compute_pi_batch(uniques, n_D.astype(float), n_E.astype(float),
                                 n_H.astype(float), n_C.astype(float),
                                 n_Y.astype(float), n_K.astype(float),
                                 n_R.astype(float))

    nterm_is_basic   = np.isin(nterm_code_u, [ord("R"), ord("K"), ord("H")])
    nterm_is_pyroglu = np.isin(nterm_code_u, [ord("Q"), ord("E")])

    df["nterm_basic"]            = nterm_is_basic[codes].astype(float)
    df["peptide_pi"]             = pi_u[codes]
    df["has_oxidized_met"]       = (n_M > 0)[codes].astype(float)
    df["has_cys"]                = (n_C > 0)[codes].astype(float)
    df["n_proline"]              = n_P[codes].astype(float)
    df["nterm_pyroglu_risk"]     = nterm_is_pyroglu[codes].astype(float)
    df["acidic_residue_density"] = ((n_D + n_E).astype(float) / np.maximum(seq_lens_u, 1))[codes]
    df["n_tryptophan"]           = n_W[codes].astype(float)
    df["n_tyrosine"]             = n_Y[codes].astype(float)

    return df


# ---------------------------------------------------------------------------
# E1 — Isotopologue spatial co-localization
# ---------------------------------------------------------------------------

def compute_isotopologue_colocalization(
    df: pd.DataFrame,
    ion_images: np.ndarray,
    ion_image_mzs: np.ndarray,
    ppm_tolerance: float = 10.0,
) -> pd.DataFrame:
    """
    Pearson r between the M0 ion image and the M+1 / M+2 ion images (E1).

    For each MALDI feature at m/z M, looks up ion images at M + NEUTRON and
    M + 2*NEUTRON within ``ppm_tolerance``. The correlation is a per-feature
    scalar broadcast to all candidates.

    Features added
    --------------
    isotope_image_colocalization_m1     Pearson r(M0, M+1)
    isotope_image_colocalization_m2     Pearson r(M0, M+2)
    isotope_image_colocalization_mean   mean of available correlations
    """
    mz_arr = np.asarray(ion_image_mzs, dtype=float)

    # Pre-flatten and cache only images with non-zero variance
    flat_cache: dict[float, np.ndarray] = {}
    for i, mz in enumerate(mz_arr):
        flat = ion_images[i].flatten().astype(np.float64)
        if flat.std() > 1e-10:
            flat_cache[float(mz)] = flat

    r_m1_map: dict[float, float] = {}
    r_m2_map: dict[float, float] = {}
    r_mean_map: dict[float, float] = {}

    for feat_mz in df["feature_mz"].unique():
        feat_mz_f = float(feat_mz)
        if feat_mz_f not in flat_cache:
            continue
        flat_m0 = flat_cache[feat_mz_f]

        def _find_key(target: float) -> float | None:
            ppm = np.abs(mz_arr - target) / target * 1e6
            idx = int(np.argmin(ppm))
            return float(mz_arr[idx]) if ppm[idx] <= ppm_tolerance else None

        rs: list[float] = []
        r_m1 = float("nan")
        r_m2 = float("nan")

        key_m1 = _find_key(feat_mz_f + NEUTRON)
        if key_m1 is not None and key_m1 in flat_cache:
            r_m1 = float(np.corrcoef(flat_m0, flat_cache[key_m1])[0, 1])
            rs.append(r_m1)

        key_m2 = _find_key(feat_mz_f + 2.0 * NEUTRON)
        if key_m2 is not None and key_m2 in flat_cache:
            r_m2 = float(np.corrcoef(flat_m0, flat_cache[key_m2])[0, 1])
            rs.append(r_m2)

        r_m1_map[feat_mz_f] = r_m1
        r_m2_map[feat_mz_f] = r_m2
        r_mean_map[feat_mz_f] = float(np.nanmean(rs)) if rs else float("nan")

    for col, mapping in [
        ("isotope_image_colocalization_m1", r_m1_map),
        ("isotope_image_colocalization_m2", r_m2_map),
        ("isotope_image_colocalization_mean", r_mean_map),
    ]:
        df[col] = df["feature_mz"].map(mapping)
        valid = df[col].dropna()
        df[col] = df[col].fillna(float(valid.median()) if len(valid) > 0 else 0.0)

    logger.info(
        f"Isotopologue colocalization (E1): {sum(not np.isnan(v) for v in r_m1_map.values())} "
        f"features with M+1 image"
    )
    return df


# ---------------------------------------------------------------------------
# E2 — Adduct co-localization
# ---------------------------------------------------------------------------

def compute_adduct_colocalization(
    df: pd.DataFrame,
    ion_images: np.ndarray,
    ion_image_mzs: np.ndarray,
    ppm_tolerance: float = 10.0,
) -> pd.DataFrame:
    """
    Pearson r between the M0 ion image and Na/K/CHCA adduct ion images (E2).

    Features added
    --------------
    adduct_colocalization_na    Pearson r with [M+Na]+ image
    adduct_colocalization_k     Pearson r with [M+K]+ image
    adduct_colocalization_chca  Pearson r with [M+CHCA+H-H2O]+ image
    """
    mz_arr = np.asarray(ion_image_mzs, dtype=float)

    flat_cache: dict[float, np.ndarray] = {}
    for i, mz in enumerate(mz_arr):
        flat = ion_images[i].flatten().astype(np.float64)
        if flat.std() > 1e-10:
            flat_cache[float(mz)] = flat

    results: dict[str, dict[float, float]] = {name: {} for name in _ADDUCT_DELTAS}

    for feat_mz in df["feature_mz"].unique():
        feat_mz_f = float(feat_mz)
        if feat_mz_f not in flat_cache:
            continue
        flat_m0 = flat_cache[feat_mz_f]

        for adduct_name, delta in _ADDUCT_DELTAS.items():
            target = feat_mz_f + delta
            ppm = np.abs(mz_arr - target) / target * 1e6
            idx = int(np.argmin(ppm))
            if ppm[idx] <= ppm_tolerance and float(mz_arr[idx]) in flat_cache:
                r = float(np.corrcoef(flat_m0, flat_cache[float(mz_arr[idx])])[0, 1])
                results[adduct_name][feat_mz_f] = r

    for adduct_name in _ADDUCT_DELTAS:
        col = f"adduct_colocalization_{adduct_name}"
        df[col] = df["feature_mz"].map(results[adduct_name])
        valid = df[col].dropna()
        df[col] = df[col].fillna(float(valid.median()) if len(valid) > 0 else 0.0)

    return df


# ---------------------------------------------------------------------------
# E5/E6 — Moran's I and Geary's C (full spatial autocorrelation)
# ---------------------------------------------------------------------------

def compute_spatial_autocorrelation_full(
    df: pd.DataFrame,
    ion_images: np.ndarray,
    ion_image_mzs: np.ndarray,
) -> pd.DataFrame:
    """
    Compute Moran's I and Geary's C for each MALDI feature ion image (E5/E6).

    Uses queen contiguity (8-neighbour) weights computed via 2-D convolution
    with a 3×3 kernel — avoids building a large sparse matrix. Results are
    per-feature scalars broadcast to all candidates.

    Features added
    --------------
    spatial_morans_i    Moran's I (positive = clustered, negative = dispersed)
    spatial_gearys_c    Geary's C (< 1 clustered, ≈ 1 random, > 1 dispersed)

    Requires scipy. Silently skips if not installed.
    """
    try:
        from scipy.signal import convolve2d
    except ImportError:
        logger.debug("scipy not installed — skipping Moran's I / Geary's C (E5/E6)")
        return df

    if ion_images is None or len(ion_images) == 0:
        return df

    _, height, width = ion_images.shape
    N = height * width

    # Queen neighbourhood kernel (3×3, centre = 0)
    kernel = np.ones((3, 3), dtype=float)
    kernel[1, 1] = 0.0

    # Neighbour counts per pixel (precomputed — same for all images)
    neighbor_counts = convolve2d(
        np.ones((height, width), dtype=float),
        kernel,
        mode="same",
        boundary="fill",
        fillvalue=0,
    ).flatten()
    W_sum = float(neighbor_counts.sum())

    mz_arr = np.asarray(ion_image_mzs, dtype=float)
    mz_to_idx = {float(mz): i for i, mz in enumerate(mz_arr)}

    morans_map: dict[float, float] = {}
    gearys_map: dict[float, float] = {}

    for feat_mz in df["feature_mz"].unique():
        feat_mz_f = float(feat_mz)
        idx = mz_to_idx.get(feat_mz_f)
        if idx is None:
            continue

        img = ion_images[idx].astype(float)
        x = img.flatten()
        x_bar = float(x.mean())
        z = x - x_bar
        z_sq_sum = float(z @ z)

        if z_sq_sum < 1e-12 or W_sum < 1.0:
            morans_map[feat_mz_f] = 0.0
            gearys_map[feat_mz_f] = 1.0
            continue

        # Σⱼ wᵢⱼ zⱼ for each pixel i
        Wz = convolve2d(
            z.reshape(height, width), kernel, mode="same", boundary="fill", fillvalue=0
        ).flatten()

        # Moran's I = (N / W) * (z^T W z) / (z^T z)
        morans_map[feat_mz_f] = float((N / W_sum) * (z @ Wz) / z_sq_sum)

        # Geary's C = ((N-1) / W) * (x^T(D-W)x) / (z^T z)
        # where (D-W)x = row_sums * x - W*x
        Wx = convolve2d(
            x.reshape(height, width), kernel, mode="same", boundary="fill", fillvalue=0
        ).flatten()
        xTDx = float(np.dot(x ** 2, neighbor_counts))
        xTWx = float(x @ Wx)
        gearys_map[feat_mz_f] = float(((N - 1) / W_sum) * (xTDx - xTWx) / z_sq_sum)

    df["spatial_morans_i"] = df["feature_mz"].map(morans_map).fillna(0.0)
    df["spatial_gearys_c"] = df["feature_mz"].map(gearys_map).fillna(1.0)

    logger.info(
        f"Spatial autocorrelation (E5/E6): computed for {len(morans_map)} features"
    )
    return df
