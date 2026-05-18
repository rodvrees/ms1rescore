"""
Probabilistic generative scorer for MALDI-MSI candidates.

This module is self-contained: it takes the candidates DataFrame already
populated with all features by compute_all_features() and returns score and
FDR columns. It performs no training, no mokapot, no catboost.

The generative model assumes that each observed MALDI feature is either a
true peptide signal or noise, and assigns a log-likelihood to each candidate
based on independent component likelihoods for mass accuracy, isotope pattern,
CCS (if available), and spatial autocorrelation (if available).

Noise parameters (half-normal/normal widths) are estimated label-free from a
proxy set of likely-correct identifications: the best-ppm non-decoy candidate
per MALDI feature.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Parameter estimation
# ---------------------------------------------------------------------------

def estimate_noise_params(candidates_df: pd.DataFrame) -> dict:
    """Estimate generative model parameters from a label-free proxy set.

    Proxy set = the best-ppm non-decoy candidate per MALDI feature. These
    are likely true positives (or at minimum the best available matches), so
    their feature distributions characterise the signal component.

    Half-normal sigma is estimated as RMS of the proxy values (equivalent to
    the MLE sigma of a half-normal with zero mean).

    Returns
    -------
    dict with keys:
        ppm_sigma, isotope_sigma,
        ccs_mu, ccs_sigma (if im2deep_delta_ccs present),
        spatial_mu, spatial_sigma (if spatial_autocorrelation present)
    """
    targets = candidates_df[~candidates_df["is_decoy"]]
    if "feature_idx" in targets.columns:
        group_col = "feature_idx"
    elif "feature_mz" in targets.columns:
        group_col = "feature_mz"
    else:
        group_col = targets.columns[0]

    proxy_idx = targets.groupby(group_col)["ppm_error_abs"].idxmin()
    proxy = targets.loc[proxy_idx]

    params: dict = {}

    ppm_vals = proxy["ppm_error_abs"].dropna().values
    if len(ppm_vals) > 0:
        params["ppm_sigma"] = float(np.sqrt(np.mean(ppm_vals**2)))
    else:
        params["ppm_sigma"] = 5.0
    params["ppm_sigma"] = max(params["ppm_sigma"], 0.1)

    if "theo_isotope_cosine" in proxy.columns:
        iso_dev = np.maximum(1.0 - proxy["theo_isotope_cosine"].dropna().values, 0.0)
        params["isotope_sigma"] = (
            float(np.sqrt(np.mean(iso_dev**2))) if len(iso_dev) > 0 else 0.3
        )
        params["isotope_sigma"] = max(params["isotope_sigma"], 0.01)
    else:
        params["isotope_sigma"] = 0.3

    if "im2deep_delta_ccs" in proxy.columns:
        ccs_vals = proxy["im2deep_delta_ccs"].dropna().values
        params["ccs_mu"] = float(np.mean(ccs_vals)) if len(ccs_vals) > 0 else 0.0
        params["ccs_sigma"] = float(np.std(ccs_vals)) if len(ccs_vals) > 1 else 10.0
        params["ccs_sigma"] = max(params["ccs_sigma"], 0.5)
    else:
        params["ccs_mu"] = 0.0
        params["ccs_sigma"] = 10.0

    if "spatial_autocorrelation" in proxy.columns:
        sa_vals = proxy["spatial_autocorrelation"].dropna().values
        params["spatial_mu"] = float(np.mean(sa_vals)) if len(sa_vals) > 0 else 0.0
        params["spatial_sigma"] = (
            float(np.std(sa_vals)) if len(sa_vals) > 1 else 0.5
        )
        params["spatial_sigma"] = max(params["spatial_sigma"], 0.01)
    else:
        params["spatial_mu"] = 0.0
        params["spatial_sigma"] = 0.5

    logger.info(
        f"  Generative params: ppm_sigma={params['ppm_sigma']:.3f}, "
        f"isotope_sigma={params['isotope_sigma']:.3f}, "
        f"ccs_sigma={params['ccs_sigma']:.3f}, "
        f"spatial_mu={params['spatial_mu']:.3f}, "
        f"spatial_sigma={params['spatial_sigma']:.3f}"
    )
    return params


# ---------------------------------------------------------------------------
# Component likelihoods
# ---------------------------------------------------------------------------

def _ppm_likelihood(ppm_error_abs: np.ndarray, sigma: float) -> np.ndarray:
    """Half-normal likelihood for mass accuracy. L(0) = 1.0, falls with error."""
    return np.exp(-0.5 * (ppm_error_abs / sigma) ** 2)


def _isotope_likelihood(cosine: np.ndarray, sigma: float) -> np.ndarray:
    """Half-normal likelihood for isotope cosine. L(cosine=1) = 1.0."""
    dev = np.maximum(1.0 - cosine, 0.0)
    return np.exp(-0.5 * (dev / sigma) ** 2)


def _ccs_likelihood(delta_ccs: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """Normal likelihood for CCS deviation from calibration mean."""
    return np.exp(-0.5 * ((delta_ccs - mu) / sigma) ** 2)


def _spatial_likelihood(
    spatial_score: np.ndarray, mu: float, sigma: float
) -> np.ndarray:
    """Normal likelihood for spatial autocorrelation centered at proxy-set mean."""
    return np.exp(-0.5 * ((spatial_score - mu) / sigma) ** 2)


# ---------------------------------------------------------------------------
# Score computation
# ---------------------------------------------------------------------------

def compute_generative_scores(
    candidates_df: pd.DataFrame,
    noise_params: dict,
) -> tuple[pd.Series, dict]:
    """Compute log-sum generative score for each candidate.

    Components are log-additive (independent likelihood assumption). Each
    component is included only when its feature column is present and has at
    least one non-NaN value. Missing values per row are filled with the
    worst-case value for that component (0 likelihood → very negative log).

    Returns
    -------
    (log_scores, components_used)
        log_scores   : pd.Series aligned with candidates_df index
        components_used : dict mapping component name → bool
    """
    n = len(candidates_df)
    log_score = np.zeros(n, dtype=np.float64)
    components_used: dict[str, bool] = {}

    _LOG_FLOOR = np.log(1e-300)

    if "ppm_error_abs" in candidates_df.columns:
        vals = candidates_df["ppm_error_abs"].fillna(noise_params["ppm_sigma"] * 10).values
        log_score += np.maximum(
            np.log(_ppm_likelihood(vals, noise_params["ppm_sigma"])), _LOG_FLOOR
        )
        components_used["ppm"] = True

    if "theo_isotope_cosine" in candidates_df.columns:
        vals = candidates_df["theo_isotope_cosine"].fillna(0.0).values
        log_score += np.maximum(
            np.log(_isotope_likelihood(vals, noise_params["isotope_sigma"])), _LOG_FLOOR
        )
        components_used["isotope"] = True

    if (
        "im2deep_delta_ccs" in candidates_df.columns
        and candidates_df["im2deep_delta_ccs"].notna().any()
    ):
        fill = noise_params["ccs_mu"]
        vals = candidates_df["im2deep_delta_ccs"].fillna(fill).values
        log_score += np.maximum(
            np.log(_ccs_likelihood(vals, noise_params["ccs_mu"], noise_params["ccs_sigma"])),
            _LOG_FLOOR,
        )
        components_used["ccs"] = True

    if (
        "spatial_autocorrelation" in candidates_df.columns
        and candidates_df["spatial_autocorrelation"].notna().any()
    ):
        fill = noise_params["spatial_mu"]
        vals = candidates_df["spatial_autocorrelation"].fillna(fill).values
        log_score += np.maximum(
            np.log(
                _spatial_likelihood(vals, noise_params["spatial_mu"], noise_params["spatial_sigma"])
            ),
            _LOG_FLOOR,
        )
        components_used["spatial"] = True

    logger.info(
        f"  Generative score components: {list(components_used.keys())}"
    )
    return pd.Series(log_score, index=candidates_df.index), components_used


# ---------------------------------------------------------------------------
# Ranking features
# ---------------------------------------------------------------------------

def add_ranking_features(
    candidates_df: pd.DataFrame,
    log_scores: pd.Series,
    feature_col: str = "feature_idx",
) -> pd.DataFrame:
    """Add per-MALDI-feature ranking features derived from generative scores.

    New columns added to a copy of candidates_df:
        generative_score       : raw log-likelihood
        generative_score_rank  : rank within the feature (1 = best)
        generative_score_gap   : score(rank-1) - score(rank-2); NaN for single-candidate features, 0 for non-winners
        generative_score_z     : z-score within the feature's candidate set

    Returns a copy of candidates_df with these columns appended.
    """
    df = candidates_df.copy()
    df["generative_score"] = log_scores.values

    if feature_col not in df.columns:
        feature_col = "feature_mz" if "feature_mz" in df.columns else df.columns[0]

    df["generative_score_rank"] = (
        df.groupby(feature_col)["generative_score"]
        .rank(method="min", ascending=False)
        .astype(int)
    )

    def _zscore(grp: pd.Series) -> pd.Series:
        mu = grp.mean()
        sigma = grp.std()
        if np.isnan(sigma) or sigma < 1e-12:
            return pd.Series(np.nan, index=grp.index)
        return (grp - mu) / sigma

    df["generative_score_z"] = df.groupby(feature_col)["generative_score"].transform(
        _zscore
    )

    def _gap(grp: pd.Series) -> pd.Series:
        scores = grp.values
        if len(scores) < 2:
            return pd.Series(np.nan, index=grp.index)
        top2 = np.partition(scores, -2)[-2:]  # two largest
        best, second = top2[1], top2[0]
        gap = best - second
        return pd.Series(np.where(scores == best, gap, 0.0), index=grp.index)

    df["generative_score_gap"] = df.groupby(feature_col)["generative_score"].transform(
        _gap
    )

    return df


# ---------------------------------------------------------------------------
# FDR estimation
# ---------------------------------------------------------------------------

def estimate_fdr(
    candidates_df: pd.DataFrame,
    score_col: str = "generative_score",
    feature_col: str = "feature_idx",
) -> pd.DataFrame:
    """Per-feature target-decoy competition (TDC) with Tm-based ranking.

    For each MALDI feature, collapses the candidate set to two values:
    Tm = best target score, Dm = best decoy score. Features are ranked
    by descending Tm for FDR estimation. This is standard TDC applied
    at the feature level: at threshold τ, FDR(τ) = #{features where
    Dm ≥ τ} / #{features where Tm ≥ τ}.

    Delta_m = Tm - Dm is computed as a supplementary diagnostic column
    (useful for filtering near-ties) but is not the primary ranking
    statistic.

    Steps
    -----
    1. Per feature: Tm = max score over targets, Dm = max score over decoys.
       Track the target candidate achieving Tm (the "winning peptide").
    2. Delta_m = Tm - Dm (supplementary).
    3. Sort features by descending Tm.
    4. TDC q-value: q(i) = (n_decoy_wins_at_or_better + 1)
                             / n_target_wins_at_or_better
       Monotonized by cumulative minimum from best to worst Tm.
    5. Assign q-values to all candidates at a feature (non-winners get
       the same q-value as their feature; identification = winning peptide).
    6. Local PEP via isotonic regression on Tm-sorted target/decoy win
       binary labels (requires scikit-learn; NaN if unavailable).

    Added/replaced columns
    ----------------------
    Tm                 : best target score for this feature
    Dm                 : best decoy score for this feature
    Delta_m            : Tm - Dm (supplementary)
    generative_q_value : TDC q-value based on Tm ranking
    generative_pep     : local PEP from isotonic regression on Tm
    is_tdc_winner      : True for the single winning target per feature
    """
    if feature_col not in candidates_df.columns:
        feature_col = "feature_mz" if "feature_mz" in candidates_df.columns else candidates_df.columns[0]

    targets = candidates_df[candidates_df["is_decoy"] == 0]
    decoys  = candidates_df[candidates_df["is_decoy"] == 1]

    Tm = targets.groupby(feature_col)[score_col].max().rename("Tm")
    Dm = decoys.groupby(feature_col)[score_col].max().rename("Dm")

    feature_df = pd.concat([Tm, Dm], axis=1)
    feature_df["Dm"] = feature_df["Dm"].fillna(-np.inf)
    feature_df = feature_df.dropna(subset=["Tm"])
    feature_df["Delta_m"] = feature_df["Tm"] - feature_df["Dm"]
    feature_df["target_wins"] = feature_df["Tm"] > feature_df["Dm"]

    # TDC q-values over features sorted by descending Tm
    feature_df = feature_df.sort_values("Tm", ascending=False)
    n_target_wins = feature_df["target_wins"].cumsum()
    n_decoy_wins  = (~feature_df["target_wins"]).cumsum()
    raw_q = (n_decoy_wins + 1) / n_target_wins.clip(lower=1)
    feature_df["generative_q_value"] = raw_q[::-1].cummin()[::-1].clip(upper=1.0)

    # Local PEP via isotonic regression (decreasing: higher Tm → lower PEP)
    try:
        from sklearn.isotonic import IsotonicRegression
        ir = IsotonicRegression(out_of_bounds="clip", increasing=False)
        feature_df["generative_pep"] = ir.fit_transform(
            feature_df["Tm"].values,
            (~feature_df["target_wins"]).astype(float).values,
        )
    except ImportError:
        feature_df["generative_pep"] = float("nan")

    # Winning target index per feature (idxmax on targets only)
    winning_idx = targets.groupby(feature_col)[score_col].idxmax()

    # Map feature-level stats back to candidate rows
    result = candidates_df.copy()
    _overwrite_cols = ["Tm", "Dm", "Delta_m", "generative_q_value", "generative_pep", "is_tdc_winner"]
    result.drop(columns=[c for c in _overwrite_cols if c in result.columns], inplace=True)
    feat_map = feature_df[["Tm", "Dm", "Delta_m", "generative_q_value", "generative_pep"]]
    result = result.join(feat_map, on=feature_col, how="left")

    result["is_tdc_winner"] = False
    valid_winners = winning_idx[winning_idx.index.isin(feature_df.index)]
    result.loc[valid_winners.values, "is_tdc_winner"] = True

    n_targets_01 = (
        result["is_tdc_winner"]
        & (result["generative_q_value"] <= 0.01)
    ).sum()
    n_targets_05 = (
        result["is_tdc_winner"]
        & (result["generative_q_value"] <= 0.05)
    ).sum()
    logger.info(
        f"  Generative FDR: {n_targets_01} winners at 1% FDR, "
        f"{n_targets_05} winners at 5% FDR"
    )
    return result


# ---------------------------------------------------------------------------
# Top-level convenience
# ---------------------------------------------------------------------------

def run_generative_scoring(
    candidates_df: pd.DataFrame,
    feature_col: str = "feature_idx",
) -> pd.DataFrame:
    """Run the full generative scoring pipeline on a features DataFrame.

    Estimates noise parameters, computes log-scores, adds ranking features,
    and runs margin-based per-feature TDC FDR estimation. Returns
    candidates_df with added columns:
        generative_score, generative_score_rank, generative_score_gap,
        generative_score_z, Tm, Dm, Delta_m, generative_q_value,
        generative_pep, is_tdc_winner

    This is the function called by pipeline.rescore() for model="generative"
    and also when compute_generative=True before SVM/CatBoost rescoring.
    """
    noise_params = estimate_noise_params(candidates_df)
    log_scores, _ = compute_generative_scores(candidates_df, noise_params)
    df = add_ranking_features(candidates_df, log_scores, feature_col=feature_col)
    df = estimate_fdr(df, score_col="generative_score", feature_col=feature_col)
    return df
