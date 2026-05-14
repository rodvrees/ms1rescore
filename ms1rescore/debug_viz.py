"""
Debug visualization for the MALDI-MSI rescoring pipeline.

Six subsystems:
  1. Ion image colocalization  — per-candidate precursor + protein co-feature images
  2. Feature diagnostics       — per-candidate 3×3 panel figure
  3. Isotope envelopes         — per-candidate spectrum-style envelope comparison
  4. Feature importance        — global sorted bar plots (rounds 1 and 2)
  5. Feature distributions     — per-feature target/decoy histograms (all + R2)
  6. CCS scatter               — observed vs predicted CCS for all candidates

Entry point: save_debug_figures()
"""

import logging
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _sample_subset(
    features_df: pd.DataFrame,
    result_df: pd.DataFrame,
    n: int = 50,
    seed: int = 42,
    fdr_threshold: float = 0.01,
) -> pd.DataFrame:
    """
    Stratified sample of n rows, guaranteeing at least one from each non-empty group:
      ID  — round-2 winner with reweighted_q_value <= fdr_threshold
      R1  — round-1 winner but does not pass FDR
      L   — not a winner

    Groups are computed exclusively from result_df to avoid conflicts with
    features_df, which may contain its own is_tdc_winner column from the
    generative pre-scoring step.
    """
    rng = np.random.default_rng(seed)
    N = len(features_df)

    feat = features_df.reset_index(drop=True)
    res = result_df.reset_index(drop=True)

    is_winner = (
        res.get("is_tdc_winner", pd.Series(False, index=res.index))
        .fillna(False)
        .astype(bool)
    )
    rw_q = pd.to_numeric(
        res.get("reweighted_q_value", pd.Series(float("nan"), index=res.index)),
        errors="coerce",
    )
    passes = is_winner & (rw_q <= fdr_threshold)
    groups = np.where(passes, "ID", np.where(is_winner.values, "R1", "L"))

    is_decoy_arr = (
        feat.get("is_decoy", pd.Series(False, index=feat.index))
        .fillna(False)
        .astype(bool)
        .values
    )
    td_labels = np.where(is_decoy_arr, "D", "T")

    # Stratify across 6 strata: {ID, R1, L} × {T, D}
    strata = [(grp, td) for grp in ("ID", "R1", "L") for td in ("T", "D")]
    per_stratum = max(1, n // len(strata))
    sampled_idx: list[int] = []
    for grp, td in strata:
        stratum_idx = np.where((groups == grp) & (td_labels == td))[0].tolist()
        if stratum_idx:
            k = min(per_stratum, len(stratum_idx))
            sampled_idx.extend(rng.choice(stratum_idx, size=k, replace=False).tolist())

    remaining = n - len(sampled_idx)
    if remaining > 0:
        sampled_set = set(sampled_idx)
        unsampled = [i for i in range(N) if i not in sampled_set]
        if unsampled:
            sampled_idx.extend(
                rng.choice(unsampled, size=min(remaining, len(unsampled)), replace=False).tolist()
            )

    sampled_idx = rng.permutation(sampled_idx).tolist()

    subset = feat.iloc[sampled_idx].copy().reset_index(drop=True)
    subset["_group"] = [groups[i] for i in sampled_idx]
    subset["_td"] = [td_labels[i] for i in sampled_idx]

    score_r1_cols = [c for c in result_df.columns if c.endswith("_score_r1")]
    score_r2_cols = [c for c in result_df.columns if c.endswith("_score_r2")]
    display_cols = score_r1_cols + score_r2_cols + [
        c for c in ["q_value", "is_tdc_winner", "reweighted_score", "reweighted_q_value"]
        if c in result_df.columns
    ]
    res_sampled = res.iloc[sampled_idx][display_cols].reset_index(drop=True)
    for col in display_cols:
        subset[col] = res_sampled[col].values

    if score_r1_cols:
        subset["_score_r1"] = subset[score_r1_cols[0]]
    else:
        subset["_score_r1"] = np.nan

    subset["_rank"] = (
        subset["_score_r1"]
        .rank(ascending=False, method="min", na_option="bottom")
        .astype(int)
    )
    subset["_total"] = N
    return subset


def _find_image_idx(feature_mz: float, ion_image_mzs: np.ndarray, ppm: float = 25.0) -> int | None:
    if ion_image_mzs is None or len(ion_image_mzs) == 0:
        return None
    diffs = np.abs(ion_image_mzs - feature_mz) / feature_mz * 1e6
    best = int(np.argmin(diffs))
    return best if diffs[best] < ppm else None


def _pearson_r(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.ravel().astype(float), b.ravel().astype(float)
    if a.std() < 1e-9 or b.std() < 1e-9:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _candidate_title(row: pd.Series) -> str:
    peptide = row.get("peptide", "?")
    protein = row.get("protein", "?")
    fmz = row.get("feature_mz", float("nan"))
    r1 = row.get("_score_r1", float("nan"))
    rank = row.get("_rank", "?")
    total = row.get("_total", "?")
    q = _get(row, "reweighted_q_value")
    winner = bool(row.get("is_tdc_winner", False))
    passes = winner and np.isfinite(q) and q <= 0.01
    label = "PASS" if passes else "FAIL"
    score_str = f"score={r1:.3f} | " if np.isfinite(r1) else ""
    q_str = f"q={q:.3f} | " if np.isfinite(q) else ""
    return f"{peptide} | {protein} | m/z {fmz:.4f} | {score_str}rank {rank}/{total} | {q_str}{label}"


def _safe_fname(s: str, maxlen: int = 45) -> str:
    return "".join(c if (c.isalnum() or c in "-_.") else "_" for c in s)[:maxlen]


def _get(row: pd.Series, col: str) -> float:
    v = row.get(col, float("nan"))
    try:
        v = float(v)
    except (TypeError, ValueError):
        v = float("nan")
    return v


# ---------------------------------------------------------------------------
# Subsystem 1: Ion image colocalization
# ---------------------------------------------------------------------------

def plot_ion_image_colocalization(
    subset: pd.DataFrame,
    features_df: pd.DataFrame,
    ion_images: np.ndarray,
    ion_image_mzs: np.ndarray,
    out_dir: str,
    max_co_features: int = 4,
) -> None:
    """
    Per-candidate figure: precursor ion image + protein co-feature images + protein mean.

    Files are saved as ``{out_dir}/{rank:03d}_{peptide}_{feature_mz:.4f}.png``.
    """
    os.makedirs(out_dir, exist_ok=True)

    for _, row in subset.iterrows():
        feature_mz = row.get("feature_mz")
        if feature_mz is None or not np.isfinite(float(feature_mz)):
            continue
        feature_mz = float(feature_mz)
        rank = int(row.get("_rank", 0))
        peptide = str(row.get("peptide", "unknown"))
        protein = row.get("protein")

        prefix = str(row.get("_group", "L"))
        td = str(row.get("_td", "T"))
        prec_idx = _find_image_idx(feature_mz, ion_image_mzs)
        if prec_idx is None:
            continue
        prec_img = ion_images[prec_idx]

        # Collect co-feature images for the same protein
        co_imgs: list[np.ndarray] = []
        co_mzs: list[float] = []
        if protein and "protein" in features_df.columns and "feature_mz" in features_df.columns:
            prot_mzs = (
                features_df.loc[features_df["protein"] == protein, "feature_mz"]
                .dropna()
                .unique()
            )
            for mz in sorted(float(m) for m in prot_mzs if abs(float(m) - feature_mz) > 1e-6):
                if len(co_imgs) >= max_co_features:
                    break
                idx = _find_image_idx(mz, ion_image_mzs)
                if idx is not None:
                    co_imgs.append(ion_images[idx])
                    co_mzs.append(mz)

        # Protein mean image across precursor + co-features
        all_imgs = [prec_img] + co_imgs
        prot_mean = np.mean(all_imgs, axis=0)

        n_panels = 1 + len(co_imgs) + 1
        fig, axes = plt.subplots(1, n_panels, figsize=(3.2 * n_panels, 3.8))
        if n_panels == 1:
            axes = [axes]

        def _panel(ax: plt.Axes, img: np.ndarray, title: str, r: float | None = None) -> None:
            im = ax.imshow(img, cmap="hot", aspect="auto")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cap = title if r is None else f"{title}\nr={r:.2f}"
            ax.set_title(cap, fontsize=7)
            ax.axis("off")

        _panel(axes[0], prec_img, f"Precursor\n{feature_mz:.4f}")
        for i, (cimg, cmz) in enumerate(zip(co_imgs, co_mzs)):
            _panel(axes[1 + i], cimg, f"Co-feat\n{cmz:.4f}", r=_pearson_r(prec_img, cimg))
        _panel(axes[-1], prot_mean, f"Protein mean\n({len(all_imgs)} imgs)", r=_pearson_r(prec_img, prot_mean))

        fig.suptitle(_candidate_title(row), fontsize=8, y=1.01)
        plt.tight_layout()
        fname = f"{prefix}_{td}_{rank:03d}_{_safe_fname(peptide)}_{feature_mz:.4f}.png"
        fig.savefig(os.path.join(out_dir, fname), dpi=100, bbox_inches="tight")
        plt.close(fig)


# ---------------------------------------------------------------------------
# Subsystem 2: Feature diagnostics
# ---------------------------------------------------------------------------

def plot_feature_diagnostics(
    subset: pd.DataFrame,
    features_df: pd.DataFrame,
    ion_images: np.ndarray | None,
    ion_image_mzs: np.ndarray | None,
    maldi_envelopes: dict | None,
    out_dir: str,
) -> None:
    """
    Per-candidate 3×3 diagnostic figure.

    Panels:
      [0,0] Ion image heatmap + fraction_detected / CV / Moran's I
      [0,1] Mass accuracy horizontal bar with ±2 / ±5 ppm shading
      [0,2] Observed vs theoretical isotope envelope
      [1,0] Peptide properties (bar chart)
      [1,1] Spatial statistics (bar chart)
      [1,2] LC-MS/MS features (bar chart, or "N/A")
      [2,0] CHCA cluster proximity gauge (chca_cluster_distance_ppm)
      [2,1] CHCA adduct colocalization (ion image thumbnails or Pearson r gauge)
      [2,2] Monoisotopic confidence gauge (monoisotopic_confidence)
    """
    from ms1rescore.utils import theoretical_isotope_distribution

    os.makedirs(out_dir, exist_ok=True)

    for _, row in subset.iterrows():
        feature_mz = row.get("feature_mz")
        if feature_mz is not None:
            feature_mz = float(feature_mz)
        prefix = str(row.get("_group", "L"))
        td = str(row.get("_td", "T"))
        rank = int(row.get("_rank", 0))
        peptide = str(row.get("peptide", "unknown"))

        fig = plt.figure(figsize=(15, 12))
        gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.55, wspace=0.38)
        ax = [[fig.add_subplot(gs[r, c]) for c in range(3)] for r in range(3)]

        # ------------------------------------------------------------------
        # [0,0] Ion image
        # ------------------------------------------------------------------
        if ion_images is not None and feature_mz is not None:
            prec_idx = _find_image_idx(feature_mz, ion_image_mzs)
        else:
            prec_idx = None

        if prec_idx is not None:
            im = ax[0][0].imshow(ion_images[prec_idx], cmap="hot", aspect="auto")
            plt.colorbar(im, ax=ax[0][0], fraction=0.046, pad=0.04)
            frac = _get(row, "fraction_detected")
            cv = _get(row, "intensity_cv")
            mi = _get(row, "spatial_autocorrelation")
            frac_s = f"f={frac:.2f}" if np.isfinite(frac) else ""
            cv_s = f"CV={cv:.2f}" if np.isfinite(cv) else ""
            mi_s = f"Moran={mi:.2f}" if np.isfinite(mi) else ""
            subtitle = "  ".join(s for s in [frac_s, cv_s, mi_s] if s)
            ax[0][0].set_title(f"Ion image\n{subtitle}", fontsize=8)
        else:
            ax[0][0].text(0.5, 0.5, "No ion image", ha="center", va="center", transform=ax[0][0].transAxes)
            ax[0][0].set_title("Ion image", fontsize=8)
        ax[0][0].axis("off")

        # ------------------------------------------------------------------
        # [0,1] Mass accuracy
        # ------------------------------------------------------------------
        ppm_abs = _get(row, "ppm_error_abs")
        ppm_signed = _get(row, "ppm_error")
        if not np.isfinite(ppm_signed):
            ppm_signed = ppm_abs  # fall back to unsigned if signed not available
        ax[0][1].axvspan(-2, 2, alpha=0.22, color="steelblue", label="±2 ppm")
        ax[0][1].axvspan(-5, 5, alpha=0.10, color="steelblue", label="±5 ppm")
        ax[0][1].axvline(0, color="gray", lw=0.8, ls="--")
        if np.isfinite(ppm_signed):
            color = "tomato" if abs(ppm_signed) > 5 else "steelblue"
            ax[0][1].barh([0], [ppm_signed], height=0.5, color=color, alpha=0.85)
            ax[0][1].set_xlim(-max(10, abs(ppm_signed) * 1.4), max(10, abs(ppm_signed) * 1.4))
        ax[0][1].set_yticks([])
        ax[0][1].set_xlabel("ppm error", fontsize=8)
        title_ppm = f"Mass accuracy\n{ppm_abs:.2f} ppm" if np.isfinite(ppm_abs) else "Mass accuracy"
        ax[0][1].set_title(title_ppm, fontsize=8)
        ax[0][1].legend(fontsize=6, loc="upper right")

        # ------------------------------------------------------------------
        # [0,2] Isotope envelope comparison
        # ------------------------------------------------------------------
        obs_env = None
        if maldi_envelopes is not None and feature_mz is not None:
            obs_env = maldi_envelopes.get(feature_mz)

        theo_env = None
        comp_cols = ["n_C", "n_H", "n_N", "n_O", "n_S"]
        if all(c in row.index for c in comp_cols):
            try:
                comp = tuple(int(_get(row, c)) for c in comp_cols)
                if all(v >= 0 for v in comp):
                    theo_env = theoretical_isotope_distribution(*comp, n_peaks=3)
            except Exception:
                pass

        if obs_env is not None or theo_env is not None:
            x = np.arange(3)
            w = 0.35
            if theo_env is not None:
                te = np.asarray(theo_env[:3], dtype=float)
                te = te / te.max() if te.max() > 0 else te
                ax[0][2].bar(x - w / 2, te, width=w, label="Theoretical", color="steelblue", alpha=0.8)
            if obs_env is not None:
                oe = np.asarray(obs_env[:3], dtype=float)
                oe = oe / oe.max() if oe.max() > 0 else oe
                ax[0][2].bar(x + w / 2, oe, width=w, label="Observed", color="tomato", alpha=0.8)
            ax[0][2].set_xticks(x)
            ax[0][2].set_xticklabels(["M", "M+1", "M+2"], fontsize=8)
            ax[0][2].set_ylabel("Norm. intensity", fontsize=8)
            cosine = _get(row, "theo_isotope_cosine")
            cosine_s = f"cosine={cosine:.3f}" if np.isfinite(cosine) else ""
            ax[0][2].set_title(f"Isotope envelope\n{cosine_s}", fontsize=8)
            ax[0][2].legend(fontsize=6)
        else:
            ax[0][2].text(0.5, 0.5, "No envelope data", ha="center", va="center",
                          transform=ax[0][2].transAxes)
            ax[0][2].set_title("Isotope envelope", fontsize=8)

        # ------------------------------------------------------------------
        # [1,0] Peptide properties
        # ------------------------------------------------------------------
        _prop_cols = [
            ("n_arginine", "Arg count"),
            ("n_basic_residues", "Basic residues"),
            ("gravy_score", "GRAVY"),
            ("peptide_pi", "pI"),
            ("peptide_length", "Length"),
            ("n_missed_cleavages", "Missed cleavages"),
        ]
        pnames, pvals = [], []
        for col, lab in _prop_cols:
            v = _get(row, col)
            if np.isfinite(v):
                pnames.append(lab)
                pvals.append(v)
        if pnames:
            ax[1][0].barh(range(len(pnames)), pvals, color="steelblue", alpha=0.75)
            ax[1][0].set_yticks(range(len(pnames)))
            ax[1][0].set_yticklabels(pnames, fontsize=7)
        else:
            ax[1][0].text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax[1][0].transAxes)
        ax[1][0].set_title("Peptide properties", fontsize=8)

        # ------------------------------------------------------------------
        # [1,1] Spatial statistics
        # ------------------------------------------------------------------
        _spatial_cols = [
            ("spatial_autocorrelation", "Moran's I"),
            ("spatial_morans_i", "Moran's I (full)"),
            ("spatial_gearys_c", "Geary's C"),
            ("fraction_detected", "Fraction detected"),
            ("intensity_cv", "Intensity CV"),
            ("spatial_entropy", "Entropy"),
        ]
        snames, svals = [], []
        for col, lab in _spatial_cols:
            v = _get(row, col)
            if np.isfinite(v):
                snames.append(lab)
                svals.append(v)
        if snames:
            ax[1][1].barh(range(len(snames)), svals, color="seagreen", alpha=0.75)
            ax[1][1].set_yticks(range(len(snames)))
            ax[1][1].set_yticklabels(snames, fontsize=7)
        else:
            ax[1][1].text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax[1][1].transAxes)
        ax[1][1].set_title("Spatial statistics", fontsize=8)

        # ------------------------------------------------------------------
        # [1,2] LC-MS/MS + CCS features
        # ------------------------------------------------------------------
        _lcms_cols = [
            ("lcms_ms2_spectral_angle", "MS2 spectral angle"),
            ("lcms_ms1_intensity", "MS1 intensity"),
            ("lcms_ms1_snr", "MS1 SNR"),
            ("lcms_ms1_isotope_cosine", "MS1 iso cosine"),
            ("theo_m1_ratio_diff_lcms", "M+1 ratio diff (LC-MS)"),
            ("isotope_envelope_cosine", "Envelope cosine"),
            ("lcms_q_value", "LC-MS q-value"),
            ("im2deep_delta_ccs", "Δ CCS (Å²)"),
            ("im2deep_abs_delta_ccs_pct", "|Δ CCS| (%)"),
            ("im2deep_ccs_zscore", "CCS z-score"),
            ("im2deep_ccs_rank", "CCS rank"),
        ]
        lnames, lvals, lcolors = [], [], []
        for col, lab in _lcms_cols:
            v = _get(row, col)
            if np.isfinite(v):
                lnames.append(lab)
                lvals.append(v)
                lcolors.append("darkorange" if col.startswith("im2deep") else "mediumpurple")
        if lnames:
            ax[1][2].barh(range(len(lnames)), lvals, color=lcolors, alpha=0.75)
            ax[1][2].set_yticks(range(len(lnames)))
            ax[1][2].set_yticklabels(lnames, fontsize=7)
            ax[1][2].axvline(0, color="gray", lw=0.6, ls="--")
        else:
            ax[1][2].text(0.5, 0.5, "N/A", ha="center", va="center",
                          transform=ax[1][2].transAxes, fontsize=14, color="gray")
        ax[1][2].set_title("LC-MS/MS + CCS features", fontsize=8)

        # ------------------------------------------------------------------
        # [2,0] CHCA cluster proximity
        # ------------------------------------------------------------------
        chca_dist = _get(row, "chca_cluster_distance_ppm")
        if feature_mz is not None and feature_mz > 0:
            _ppm_dists = np.abs(_CHCA_CLUSTER_MZS - feature_mz) / feature_mz * 1e6
            _nearest_c_idx = int(np.argmin(_ppm_dists))
            nearest_cluster_mz = float(_CHCA_CLUSTER_MZS[_nearest_c_idx])
        else:
            nearest_cluster_mz = float("nan")

        if np.isfinite(chca_dist):
            _disp = min(chca_dist, 100.0)
            _bc = "tomato" if chca_dist < 20.0 else ("orange" if chca_dist < 50.0 else "seagreen")
            ax[2][0].barh([0], [_disp], height=0.5, color=_bc, alpha=0.85)
            ax[2][0].set_xlim(0, 100)
            ax[2][0].axvline(20.0, color="tomato", lw=1.0, ls="--", alpha=0.6, label="20 ppm")
            ax[2][0].axvline(50.0, color="orange", lw=1.0, ls="--", alpha=0.6, label="50 ppm")
            if chca_dist > 100.0:
                ax[2][0].text(96, 0, f">{chca_dist:.0f}", ha="right", va="center", fontsize=7)
            ax[2][0].set_yticks([])
            ax[2][0].set_xlabel("ppm to nearest CHCA cluster", fontsize=8)
            ax[2][0].legend(fontsize=6, loc="upper left")
            _ncmz_s = f" (CHCA@{nearest_cluster_mz:.4f})" if np.isfinite(nearest_cluster_mz) else ""
            ax[2][0].set_title(f"CHCA proximity\n{chca_dist:.1f} ppm{_ncmz_s}", fontsize=8)
        else:
            ax[2][0].text(0.5, 0.5, "N/A", ha="center", va="center",
                          transform=ax[2][0].transAxes, fontsize=14, color="gray")
            ax[2][0].set_title("CHCA proximity", fontsize=8)

        # ------------------------------------------------------------------
        # [2,1] CHCA adduct colocalization
        # ------------------------------------------------------------------
        chca_corr = _get(row, "adduct_colocalization_chca")
        _adduct_mz = (feature_mz + _CHCA_ADDUCT_DELTA) if feature_mz is not None else None
        _adduct_idx = (
            _find_image_idx(_adduct_mz, ion_image_mzs)
            if (_adduct_mz is not None and ion_image_mzs is not None)
            else None
        )

        if _adduct_idx is not None and prec_idx is not None:
            ax[2][1].axis("off")
            _axl = ax[2][1].inset_axes([0.02, 0.10, 0.44, 0.78])
            _axl.imshow(ion_images[prec_idx], cmap="hot", aspect="auto")
            _axl.set_title(f"Precursor\n{feature_mz:.4f}", fontsize=6)
            _axl.axis("off")
            _axr = ax[2][1].inset_axes([0.54, 0.10, 0.44, 0.78])
            _axr.imshow(ion_images[_adduct_idx], cmap="hot", aspect="auto")
            _axr.set_title(f"CHCA adduct\n{_adduct_mz:.4f}", fontsize=6)
            _axr.axis("off")
            _corr_s = f"r={chca_corr:.3f}" if np.isfinite(chca_corr) else "r=N/A"
            ax[2][1].set_title(f"CHCA adduct colocalization\n{_corr_s}", fontsize=8)
        else:
            if np.isfinite(chca_corr):
                _bar_color_chca = plt.cm.RdYlGn_r((chca_corr + 1.0) / 2.0)
                ax[2][1].barh([0], [chca_corr], height=0.5, color=_bar_color_chca, alpha=0.85)
                ax[2][1].set_xlim(-1.0, 1.0)
                ax[2][1].axvline(0.0, color="gray", lw=0.8, ls="--")
                ax[2][1].axvline(0.5, color="orange", lw=0.8, ls=":", alpha=0.7, label="r=0.5")
                ax[2][1].set_yticks([])
                ax[2][1].set_xlabel("Pearson r (precursor vs CHCA adduct)", fontsize=8)
                ax[2][1].legend(fontsize=6, loc="upper left")
                _corr_s = f"r={chca_corr:.3f}"
            else:
                ax[2][1].text(0.5, 0.5, "N/A", ha="center", va="center",
                              transform=ax[2][1].transAxes, fontsize=14, color="gray")
                _corr_s = "N/A"
            ax[2][1].set_title(f"CHCA adduct colocalization\n{_corr_s}", fontsize=8)

        # ------------------------------------------------------------------
        # [2,2] Monoisotopic confidence
        # ------------------------------------------------------------------
        mono_conf = _get(row, "monoisotopic_confidence")
        mono_idx_val = _get(row, "mono_isotope_index")

        if np.isfinite(mono_conf):
            _bar_color_mono = plt.cm.RdYlGn(float(mono_conf))
            ax[2][2].barh([0], [mono_conf], height=0.5, color=_bar_color_mono, alpha=0.85)
            ax[2][2].set_xlim(0.0, 1.0)
            ax[2][2].axvline(0.5, color="orange", lw=1.0, ls="--", alpha=0.8, label="0.5")
            ax[2][2].set_yticks([])
            ax[2][2].set_xlabel("Monoisotopic confidence", fontsize=8)
            ax[2][2].legend(fontsize=6, loc="upper left")
            if np.isfinite(mono_idx_val):
                _idx_label = {
                    0: "M₀ (correct)",
                    1: "M+1 detected as M₀",
                    2: "M+2 detected as M₀",
                }.get(int(mono_idx_val), f"index={int(mono_idx_val)}")
            else:
                _idx_label = ""
            _title_mono = f"Monoisotopic confidence\n{mono_conf:.3f}"
            if _idx_label:
                _title_mono += f"  —  {_idx_label}"
            ax[2][2].set_title(_title_mono, fontsize=8)
        else:
            ax[2][2].text(0.5, 0.5, "N/A", ha="center", va="center",
                          transform=ax[2][2].transAxes, fontsize=14, color="gray")
            ax[2][2].set_title("Monoisotopic confidence", fontsize=8)

        # ------------------------------------------------------------------
        fig.suptitle(_candidate_title(row), fontsize=9, y=1.01)
        plt.tight_layout()
        fname = f"{prefix}_{td}_{rank:03d}_{_safe_fname(peptide)}_{feature_mz:.4f}.png" if feature_mz is not None else f"{prefix}_{td}_{rank:03d}_{_safe_fname(peptide)}.png"
        fig.savefig(os.path.join(out_dir, fname), dpi=100, bbox_inches="tight")
        plt.close(fig)


# ---------------------------------------------------------------------------
# Subsystem 3: Isotope envelope
# ---------------------------------------------------------------------------

_NEUTRON = 1.003355
_ISOTOPE_DISPLAY_FWHM = 0.05  # Da — chosen for visual clarity, not instrument-specific

_CHCA_CLUSTER_MZS = np.array([
    172.0393, 190.0499, 212.0318, 228.0058,
    379.0925, 401.0744, 568.1351, 757.1777,
])
_CHCA_ADDUCT_DELTA = 172.0392  # [M+H]+ → [M+CHCA+H-H2O]+ m/z offset


def _gauss(mz_arr: np.ndarray, center: float, intensity: float) -> np.ndarray:
    sigma = _ISOTOPE_DISPLAY_FWHM / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    return intensity * np.exp(-0.5 * ((mz_arr - center) / sigma) ** 2)


def plot_isotope_envelope_figures(
    subset: pd.DataFrame,
    maldi_envelopes: dict | None,
    out_dir: str,
) -> None:
    """
    Per-candidate isotope envelope figure in spectrum style.

    Simulates peak shapes as Gaussians at the actual M, M+1, M+2 m/z positions
    so the x-axis is m/z (not categorical).  Observed MALDI envelope is shown as
    a filled blue trace; theoretical (from elemental composition, normalized to
    M0 = 1) is overlaid as a dashed red line.  Peak windows are shaded in orange
    (style matches optimize_maldi_params.py interval shading), and the detected
    apex positions are marked with triangles.
    """
    from ms1rescore.utils import theoretical_isotope_distribution

    os.makedirs(out_dir, exist_ok=True)

    for _, row in subset.iterrows():
        feature_mz = row.get("feature_mz")
        if feature_mz is not None:
            feature_mz = float(feature_mz)
        prefix = str(row.get("_group", "L"))
        td = str(row.get("_td", "T"))
        rank = int(row.get("_rank", 0))
        peptide = str(row.get("peptide", "unknown"))

        obs_env = None
        if maldi_envelopes is not None and feature_mz is not None:
            raw = maldi_envelopes.get(feature_mz)
            if raw is not None:
                obs_env = np.asarray(raw[:3], dtype=float)

        theo_env = None
        comp_cols = ["n_C", "n_H", "n_N", "n_O", "n_S"]
        if all(c in row.index for c in comp_cols):
            try:
                comp = tuple(int(_get(row, c)) for c in comp_cols)
                if all(v >= 0 for v in comp):
                    theo_env = np.asarray(
                        theoretical_isotope_distribution(*comp, n_peaks=3), dtype=float
                    )
            except Exception:
                pass

        if obs_env is None and theo_env is None:
            continue
        if feature_mz is None:
            continue

        n_peaks = 3
        peak_mzs = np.array([feature_mz + i * _NEUTRON for i in range(n_peaks)])

        # Normalize: M0 = 1 for whichever series is available
        obs_norm = None
        if obs_env is not None and obs_env[0] > 0:
            obs_norm = obs_env / obs_env[0]

        theo_norm = None
        if theo_env is not None and theo_env[0] > 0:
            theo_norm = theo_env / theo_env[0]

        # Fine m/z grid spanning all isotope peaks
        mz_lo = feature_mz - 3 * _ISOTOPE_DISPLAY_FWHM
        mz_hi = peak_mzs[-1] + 3 * _ISOTOPE_DISPLAY_FWHM
        mz_fine = np.linspace(mz_lo, mz_hi, 2000)

        fig, ax = plt.subplots(figsize=(8, 4))

        # Orange shading for peak windows (±FWHM around each peak)
        for mz_peak in peak_mzs:
            ax.axvspan(
                mz_peak - _ISOTOPE_DISPLAY_FWHM,
                mz_peak + _ISOTOPE_DISPLAY_FWHM,
                alpha=0.12, color="orange",
            )

        # Observed trace
        if obs_norm is not None:
            obs_spectrum = sum(_gauss(mz_fine, mz_peak, h)
                               for mz_peak, h in zip(peak_mzs, obs_norm))
            ax.fill_between(mz_fine, 0, obs_spectrum, color="steelblue", alpha=0.55,
                            label="Observed (MALDI)")
            ax.plot(mz_fine, obs_spectrum, color="steelblue", lw=1.0)
            # Apex markers
            for mz_peak, h in zip(peak_mzs, obs_norm):
                ax.plot(mz_peak, h, "^", color="steelblue", ms=8, zorder=5)

        # Theoretical overlay
        if theo_norm is not None:
            theo_spectrum = sum(_gauss(mz_fine, mz_peak, h)
                                for mz_peak, h in zip(peak_mzs, theo_norm))
            ax.plot(mz_fine, theo_spectrum, color="tomato", lw=1.8, ls="--",
                    label="Theoretical", zorder=4)
            for mz_peak, h in zip(peak_mzs, theo_norm):
                ax.axvline(mz_peak, color="tomato", lw=0.7, ls=":", alpha=0.6)

        # Peak m/z labels on x-axis
        ax.set_xticks(peak_mzs)
        ax.set_xticklabels(
            [f"M+{i}\n{mz:.4f}" for i, mz in enumerate(peak_mzs)], fontsize=8
        )
        ax.set_xlim(mz_lo, mz_hi)
        ax.set_ylim(bottom=0)
        ax.set_xlabel("m/z", fontsize=9)
        ax.set_ylabel("Relative intensity (M0 = 1)", fontsize=9)

        cosine = _get(row, "theo_isotope_cosine")
        m1_diff = _get(row, "theo_m1_ratio_diff")
        m2_diff = _get(row, "theo_m2_ratio_diff")
        parts = []
        if np.isfinite(cosine):
            parts.append(f"cosine={cosine:.3f}")
        if np.isfinite(m1_diff):
            parts.append(f"ΔM+1={m1_diff:+.3f}")
        if np.isfinite(m2_diff):
            parts.append(f"ΔM+2={m2_diff:+.3f}")
        ax.set_title("  ".join(parts), fontsize=8)
        ax.legend(fontsize=8)

        fig.suptitle(_candidate_title(row), fontsize=9, y=1.01)
        plt.tight_layout()
        fname = (
            f"{prefix}_{td}_{rank:03d}_{_safe_fname(peptide)}_{feature_mz:.4f}.png"
            if feature_mz is not None
            else f"{prefix}_{td}_{rank:03d}_{_safe_fname(peptide)}.png"
        )
        fig.savefig(os.path.join(out_dir, fname), dpi=100, bbox_inches="tight")
        plt.close(fig)


# ---------------------------------------------------------------------------
# Subsystem 4: Feature importance
# ---------------------------------------------------------------------------

def plot_feature_importance(
    feature_names: list[str],
    importances_r1: np.ndarray | None,
    importances_r2: np.ndarray | None,
    out_dir: str,
    model_name: str = "model",
    top_n: int = 30,
) -> None:
    """
    Save sorted horizontal bar charts of feature importances for rounds 1 and 2.

    Positive values are shown in steelblue, negative in tomato.
    Files: ``{out_dir}/{model_name}_round1_feature_importance.png`` etc.
    """
    os.makedirs(out_dir, exist_ok=True)

    def _one(importances: np.ndarray | None, suffix: str) -> None:
        if importances is None or len(importances) == 0:
            return
        importances = np.asarray(importances, dtype=float)
        if len(importances) != len(feature_names):
            logger.warning(
                "Feature importance length mismatch: %d importances vs %d names — skipping %s",
                len(importances), len(feature_names), suffix,
            )
            return
        order = np.argsort(np.abs(importances))[-top_n:]
        names = [feature_names[i] for i in order]
        vals = importances[order]
        colors = ["steelblue" if v >= 0 else "tomato" for v in vals]

        fig, axi = plt.subplots(figsize=(9, max(4, len(names) * 0.32)))
        axi.barh(range(len(names)), vals, color=colors, alpha=0.82)
        axi.set_yticks(range(len(names)))
        axi.set_yticklabels(names, fontsize=7)
        axi.axvline(0, color="black", lw=0.8)
        axi.set_xlabel("Importance", fontsize=9)
        round_label = suffix.replace("_", " ").title()
        axi.set_title(f"{model_name} — {round_label} feature importance (top {len(names)})", fontsize=10)
        plt.tight_layout()
        fig.savefig(
            os.path.join(out_dir, f"{model_name}_{suffix}_feature_importance.png"),
            dpi=100, bbox_inches="tight",
        )
        plt.close(fig)

    _one(importances_r1, "round1")
    _one(importances_r2, "round2")


# ---------------------------------------------------------------------------
# Subsystem 5: Feature distributions
# ---------------------------------------------------------------------------

_DIST_SKIP = frozenset({
    "is_decoy", "peptide", "protein", "feature_mz", "feature_idx", "source",
    "_group", "_td", "_rank", "_total", "_score_r1",
    "is_tdc_winner", "reweighted_score", "reweighted_q_value", "q_value",
})


def plot_feature_distributions(
    features_df: pd.DataFrame,
    result_df: pd.DataFrame,
    out_dir: str,
    feature_names: list[str] | None = None,
    gt_peptides: list[str] | None = None,
) -> None:
    """
    Per-feature target/decoy distribution figures.

    Two subplots per figure:
      Top    — all candidates
      Bottom — round-2 candidates (is_tdc_winner == True)

    Overlapping histograms are drawn in steelblue (target) and tomato (decoy)
    with dashed median lines.  When ``gt_peptides`` is provided, a solid green
    vertical line is drawn at each GT candidate's value on both subplots.

    Files: ``{out_dir}/{feature_name}.png``
    """
    os.makedirs(out_dir, exist_ok=True)

    feat = features_df.reset_index(drop=True)
    res = result_df.reset_index(drop=True)

    is_decoy = (
        feat.get("is_decoy", pd.Series(False, index=feat.index))
        .fillna(False).astype(bool).values
    )
    is_winner = (
        res.get("is_tdc_winner", pd.Series(False, index=res.index))
        .fillna(False).astype(bool).values
    )
    target_mask = ~is_decoy
    decoy_mask = is_decoy
    winner_target_mask = is_winner & target_mask
    winner_decoy_mask = is_winner & decoy_mask

    gt_mask = np.zeros(len(feat), dtype=bool)
    if gt_peptides and "peptide" in feat.columns:
        gt_set = set(gt_peptides)
        gt_mask = target_mask & feat["peptide"].isin(gt_set).values

    if feature_names is None:
        feature_names = [
            c for c in feat.columns
            if c not in _DIST_SKIP and pd.api.types.is_numeric_dtype(feat[c])
        ]

    def _draw(ax: plt.Axes, t_vals: np.ndarray, d_vals: np.ndarray,
               bins: np.ndarray, subtitle: str) -> None:
        ax.set_title(subtitle, fontsize=8)
        if len(t_vals) > 0:
            ax.hist(t_vals, bins=bins, density=True, alpha=0.55,
                    color="steelblue", label=f"Target (n={len(t_vals)})")
            ax.axvline(float(np.nanmedian(t_vals)), color="steelblue",
                       lw=1.3, ls="--", alpha=0.85)
        if len(d_vals) > 0:
            ax.hist(d_vals, bins=bins, density=True, alpha=0.55,
                    color="tomato", label=f"Decoy (n={len(d_vals)})")
            ax.axvline(float(np.nanmedian(d_vals)), color="tomato",
                       lw=1.3, ls="--", alpha=0.85)
        if len(t_vals) == 0 and len(d_vals) == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes, color="gray")
            return
        ax.legend(fontsize=7)
        ax.set_ylabel("Density", fontsize=8)
        ax.tick_params(labelsize=7)

    def _draw_gt(ax: plt.Axes, gt_vals: np.ndarray) -> None:
        for i, v in enumerate(gt_vals):
            ax.axvline(
                v, color="limegreen", lw=1.5, ls="-.", alpha=0.85,
                label=f"GT (n={len(gt_vals)})" if i == 0 else "_nolegend_",
            )
        if len(gt_vals) > 0:
            ax.legend(fontsize=7)

    for feat_col in feature_names:
        col = feat.get(feat_col)
        if col is None:
            continue
        vals = pd.to_numeric(col, errors="coerce").values.astype(float)

        finite_mask = np.isfinite(vals)
        t_all = vals[target_mask & finite_mask]
        d_all = vals[decoy_mask & finite_mask]
        t_r2 = vals[winner_target_mask & finite_mask]
        d_r2 = vals[winner_decoy_mask & finite_mask]
        gt_vals = vals[gt_mask & finite_mask]

        all_finite = vals[finite_mask]
        if len(all_finite) == 0:
            continue

        lo = float(np.percentile(all_finite, 1))
        hi = float(np.percentile(all_finite, 99))
        if lo >= hi:
            lo, hi = float(all_finite.min()), float(all_finite.max())
        if lo >= hi:
            lo, hi = lo - 1.0, hi + 1.0
        bins = np.linspace(lo, hi, 51)

        fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        fig.suptitle(feat_col, fontsize=10)

        _draw(ax_top, t_all, d_all, bins,
              f"All candidates  (T={len(t_all)}, D={len(d_all)})")
        _draw(ax_bot, t_r2, d_r2, bins,
              f"Round-2 candidates  (T={len(t_r2)}, D={len(d_r2)})")

        _draw_gt(ax_top, gt_vals)
        _draw_gt(ax_bot, gt_vals)

        ax_bot.set_xlabel(feat_col, fontsize=8)
        plt.tight_layout()
        fig.savefig(
            os.path.join(out_dir, f"{_safe_fname(feat_col, maxlen=80)}.png"),
            dpi=100, bbox_inches="tight",
        )
        plt.close(fig)


# ---------------------------------------------------------------------------
# Subsystem 6: CCS scatter
# ---------------------------------------------------------------------------


def plot_ccs_scatter(
    features_df: pd.DataFrame,
    result_df: pd.DataFrame,
    out_dir: str,
    fdr_threshold: float = 0.01,
    gt_peptides: list[str] | None = None,
) -> None:
    """
    Scatter plot of observed vs predicted CCS for all candidates.

    Requires ``im2deep_observed_ccs`` and ``im2deep_predicted_ccs`` columns in
    ``features_df`` (added by ``compute_im2deep_features``). Silently skips if
    neither column is present.

    Points are coloured by target/decoy status. "R2 winner" means the feature's
    best candidate AND reweighted_q_value <= fdr_threshold. R1 winners (best
    candidate but below FDR threshold) are shown at intermediate size.
    Saved to ``{out_dir}/ccs_scatter.png``.
    """
    if "im2deep_observed_ccs" not in features_df.columns:
        return
    if "im2deep_predicted_ccs" not in features_df.columns:
        return

    os.makedirs(out_dir, exist_ok=True)

    feat = features_df.reset_index(drop=True)
    res = result_df.reset_index(drop=True)

    obs = pd.to_numeric(feat["im2deep_observed_ccs"], errors="coerce").values
    pred = pd.to_numeric(feat["im2deep_predicted_ccs"], errors="coerce").values
    is_decoy = feat.get("is_decoy", pd.Series(False, index=feat.index)).fillna(False).astype(bool).values
    is_winner = res.get("is_tdc_winner", pd.Series(False, index=res.index)).fillna(False).astype(bool).values
    rw_q = pd.to_numeric(
        res.get("reweighted_q_value", pd.Series(float("nan"), index=res.index)),
        errors="coerce",
    ).values
    # R2 winner = round-2 TDC winner AND passes FDR; R1 = winner but below FDR
    passes_fdr = is_winner & (rw_q <= fdr_threshold)
    r1_only = is_winner & ~passes_fdr

    valid = np.isfinite(obs) & np.isfinite(pred)
    if not valid.any():
        return

    obs_v, pred_v = obs[valid], pred[valid]
    decoy_v = is_decoy[valid]
    fdr_v = passes_fdr[valid]
    r1_v = r1_only[valid]
    bg_v = ~fdr_v & ~r1_v

    fig, ax = plt.subplots(figsize=(7, 6))

    # Background: non-winner candidates
    ax.scatter(
        pred_v[bg_v & ~decoy_v], obs_v[bg_v & ~decoy_v],
        s=6, alpha=0.25, color="steelblue", linewidths=0, label="Target",
    )
    ax.scatter(
        pred_v[bg_v & decoy_v], obs_v[bg_v & decoy_v],
        s=6, alpha=0.25, color="tomato", linewidths=0, label="Decoy",
    )

    # Mid-layer: R1 winners (best per feature, but below FDR)
    if r1_v.any():
        ax.scatter(
            pred_v[r1_v & ~decoy_v], obs_v[r1_v & ~decoy_v],
            s=15, alpha=0.5, color="steelblue", linewidths=0, label="Target (R1 winner)",
        )
        ax.scatter(
            pred_v[r1_v & decoy_v], obs_v[r1_v & decoy_v],
            s=15, alpha=0.5, color="tomato", linewidths=0, label="Decoy (R1 winner)",
        )

    # Foreground: FDR-passing winners
    ax.scatter(
        pred_v[fdr_v & ~decoy_v], obs_v[fdr_v & ~decoy_v],
        s=50, alpha=0.9, color="steelblue", edgecolors="navy", linewidths=0.7,
        label=f"Target (FDR ≤ {fdr_threshold:.0%})", zorder=5,
    )
    ax.scatter(
        pred_v[fdr_v & decoy_v], obs_v[fdr_v & decoy_v],
        s=50, alpha=0.9, color="tomato", edgecolors="darkred", linewidths=0.7,
        label=f"Decoy (FDR ≤ {fdr_threshold:.0%})", zorder=5,
    )

    # GT peptide overlay
    if gt_peptides:
        gt_set = set(gt_peptides)
        pep_col = feat.get("peptide", pd.Series(dtype=str)).values
        gt_mask = np.array([p in gt_set for p in pep_col], dtype=bool) & valid
        if gt_mask.any():
            gt_pred = pred[gt_mask]
            gt_obs = obs[gt_mask]
            gt_names = pep_col[gt_mask]
            ax.scatter(
                gt_pred, gt_obs,
                s=150, marker="*", color="black", edgecolors="darkorange", linewidths=0.8,
                zorder=10,
            )

    # y = x reference line
    lo = min(float(pred_v.min()), float(obs_v.min()))
    hi = max(float(pred_v.max()), float(obs_v.max()))
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.0, alpha=0.5, label="y = x")

    # Linear regression across all valid points
    try:
        coeffs = np.polyfit(pred_v, obs_v, 1)
        x_fit = np.linspace(float(pred_v.min()), float(pred_v.max()), 200)
        ax.plot(x_fit, np.polyval(coeffs, x_fit), color="gray", lw=1.5,
                ls="-", alpha=0.7,
                label=f"fit: y={coeffs[0]:.3f}x{coeffs[1]:+.1f}")
    except Exception:
        pass

    # Correlation annotation
    corr_all = float(np.corrcoef(pred_v, obs_v)[0, 1]) if len(pred_v) > 1 else float("nan")
    if fdr_v.sum() > 1:
        corr_fdr = float(np.corrcoef(pred_v[fdr_v], obs_v[fdr_v])[0, 1])
        corr_str = f"r (all) = {corr_all:.3f}   r (FDR ≤ {fdr_threshold:.0%}) = {corr_fdr:.3f}"
    else:
        corr_str = f"r = {corr_all:.3f}"

    ax.set_xlabel("Predicted CCS (Å²)", fontsize=10)
    ax.set_ylabel("Observed CCS (Å²)", fontsize=10)
    ax.set_title(f"Observed vs Predicted CCS\n{corr_str}", fontsize=10)
    ax.legend(fontsize=7, markerscale=1.5)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "ccs_scatter.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Ground-truth helpers
# ---------------------------------------------------------------------------

def _make_gt_subset(
    gt_peptides: list[str],
    features_df: pd.DataFrame,
    result_df: pd.DataFrame,
) -> tuple["pd.DataFrame | None", list[str]]:
    """
    Build a subset DataFrame for GT peptides, parallel to _sample_subset output.

    Returns (gt_subset, not_found) where not_found lists GT peptides absent from
    features_df entirely.
    """
    feat = features_df.reset_index(drop=True)
    res = result_df.reset_index(drop=True)

    peptide_vals = feat.get("peptide", pd.Series(dtype=str)).values
    gt_set = set(gt_peptides)
    not_found = [p for p in gt_peptides if p not in peptide_vals]

    matched_idx = [i for i, p in enumerate(peptide_vals) if p in gt_set]
    if not matched_idx:
        return None, list(gt_set)

    subset = feat.iloc[matched_idx].copy().reset_index(drop=True)
    subset["_group"] = "GT"
    is_dec = (
        subset.get("is_decoy", pd.Series(False, index=subset.index))
        .fillna(False).astype(bool).values
    )
    subset["_td"] = np.where(is_dec, "D", "T")

    score_r1_cols = [c for c in result_df.columns if c.endswith("_score_r1")]
    score_r2_cols = [c for c in result_df.columns if c.endswith("_score_r2")]
    display_cols = score_r1_cols + score_r2_cols + [
        c for c in ["q_value", "is_tdc_winner", "reweighted_score", "reweighted_q_value"]
        if c in result_df.columns
    ]
    res_matched = res.iloc[matched_idx][display_cols].reset_index(drop=True)
    for col in display_cols:
        subset[col] = res_matched[col].values

    if score_r1_cols:
        subset["_score_r1"] = subset[score_r1_cols[0]]
    else:
        subset["_score_r1"] = np.nan

    subset["_rank"] = (
        subset["_score_r1"]
        .rank(ascending=False, method="min", na_option="bottom")
        .astype(int)
    )
    subset["_total"] = len(feat)
    return subset, not_found


def _save_gt_not_found_figures(peptides: list[str], subdirs: list[str]) -> None:
    """Save a 'not a candidate' placeholder figure for each unfound GT peptide."""
    for sub in subdirs:
        os.makedirs(sub, exist_ok=True)
        for pep in peptides:
            fig, ax = plt.subplots(figsize=(6, 2))
            ax.text(
                0.5, 0.5,
                f"GT peptide '{pep}' is not a candidate",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=12, color="gray",
            )
            ax.axis("off")
            fig.suptitle(f"GT: {pep}", fontsize=10)
            fname = f"GT_T_000_{_safe_fname(pep)}.png"
            fig.savefig(os.path.join(sub, fname), dpi=80, bbox_inches="tight")
            plt.close(fig)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def save_debug_figures(
    features_df: pd.DataFrame,
    result_df: pd.DataFrame,
    *,
    ion_images: np.ndarray | None = None,
    ion_image_mzs: np.ndarray | None = None,
    maldi_envelopes: dict | None = None,
    feature_names: list[str] | None = None,
    model_name: str = "model",
    importances_r1: np.ndarray | None = None,
    importances_r2: np.ndarray | None = None,
    importance_names: list[str] | None = None,
    debug_dir: str = "debug",
    n_subset: int = 50,
    seed: int = 42,
    gt_peptides: list[str] | None = None,
) -> None:
    """
    Generate all debug figures and save them under ``debug_dir``.

    Parameters
    ----------
    features_df
        Full candidate DataFrame (all rows, before any filtering), same row
        order as ``result_df``.
    result_df
        Scoring output from ``rescore()`` (same row order as ``features_df``).
    ion_images
        MALDI ion images array, shape (n_features, H, W) float32.
    ion_image_mzs
        m/z values corresponding to ``ion_images`` rows (length n_features).
    maldi_envelopes
        Dict mapping float feature_mz → [m0_mean, m1_mean, m2_mean].
    feature_names
        Feature names used by the model (for importance plots).
    model_name
        Scoring model identifier used in output file names.
    importances_r1, importances_r2
        Feature importance arrays aligned with ``importance_names``.
    importance_names
        Feature names aligned with importance arrays; defaults to
        ``feature_names`` when omitted.
    debug_dir
        Root output directory.  Sub-directories are created automatically.
    n_subset
        Number of candidates to sample for per-candidate figures.
    seed
        Random seed for reproducible sampling.
    """
    os.makedirs(debug_dir, exist_ok=True)

    subset = _sample_subset(features_df, result_df, n=n_subset, seed=seed)
    logger.info("Debug viz: sampled %d candidates from %d", len(subset), len(features_df))

    if ion_images is not None:
        try:
            # Build a full (unsampled) set of FDR ≤ 1% winners for ion images.
            feat_aligned = features_df.reset_index(drop=True)
            res_aligned = result_df.reset_index(drop=True)
            _is_winner = (
                res_aligned.get("is_tdc_winner", pd.Series(False, index=res_aligned.index))
                .fillna(False).astype(bool)
            )
            _rw_q = pd.to_numeric(
                res_aligned.get("reweighted_q_value", pd.Series(float("nan"), index=res_aligned.index)),
                errors="coerce",
            )
            _id_idx = np.where((_is_winner & (_rw_q <= 0.01)).values)[0].tolist()
            _score_r1_cols = [c for c in result_df.columns if c.endswith("_score_r1")]
            _display_cols = _score_r1_cols + [
                c for c in ["q_value", "is_tdc_winner", "reweighted_score", "reweighted_q_value"]
                if c in result_df.columns
            ]
            id_subset = feat_aligned.iloc[_id_idx].copy().reset_index(drop=True)
            id_subset["_group"] = "ID"
            _is_dec = (
                id_subset.get("is_decoy", pd.Series(False, index=id_subset.index))
                .fillna(False).astype(bool).values
            )
            id_subset["_td"] = np.where(_is_dec, "D", "T")
            _res_id = res_aligned.iloc[_id_idx][_display_cols].reset_index(drop=True)
            for _col in _display_cols:
                id_subset[_col] = _res_id[_col].values
            id_subset["_score_r1"] = id_subset[_score_r1_cols[0]] if _score_r1_cols else np.nan
            id_subset["_rank"] = (
                id_subset["_score_r1"]
                .rank(ascending=False, method="min", na_option="bottom")
                .astype(int)
            )
            id_subset["_total"] = len(feat_aligned)

            # Combine: all ID rows + sampled R1/L rows from the existing subset.
            subset_for_images = pd.concat(
                [id_subset, subset[subset["_group"].isin(["R1", "L"])].copy()],
                ignore_index=True,
            )
            logger.info(
                "Ion image colocalization: %d FDR winners + %d R1/L sampled = %d total",
                len(id_subset),
                len(subset_for_images) - len(id_subset),
                len(subset_for_images),
            )
            plot_ion_image_colocalization(
                subset_for_images, features_df, ion_images, ion_image_mzs,
                out_dir=os.path.join(debug_dir, "ion_images"),
            )
            logger.info("Ion image colocalization figures saved to %s/ion_images/", debug_dir)
        except Exception as exc:
            logger.warning("Ion image colocalization figures failed: %s", exc)

    try:
        plot_feature_diagnostics(
            subset, features_df, ion_images, ion_image_mzs, maldi_envelopes,
            out_dir=os.path.join(debug_dir, "features"),
        )
        logger.info("Feature diagnostic figures saved to %s/features/", debug_dir)
    except Exception as exc:
        logger.warning("Feature diagnostic figures failed: %s", exc)

    try:
        plot_isotope_envelope_figures(
            subset, maldi_envelopes,
            out_dir=os.path.join(debug_dir, "isotope_envelopes"),
        )
        logger.info("Isotope envelope figures saved to %s/isotope_envelopes/", debug_dir)
    except Exception as exc:
        logger.warning("Isotope envelope figures failed: %s", exc)

    try:
        plot_feature_distributions(
            features_df, result_df,
            out_dir=os.path.join(debug_dir, "feature_distributions"),
            feature_names=feature_names,
            gt_peptides=gt_peptides,
        )
        logger.info("Feature distribution figures saved to %s/feature_distributions/", debug_dir)
    except Exception as exc:
        logger.warning("Feature distribution figures failed: %s", exc)

    try:
        plot_ccs_scatter(
            features_df, result_df,
            out_dir=debug_dir,
            gt_peptides=gt_peptides,
        )
        if "im2deep_observed_ccs" in features_df.columns:
            logger.info("CCS scatter saved to %s/ccs_scatter.png", debug_dir)
    except Exception as exc:
        logger.warning("CCS scatter failed: %s", exc)

    if importances_r1 is not None or importances_r2 is not None:
        imp_names = importance_names or feature_names or []
        try:
            plot_feature_importance(
                imp_names, importances_r1, importances_r2,
                out_dir=os.path.join(debug_dir, "feature_importance"),
                model_name=model_name,
            )
            logger.info("Feature importance figures saved to %s/feature_importance/", debug_dir)
        except Exception as exc:
            logger.warning("Feature importance figures failed: %s", exc)

    # --- Ground-truth peptide figures ---
    if gt_peptides:
        try:
            gt_subset, not_found = _make_gt_subset(gt_peptides, features_df, result_df)
            if not_found:
                _save_gt_not_found_figures(
                    not_found,
                    subdirs=[
                        os.path.join(debug_dir, "features"),
                        os.path.join(debug_dir, "isotope_envelopes"),
                    ],
                )
                logger.info(
                    "GT peptides not found as candidates (%d): %s",
                    len(not_found), ", ".join(not_found),
                )
            if gt_subset is not None:
                logger.info(
                    "GT debug viz: %d rows for %d GT peptides",
                    len(gt_subset), len(gt_peptides) - len(not_found),
                )
                try:
                    plot_feature_diagnostics(
                        gt_subset, features_df, ion_images, ion_image_mzs, maldi_envelopes,
                        out_dir=os.path.join(debug_dir, "features"),
                    )
                except Exception as exc:
                    logger.warning("GT feature diagnostic figures failed: %s", exc)
                try:
                    plot_isotope_envelope_figures(
                        gt_subset, maldi_envelopes,
                        out_dir=os.path.join(debug_dir, "isotope_envelopes"),
                    )
                except Exception as exc:
                    logger.warning("GT isotope envelope figures failed: %s", exc)
                if ion_images is not None:
                    try:
                        plot_ion_image_colocalization(
                            gt_subset, features_df, ion_images, ion_image_mzs,
                            out_dir=os.path.join(debug_dir, "ion_images"),
                        )
                    except Exception as exc:
                        logger.warning("GT ion image figures failed: %s", exc)
        except Exception as exc:
            logger.warning("GT debug figures failed: %s", exc)
