"""
Debug visualization for the MALDI-MSI rescoring pipeline.

Fourteen subsystems:
  1. Ion image colocalization  — per-candidate precursor + ALL same-protein co-feature images; each panel framed by ID FDR (dark green ≤1%, light green ≤5%, white otherwise)
  2. Feature diagnostics       — per-candidate 4×3 panel figure (incl. m/z-detrended CCS, ion-image colocalization, theoretical isotope/mass defect)
  3. Isotope envelopes         — per-candidate spectrum-style envelope comparison
  4. Feature importance        — global sorted bar plots (rounds 1 and 2)
  5. Feature distributions     — per-feature target/decoy histograms (all + R2)
  6. CCS scatter               — observed vs predicted CCS for all candidates
  7. IDs vs FDR curve          — target identifications as a function of FDR threshold
  8. Protein colocalization    — colocalization values split by scoring group
  9. T/D m/z distribution      — target vs decoy m/z coverage and competition status
 10. Candidate competition     — target/decoy candidate counts per feature (with CCS-filter note)
 11. Score PP plot             — empirical CDF of decoy scores vs target scores
 12. Score distributions       — target/decoy score histograms at R1, R2, and reweighted
 13. Pearson r distribution    — same-protein vs different-protein ion image Pearson r at 5% FDR
 14. Protein spatial coherence — per-protein peptide count vs mean ion image Pearson r at 5% FDR
 15. Region ion-image panels   — per-protein ion images + region overlay + profile bar (region-coloc debug folder)
 16. Isotope-envelope CCS      — spread by class, per-feature CCS profiles, spread vs intensity, peak counts

Entry point: save_debug_figures()
"""

import logging
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings(
    "ignore",
    message="This figure includes Axes that are not compatible with tight_layout",
    category=UserWarning,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _save_and_close(fig, path, dpi=120):
    """Save a figure with the standard tight bounding box and close it."""
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


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

    # If no candidates pass 1% FDR, seed the ID stratum from the 5 winners
    # with the lowest PEP so at least one high-confidence example appears in
    # every per-candidate debug figure.
    pep_col = pd.to_numeric(
        res.get("pep", pd.Series(float("nan"), index=res.index)),
        errors="coerce",
    )
    if not passes.any() and "pep" in res.columns:
        winner_indices = np.where(is_winner.values)[0]
        if len(winner_indices) > 0:
            winner_pep = pep_col.values[winner_indices]
            finite_mask = np.isfinite(winner_pep)
            if finite_mask.any():
                ranked = np.argsort(winner_pep[finite_mask])
                top5 = winner_indices[np.where(finite_mask)[0][ranked[:5]]]
                groups[top5] = "ID"
                passes = pd.Series(groups == "ID", index=res.index)

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


def _fdr_frame_color(
    qval: float, fdr_strict: float = 0.01, fdr_loose: float = 0.05
) -> str:
    """Frame colour for an ion-image panel by the identified peptide's FDR:
    dark green at q ≤ 1%, light green at q ≤ 5%, white otherwise (incl. NaN)."""
    try:
        q = float(qval)
    except (TypeError, ValueError):
        return "white"
    if not np.isfinite(q):
        return "white"
    if q <= fdr_strict:
        return "#006400"  # dark green
    if q <= fdr_loose:
        return "#90EE90"  # light green
    return "white"


# ---------------------------------------------------------------------------
# Subsystem 1: Ion image colocalization
# ---------------------------------------------------------------------------

def _mz_diverse_order(df: pd.DataFrame, mz_col: str = "feature_mz") -> pd.DataFrame:
    """
    Reorder rows so that features with the most spread-out m/z values appear
    first.  Uses greedy farthest-point selection: seed with the row closest to
    the median m/z, then iteratively pick the row whose m/z is farthest from
    all already-selected rows.  Rows without a finite m/z sink to the end.

    This prevents badly-extracted features (the same peptide peak split into
    several nearby m/z entries) from dominating the beginning of the figure
    output, where the most distinct/informative images should appear.
    """
    if len(df) <= 1 or mz_col not in df.columns:
        return df
    mzs = pd.to_numeric(df[mz_col], errors="coerce").values
    valid_idx = np.where(np.isfinite(mzs))[0]
    invalid_idx = np.where(~np.isfinite(mzs))[0]
    if len(valid_idx) <= 1:
        return df

    mzs_v = mzs[valid_idx]
    nv = len(mzs_v)
    selected = np.zeros(nv, dtype=bool)
    min_dist = np.full(nv, np.inf)

    # Seed: valid row closest to the median m/z.
    seed = int(np.argmin(np.abs(mzs_v - float(np.median(mzs_v)))))
    order_v: list[int] = [seed]
    selected[seed] = True
    min_dist = np.abs(mzs_v - mzs_v[seed])

    while len(order_v) < nv:
        # Among unselected valid rows, pick the one farthest from all selected.
        available = np.where(selected, -np.inf, min_dist)
        nxt = int(np.argmax(available))
        order_v.append(nxt)
        selected[nxt] = True
        min_dist = np.minimum(min_dist, np.abs(mzs_v - mzs_v[nxt]))

    # Map local indices back to DataFrame row indices; invalid rows go last.
    full_order = valid_idx[order_v].tolist() + invalid_idx.tolist()
    return df.iloc[full_order].reset_index(drop=True)

def plot_ion_image_colocalization(
    subset: pd.DataFrame,
    features_df: pd.DataFrame,
    ion_images: np.ndarray,
    ion_image_mzs: np.ndarray,
    out_dir: str,
    feature_qvals: dict | None = None,
    feature_peptides: dict | None = None,
) -> None:
    """
    One figure **per protein** (the caller collapses ``subset`` to one
    representative row — the lowest-q peptide — per protein): the representative
    feature's ion image + ALL same-protein co-feature images (ranked by
    reweighted q-value ascending) + protein mean. Per-peptide figures of the same
    protein would show the identical feature set in a different order, so only the
    protein-level figure is emitted.

    Co-feature panels show the same-protein candidate peptide as the label.  When
    a different-protein peptide is the TDC winner at that feature, it is annotated
    as "(not winner: <winner>)" so mass-coincidence competitors are visible.

    Files are saved as ``{out_dir}/{T|D}_{rank:03d}_{protein}.png`` (rank = protein
    rank by best q-value).
    Panels are arranged in a grid of up to 8 columns.  Each ion-image panel is
    framed by the identified peptide's FDR at that feature (``feature_qvals``):
    dark green at q ≤ 1%, light green at q ≤ 5%, white (no frame) otherwise.
    """
    os.makedirs(out_dir, exist_ok=True)

    n_saved = 0
    for _, row in subset.iterrows():
        try:
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

            # Collect co-feature images for the same protein, ranked by reweighted q-value.
            co_imgs: list[np.ndarray] = []
            co_mzs: list[float] = []
            co_pep_labels: list[str] = []
            if protein and "protein" in features_df.columns and "feature_mz" in features_df.columns:
                prot_mzs = (
                    features_df.loc[features_df["protein"] == protein, "feature_mz"]
                    .dropna()
                    .unique()
                )
                co_mz_candidates = [float(m) for m in prot_mzs if abs(float(m) - feature_mz) > 1e-6]
                co_mz_candidates.sort(
                    key=lambda m: (
                        0 if (feature_qvals and np.isfinite(feature_qvals.get(m, float("nan")))) else 1,
                        feature_qvals.get(m, float("inf")) if feature_qvals else m,
                    )
                )
                for mz in co_mz_candidates:
                    co_img_idx = _find_image_idx(mz, ion_image_mzs)
                    if co_img_idx is not None:
                        same_prot_peps = features_df.loc[
                            (features_df["feature_mz"] == mz) & (features_df["protein"] == protein),
                            "peptide",
                        ]
                        co_pep = str(same_prot_peps.iloc[0]) if len(same_prot_peps) > 0 else ""
                        winner_pep = feature_peptides.get(mz, "") if feature_peptides else ""
                        label = co_pep
                        if co_pep and winner_pep and co_pep != winner_pep:
                            label += f"\n(not winner: {winner_pep})"
                        co_imgs.append(ion_images[co_img_idx])
                        co_mzs.append(mz)
                        co_pep_labels.append(label)

            all_imgs = [prec_img] + co_imgs
            prot_mean = np.mean(all_imgs, axis=0)

            n_panels = 1 + len(co_imgs) + 1
            _ncols = min(n_panels, 8)
            _nrows = (n_panels + _ncols - 1) // _ncols
            fig, _axes_grid = plt.subplots(
                _nrows, _ncols,
                figsize=(3.2 * _ncols, 3.8 * _nrows),
                squeeze=False,
            )
            axes = _axes_grid.ravel()

            def _panel(
                ax: plt.Axes, img: np.ndarray, title: str,
                r: float | None = None, qval: float = float("nan"),
            ) -> None:
                im = ax.imshow(img, cmap="hot", aspect="auto")
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cap = title if r is None else f"{title}\nr={r:.2f}"
                ax.set_title(cap, fontsize=7)
                # FDR-coded frame: dark green at ≤1% FDR, light green at ≤5%, white otherwise.
                ax.set_xticks([])
                ax.set_yticks([])
                color = _fdr_frame_color(qval)
                lw = 0.0 if color == "white" else 3.5
                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_color(color)
                    spine.set_linewidth(lw)

            _prec_q = feature_qvals.get(feature_mz, float("nan")) if feature_qvals else float("nan")
            _prec_q_s = f"\nq={_prec_q:.3f}" if np.isfinite(_prec_q) else ""
            _panel(axes[0], prec_img, f"Precursor\n{feature_mz:.4f}{_prec_q_s}", qval=_prec_q)
            for i, (cimg, cmz, clabel) in enumerate(zip(co_imgs, co_mzs, co_pep_labels)):
                _q = feature_qvals.get(cmz, float("nan")) if feature_qvals else float("nan")
                _q_s = f"\nq={_q:.3f}" if np.isfinite(_q) else ""
                _co_pep_s = f"\n{clabel}" if clabel else ""
                _panel(axes[1 + i], cimg, f"{cmz:.4f}{_co_pep_s}{_q_s}",
                       r=_pearson_r(prec_img, cimg), qval=_q)
            _panel(axes[n_panels - 1], prot_mean, f"Protein mean\n({len(all_imgs)} imgs)",
                   r=_pearson_r(prec_img, prot_mean))
            for ax in axes[n_panels:]:
                ax.axis("off")

            fig.suptitle(_candidate_title(row), fontsize=8, y=1.01)
            plt.tight_layout()
            _prot_tag = _safe_fname(str(protein)) if protein else _safe_fname(peptide)
            fname = f"{td}_{rank:03d}_{_prot_tag}.png"
            _save_and_close(fig, os.path.join(out_dir, fname), dpi=100)
            n_saved += 1
        except Exception as _row_exc:
            logger.debug(
                "Ion image colocalization: skipped row (feature_mz=%s): %s",
                row.get("feature_mz"),
                _row_exc,
            )
            try:
                plt.close("all")
            except Exception:
                pass
    if n_saved == 0:
        logger.warning(
            "Ion image colocalization: 0 figures saved from %d candidates "
            "(ion_image_mzs has %d entries; check feature_mz alignment)",
            len(subset),
            len(ion_image_mzs) if ion_image_mzs is not None else 0,
        )


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
    Per-candidate 4×3 diagnostic figure.

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
      [3,0] CCS: raw vs m/z-detrended (``im2deep_*`` vs ``im2deep_*_resid``)
      [3,1] Ion-image colocalization (isotopologue + adduct Pearson r, incl. ``_mob``)
      [3,2] Theoretical isotope + mass-defect detail (bar chart)

    The bottom row surfaces features introduced after the original 3×3 layout:
    the m/z-detrended CCS variants (the ``mz_shuffle`` decoy-leak fix), the
    isotopologue/adduct ion-image colocalizations, and the theoretical-isotope
    and mass-defect quantities.
    """
    from msi_picasso.utils import theoretical_isotope_distribution

    os.makedirs(out_dir, exist_ok=True)

    for _, row in subset.iterrows():
        feature_mz = row.get("feature_mz")
        if feature_mz is not None:
            feature_mz = float(feature_mz)
        prefix = str(row.get("_group", "L"))
        td = str(row.get("_td", "T"))
        rank = int(row.get("_rank", 0))
        peptide = str(row.get("peptide", "unknown"))

        fig = plt.figure(figsize=(15, 16))
        gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.6, wspace=0.38)
        ax = [[fig.add_subplot(gs[r, c]) for c in range(3)] for r in range(4)]

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
        # [3,0] CCS: raw vs m/z-detrended (im2deep_* vs im2deep_*_resid)
        # ------------------------------------------------------------------
        # The *_resid variants subtract the expected m/z-gap CCS difference, so
        # for relocated decoys (mz_shift / mz_shuffle / entrapment) they remove
        # the trivial m/z-baseline separation that the raw deltas would leak.
        _ccs_pairs = [
            ("im2deep_delta_ccs", "im2deep_delta_ccs_resid", "Δ CCS (Å²)"),
            ("im2deep_abs_delta_ccs_pct", "im2deep_abs_delta_ccs_pct_resid", "|Δ CCS| (%)"),
            ("im2deep_ccs_zscore", "im2deep_ccs_zscore_resid", "CCS z-score"),
            ("im2deep_ccs_rank", "im2deep_ccs_rank_resid", "CCS rank"),
        ]
        _ccs_names, _ccs_raw, _ccs_res = [], [], []
        for raw_col, res_col, lab in _ccs_pairs:
            rv, sv = _get(row, raw_col), _get(row, res_col)
            if np.isfinite(rv) or np.isfinite(sv):
                _ccs_names.append(lab)
                _ccs_raw.append(rv if np.isfinite(rv) else 0.0)
                _ccs_res.append(sv if np.isfinite(sv) else 0.0)
        if _ccs_names:
            _y = np.arange(len(_ccs_names))
            _h = 0.38
            ax[3][0].barh(_y + _h / 2, _ccs_raw, height=_h, color="darkorange",
                          alpha=0.85, label="raw")
            ax[3][0].barh(_y - _h / 2, _ccs_res, height=_h, color="teal",
                          alpha=0.85, label="m/z-detrended")
            ax[3][0].set_yticks(_y)
            ax[3][0].set_yticklabels(_ccs_names, fontsize=7)
            ax[3][0].axvline(0, color="gray", lw=0.6, ls="--")
            ax[3][0].legend(fontsize=6, loc="best")
        else:
            ax[3][0].text(0.5, 0.5, "N/A", ha="center", va="center",
                          transform=ax[3][0].transAxes, fontsize=14, color="gray")
        ax[3][0].set_title("CCS: raw vs m/z-detrended", fontsize=8)

        # ------------------------------------------------------------------
        # [3,1] Ion-image colocalization (isotopologue + adduct Pearson r)
        # ------------------------------------------------------------------
        _coloc_cols = [
            ("isotope_image_colocalization_m1", "iso M+1"),
            ("isotope_image_colocalization_m2", "iso M+2"),
            ("isotope_image_colocalization_mean", "iso mean"),
            ("adduct_colocalization_na", "adduct Na"),
            ("adduct_colocalization_k", "adduct K"),
            ("adduct_colocalization_chca", "adduct CHCA"),
            ("protein_colocalization", "protein (mean)"),
            # Mobility-gated variants (raw-query / --mob-coloc only).
            ("isotope_colocalization_mean_mob", "iso mean (mob)"),
            ("adduct_colocalization_chca_mob", "adduct CHCA (mob)"),
        ]
        _cnames, _cvals = [], []
        for col, lab in _coloc_cols:
            v = _get(row, col)
            if np.isfinite(v):
                _cnames.append(lab)
                _cvals.append(v)
        if _cnames:
            _ccolors = ["seagreen" if v >= 0 else "tomato" for v in _cvals]
            ax[3][1].barh(range(len(_cnames)), _cvals, color=_ccolors, alpha=0.78)
            ax[3][1].set_yticks(range(len(_cnames)))
            ax[3][1].set_yticklabels(_cnames, fontsize=7)
            ax[3][1].set_xlim(-1.0, 1.0)
            ax[3][1].axvline(0.0, color="gray", lw=0.8, ls="--")
            ax[3][1].axvline(0.5, color="orange", lw=0.7, ls=":", alpha=0.7)
            ax[3][1].set_xlabel("Pearson r", fontsize=8)
        else:
            ax[3][1].text(0.5, 0.5, "N/A", ha="center", va="center",
                          transform=ax[3][1].transAxes, fontsize=14, color="gray")
        ax[3][1].set_title("Ion-image colocalization", fontsize=8)

        # ------------------------------------------------------------------
        # [3,2] Theoretical isotope + mass-defect detail
        # ------------------------------------------------------------------
        _theo_cols = [
            ("theo_isotope_cosine", "iso cosine"),
            ("theo_isotope_chi2", "iso χ²"),
            ("theo_isotope_kl", "iso KL"),
            ("averagine_deviation", "averagine dev"),
            ("averagine_deviation_sulfur", "averagine dev (S)"),
            ("theo_m1_ratio_diff", "ΔM+1 ratio"),
            ("theo_m2_ratio_diff", "ΔM+2 ratio"),
            ("kendrick_mass_defect", "Kendrick defect"),
            ("mass_defect_residual", "mass-defect resid"),
        ]
        _tnames, _tvals = [], []
        for col, lab in _theo_cols:
            v = _get(row, col)
            if np.isfinite(v):
                _tnames.append(lab)
                _tvals.append(v)
        if _tnames:
            ax[3][2].barh(range(len(_tnames)), _tvals, color="slateblue", alpha=0.78)
            ax[3][2].set_yticks(range(len(_tnames)))
            ax[3][2].set_yticklabels(_tnames, fontsize=7)
            ax[3][2].axvline(0, color="gray", lw=0.6, ls="--")
        else:
            ax[3][2].text(0.5, 0.5, "N/A", ha="center", va="center",
                          transform=ax[3][2].transAxes, fontsize=14, color="gray")
        ax[3][2].set_title("Theoretical isotope + mass defect", fontsize=8)

        # ------------------------------------------------------------------
        fig.suptitle(_candidate_title(row), fontsize=9, y=1.01)
        plt.tight_layout()
        fname = f"{prefix}_{td}_{rank:03d}_{_safe_fname(peptide)}_{feature_mz:.4f}.png" if feature_mz is not None else f"{prefix}_{td}_{rank:03d}_{_safe_fname(peptide)}.png"
        _save_and_close(fig, os.path.join(out_dir, fname), dpi=100)


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
    from msi_picasso.utils import theoretical_isotope_distribution

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
        _save_and_close(fig, os.path.join(out_dir, fname), dpi=100)


# ---------------------------------------------------------------------------
# Subsystem 4: Feature importance
# ---------------------------------------------------------------------------

def plot_feature_importance(
    names_r1: list[str],
    importances_r1: np.ndarray | None,
    importances_r2: np.ndarray | None,
    out_dir: str,
    model_name: str = "model",
    top_n: int = 30,
    names_r2: list[str] | None = None,
    structure_coefs_r1: np.ndarray | None = None,
    structure_names_r1: list[str] | None = None,
    structure_coefs_r2: np.ndarray | None = None,
    structure_names_r2: list[str] | None = None,
) -> None:
    """
    Save feature importance figures for rounds 1 and 2.

    When structure coefficients are provided, each round produces a two-panel
    figure (paired horizontal bar chart):
      Left  — raw LDA coefficient, normalised to [-1, 1] by the maximum absolute
               value.  Can be inflated by collinearity between features.
      Right — structure coefficient: Pearson r between each (scaled) feature and
               the discriminant score.  Bounded in [-1, 1] and unaffected by
               collinearity.  Features are sorted top-to-bottom by |structure coef|.

    When structure coefficients are absent the original single-panel plot is
    produced.  Blue = positive (target-like), red = negative (decoy-like).

    Files: ``{out_dir}/{model_name}_round1_feature_importance.png`` etc.
    """
    os.makedirs(out_dir, exist_ok=True)

    def _one(
        importances: np.ndarray | None,
        names: list[str],
        suffix: str,
        struct_coefs: np.ndarray | None = None,
        struct_names: list[str] | None = None,
    ) -> None:
        if importances is None or len(importances) == 0:
            return
        importances = np.asarray(importances, dtype=float)
        if len(importances) != len(names):
            logger.warning(
                "Feature importance length mismatch: %d importances vs %d names — skipping %s",
                len(importances), len(names), suffix,
            )
            return

        has_struct = (
            struct_coefs is not None
            and struct_names is not None
            and len(struct_coefs) == len(struct_names)
            and len(struct_coefs) > 0
        )

        round_label = suffix.replace("_", " ").title()

        if has_struct:
            struct_coefs_arr = np.asarray(struct_coefs, dtype=float)
            # Map raw coef by feature name; handles poly expansion where
            # struct_names ⊆ names (original features ⊂ expanded names).
            name_to_raw = dict(zip(names, importances))
            raw_for_struct = np.array([name_to_raw.get(n, np.nan) for n in struct_names])

            # Sort by |structure coef|, largest at top (barh: index 0 = bottom).
            order = np.argsort(np.abs(struct_coefs_arr))[-top_n:]
            plot_names = [struct_names[i] for i in order]
            s_vals = struct_coefs_arr[order]
            r_vals = raw_for_struct[order]

            # Normalise raw coefs to [-1, 1] so both axes share the same scale.
            r_max = np.nanmax(np.abs(r_vals)) if np.any(np.isfinite(r_vals)) else 1.0
            r_norm = r_vals / (r_max + 1e-12)

            n_feats = len(plot_names)
            fig, (ax_raw, ax_struct) = plt.subplots(
                1, 2,
                figsize=(14, max(5, n_feats * 0.38)),
                sharey=True,
            )

            raw_colors = ["steelblue" if v >= 0 else "tomato" for v in r_norm]
            ax_raw.barh(range(n_feats), r_norm, color=raw_colors, alpha=0.80, height=0.65)
            ax_raw.axvline(0, color="black", lw=0.8)
            ax_raw.set_xlim(-1.12, 1.12)
            ax_raw.set_yticks(range(n_feats))
            ax_raw.set_yticklabels(plot_names, fontsize=7)
            ax_raw.set_xlabel("Raw LDA coef  (normalised to max abs)", fontsize=9)
            ax_raw.set_title("Raw LDA coefficient\n(can be inflated by collinearity)", fontsize=9)

            struct_colors = ["steelblue" if v >= 0 else "tomato" for v in s_vals]
            ax_struct.barh(range(n_feats), s_vals, color=struct_colors, alpha=0.80, height=0.65)
            ax_struct.axvline(0, color="black", lw=0.8)
            ax_struct.set_xlim(-1.12, 1.12)
            ax_struct.set_xlabel("Structure coef  r(feature, discriminant score)", fontsize=9)
            ax_struct.set_title("Structure coefficient\n(collinearity-robust, bounded [−1, 1])", fontsize=9)
            ax_struct.tick_params(labelleft=False)

            fig.suptitle(
                f"{model_name} — {round_label}: raw vs structure importance "
                f"(top {n_feats} by |structure coef|  ·  blue = target-like / red = decoy-like)",
                fontsize=9, y=1.01,
            )
        else:
            order = np.argsort(np.abs(importances))[-top_n:]
            plot_names = [names[i] for i in order]
            vals = importances[order]
            colors = ["steelblue" if v >= 0 else "tomato" for v in vals]

            fig, ax_raw = plt.subplots(figsize=(9, max(4, len(plot_names) * 0.32)))
            ax_raw.barh(range(len(plot_names)), vals, color=colors, alpha=0.82)
            ax_raw.set_yticks(range(len(plot_names)))
            ax_raw.set_yticklabels(plot_names, fontsize=7)
            ax_raw.axvline(0, color="black", lw=0.8)
            ax_raw.set_xlabel("Importance", fontsize=9)
            ax_raw.set_title(
                f"{model_name} — {round_label} feature importance (top {len(plot_names)})",
                fontsize=10,
            )

        plt.tight_layout()
        _save_and_close(fig, os.path.join(out_dir, f"{model_name}_{suffix}_feature_importance.png"), dpi=100)

    _one(importances_r1, names_r1, "round1",
         struct_coefs=structure_coefs_r1, struct_names=structure_names_r1)
    _one(importances_r2, names_r2 if names_r2 is not None else names_r1, "round2",
         struct_coefs=structure_coefs_r2, struct_names=structure_names_r2)


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
    single_round: bool = False,
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
    # Entrapment pseudo-targets: is_decoy=False but source=="entrapment_shuffled"
    _src = feat.get("source", pd.Series("", index=feat.index)).fillna("").values
    entrapment_mask = (~is_decoy) & (_src == "entrapment_shuffled")
    has_entrapment = entrapment_mask.any()

    target_mask = (~is_decoy) & (~entrapment_mask)
    decoy_mask = is_decoy
    winner_target_mask = is_winner & target_mask
    winner_decoy_mask = is_winner & decoy_mask
    winner_ent_mask = is_winner & entrapment_mask

    gt_mask = np.zeros(len(feat), dtype=bool)
    if gt_peptides and "peptide" in feat.columns:
        gt_set = set(gt_peptides)
        all_gt_mask = target_mask & feat["peptide"].isin(gt_set).values
        if all_gt_mask.any():
            # Each GT peptide may appear at multiple MALDI features; keep only the
            # row with the highest round-1 score so each peptide contributes exactly
            # one vertical line per feature distribution plot.
            _r1_col = next(
                (c for c in res.columns if c.endswith("_r1") and pd.api.types.is_numeric_dtype(res[c])),
                None,
            )
            if _r1_col is None:
                gt_mask = all_gt_mask
            else:
                gt_idx = np.where(all_gt_mask)[0]
                tmp = pd.DataFrame({
                    "row": gt_idx,
                    "peptide": feat["peptide"].values[gt_idx],
                    "score": pd.to_numeric(res[_r1_col].iloc[gt_idx].values, errors="coerce"),
                })
                best = tmp.sort_values("score", ascending=False).drop_duplicates("peptide")
                gt_mask[best["row"].values] = True

    # Always plot every numeric column in features_df that is not in _DIST_SKIP.
    # Explicitly listed features (the ranker inputs) come first; the remaining
    # numeric columns — LC-MS/MS prior features, spatial prior features, and
    # optional intrinsic features dropped before training — follow.
    _explicit = list(feature_names) if feature_names is not None else []
    _explicit_set = set(_explicit)
    _extra = [
        c for c in feat.columns
        if c not in _DIST_SKIP
        and c not in _explicit_set
        and pd.api.types.is_numeric_dtype(feat[c])
    ]
    feature_names = _explicit + _extra

    def _draw(ax: plt.Axes, t_vals: np.ndarray, d_vals: np.ndarray,
               bins: np.ndarray, subtitle: str,
               e_vals: np.ndarray | None = None) -> None:
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
        if e_vals is not None and len(e_vals) > 0:
            ax.hist(e_vals, bins=bins, density=True, alpha=0.55,
                    color="goldenrod", label=f"Entrapment (n={len(e_vals)})")
            ax.axvline(float(np.nanmedian(e_vals)), color="goldenrod",
                       lw=1.3, ls="--", alpha=0.85)
        if len(t_vals) == 0 and len(d_vals) == 0 and (e_vals is None or len(e_vals) == 0):
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
        e_all = vals[entrapment_mask & finite_mask] if has_entrapment else None
        e_r2  = vals[winner_ent_mask & finite_mask]  if has_entrapment else None
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

        _bot_label = "Winners" if single_round else "Round-2 candidates"
        _e_all_n = len(e_all) if e_all is not None else 0
        _e_r2_n  = len(e_r2)  if e_r2  is not None else 0
        _top_title = (
            f"All candidates  (T={len(t_all)}, D={len(d_all)}, E={_e_all_n})"
            if has_entrapment
            else f"All candidates  (T={len(t_all)}, D={len(d_all)})"
        )
        _bot_title = (
            f"{_bot_label}  (T={len(t_r2)}, D={len(d_r2)}, E={_e_r2_n})"
            if has_entrapment
            else f"{_bot_label}  (T={len(t_r2)}, D={len(d_r2)})"
        )
        _draw(ax_top, t_all, d_all, bins, _top_title, e_vals=e_all)
        _draw(ax_bot, t_r2, d_r2, bins, _bot_title, e_vals=e_r2)

        _draw_gt(ax_top, gt_vals)
        _draw_gt(ax_bot, gt_vals)

        ax_bot.set_xlabel(feat_col, fontsize=8)
        plt.tight_layout()
        _save_and_close(fig, os.path.join(out_dir, f"{_safe_fname(feat_col, maxlen=80)}.png"), dpi=100)


# ---------------------------------------------------------------------------
# Subsystem 6: CCS scatter
# ---------------------------------------------------------------------------


def plot_ccs_scatter(
    features_df: pd.DataFrame,
    result_df: pd.DataFrame,
    out_dir: str,
    fdr_threshold: float = 0.01,
    gt_peptides: list[str] | None = None,
    ccs_tol_pct: float | None = None,
    title_extra: str = "",
    filename: str = "ccs_scatter.png",
) -> None:
    """
    Scatter plot of observed vs predicted CCS for all candidates.

    Requires ``im2deep_observed_ccs`` and ``im2deep_predicted_ccs`` columns in
    ``features_df`` (added by ``compute_im2deep_features``). Silently skips if
    neither column is present.

    Points are coloured by target/decoy status. "R2 winner" means the feature's
    best candidate AND reweighted_q_value <= fdr_threshold. R1 winners (best
    candidate but below FDR threshold) are shown at intermediate size.

    When ``ccs_tol_pct`` is provided, fan-shaped CCS filter boundaries are drawn:
    ``obs = pred × (1 ± ccs_tol_pct/100)``. These diverge from the origin.
    Saved to ``{out_dir}/{filename}``.
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

    # Fan-shaped CCS tolerance boundaries (diverge from origin)
    if ccs_tol_pct is not None:
        pred_range = np.linspace(float(pred_v.min()), float(pred_v.max()), 300)
        fac = ccs_tol_pct / 100.0
        ax.plot(pred_range, pred_range * (1 + fac), color="darkorange", lw=1.2,
                ls="--", alpha=0.85, label=f"±{ccs_tol_pct:.1f}% CCS threshold")
        ax.plot(pred_range, pred_range * (1 - fac), color="darkorange", lw=1.2,
                ls="--", alpha=0.85)

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

    title = f"Observed vs Predicted CCS\n{corr_str}"
    if title_extra:
        title += f"\n{title_extra}"
    ax.set_xlabel("Predicted CCS (Å²)", fontsize=10)
    ax.set_ylabel("Observed CCS (Å²)", fontsize=10)
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7, markerscale=1.5)
    plt.tight_layout()
    _save_and_close(fig, os.path.join(out_dir, filename))


# ---------------------------------------------------------------------------
# Subsystem 7: IDs vs FDR curve
# ---------------------------------------------------------------------------


def plot_ids_vs_fdr(
    result_df: pd.DataFrame,
    out_dir: str,
    fdr_max: float = 0.20,
    pi0: float | None = None,
) -> None:
    """
    Save a curve of target identifications as a function of FDR threshold.

    Backend-agnostic: the model name is inferred from the ``*_score_r1``
    column in ``result_df``; no explicit model identifier is required.  Plots
    both the TDC q-value and the reweighted q-value (when present) so the
    effect of the LC-MS/MS prior is immediately visible.  Vertical lines mark
    1 % and 5 % FDR.  Only TDC winner target rows are considered.

    Output: ``{out_dir}/ids_vs_fdr.png``
    """
    os.makedirs(out_dir, exist_ok=True)

    # Infer model name from the first *_score_r1 column present.
    r1_cols = [c for c in result_df.columns if c.endswith("_score_r1")]
    model_name = r1_cols[0].removesuffix("_score_r1") if r1_cols else "model"

    is_winner = result_df.get("is_tdc_winner", pd.Series(False, index=result_df.index)).fillna(False).astype(bool)
    is_decoy = result_df.get("is_decoy", pd.Series(False, index=result_df.index)).fillna(False).astype(bool)
    target_winners = result_df[is_winner & ~is_decoy].copy()

    pi0_label = f" (π₀={pi0:.3f})" if pi0 is not None else ""
    curves: list[tuple[str, str, str]] = []  # (column, label, colour)
    if "q_value" in target_winners.columns:
        curves.append(("q_value", "TDC q-value", "steelblue"))
    if "reweighted_q_value" in target_winners.columns and target_winners["reweighted_q_value"].notna().any():
        curves.append(("reweighted_q_value", "Reweighted q-value", "darkorange"))
    if "storey_q_value" in target_winners.columns and target_winners["storey_q_value"].notna().any():
        curves.append(("storey_q_value", f"Storey q-value{pi0_label}", "seagreen"))
    if "storey_reweighted_q_value" in target_winners.columns and target_winners["storey_reweighted_q_value"].notna().any():
        curves.append(("storey_reweighted_q_value", f"Storey reweighted{pi0_label}", "tomato"))

    if not curves:
        logger.warning("plot_ids_vs_fdr: no q_value or reweighted_q_value column — skipping")
        return

    fdr_grid = np.linspace(0.0, fdr_max, 500)

    fig, ax = plt.subplots(figsize=(7, 4.5))

    for col, label, colour in curves:
        vals = pd.to_numeric(target_winners[col], errors="coerce").dropna().values
        if len(vals) == 0:
            continue
        n_ids = np.array([(vals <= t).sum() for t in fdr_grid])
        ax.plot(fdr_grid * 100, n_ids, label=label, color=colour, lw=2)

        for thresh, ls in [(0.01, "--"), (0.05, ":")]:
            if thresh <= fdr_max:
                n_at = int((vals <= thresh).sum())
                ax.axvline(thresh * 100, color=colour, lw=0.8, ls=ls, alpha=0.6)
                ax.annotate(
                    f"{n_at}",
                    xy=(thresh * 100, n_at),
                    xytext=(4, 4),
                    textcoords="offset points",
                    fontsize=7,
                    color=colour,
                )

    ax.set_xlabel("FDR threshold (%)", fontsize=10)
    ax.set_ylabel("Target identifications", fontsize=10)
    ax.set_title(f"{model_name} — IDs vs FDR", fontsize=11)
    ax.set_xlim(0, fdr_max * 100)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=9)
    ax.grid(True, lw=0.4, alpha=0.4)
    plt.tight_layout()
    _save_and_close(fig, os.path.join(out_dir, "ids_vs_fdr.png"))


# ---------------------------------------------------------------------------
# Subsystem 8: Protein colocalization by scoring group
# ---------------------------------------------------------------------------

_COLOC_COLS = [
    ("protein_colocalization",         "Protein coloc. (mean r)"),
    ("protein_colocalization_max",     "Protein coloc. (max r)"),
    ("protein_colocalization_median",  "Protein coloc. (median r)"),
    ("protein_colocalization_n_partners", "Protein coloc. (n partners)"),
    ("protein_region_colocalization",  "Region coloc. (mean r)"),
]

_GROUP_ORDER  = ["ID @ 1% FDR", "ID @ 5% FDR", "R1 winner (below FDR)", "Non-winner"]
_GROUP_COLORS = ["seagreen",    "mediumseagreen", "darkorange",          "steelblue"]


def plot_protein_colocalization_by_group(
    features_df: pd.DataFrame,
    result_df: pd.DataFrame,
    out_dir: str,
    fdr_threshold: float = 0.01,
    fdr_threshold_loose: float = 0.05,
) -> None:
    """
    Box + strip plot of protein-level colocalization values split into four groups:
      - ID @ 1% FDR  : round-2 TDC winner with reweighted_q_value <= fdr_threshold
      - ID @ 5% FDR  : round-2 TDC winner with fdr_threshold < reweighted_q_value <= fdr_threshold_loose
      - R1 winner     : round-2 TDC winner, but reweighted_q_value > fdr_threshold_loose
      - Non-winner    : did not make it to round 2

    Only target (non-decoy) rows are shown, since decoy colocalization values
    reflect the null model rather than biology.

    Skips silently if none of the four colocalization columns are present.
    Output: ``{out_dir}/protein_colocalization_by_group.png``
    """
    present = [col for col, _ in _COLOC_COLS if col in features_df.columns]
    if not present:
        return

    os.makedirs(out_dir, exist_ok=True)

    feat = features_df.reset_index(drop=True)
    res  = result_df.reset_index(drop=True)

    is_decoy = (
        feat.get("is_decoy", pd.Series(False, index=feat.index))
        .fillna(False).astype(bool).values
    )
    is_winner = (
        res.get("is_tdc_winner", pd.Series(False, index=res.index))
        .fillna(False).astype(bool).values
    )
    rw_q = pd.to_numeric(
        res.get("reweighted_q_value", pd.Series(float("nan"), index=res.index)),
        errors="coerce",
    ).values

    passes_fdr_strict = is_winner & (rw_q <= fdr_threshold)
    passes_fdr_loose  = is_winner & (rw_q > fdr_threshold) & (rw_q <= fdr_threshold_loose)
    r1_only           = is_winner & (rw_q > fdr_threshold_loose)

    group_label = np.where(
        passes_fdr_strict, _GROUP_ORDER[0],
        np.where(passes_fdr_loose, _GROUP_ORDER[1],
        np.where(r1_only, _GROUP_ORDER[2], _GROUP_ORDER[3])),
    )

    # Restrict to targets only
    target_mask = ~is_decoy

    n_cols = len(present)
    fig, axes = plt.subplots(1, n_cols, figsize=(4.5 * n_cols, 5), sharey=False)
    if n_cols == 1:
        axes = [axes]

    rng = np.random.default_rng(0)

    for ax, col in zip(axes, present):
        label = dict(_COLOC_COLS)[col]
        vals  = pd.to_numeric(feat[col], errors="coerce").values.astype(float)

        group_data: dict[str, np.ndarray] = {}
        for grp in _GROUP_ORDER:
            mask = target_mask & (group_label == grp)
            v = vals[mask]
            group_data[grp] = v[np.isfinite(v)]

        positions = list(range(len(_GROUP_ORDER)))

        # Box plots
        bp_data = [group_data[g] for g in _GROUP_ORDER]
        bp = ax.boxplot(
            bp_data,
            positions=positions,
            widths=0.45,
            patch_artist=True,
            showfliers=False,
            medianprops=dict(color="black", lw=1.8),
            whiskerprops=dict(lw=1.0),
            capprops=dict(lw=1.0),
            boxprops=dict(lw=1.0),
        )
        for patch, color in zip(bp["boxes"], _GROUP_COLORS):
            patch.set_facecolor(color)
            patch.set_alpha(0.45)

        # Strip (jitter) overlay
        for pos, (grp, color) in enumerate(zip(_GROUP_ORDER, _GROUP_COLORS)):
            v = group_data[grp]
            if len(v) == 0:
                continue
            jitter = rng.uniform(-0.18, 0.18, size=len(v))
            ax.scatter(
                pos + jitter, v,
                s=12, alpha=0.55, color=color, linewidths=0, zorder=3,
            )

        ax.set_xticks(positions)
        ax.set_xticklabels(
            [g.replace(" ", "\n") for g in _GROUP_ORDER],
            fontsize=7,
        )
        ax.set_ylabel(label, fontsize=8)
        ax.set_title(label, fontsize=9)
        ax.tick_params(axis="y", labelsize=7)

        # Horizontal reference line at r=0 for correlation columns
        if "n_partners" not in col:
            ax.axhline(0.0, color="gray", lw=0.8, ls="--", alpha=0.6)

    # Sample size annotations — drawn after tight_layout sets final y limits.
    for ax, col in zip(axes, present):
        vals = pd.to_numeric(feat[col], errors="coerce").values.astype(float)
        y_lo, y_top = ax.get_ylim()
        y_ann = y_top + (y_top - y_lo) * 0.02
        for pos, grp in enumerate(_GROUP_ORDER):
            mask = target_mask & (group_label == grp)
            n = int(np.isfinite(vals[mask]).sum())
            ax.text(pos, y_ann, f"n={n}", ha="center", va="bottom", fontsize=7)

    fig.suptitle(
        f"Protein colocalization by scoring group "
        f"(targets only, strict FDR {fdr_threshold:.0%} / loose FDR {fdr_threshold_loose:.0%})",
        fontsize=10, y=1.02,
    )
    plt.tight_layout()
    _save_and_close(fig, os.path.join(out_dir, "protein_colocalization_by_group.png"))


def plot_region_colocalization(
    features_df: pd.DataFrame,
    region_debug: dict,
    ion_image_shape: tuple[int, int] | None = None,
    out_dir: str = "debug",
    n_proteins: int = 4,
    max_target_rows: int = 8,
    max_decoy_rows: int = 3,
) -> None:
    """Visualize *how* region-profile colocalization worked (opt-in ``--region-coloc``).

    Two figures, from the ``region_debug`` dict populated by
    ``compute_region_colocalization_features``:

    1. ``region_segmentation.png`` — the k-means region map over the tissue
       (off-tissue pixels greyed), so the discovered compartments are visible.
    2. ``region_profiles.png`` — for the proteins with the largest target-vs-decoy
       region-coloc delta, a heatmap of per-region composition with one row per
       peptide (targets ``T``, decoys ``D``). Same-protein target peptides share a
       region fingerprint; the decoy rows (relocated to foreign m/z) differ — the
       visual analog of the target-r > decoy-r the feature scores.
    """
    import matplotlib as mpl

    labels = region_debug.get("region_labels")
    profiles = region_debug.get("region_profiles")
    prof_mzs = region_debug.get("region_profile_mzs")
    if labels is None or profiles is None or prof_mzs is None:
        return
    os.makedirs(out_dir, exist_ok=True)
    profiles = np.asarray(profiles)

    # ---- Figure 1: segmentation map ----
    if ion_image_shape is not None:
        H, W = ion_image_shape
        seg = labels.reshape(H, W).astype(float)
        seg[seg < 0] = np.nan
        kmax = np.nanmax(seg)
        k = int(kmax) + 1 if np.isfinite(kmax) else 1
        fig, ax = plt.subplots(figsize=(6, 5))
        cmap = mpl.colormaps["tab20"].resampled(max(k, 1))
        cmap.set_bad("0.9")
        im = ax.imshow(seg, cmap=cmap, interpolation="nearest")
        ax.set_title(f"Region segmentation (k={k} regions; off-tissue grey)", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="region id")
        _save_and_close(fig, os.path.join(out_dir, "region_segmentation.png"))

    # ---- Figure 2: per-protein region-profile fingerprints ----
    if (
        "protein" not in features_df.columns
        or "protein_region_colocalization" not in features_df.columns
        or "is_decoy" not in features_df.columns
    ):
        return
    mz_to_row = {float(m): i for i, m in enumerate(np.asarray(prof_mzs))}
    base = features_df.assign(
        base_protein=features_df["protein"]
        .str.replace("DECOY_", "", regex=False)
        .str.replace("ENTRAPMENT_", "", regex=False)
    )
    tgt = base[~base["is_decoy"]]
    grp = base.groupby(["base_protein", "is_decoy"])["protein_region_colocalization"].mean().unstack()
    npep = tgt.groupby("base_protein")["peptide"].nunique()

    cand = []
    for prot in grp.index:
        if int(npep.get(prot, 0)) < 3:
            continue
        t = grp.loc[prot].get(False, np.nan)
        d = grp.loc[prot].get(True, np.nan)
        delta = (t - d) if (t == t and d == d) else (t if t == t else float("-inf"))
        cand.append((prot, delta))
    cand.sort(key=lambda x: (x[1] if x[1] == x[1] else float("-inf")), reverse=True)
    prots = [p for p, _ in cand[:n_proteins]]
    if not prots:
        return

    fig, axes = plt.subplots(len(prots), 1, figsize=(8, 2.4 * len(prots)), squeeze=False)
    for ri, prot in enumerate(prots):
        ax = axes[ri][0]
        rows, ylabels = [], []
        for _, r in tgt[tgt["base_protein"] == prot].drop_duplicates("peptide").head(max_target_rows).iterrows():
            idx = mz_to_row.get(float(r["feature_mz"]))
            if idx is not None:
                rows.append(profiles[idx]); ylabels.append(f"T {str(r['peptide'])[:12]}")
        decoy_rows = base[(base["is_decoy"]) & (base["base_protein"] == prot)]
        for _, r in decoy_rows.drop_duplicates("peptide").head(max_decoy_rows).iterrows():
            idx = mz_to_row.get(float(r["feature_mz"]))
            if idx is not None:
                rows.append(profiles[idx]); ylabels.append(f"D {str(r['peptide'])[:12]}")
        if not rows:
            ax.axis("off"); continue
        M = np.asarray(rows)
        im = ax.imshow(M, aspect="auto", cmap="viridis", interpolation="nearest")
        ax.set_yticks(range(len(ylabels))); ax.set_yticklabels(ylabels, fontsize=6)
        ax.set_xlabel("region id", fontsize=7)
        ax.set_title(f"{prot}: per-region composition fingerprint (T=target, D=decoy)", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    fig.suptitle(
        "Region-profile fingerprints — same-protein targets share a pattern; decoys differ",
        fontsize=10,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    _save_and_close(fig, os.path.join(out_dir, "region_profiles.png"))


def plot_region_ion_images(
    features_df: pd.DataFrame,
    region_debug: dict,
    ion_images: np.ndarray,
    ion_image_mzs: np.ndarray,
    out_dir: str = "debug/region_ion_images",
    n_proteins: int = 6,
    max_target_rows: int = 8,
    max_decoy_rows: int = 3,
) -> None:
    """Per-protein ion-image panels with region overlay.

    For the top ``n_proteins`` proteins by target-vs-decoy
    ``protein_region_colocalization`` delta, writes one PNG per protein to
    ``out_dir/<protein>.png``.  Each row in the figure is one peptide (target
    rows first, then decoy rows).  Three columns per row:

    * **Ion image** — raw spatial distribution (hot colourmap, γ=0.5).
    * **Region overlay** — same ion image with the k-means region map blended
      on top (tab20 per-region colours, semi-transparent).  Shows which tissue
      compartments carry each peptide's signal.
    * **Region profile** — horizontal bar chart of the per-region mean
      intensity fingerprint (the vector that ``protein_region_colocalization``
      correlates between peptides).  Same colours as the overlay.

    Same-protein target peptides should share a similar profile; decoy rows
    (at a foreign m/z) should diverge.
    """
    import matplotlib as mpl

    labels = region_debug.get("region_labels")
    profiles = region_debug.get("region_profiles")
    prof_mzs = region_debug.get("region_profile_mzs")
    if labels is None or profiles is None or prof_mzs is None:
        return
    if (
        "protein" not in features_df.columns
        or "protein_region_colocalization" not in features_df.columns
        or "is_decoy" not in features_df.columns
    ):
        return

    profiles = np.asarray(profiles)
    ion_image_mzs_arr = np.asarray(ion_image_mzs)
    H, W = ion_images.shape[1], ion_images.shape[2]
    seg = labels.reshape(H, W)
    n_regions = int(np.max(seg[seg >= 0])) + 1 if (seg >= 0).any() else 1
    cmap_tab = mpl.colormaps["tab20"].resampled(max(n_regions, 2))

    # Build semi-transparent region RGBA overlay (off-tissue fully transparent)
    overlay_rgba = np.zeros((H, W, 4), dtype=np.float32)
    for k in range(n_regions):
        r, g, b, _ = cmap_tab(k / max(n_regions - 1, 1))
        overlay_rgba[seg == k] = [r, g, b, 0.55]

    mz_to_prof_row = {float(m): i for i, m in enumerate(np.asarray(prof_mzs))}

    def _find_img_idx(mz, ppm=20.0):
        idx = int(np.searchsorted(ion_image_mzs_arr, mz))
        for c in (idx, idx - 1):
            if 0 <= c < len(ion_image_mzs_arr):
                if abs(ion_image_mzs_arr[c] - mz) / mz * 1e6 < ppm:
                    return c
        return None

    base = features_df.assign(
        base_protein=features_df["protein"]
        .str.replace("DECOY_", "", regex=False)
        .str.replace("ENTRAPMENT_", "", regex=False)
    )
    tgt = base[~base["is_decoy"]]
    grp = base.groupby(["base_protein", "is_decoy"])["protein_region_colocalization"].mean().unstack()
    npep = tgt.groupby("base_protein")["peptide"].nunique()

    cand = []
    for prot in grp.index:
        if int(npep.get(prot, 0)) < 2:
            continue
        t = grp.loc[prot].get(False, np.nan)
        d = grp.loc[prot].get(True, np.nan)
        if t == t and d == d:
            delta = t - d
        elif t == t:
            delta = t
        else:
            delta = float("-inf")
        cand.append((prot, delta, float(t) if t == t else float("nan"), float(d) if d == d else float("nan")))
    cand.sort(key=lambda x: x[1] if x[1] == x[1] else float("-inf"), reverse=True)

    os.makedirs(out_dir, exist_ok=True)

    for prot, delta, t_mean, d_mean in cand[:n_proteins]:
        t_rows = (
            tgt[tgt["base_protein"] == prot]
            .drop_duplicates("peptide")
            .head(max_target_rows)
        )
        d_rows = (
            base[(base["is_decoy"]) & (base["base_protein"] == prot)]
            .drop_duplicates("peptide")
            .head(max_decoy_rows)
        )
        all_rows = [(r, False) for r in t_rows.itertuples()] + [(r, True) for r in d_rows.itertuples()]
        if not all_rows:
            continue

        n_rows = len(all_rows)
        fig, axes = plt.subplots(
            n_rows, 3, figsize=(9.0, 2.2 * n_rows),
            gridspec_kw={"width_ratios": [1, 1, 1.2]},
        )
        if n_rows == 1:
            axes = axes[np.newaxis, :]

        axes[0, 0].set_title("ion image", fontsize=8)
        axes[0, 1].set_title("region overlay", fontsize=8)
        axes[0, 2].set_title("region profile", fontsize=8)

        for ri, (row, is_d) in enumerate(all_rows):
            mz = float(row.feature_mz)
            img_idx = _find_img_idx(mz)
            prof_idx = mz_to_prof_row.get(mz)
            rc_val = getattr(row, "protein_region_colocalization", float("nan"))
            rc_str = f"r={rc_val:.2f}" if rc_val == rc_val else ""
            label = f"{'D' if is_d else 'T'} {str(row.peptide)[:16]} {rc_str}"

            ax0, ax1, ax2 = axes[ri, 0], axes[ri, 1], axes[ri, 2]

            def _show_img(ax, img_idx, cmap="hot"):
                if img_idx is not None:
                    img = ion_images[img_idx].astype(float)
                    pos = img[img > 0]
                    vmax = float(np.percentile(pos, 99)) if len(pos) else 1.0
                    norm = np.clip(img / max(vmax, 1e-9), 0.0, 1.0) ** 0.5
                    ax.imshow(norm, cmap=cmap, interpolation="nearest")
                else:
                    ax.imshow(np.zeros((H, W)), cmap=cmap, interpolation="nearest")
                    ax.text(W / 2, H / 2, "no image", ha="center", va="center", fontsize=6, color="white")
                ax.set_xticks([]); ax.set_yticks([])

            _show_img(ax0, img_idx, cmap="hot")
            ax0.set_title(label, fontsize=6.5, loc="left", pad=2)

            # region overlay: grey base + coloured region alpha
            _show_img(ax1, img_idx, cmap="gray")
            ax1.imshow(overlay_rgba, interpolation="nearest")
            ax1.set_xticks([]); ax1.set_yticks([])

            # region profile bar
            if prof_idx is not None:
                prof = profiles[prof_idx]
                colors = [cmap_tab(k / max(n_regions - 1, 1)) for k in range(len(prof))]
                ax2.barh(range(len(prof)), prof, color=colors, height=0.8)
                ax2.invert_yaxis()
                ax2.set_yticks(range(n_regions))
                ax2.set_yticklabels([str(k) for k in range(n_regions)], fontsize=5)
                ax2.tick_params(axis="x", labelsize=5)
            else:
                ax2.text(0.5, 0.5, "n/a", ha="center", va="center", transform=ax2.transAxes, fontsize=7)
                ax2.axis("off")

        t_str = f"{t_mean:.3f}" if t_mean == t_mean else "n/a"
        d_str = f"{d_mean:.3f}" if d_mean == d_mean else "n/a"
        fig.suptitle(
            f"{prot}  (target region_coloc={t_str}, decoy={d_str})",
            fontsize=9,
        )
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in prot)[:60]
        _save_and_close(fig, os.path.join(out_dir, f"{safe}.png"))


# ---------------------------------------------------------------------------
# Subsystem 9: Target vs Decoy m/z distribution
# ---------------------------------------------------------------------------


def plot_target_decoy_mz_distribution(
    features_df: pd.DataFrame,
    result_df: pd.DataFrame,
    out_dir: str,
    fdr_threshold: float = 0.01,
    n_bins: int = 60,
) -> None:
    """
    Three-panel figure showing target vs decoy m/z coverage.

    Panel 1 — Per-feature competition status (histogram):
      Each MALDI feature is counted once:
        - steelblue  : target-only  (no decoy candidate matched this feature)
        - mediumpurple: contested   (at least one target AND one decoy)
        - tomato     : decoy-only   (no target candidate matched this feature)

    Panel 2 — All candidates, per-candidate density (target vs decoy):
      Overlapping density histograms with dashed median lines.

    Panel 3 — R1 winners only (best candidate per feature):
      Same layout as Panel 2 but restricted to is_tdc_winner rows.
      Shows the effective T:D ratio entering FDR estimation.

    Output: ``{out_dir}/target_decoy_mz_distribution.png``
    """
    if "feature_mz" not in features_df.columns:
        return

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

    fmz = pd.to_numeric(feat["feature_mz"], errors="coerce").values
    finite = np.isfinite(fmz)

    # --- Per-feature competition classification ---
    target_fmz_set = set(fmz[~is_decoy & finite].tolist())
    decoy_fmz_set  = set(fmz[is_decoy  & finite].tolist())

    contested_arr   = np.array(sorted(target_fmz_set & decoy_fmz_set))
    target_only_arr = np.array(sorted(target_fmz_set - decoy_fmz_set))
    decoy_only_arr  = np.array(sorted(decoy_fmz_set  - target_fmz_set))
    all_fmz_arr     = np.concatenate([contested_arr, target_only_arr, decoy_only_arr])

    # Per-candidate m/z arrays
    t_mz     = fmz[~is_decoy & finite]
    d_mz     = fmz[ is_decoy & finite]
    t_win    = fmz[~is_decoy & is_winner & finite]
    d_win    = fmz[ is_decoy & is_winner & finite]

    if len(all_fmz_arr) == 0:
        return

    lo = float(np.percentile(all_fmz_arr, 1))
    hi = float(np.percentile(all_fmz_arr, 99))
    if lo >= hi:
        lo, hi = float(all_fmz_arr.min()), float(all_fmz_arr.max())
    bins = np.linspace(lo, hi, n_bins + 1)

    fig, axes = plt.subplots(3, 1, figsize=(10, 11))

    # ------------------------------------------------------------------
    # Panel 1: per-feature competition status
    # ------------------------------------------------------------------
    ax = axes[0]
    if len(target_only_arr):
        ax.hist(target_only_arr, bins=bins, alpha=0.70, color="steelblue",
                label=f"Target-only ({len(target_only_arr)})")
    if len(contested_arr):
        ax.hist(contested_arr, bins=bins, alpha=0.70, color="mediumpurple",
                label=f"Contested ({len(contested_arr)})")
    if len(decoy_only_arr):
        ax.hist(decoy_only_arr, bins=bins, alpha=0.70, color="tomato",
                label=f"Decoy-only ({len(decoy_only_arr)})")

    n_total = len(target_only_arr) + len(contested_arr) + len(decoy_only_arr)
    ax.set_title(
        f"Per-feature competition status  "
        f"({n_total} features: {len(target_only_arr)} T-only, "
        f"{len(contested_arr)} contested, {len(decoy_only_arr)} D-only)",
        fontsize=9,
    )
    ax.set_ylabel("Number of features", fontsize=9)
    ax.legend(fontsize=8)
    ax.tick_params(labelsize=8)

    # ------------------------------------------------------------------
    # Panel 2: all candidates
    # ------------------------------------------------------------------
    ax = axes[1]
    for arr, color, label in [
        (t_mz, "steelblue", f"Target (n={len(t_mz)})"),
        (d_mz, "tomato",    f"Decoy (n={len(d_mz)})"),
    ]:
        if len(arr):
            ax.hist(arr, bins=bins, density=True, alpha=0.50, color=color, label=label)
            ax.axvline(float(np.median(arr)), color=color, lw=1.5, ls="--", alpha=0.85)

    ratio_str = (
        f"T:D = {len(t_mz)}/{len(d_mz)} = {len(t_mz)/len(d_mz):.2f}:1"
        if len(d_mz) else f"T:D = {len(t_mz)}/0"
    )
    ax.set_title(f"All candidates  ({ratio_str})", fontsize=9)
    ax.set_ylabel("Density", fontsize=9)
    ax.legend(fontsize=8)
    ax.tick_params(labelsize=8)

    # ------------------------------------------------------------------
    # Panel 3: R1 winners
    # ------------------------------------------------------------------
    ax = axes[2]
    for arr, color, label in [
        (t_win, "steelblue", f"Target R1 winners (n={len(t_win)})"),
        (d_win, "tomato",    f"Decoy R1 winners (n={len(d_win)})"),
    ]:
        if len(arr):
            ax.hist(arr, bins=bins, density=True, alpha=0.50, color=color, label=label)
            ax.axvline(float(np.median(arr)), color=color, lw=1.5, ls="--", alpha=0.85)

    win_ratio_str = (
        f"T:D = {len(t_win)}/{len(d_win)} = {len(t_win)/len(d_win):.2f}:1"
        if len(d_win) else f"T:D = {len(t_win)}/0"
    )
    ax.set_title(f"R1 winners (best per feature)  ({win_ratio_str})", fontsize=9)
    ax.set_xlabel("Feature m/z", fontsize=9)
    ax.set_ylabel("Density", fontsize=9)
    ax.legend(fontsize=8)
    ax.tick_params(labelsize=8)

    fig.suptitle("Target vs Decoy m/z Distributions", fontsize=11)
    plt.tight_layout()
    _save_and_close(fig, os.path.join(out_dir, "target_decoy_mz_distribution.png"))


# ---------------------------------------------------------------------------
# Subsystem 10: Candidate competition per feature
# ---------------------------------------------------------------------------


def plot_candidate_competition(
    features_df: pd.DataFrame,
    result_df: pd.DataFrame,
    out_dir: str,
    ccs_tol_pct: float | None = None,
) -> None:
    """
    Four-panel figure showing how many target and decoy candidates compete at
    each MALDI m/z feature.

    Panel [0,0] — Target candidate count distribution:
        Histogram of how many features have 0, 1, 2, 3, 4+ target candidates.
    Panel [0,1] — Decoy candidate count distribution:
        Same for decoy candidates.
    Panel [1,0] — T vs D balance scatter:
        Each point = one (n_targets, n_decoys) combination; size ∝ number of
        features at that combination.  Diagonal marks perfect 1:1 balance.
    Panel [1,1] — Sorted competition landscape:
        Each feature as a vertical pair of bars: n_targets (blue, above axis)
        and n_decoys (orange, below axis), sorted by total candidates
        descending.  Capped at the 200 most-contested features for clarity.

    When ``ccs_tol_pct`` is not None the title notes that a CCS filter was
    applied before calling this function.

    Output: ``{out_dir}/candidate_competition.png``
    """
    if "feature_mz" not in features_df.columns or "is_decoy" not in features_df.columns:
        return

    os.makedirs(out_dir, exist_ok=True)

    feat = features_df.reset_index(drop=True)
    is_decoy = feat["is_decoy"].fillna(False).astype(bool)

    feat_col = "feature_idx" if "feature_idx" in feat.columns else "feature_mz"

    # --- Per-feature target/decoy counts ---
    n_tgt = feat[~is_decoy].groupby(feat_col).size().rename("n_targets")
    n_dec = feat[is_decoy].groupby(feat_col).size().rename("n_decoys")
    all_features = feat[feat_col].unique()
    per_feat = (
        pd.DataFrame(index=all_features)
        .join(n_tgt, how="left")
        .join(n_dec, how="left")
        .fillna(0)
        .astype(int)
    )
    per_feat["n_total"] = per_feat["n_targets"] + per_feat["n_decoys"]

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle(
        "Candidate competition per MALDI feature"
        + (f"  [CCS filter applied: ±{ccs_tol_pct:.1f}%]" if ccs_tol_pct is not None else ""),
        fontsize=11,
    )

    # ------------------------------------------------------------------ #
    # [0,0]  Target candidate count distribution                          #
    # ------------------------------------------------------------------ #
    ax = axes[0][0]
    max_n = int(per_feat["n_targets"].max()) if len(per_feat) else 4
    cap = min(max_n, 6)
    bins_t = np.arange(0, cap + 2) - 0.5
    vals_t = np.clip(per_feat["n_targets"].values, 0, cap)
    ax.hist(vals_t, bins=bins_t, color="steelblue", edgecolor="white", linewidth=0.5)
    ax.set_xticks(np.arange(0, cap + 1))
    ax.set_xticklabels([str(i) if i < cap else f"{cap}+" for i in range(cap + 1)])
    ax.set_xlabel("Target candidates per feature")
    ax.set_ylabel("Number of features")
    ax.set_title(
        f"Target candidate distribution\n"
        f"median={per_feat['n_targets'].median():.1f}  "
        f"mean={per_feat['n_targets'].mean():.2f}  "
        f"0-target features: {int((per_feat['n_targets']==0).sum())}",
        fontsize=8,
    )

    # ------------------------------------------------------------------ #
    # [0,1]  Decoy candidate count distribution                           #
    # ------------------------------------------------------------------ #
    ax = axes[0][1]
    max_d = int(per_feat["n_decoys"].max()) if len(per_feat) else 4
    cap_d = min(max_d, 6)
    bins_d = np.arange(0, cap_d + 2) - 0.5
    vals_d = np.clip(per_feat["n_decoys"].values, 0, cap_d)
    ax.hist(vals_d, bins=bins_d, color="tomato", edgecolor="white", linewidth=0.5)
    ax.set_xticks(np.arange(0, cap_d + 1))
    ax.set_xticklabels([str(i) if i < cap_d else f"{cap_d}+" for i in range(cap_d + 1)])
    ax.set_xlabel("Decoy candidates per feature")
    ax.set_ylabel("Number of features")
    ax.set_title(
        f"Decoy candidate distribution\n"
        f"median={per_feat['n_decoys'].median():.1f}  "
        f"mean={per_feat['n_decoys'].mean():.2f}  "
        f"0-decoy features: {int((per_feat['n_decoys']==0).sum())}",
        fontsize=8,
    )

    # ------------------------------------------------------------------ #
    # [1,0]  T vs D balance scatter                                       #
    # ------------------------------------------------------------------ #
    ax = axes[1][0]
    td_counts = per_feat.groupby(["n_targets", "n_decoys"]).size().reset_index(name="count")
    sc = ax.scatter(
        td_counts["n_targets"],
        td_counts["n_decoys"],
        s=np.clip(td_counts["count"], 1, None) * 12,
        c=np.log1p(td_counts["count"]),
        cmap="Blues",
        edgecolors="steelblue",
        linewidths=0.6,
        alpha=0.85,
    )
    plt.colorbar(sc, ax=ax, label="log(n features + 1)", fraction=0.046, pad=0.04)
    # Ideal 1:1 diagonal
    _lim = max(int(td_counts[["n_targets", "n_decoys"]].max().max()), 1)
    ax.plot([0, _lim], [0, _lim], color="gray", lw=0.8, ls="--", alpha=0.6, label="1:1")
    ax.set_xlim(left=-0.3)
    ax.set_ylim(bottom=-0.3)
    ax.set_xlabel("n_targets per feature")
    ax.set_ylabel("n_decoys per feature")
    ax.set_title("Target vs decoy balance per feature\n(bubble size ∝ n features)", fontsize=8)
    ax.legend(fontsize=7)

    # ------------------------------------------------------------------ #
    # [1,1]  Sorted competition landscape (top-N most contested)          #
    # ------------------------------------------------------------------ #
    ax = axes[1][1]
    _TOP = 150
    sorted_pf = per_feat.sort_values("n_total", ascending=False).head(_TOP).reset_index(drop=True)
    x = np.arange(len(sorted_pf))
    ax.bar(x, sorted_pf["n_targets"].values, color="steelblue", label="Targets", width=1.0)
    ax.bar(x, -sorted_pf["n_decoys"].values, color="tomato", label="Decoys", width=1.0)
    ax.axhline(0, color="black", lw=0.6)
    ax.axhline(1, color="steelblue", lw=0.7, ls=":", alpha=0.5)
    ax.axhline(-1, color="tomato", lw=0.7, ls=":", alpha=0.5)
    ax.set_xlabel(f"Feature rank (by total candidates, top {min(_TOP, len(per_feat))} shown)")
    ax.set_ylabel("n_candidates  (targets ↑  /  decoys ↓)")
    _n_feat = len(per_feat)
    _td_ratio = per_feat["n_decoys"].sum() / max(per_feat["n_targets"].sum(), 1)
    ax.set_title(
        f"Competition landscape  ({_n_feat} features total)\n"
        f"total T={per_feat['n_targets'].sum()}  D={per_feat['n_decoys'].sum()}  "
        f"D:T ratio={_td_ratio:.2f}",
        fontsize=8,
    )
    ax.legend(fontsize=7, loc="upper right")

    plt.tight_layout()
    _save_and_close(fig, os.path.join(out_dir, "candidate_competition.png"))


# ---------------------------------------------------------------------------
# Subsystem 11: Score PP plot
# ---------------------------------------------------------------------------


def _pp_curve(
    target_scores: np.ndarray,
    decoy_scores: np.ndarray,
    n_points: int = 500,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (F_decoy(t), F_target(t)) evaluated on a grid of n_points thresholds."""
    t_sorted = np.sort(target_scores)
    d_sorted = np.sort(decoy_scores)
    lo = float(min(t_sorted[0], d_sorted[0]))
    hi = float(max(t_sorted[-1], d_sorted[-1]))
    if lo >= hi:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])
    thresholds = np.linspace(lo, hi, n_points)
    x = np.searchsorted(d_sorted, thresholds, side="right") / len(d_sorted)
    y = np.searchsorted(t_sorted, thresholds, side="right") / len(t_sorted)
    # Prepend (0,0) so the curve starts at the origin
    x = np.concatenate([[0.0], x])
    y = np.concatenate([[0.0], y])
    return x, y


def _draw_pp_panel(
    ax: "matplotlib.axes.Axes",
    target_scores: np.ndarray,
    decoy_scores: np.ndarray,
    title: str,
    n_points: int = 500,
    pi0: float | None = None,
) -> None:
    """Draw a single PP-plot panel onto ax."""
    n_t = len(target_scores)
    n_d = len(decoy_scores)
    if n_t == 0 or n_d == 0:
        ax.set_visible(False)
        return

    x, y = _pp_curve(target_scores, decoy_scores, n_points=n_points)

    # Reference line 1: diagonal y = x (what a fully null distribution would look like)
    ax.plot([0, 1], [0, 1], color="grey", lw=1.0, ls="--", label="y = x (null)")

    # Reference line 2: y = (1 − π₁) × x — expected slope in the null-dominated region.
    # π₁ estimated under the TDC assumption: decoys approximate null targets.
    pi1_hat = max(0.0, 1.0 - n_d / n_t)
    slope = 1.0 - pi1_hat
    ax.plot(
        [0, 1], [0, slope],
        color="darkorange", lw=1.2, ls=":",
        label=f"y = (1−π₁)·x  [π₁≈{pi1_hat:.2f}]",
    )

    # Reference line 3 (optional): y = pi0 × x from Storey pi0 estimate.
    # The PP curve should track this line in the null-dominated (low-score) region.
    if pi0 is not None:
        ax.plot(
            [0, 1], [0, pi0],
            color="seagreen", lw=1.4, ls="--",
            label=f"expected (pi0={pi0:.2f})",
        )

    ax.plot(x, y, color="steelblue", lw=1.8, label=f"T (n={n_t})  vs  D (n={n_d})")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("F_decoy(t)", fontsize=9)
    ax.set_ylabel("F_target(t)", fontsize=9)
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7, loc="upper left")
    ax.tick_params(labelsize=8)
    ax.set_aspect("equal", adjustable="box")


def plot_score_pp(
    features_df: pd.DataFrame,
    result_df: pd.DataFrame,
    out_dir: str,
    n_points: int = 500,
    pi0: float | None = None,
    single_round: bool = False,
) -> None:
    """
    PP plot of score distributions: F_decoy(t) on the x-axis vs F_target(t)
    on the y-axis, sweeping threshold t across all observed scores.

    Three panels:
      Left   — Round-1 scores on all candidates.
      Centre — Round-2 scores on R1 winners (the TDC input set).
      Right  — Reweighted R2 scores on R1 winners (after LC-MS/MS + spatial prior).

    Reference lines on each panel:
      - Dashed grey  y = x: the curve if targets and decoys were identically
        distributed (pure null, no true positives).
      - Dotted orange y = (1−π₁)·x: expected null-component slope in the
        bottom-left region, where π₁ = max(0, 1 − n_decoy / n_target) is
        the fraction of estimated true positives among targets.

    A healthy TDC run: the curve starts at (0, 0), tracks the orange dotted
    line in the low-score region (where both distributions are dominated by
    incorrect matches), then peels away toward (1, 1) as the target CDF
    accumulates true positives in the high-score tail.

    Output: ``{out_dir}/score_pp_plot.png``
    """
    os.makedirs(out_dir, exist_ok=True)

    feat = features_df.reset_index(drop=True)
    res  = result_df.reset_index(drop=True)

    assert len(feat) == len(res), f"Length mismatch: {len(feat)} vs {len(res)}"

    is_decoy = (
        feat.get("is_decoy", pd.Series(False, index=feat.index))
        .fillna(False).astype(bool).values
    )
    is_winner = (
        res.get("is_tdc_winner", pd.Series(False, index=res.index))
        .fillna(False).astype(bool).values
    )

    # Detect score columns dynamically
    r2_cols = [c for c in res.columns if c.endswith("_score_r2")]
    r1_cols = [c for c in res.columns if c.endswith("_score_r1")]
    has_reweighted = "reweighted_score" in res.columns
    if not r2_cols and not r1_cols:
        logger.debug("score_pp: no score columns found, skipping")
        return

    r2_col = r2_cols[0] if r2_cols else None
    r1_col = r1_cols[0] if r1_cols else None

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Left panel: R1 scores on all candidates
    ax = axes[0]
    if r1_col is not None:
        r1_scores = pd.to_numeric(res[r1_col], errors="coerce").values
        finite_r1 = np.isfinite(r1_scores)
        t_r1 = r1_scores[~is_decoy & finite_r1]
        d_r1 = r1_scores[ is_decoy & finite_r1]
        _draw_pp_panel(ax, t_r1, d_r1, f"All candidates — {r1_col}", n_points=n_points)
    else:
        ax.set_visible(False)

    # Centre panel: final (round-2, or round-1 in single-round) scores on winners
    ax = axes[1]
    if r2_col is not None:
        r2_scores = pd.to_numeric(res[r2_col], errors="coerce").values
        finite_w = is_winner & np.isfinite(r2_scores)
        t_r2 = r2_scores[~is_decoy & finite_w]
        d_r2 = r2_scores[ is_decoy & finite_w]
        _ttl = "Winners — final score (R1)" if single_round else f"R1 winners — {r2_col}"
        _draw_pp_panel(ax, t_r2, d_r2, _ttl, n_points=n_points, pi0=pi0)
    else:
        ax.set_visible(False)

    # Right panel: reweighted R2 scores on R1 winners
    ax = axes[2]
    if has_reweighted:
        rw_scores = pd.to_numeric(res["reweighted_score"], errors="coerce").values
        finite_rw = is_winner & np.isfinite(rw_scores)
        t_rw = rw_scores[~is_decoy & finite_rw]
        d_rw = rw_scores[ is_decoy & finite_rw]
        _draw_pp_panel(ax, t_rw, d_rw, "R1 winners — reweighted_score", n_points=n_points, pi0=pi0)
    else:
        ax.set_visible(False)

    fig.suptitle("Score PP plot: target vs decoy empirical CDFs", fontsize=11)
    plt.tight_layout()
    _save_and_close(fig, os.path.join(out_dir, "score_pp_plot.png"))


# ---------------------------------------------------------------------------
# Score distributions
# ---------------------------------------------------------------------------

def plot_score_distributions(
    features_df: pd.DataFrame,
    result_df: pd.DataFrame,
    out_dir: str,
    n_bins: int = 60,
    single_round: bool = False,
) -> None:
    """
    Overlapping target/decoy score histograms for R1, R2, and reweighted scores.

    Three panels (left to right):
      R1  — all candidates (including non-winners).
      R2  — R1 winners only (non-winners have NaN R2 scores).
      RW  — reweighted score on R1 winners (same subset as R2).

    Each panel uses normalised counts (density=True) so target and decoy
    distributions are comparable when their sizes differ. A vertical dashed
    line marks the score threshold corresponding to q_value ≤ 0.01 on R2 (if
    computable). Panels with no finite scores are hidden.

    Output: ``{out_dir}/score_distributions.png``
    """
    os.makedirs(out_dir, exist_ok=True)

    feat = features_df.reset_index(drop=True)
    res  = result_df.reset_index(drop=True)

    is_decoy = (
        feat.get("is_decoy", pd.Series(False, index=feat.index))
        .fillna(False).astype(bool).values
    )
    is_winner = (
        res.get("is_tdc_winner", pd.Series(False, index=res.index))
        .fillna(False).astype(bool).values
    )
    _src_sd = feat.get("source", pd.Series("", index=feat.index)).fillna("").values
    entrapment_mask_sd = (~is_decoy) & (_src_sd == "entrapment_shuffled")
    has_entrapment_sd = entrapment_mask_sd.any()
    real_target_mask_sd = (~is_decoy) & (~entrapment_mask_sd)

    r1_cols = [c for c in res.columns if c.endswith("_score_r1")]
    r2_cols = [c for c in res.columns if c.endswith("_score_r2")]
    r1_col = r1_cols[0] if r1_cols else None
    r2_col = r2_cols[0] if r2_cols else None
    rw_col = "reweighted_score" if "reweighted_score" in res.columns else None

    panels = []
    if r1_col:
        s = pd.to_numeric(res[r1_col], errors="coerce").values
        panels.append((s, np.ones(len(s), dtype=bool), r1_col, "All candidates"))
    if r2_col:
        s = pd.to_numeric(res[r2_col], errors="coerce").values
        panels.append((s, is_winner, r2_col, "R1 winners"))
    if rw_col:
        s = pd.to_numeric(res[rw_col], errors="coerce").values
        panels.append((s, is_winner, "reweighted_score", "R1 winners"))

    if not panels:
        logger.debug("score_distributions: no score columns found, skipping")
        return

    # Q=0.01 threshold from R2 winners
    q_threshold_score = None
    if r2_col and "q_value" in res.columns:
        r2_scores = pd.to_numeric(res[r2_col], errors="coerce").values
        q_vals = pd.to_numeric(res["q_value"], errors="coerce").values
        mask = is_winner & real_target_mask_sd & np.isfinite(r2_scores) & np.isfinite(q_vals)
        if mask.any():
            passing = r2_scores[mask & (q_vals <= 0.01)]
            if len(passing):
                q_threshold_score = passing.min()

    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 4), squeeze=False)
    axes = axes[0]

    colours = {"T": "#2196F3", "D": "#F44336", "E": "goldenrod"}

    for ax, (scores, subset_mask, col_label, subset_label) in zip(axes, panels):
        t_scores = scores[real_target_mask_sd & subset_mask]
        d_scores = scores[is_decoy & subset_mask]
        e_scores = scores[entrapment_mask_sd & subset_mask] if has_entrapment_sd else np.array([])
        t_finite = t_scores[np.isfinite(t_scores)]
        d_finite = d_scores[np.isfinite(d_scores)]
        e_finite = e_scores[np.isfinite(e_scores)] if len(e_scores) else np.array([])

        if not len(t_finite) and not len(d_finite):
            ax.set_visible(False)
            continue

        all_finite = np.concatenate([t_finite, d_finite] + ([e_finite] if len(e_finite) else []))

        # IQR-based x-axis limits: robust to heavy-tailed and skewed distributions.
        # Whisker = Q1 - 3*IQR … Q3 + 3*IQR, then clipped to data range.
        q1, q3 = np.percentile(all_finite, [25, 75])
        iqr = q3 - q1
        lo = max(all_finite.min(), q1 - 3.0 * iqr)
        hi = min(all_finite.max(), q3 + 3.0 * iqr)
        if lo >= hi:
            lo, hi = all_finite.min(), all_finite.max()
        bins = np.linspace(lo, hi, n_bins + 1)

        if len(t_finite):
            ax.hist(
                t_finite, bins=bins, density=True,
                color=colours["T"], alpha=0.45, label=f"Target (n={len(t_finite):,})",
            )
        if len(d_finite):
            ax.hist(
                d_finite, bins=bins, density=True,
                color=colours["D"], alpha=0.45, label=f"Decoy (n={len(d_finite):,})",
            )
        if len(e_finite):
            ax.hist(
                e_finite, bins=bins, density=True,
                color=colours["E"], alpha=0.45, label=f"Entrapment (n={len(e_finite):,})",
            )

        # KDE overlay for clearer shape visualization.
        try:
            from scipy.stats import gaussian_kde
            x_kde = np.linspace(lo, hi, 300)
            if len(t_finite) >= 5:
                ax.plot(x_kde, gaussian_kde(t_finite)(x_kde),
                        color=colours["T"], lw=1.5)
            if len(d_finite) >= 5:
                ax.plot(x_kde, gaussian_kde(d_finite)(x_kde),
                        color=colours["D"], lw=1.5)
            if len(e_finite) >= 5:
                ax.plot(x_kde, gaussian_kde(e_finite)(x_kde),
                        color=colours["E"], lw=1.5, linestyle="--")
        except Exception:
            pass

        if q_threshold_score is not None and col_label == r2_col:
            ax.axvline(
                q_threshold_score, color="black", linestyle="--", linewidth=1.0,
                label=f"q≤0.01 ({q_threshold_score:.3f})",
            )

        ax.set_xlabel("Score")
        ax.set_ylabel("Density")
        ax.set_xlim(lo, hi)
        _disp = "final score (R1)" if (single_round and col_label == r2_col) else col_label
        ax.set_title(f"{_disp}\n({subset_label})", fontsize=9)
        ax.legend(fontsize=8)

    fig.suptitle("Score distributions — target vs decoy", fontsize=11)
    plt.tight_layout()
    _save_and_close(fig, os.path.join(out_dir, "score_distributions.png"))


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
            _save_and_close(fig, os.path.join(sub, fname), dpi=80)


# ---------------------------------------------------------------------------
# PEP mixture model visualization
# ---------------------------------------------------------------------------

def plot_pep_mixture(
    result_df: pd.DataFrame,
    out_dir: str,
    model_name: str = "model",
    n_bins: int = 50,
    pep_method: str = "gaussian",
    single_round: bool = False,
) -> None:
    """
    Overlay histogram of target and decoy R2 scores with fitted density curves
    and a secondary y-axis PEP curve.

    The ``pep_method`` parameter must match the method passed to ``estimate_pep``
    so that the overlay curves reflect the actual PEP computation:
    - ``"gaussian"`` (default): parametric Gaussian f0/f1 (LDA, SVM, CatBoost).
    - ``"kde"``: kernel density estimates (QDA).

    X-axis uses IQR-based limits (Q1 − 3×IQR … Q3 + 3×IQR) to handle
    heavy-tailed score distributions robustly.

    Reads ``pep`` and the ``*_score_r2`` column from ``result_df`` (winners only).
    Output: ``{out_dir}/pep_mixture.png``
    """
    os.makedirs(out_dir, exist_ok=True)

    is_winner = result_df.get("is_tdc_winner", pd.Series(False, index=result_df.index)).fillna(False).astype(bool)
    winners = result_df[is_winner].copy()
    if len(winners) == 0:
        logger.debug("plot_pep_mixture: no winners, skipping")
        return

    r2_cols = [c for c in winners.columns if c.endswith("_score_r2")]
    if not r2_cols:
        logger.debug("plot_pep_mixture: no R2 score column found, skipping")
        return
    r2_col = r2_cols[0]

    if "pep" not in winners.columns:
        logger.debug("plot_pep_mixture: pep column missing, skipping")
        return

    is_decoy_w = winners["is_decoy"].fillna(False).astype(bool).values
    scores = pd.to_numeric(winners[r2_col], errors="coerce").values
    pep_vals = pd.to_numeric(winners["pep"], errors="coerce").values

    finite = np.isfinite(scores) & np.isfinite(pep_vals)
    if finite.sum() < 4:
        logger.debug("plot_pep_mixture: too few finite winners, skipping")
        return

    scores_f = scores[finite]
    pep_f = pep_vals[finite]
    is_decoy_f = is_decoy_w[finite]

    t_scores = scores_f[~is_decoy_f]
    d_scores = scores_f[is_decoy_f]

    # IQR-based x limits: robust to heavy tails from QDA or CatBoost scores.
    q1, q3 = np.percentile(scores_f, [25, 75])
    iqr = q3 - q1
    lo = max(float(scores_f.min()), q1 - 3.0 * iqr)
    hi = min(float(scores_f.max()), q3 + 3.0 * iqr)
    if lo >= hi:
        lo, hi = float(scores_f.min()) - 0.5, float(scores_f.max()) + 0.5
    score_range = np.linspace(lo, hi, 300)

    # Shared setup for both methods (mirrors estimate_pep)
    median_t = float(np.median(t_scores)) if len(t_scores) >= 2 else float(np.mean(scores_f))
    high_t = t_scores[t_scores > median_t] if len(t_scores) >= 2 else t_scores
    if len(high_t) < 2:
        high_t = t_scores
    pi0 = is_decoy_f.sum() / len(is_decoy_f)

    fig, ax1 = plt.subplots(figsize=(8, 5))
    bins = np.linspace(lo, hi, n_bins + 1)
    ax1.hist(t_scores, bins=bins, alpha=0.4, color="steelblue", label="Targets", density=True)
    ax1.hist(d_scores, bins=bins, alpha=0.4, color="tomato", label="Decoys", density=True)

    if pep_method == "kde":
        # QDA: PEP comes directly from predict_proba — no mixture model fitting needed.
        # Overlay KDE curves of the score distributions to show separation, and fit an
        # isotonic regression through the (score, PEP) scatter for the trend line.
        from scipy.stats import gaussian_kde
        try:
            if len(t_scores) >= 5:
                ax1.plot(score_range, gaussian_kde(t_scores)(score_range),
                         color="steelblue", lw=1.5, ls="--", label="target KDE")
            if len(d_scores) >= 5:
                ax1.plot(score_range, gaussian_kde(d_scores)(score_range),
                         color="tomato", lw=1.5, ls="--", label="decoy KDE")
        except Exception:
            pass

        ax2 = ax1.twinx()
        sort_idx = np.argsort(scores_f)
        ax2.scatter(scores_f[sort_idx], pep_f[sort_idx],
                    s=6, color="grey", alpha=0.35, zorder=3, label="PEP (predict_proba)")
        try:
            from sklearn.isotonic import IsotonicRegression
            ir = IsotonicRegression(increasing=False, out_of_bounds="clip")
            pep_iso = ir.fit_transform(scores_f[sort_idx], pep_f[sort_idx])
            ax2.plot(scores_f[sort_idx], pep_iso, color="black", lw=2, label="PEP (isotonic)")
        except Exception:
            pass
        title_suffix = "QDA predict_proba"
    else:
        # Gaussian mixture: reconstruct f0/f1 and PEP curve analytically.
        from scipy.stats import norm
        mu0 = float(np.mean(d_scores)) if len(d_scores) >= 2 else float(np.mean(scores_f))
        sigma0 = max(float(np.std(d_scores)), 1e-6) if len(d_scores) >= 2 else 1.0
        mu1 = float(np.mean(high_t)) if len(high_t) >= 1 else mu0 + 1.0
        sigma1 = max(float(np.std(high_t)), 1e-6) if len(high_t) >= 2 else 1.0
        f0_curve = norm.pdf(score_range, mu0, sigma0)
        f1_curve = norm.pdf(score_range, mu1, sigma1)
        numer = pi0 * f0_curve
        denom = numer + (1.0 - pi0) * f1_curve
        with np.errstate(invalid="ignore", divide="ignore"):
            pep_curve = np.where(denom > 0, numer / denom, 1.0)

        ax1.plot(score_range, f1_curve, color="steelblue", lw=1.5, ls="--", label="f1 (signal)")
        ax1.plot(score_range, f0_curve, color="tomato", lw=1.5, ls="--", label="f0 (null)")

        ax2 = ax1.twinx()
        ax2.plot(score_range, pep_curve, color="black", lw=2, label="PEP")
        sort_idx = np.argsort(scores_f)
        ax2.scatter(scores_f[sort_idx], pep_f[sort_idx],
                    s=6, color="grey", alpha=0.4, zorder=3)
        title_suffix = "Gaussian mixture"

    ax1.set_xlabel("final score (R1)" if single_round else f"R2 score ({r2_col})")
    ax1.set_ylabel("Density")
    ax1.set_xlim(lo, hi)

    ax2.set_ylabel("PEP")
    ax2.set_ylim(-0.05, 1.15)
    ax2.axhline(0.05, color="black", lw=0.8, ls=":", alpha=0.6)
    ax2.axhline(0.20, color="black", lw=0.8, ls=":", alpha=0.4)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper left")

    _winner_lbl = "winners" if single_round else "R2 winners"
    ax1.set_title(
        f"PEP — {model_name} {_winner_lbl} (n={len(scores_f)}) [{title_suffix}]"
    )
    plt.tight_layout()
    _save_and_close(fig, os.path.join(out_dir, "pep_mixture.png"))


# ---------------------------------------------------------------------------
# Subsystem 12: Ion image Pearson r distribution
# ---------------------------------------------------------------------------

def plot_ion_image_pearson_distribution(
    features_df: pd.DataFrame,
    result_df: pd.DataFrame,
    ion_images: np.ndarray,
    ion_image_mzs: np.ndarray,
    out_dir: str,
    fdr_threshold: float = 0.05,
) -> None:
    """
    Distribution of pairwise ion image Pearson r for same-protein vs different-protein
    peptide pairs, restricted to target IDs at reweighted FDR <= fdr_threshold.

    Output: ``{out_dir}/ion_image_pearson_distribution.png``
    """
    os.makedirs(out_dir, exist_ok=True)

    feat = features_df.reset_index(drop=True)
    res  = result_df.reset_index(drop=True)

    is_winner = (
        res.get("is_tdc_winner", pd.Series(False, index=res.index))
        .fillna(False).astype(bool).values
    )
    rw_q = pd.to_numeric(
        res.get("reweighted_q_value", pd.Series(float("nan"), index=res.index)),
        errors="coerce",
    ).values
    is_decoy = (
        feat.get("is_decoy", pd.Series(False, index=feat.index))
        .fillna(False).astype(bool).values
    )

    id_mask = is_winner & (rw_q <= fdr_threshold) & ~is_decoy
    id_idx = np.where(id_mask)[0]
    if len(id_idx) < 2:
        logger.debug(
            "plot_ion_image_pearson_distribution: fewer than 2 target IDs at FDR %.2f, skipping",
            fdr_threshold,
        )
        return

    id_mzs = (
        feat["feature_mz"].values[id_idx]
        if "feature_mz" in feat.columns
        else np.full(len(id_idx), float("nan"))
    )
    id_proteins = (
        feat["protein"].fillna("unknown").values[id_idx]
        if "protein" in feat.columns
        else np.full(len(id_idx), "unknown")
    )

    flat_imgs: list[np.ndarray] = []
    valid_pos: list[int] = []
    for pos, mz in enumerate(id_mzs):
        if not np.isfinite(float(mz)):
            continue
        img_idx = _find_image_idx(float(mz), ion_image_mzs)
        if img_idx is None:
            continue
        flat_imgs.append(ion_images[img_idx].ravel().astype(float))
        valid_pos.append(pos)

    if len(valid_pos) < 2:
        return

    n = len(valid_pos)
    valid_proteins = np.array([id_proteins[p] for p in valid_pos])
    images_flat = np.array(flat_imgs)  # (n, n_pixels)

    # Full Pearson r matrix via normalised dot product
    means = images_flat.mean(axis=1, keepdims=True)
    stds  = images_flat.std(axis=1, keepdims=True)
    constant = stds.ravel() < 1e-9
    stds_safe = np.where(constant[:, None], 1.0, stds)
    normed = (images_flat - means) / stds_safe
    normed[constant] = 0.0
    corr = np.clip((normed @ normed.T) / images_flat.shape[1], -1.0, 1.0)

    same_r: list[float] = []
    diff_r: list[float] = []
    for i in range(n):
        if constant[i]:
            continue
        for j in range(i + 1, n):
            if constant[j]:
                continue
            r = float(corr[i, j])
            if valid_proteins[i] == valid_proteins[j]:
                same_r.append(r)
            else:
                diff_r.append(r)

    if not same_r and not diff_r:
        return

    rng = np.random.default_rng(42)
    max_bg = 10_000
    if len(diff_r) > max_bg:
        diff_r = rng.choice(diff_r, max_bg, replace=False).tolist()

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(-1.0, 1.0, 61)
    if diff_r:
        ax.hist(
            diff_r, bins=bins, alpha=0.55, color="steelblue", density=True,
            label=f"Different protein ({len(diff_r):,} pairs)",
        )
    if same_r:
        ax.hist(
            same_r, bins=bins, alpha=0.70, color="tomato", density=True,
            label=f"Same protein ({len(same_r):,} pairs)",
        )
    ax.axvline(0, color="black", lw=0.8, ls=":")
    ax.set_xlabel("Pearson r of ion images")
    ax.set_ylabel("Density")
    ax.set_title(
        f"Ion image colocalization — peptide pairs at ≤{fdr_threshold:.0%} reweighted FDR\n"
        f"(n={n} target IDs)"
    )
    ax.legend(fontsize=9)
    plt.tight_layout()
    _save_and_close(fig, os.path.join(out_dir, "ion_image_pearson_distribution.png"))


# ---------------------------------------------------------------------------
# Subsystem 13: Protein spatial coherence scatter
# ---------------------------------------------------------------------------

def plot_protein_spatial_coherence(
    features_df: pd.DataFrame,
    result_df: pd.DataFrame,
    ion_images: np.ndarray,
    ion_image_mzs: np.ndarray,
    out_dir: str,
    fdr_threshold: float = 0.05,
) -> None:
    """
    Per-protein scatter: number of unique peptides at reweighted FDR <= fdr_threshold
    (x-axis) vs mean pairwise ion image Pearson r (y-axis).  Singletons are shown at
    y = -0.15 with jitter.  Target and decoy proteins are coloured separately.

    Output: ``{out_dir}/protein_spatial_coherence.png``
    """
    os.makedirs(out_dir, exist_ok=True)

    feat = features_df.reset_index(drop=True)
    res  = result_df.reset_index(drop=True)

    is_winner = (
        res.get("is_tdc_winner", pd.Series(False, index=res.index))
        .fillna(False).astype(bool).values
    )
    rw_q = pd.to_numeric(
        res.get("reweighted_q_value", pd.Series(float("nan"), index=res.index)),
        errors="coerce",
    ).values
    is_decoy = (
        feat.get("is_decoy", pd.Series(False, index=feat.index))
        .fillna(False).astype(bool).values
    )

    id_mask = is_winner & (rw_q <= fdr_threshold)
    id_idx = np.where(id_mask)[0]
    if len(id_idx) < 2:
        return

    id_mzs   = (
        feat["feature_mz"].values[id_idx]
        if "feature_mz" in feat.columns
        else np.full(len(id_idx), float("nan"))
    )
    id_prots  = (
        feat["protein"].fillna("unknown").values[id_idx]
        if "protein" in feat.columns
        else np.full(len(id_idx), "unknown")
    )
    id_decoys = is_decoy[id_idx]

    prot_imgs: dict[str, list[np.ndarray]] = {}
    prot_is_decoy: dict[str, bool] = {}
    for mz, prot, dec in zip(id_mzs, id_prots, id_decoys):
        if not np.isfinite(float(mz)):
            continue
        img_idx = _find_image_idx(float(mz), ion_image_mzs)
        if img_idx is None:
            continue
        img = ion_images[img_idx].ravel().astype(float)
        if prot not in prot_imgs:
            prot_imgs[prot] = []
            prot_is_decoy[prot] = bool(dec)
        prot_imgs[prot].append(img)
        if not bool(dec):
            prot_is_decoy[prot] = False  # target evidence takes priority

    rows = []
    for prot, imgs in prot_imgs.items():
        n_pep = len(imgs)
        if n_pep == 1:
            mean_r = float("nan")
        else:
            rs = [
                _pearson_r(imgs[a], imgs[b])
                for a in range(n_pep)
                for b in range(a + 1, n_pep)
            ]
            finite_rs = [r for r in rs if np.isfinite(r)]
            mean_r = float(np.mean(finite_rs)) if finite_rs else float("nan")
        rows.append({
            "protein": prot,
            "n_peptides": n_pep,
            "mean_r": mean_r,
            "is_decoy": prot_is_decoy[prot],
        })

    if not rows:
        return

    df_p  = pd.DataFrame(rows)
    multi  = df_p[df_p["n_peptides"] > 1]
    single = df_p[df_p["n_peptides"] == 1]

    rng = np.random.default_rng(42)
    fig, ax = plt.subplots(figsize=(8, 5))

    for dec, color, lbl in [(False, "tomato", "target"), (True, "steelblue", "decoy")]:
        sub = multi[multi["is_decoy"] == dec]
        if not sub.empty:
            ax.scatter(
                sub["n_peptides"], sub["mean_r"],
                c=color, alpha=0.75, s=50, zorder=3,
                label=f"Multi-peptide {lbl} (n={len(sub)})",
            )

    for dec, color, lbl in [(False, "tomato", "target"), (True, "steelblue", "decoy")]:
        sub = single[single["is_decoy"] == dec]
        if not sub.empty:
            jx = rng.uniform(-0.08, 0.08, len(sub))
            jy = rng.uniform(-0.02, 0.02, len(sub))
            ax.scatter(
                sub["n_peptides"].values + jx, -0.15 + jy,
                c=color, alpha=0.45, s=25, marker="x", zorder=2,
                label=f"Singleton {lbl} (n={len(sub)})",
            )

    if not multi.empty:
        top_n = multi.nlargest(min(10, len(multi)), "n_peptides")
        for _, r in top_n.iterrows():
            short = str(r["protein"]).split("|")[-1][:20]
            ax.annotate(
                short, (r["n_peptides"], r["mean_r"]),
                textcoords="offset points", xytext=(5, 3),
                fontsize=6, alpha=0.8,
            )

    ax.axhline(0.0, color="black", lw=0.8, ls=":")
    ax.set_xlabel("Unique peptides at FDR threshold")
    ax.set_ylabel("Mean pairwise ion image Pearson r")
    ax.set_title(
        f"Protein spatial coherence at ≤{fdr_threshold:.0%} reweighted FDR\n"
        f"(n={len(df_p)} proteins; singletons shown at y=−0.15)"
    )
    ax.set_xlim(left=0.0)
    ax.legend(fontsize=8)
    plt.tight_layout()
    _save_and_close(fig, os.path.join(out_dir, "protein_spatial_coherence.png"))


# ---------------------------------------------------------------------------
# Subsystem 15: Per-candidate SHAP explanations (LinearExplainer)
# ---------------------------------------------------------------------------

def _reshape_ion_image(
    img: np.ndarray, spatial_df: pd.DataFrame | None
) -> np.ndarray | None:
    """Return a 2D ion image.

    Ion images in this pipeline are already ``(H, W)`` (the array is indexed
    ``ion_images[feature_idx]``), so a 2D input is returned unchanged.  A 1D
    flat image is reshaped from per-pixel ``x``/``y`` coordinates in
    ``spatial_df`` when those columns are present, falling back to a near-square
    layout otherwise.
    """
    img = np.asarray(img)
    if img.ndim == 2:
        return img
    if img.ndim != 1:
        return None
    xs = ys = None
    if spatial_df is not None:
        for xc, yc in (("x", "y"), ("x_coords", "y_coords"), ("pixel_x", "pixel_y")):
            if xc in spatial_df.columns and yc in spatial_df.columns:
                xs = spatial_df[xc].to_numpy()
                ys = spatial_df[yc].to_numpy()
                break
    if xs is not None and ys is not None and len(xs) == img.size:
        x0, y0 = int(np.min(xs)), int(np.min(ys))
        w = int(np.max(xs)) - x0 + 1
        h = int(np.max(ys)) - y0 + 1
        grid = np.zeros((h, w), dtype=float)
        grid[(ys.astype(int) - y0), (xs.astype(int) - x0)] = img
        return grid
    side = int(np.ceil(np.sqrt(img.size)))
    padded = np.zeros(side * side, dtype=float)
    padded[: img.size] = img
    return padded.reshape(side, side)


def debug_pfm_explanations(
    result_df: pd.DataFrame,
    X: np.ndarray,
    svm_pipeline,
    feature_names: list[str],
    ion_images: np.ndarray | None,
    feature_mzs: np.ndarray | None,
    spatial_df: pd.DataFrame | None,
    output_dir: str,
    n_decoys: int = 10,
    fdr_threshold: float | None = None,
) -> None:
    """
    Per-PFM SHAP explanation figures for the linear rescoring model.

    For a set of selected peptide-feature matches (PFMs) — all target TDC winners
    passing FDR plus a random sample of decoy winners — this computes SHAP values
    with ``shap.LinearExplainer`` (``feature_perturbation="interventional"``) on
    the bare linear estimator inside ``svm_pipeline`` and saves a three-panel
    figure per candidate plus a summary TSV.

    ``result_df`` must be aligned row-for-row with ``X`` (the raw, pre-pipeline
    feature matrix the model was trained on, one row per winner).  ``feature_names``
    names the columns of ``X``.  ``svm_pipeline`` is the fitted sklearn ``Pipeline``
    (imputer → scaler → [poly] → linear estimator).

    Selection
    ---------
    Targets: all TDC winners with ``q_value <= fdr_threshold`` (default 0.01),
    falling back to 0.05 when fewer than one target passes at 1%.  Decoys:
    ``n_decoys`` random TDC winners with ``is_decoy=True`` (``random.seed(42)``).

    Outputs (``<output_dir>/pfm_explanations/``)
    --------------------------------------------
    One PNG per candidate, ``{rank:03d}_{peptide}_{feature_mz:.4f}_{target|decoy}.png``
    (rank by ``q_value`` for targets, sampling order for decoys), each with:
      Left   — the candidate feature's ion image (``hot`` colormap, gamma 0.5).
      Middle — SHAP waterfall: top-15 per-feature contributions sorted by |SHAP|,
               positive (toward target) in steelblue, negative in tomato, with the
               base value and final score annotated.
      Right  — percentile rank of each top-15 feature value within the training
               distribution (0–100 horizontal bar with a marker).
    Plus ``summary.tsv`` with one row per explained candidate.
    """
    import random

    try:
        import shap
    except ImportError:
        logger.warning(
            "debug_pfm_explanations: the 'shap' package is not installed — "
            "skipping PFM SHAP explanations (pip install shap)."
        )
        return

    out_dir      = os.path.join(output_dir, "pfm_explanations")
    shap_data_dir = os.path.join(out_dir, "shap_data")
    os.makedirs(out_dir,       exist_ok=True)
    os.makedirs(shap_data_dir, exist_ok=True)

    res = result_df.reset_index(drop=True)
    X = np.asarray(X, dtype=np.float64)
    if X.shape[0] != len(res):
        logger.warning(
            "debug_pfm_explanations: X has %d rows but result_df has %d — "
            "cannot align; skipping.",
            X.shape[0], len(res),
        )
        return

    # TDC-winner population (the rows passed are winners, but guard anyway).
    if "is_tdc_winner" in res.columns:
        winner_mask = res["is_tdc_winner"].fillna(False).astype(bool).values
    else:
        winner_mask = np.ones(len(res), dtype=bool)
    is_decoy = res.get("is_decoy", pd.Series(False, index=res.index)).fillna(False).astype(bool).values
    q_value = pd.to_numeric(
        res.get("q_value", pd.Series(np.nan, index=res.index)), errors="coerce"
    ).values

    # --- Select targets at FDR (fall back 1% → 5%) ---
    if fdr_threshold is None:
        thr = 0.01
        target_pos = np.where(winner_mask & ~is_decoy & (q_value <= thr))[0]
        if len(target_pos) < 1:
            thr = 0.05
            target_pos = np.where(winner_mask & ~is_decoy & (q_value <= thr))[0]
    else:
        thr = float(fdr_threshold)
        target_pos = np.where(winner_mask & ~is_decoy & (q_value <= thr))[0]
    # Rank targets by q_value ascending.
    target_pos = target_pos[np.argsort(q_value[target_pos], kind="stable")]

    # --- Sample decoy winners ---
    decoy_candidates = np.where(winner_mask & is_decoy)[0].tolist()
    random.seed(42)
    if len(decoy_candidates) > n_decoys:
        decoy_pos = sorted(random.sample(decoy_candidates, n_decoys))
    else:
        decoy_pos = decoy_candidates

    logger.info(
        "debug_pfm_explanations: explaining %d targets (q<=%.2g) and %d decoys",
        len(target_pos), thr, len(decoy_pos),
    )
    if len(target_pos) == 0 and len(decoy_pos) == 0:
        logger.warning("debug_pfm_explanations: no candidates to explain — skipping.")
        return

    # --- Pipeline decomposition: pre-processing (imputer/scaler/[poly]) + estimator ---
    try:
        pre = svm_pipeline[:-1]
        estimator = svm_pipeline[-1]
        Xt_all = np.asarray(pre.transform(X), dtype=np.float64)
        coef = np.asarray(estimator.coef_, dtype=np.float64).ravel()
        intercept = float(np.asarray(estimator.intercept_).ravel()[0])
    except Exception as exc:
        logger.warning(
            "debug_pfm_explanations: could not decompose pipeline / estimator (%s) — skipping.",
            exc,
        )
        return

    # The transformed matrix may have more columns than feature_names when a
    # polynomial-interaction step expands the inputs; in that case raw-value and
    # percentile annotations fall back to the transformed feature space.
    if Xt_all.shape[1] == len(feature_names):
        est_names = list(feature_names)
        raw_aligned = True
    else:
        est_names = [f"f{j}" for j in range(Xt_all.shape[1])]
        raw_aligned = False
        logger.info(
            "debug_pfm_explanations: transformed matrix has %d columns vs %d raw "
            "feature names (poly expansion?) — using transformed values for labels.",
            Xt_all.shape[1], len(feature_names),
        )

    # --- SHAP LinearExplainer on the bare linear estimator ---
    # feature_perturbation="interventional" with the full transformed training
    # matrix as background.
    try:
        explainer = shap.LinearExplainer(
            (coef, intercept), Xt_all, feature_perturbation="interventional"
        )
        base_value = float(np.asarray(explainer.expected_value).ravel()[0])
    except Exception as exc:
        logger.warning("debug_pfm_explanations: LinearExplainer failed (%s) — skipping.", exc)
        return

    # Per-column training distributions for percentile ranks (raw space when aligned).
    dist_matrix = X if raw_aligned else Xt_all
    n_train = dist_matrix.shape[0]

    summary_rows: list[dict] = []
    top_k = 15

    def _explain_one(pos: int, rank: int, kind: str) -> None:
        row = res.iloc[pos]
        peptide = str(row.get("peptide", "unknown"))
        protein = str(row.get("protein", ""))
        feature_mz = _get(row, "feature_mz")
        qv = _get(row, "q_value")

        xt = Xt_all[pos : pos + 1]
        shap_vals = np.asarray(explainer.shap_values(xt)).reshape(-1)
        final_score = base_value + float(shap_vals.sum())

        order = np.argsort(np.abs(shap_vals))[::-1][:top_k]
        sel_names = [est_names[j] for j in order]
        sel_shap = shap_vals[order]
        if raw_aligned:
            sel_raw = X[pos, order]
        else:
            sel_raw = Xt_all[pos, order]

        # ----- Figure -----
        _BG      = "#F7F7F7"
        _POS_COL = "#4C9BE8"   # steel blue — toward target
        _NEG_COL = "#E8654C"   # coral     — away from target

        fig, (ax_img, ax_shap, ax_pct) = plt.subplots(
            1, 3, figsize=(16, 6),
            gridspec_kw={"width_ratios": [1.0, 1.8, 0.9]},
            facecolor=_BG,
        )
        fig.patch.set_facecolor(_BG)

        # Left: ion image (gamma 0.5, hot)
        img2d = None
        if ion_images is not None and feature_mzs is not None and np.isfinite(feature_mz):
            idx = _find_image_idx(float(feature_mz), feature_mzs)
            if idx is not None:
                img2d = _reshape_ion_image(ion_images[idx], spatial_df)
        ax_img.set_facecolor("black")
        for _sp in ax_img.spines.values():
            _sp.set_visible(False)
        ax_img.set_xticks([]); ax_img.set_yticks([])
        if img2d is not None:
            _p99 = np.percentile(img2d[img2d > 0], 99) if (img2d > 0).any() else 1.0
            _imd = np.clip(img2d / _p99, 0, 1) ** 0.5
            _im  = ax_img.imshow(_imd, cmap="hot", vmin=0, vmax=1, aspect="auto",
                                 interpolation="nearest")
            _cax = ax_img.inset_axes([0.02, 0.02, 0.06, 0.35])
            _cb  = fig.colorbar(_im, cax=_cax)
            _cb.set_ticks([0, 1]); _cb.set_ticklabels(["0", "p99"], fontsize=6, color="white")
            _cb.outline.set_edgecolor("white")
            _cb.ax.tick_params(colors="white", length=2)
            ax_img.set_title(f"{peptide}\n{feature_mz:.4f} Da", fontsize=9,
                             fontweight="bold", color="#222222", pad=4)
        else:
            ax_img.text(0.5, 0.5, "No ion image", ha="center", va="center",
                        transform=ax_img.transAxes, color="gray", fontsize=9)
            ax_img.set_title(f"{peptide}", fontsize=9, fontweight="bold",
                             color="#222222", pad=4)

        # Middle: SHAP waterfall (top 15 by |SHAP|)
        ypos   = np.arange(len(order))[::-1]  # largest |SHAP| at top
        colors = [_POS_COL if v >= 0 else _NEG_COL for v in sel_shap]
        ax_shap.set_facecolor("white")
        ax_shap.barh(ypos, sel_shap, color=colors, height=0.65,
                     edgecolor="white", linewidth=0.4, zorder=3)
        ax_shap.axvline(0, color="#333333", lw=1.0, zorder=4)
        ax_shap.axvline(base_value,  color="#888888", lw=0.8, ls="--", zorder=2)
        ax_shap.axvline(final_score, color="#222222", lw=1.2, ls=":",  zorder=2)
        ax_shap.set_yticks(ypos)
        ax_shap.set_yticklabels(
            [n.replace("_", " ") for n in sel_names],
            fontsize=7.5,
        )
        ax_shap.set_xlabel("SHAP contribution  (→ target)", fontsize=8, color="#444444")
        _xlim = (
            min(sel_shap.min() - 0.3, base_value  - 0.3),
            max(sel_shap.max() + 0.3, final_score + 0.3),
        )
        ax_shap.set_xlim(*_xlim)
        ax_shap.text(base_value,  1.01, f"base\n{base_value:.3f}",
                     ha="center", va="bottom", fontsize=6.5, color="#888888",
                     transform=ax_shap.get_xaxis_transform())
        ax_shap.text(final_score, 1.01, f"score\n{final_score:.3f}",
                     ha="center", va="bottom", fontsize=6.5, color="#222222", fontweight="bold",
                     transform=ax_shap.get_xaxis_transform())
        ax_shap.tick_params(axis="x", labelsize=7, colors="#555555")
        ax_shap.tick_params(axis="y", colors="#222222", length=0)
        ax_shap.xaxis.grid(True, color="#dddddd", lw=0.5, zorder=0)
        ax_shap.set_axisbelow(True)
        for _sp in ["top", "right", "left"]: ax_shap.spines[_sp].set_visible(False)
        ax_shap.spines["bottom"].set_color("#cccccc")

        # Right: percentile rank within training distribution
        pct = np.full(len(order), np.nan)
        for i, j in enumerate(order):
            colvals = dist_matrix[:, j]
            finite  = colvals[np.isfinite(colvals)]
            v = sel_raw[i]
            if finite.size > 0 and np.isfinite(v):
                pct[i] = 100.0 * np.count_nonzero(finite <= v) / finite.size
        ax_pct.set_facecolor("white")
        ax_pct.barh(ypos, np.full(len(order), 100.0), color="#eeeeee", height=0.65, zorder=1)
        ax_pct.barh(ypos, np.nan_to_num(pct),         color="#cccccc", height=0.65, zorder=2)
        for yp, p, col in zip(ypos, pct, colors):
            if np.isfinite(p):
                ax_pct.plot(p, yp, "o", color=col, ms=7, zorder=5,
                            markeredgecolor="white", markeredgewidth=0.5)
                _offset, _ha = (-3, "right") if p > 88 else (2, "left")
                ax_pct.text(p + _offset, yp, f"{p:.0f}", va="center",
                            fontsize=6.5, color="#333333", ha=_ha)
        ax_pct.set_xlim(0, 100)
        ax_pct.set_ylim(ax_shap.get_ylim())
        ax_pct.set_yticks([])
        ax_pct.set_xticks([0, 25, 50, 75, 100])
        ax_pct.set_xticklabels(["0", "25", "50", "75", "100"], fontsize=6.5, color="#555555")
        ax_pct.set_xlabel("Percentile in training dist.", fontsize=8, color="#444444")
        ax_pct.tick_params(axis="x", length=2)
        for _sp in ["top", "right", "left"]: ax_pct.spines[_sp].set_visible(False)
        ax_pct.spines["bottom"].set_color("#cccccc")
        ax_pct.xaxis.grid(True, color="#eeeeee", lw=0.5, zorder=0)
        ax_pct.set_axisbelow(True)

        # Save per-candidate SHAP data for later reproduction
        _sel_coef = coef[order] if len(coef) == len(est_names) else np.full(len(order), np.nan)
        pd.DataFrame({
            "feature":        sel_names,
            "shap_value":     sel_shap,
            "raw_value":      sel_raw,
            "percentile_rank": pct,
            "coef":           _sel_coef,
        }).assign(
            peptide=peptide, protein=protein,
            feature_mz=feature_mz, q_value=qv,
            final_score=final_score, base_value=base_value, kind=kind,
        ).to_csv(
            os.path.join(
                shap_data_dir,
                f"{rank:03d}_{_safe_fname(peptide)}_{feature_mz:.4f}_{kind}.tsv"
                if np.isfinite(feature_mz)
                else f"{rank:03d}_{_safe_fname(peptide)}_{kind}.tsv",
            ),
            sep="\t", index=False,
        )

        _q_s = f"q = {qv:.4f}" if np.isfinite(qv) else "q = NA"
        fig.tight_layout(rect=[0, 0, 1, 0.91])
        fig.text(
            0.5, 0.97,
            f"[{kind}]  {peptide}  ·  {protein}  ·  m/z {feature_mz:.4f}"
            f"  ·  {_q_s}  ·  score = {final_score:.3f}",
            ha="center", va="top", fontsize=10, fontweight="bold",
            color="#111111",
        )
        fname = (
            f"{rank:03d}_{_safe_fname(peptide)}_{feature_mz:.4f}_{kind}.png"
            if np.isfinite(feature_mz)
            else f"{rank:03d}_{_safe_fname(peptide)}_{kind}.png"
        )
        _save_and_close(fig, os.path.join(out_dir, fname), dpi=150)

        srow = {
            "peptide": peptide,
            "protein": protein,
            "feature_mz": feature_mz,
            "q_value": qv,
            "is_decoy": bool(row.get("is_decoy", False)),
            "final_score": final_score,
        }
        for r2c in res.columns:
            if r2c.endswith("_score_r2"):
                srow[r2c] = _get(row, r2c)
        for t in range(3):
            if t < len(order):
                srow[f"shap{t+1}_feature"] = sel_names[t]
                srow[f"shap{t+1}_value"] = float(sel_shap[t])
                srow[f"shap{t+1}_feature_value"] = float(sel_raw[t])
            else:
                srow[f"shap{t+1}_feature"] = ""
                srow[f"shap{t+1}_value"] = np.nan
                srow[f"shap{t+1}_feature_value"] = np.nan
        summary_rows.append(srow)

    for rank, pos in enumerate(target_pos):
        try:
            _explain_one(int(pos), rank, "target")
        except Exception as exc:
            logger.debug("debug_pfm_explanations: target row %d failed: %s", pos, exc)
            plt.close("all")
    for rank, pos in enumerate(decoy_pos):
        try:
            _explain_one(int(pos), rank, "decoy")
        except Exception as exc:
            logger.debug("debug_pfm_explanations: decoy row %d failed: %s", pos, exc)
            plt.close("all")

    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(
            os.path.join(out_dir, "summary.tsv"), sep="\t", index=False
        )
    logger.info(
        "debug_pfm_explanations: wrote %d figures + summary.tsv to %s",
        len(summary_rows), out_dir,
    )


# ---------------------------------------------------------------------------
# Target / decoy 3D scatter: m/z × ion mobility × intensity
# ---------------------------------------------------------------------------

def plot_mz_mobility_intensity_scatter(
    features_df: pd.DataFrame,
    out_dir: str,
    filename: str = "mz_mobility_intensity_scatter.png",
) -> None:
    """
    Scatter plot of candidates in (m/z, observed CCS, log-intensity) space.

    x: ``feature_mz``; y: ``im2deep_observed_ccs`` (falls back to
    ``im2deep_predicted_ccs``); colour: log10(``feature_intensity_p90`` + 1)
    mapped to a diverging colormap; marker shape: target (circle) vs decoy
    (cross). Silently skips when neither CCS column is present.

    Saved to ``{out_dir}/{filename}``.
    """
    ccs_col = None
    for c in ("im2deep_observed_ccs", "im2deep_predicted_ccs"):
        if c in features_df.columns:
            ccs_col = c
            break
    if ccs_col is None or "feature_mz" not in features_df.columns:
        return

    os.makedirs(out_dir, exist_ok=True)

    feat = features_df.reset_index(drop=True)
    fmz = pd.to_numeric(feat["feature_mz"], errors="coerce").values
    ccs = pd.to_numeric(feat[ccs_col], errors="coerce").values
    is_decoy = feat.get("is_decoy", pd.Series(False, index=feat.index)).fillna(False).astype(bool).values

    # intensity column: p90 preferred, then raw, then ones
    int_col = next(
        (c for c in ("feature_intensity_p90", "feature_intensity") if c in feat.columns),
        None,
    )
    if int_col is not None:
        raw_int = pd.to_numeric(feat[int_col], errors="coerce").values
        raw_int = np.where(np.isfinite(raw_int) & (raw_int >= 0), raw_int, 0.0)
    else:
        raw_int = np.ones(len(feat), dtype=float)
    log_int = np.log10(raw_int + 1.0)

    valid = np.isfinite(fmz) & np.isfinite(ccs)
    if not valid.any():
        return

    fmz_v, ccs_v, li_v, dec_v = fmz[valid], ccs[valid], log_int[valid], is_decoy[valid]

    fig, ax = plt.subplots(figsize=(8, 6))

    vmin, vmax = float(np.percentile(li_v, 5)), float(np.percentile(li_v, 95))
    if vmin >= vmax:
        vmin, vmax = 0.0, max(1.0, float(li_v.max()))

    for mask, marker, label in [
        (~dec_v, "o", "Target"),
        (dec_v,  "x", "Decoy"),
    ]:
        if not mask.any():
            continue
        sc = ax.scatter(
            fmz_v[mask], ccs_v[mask],
            c=li_v[mask], cmap="viridis",
            vmin=vmin, vmax=vmax,
            s=6 if marker == "o" else 8,
            alpha=0.4 if marker == "o" else 0.6,
            marker=marker,
            linewidths=0.5,
            label=label,
        )

    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label(f"log₁₀({int_col or 'intensity'} + 1)", fontsize=9)

    ccs_label = "Observed CCS (Å²)" if ccs_col == "im2deep_observed_ccs" else "Predicted CCS (Å²)"
    ax.set_xlabel("Feature m/z (Da)", fontsize=10)
    ax.set_ylabel(ccs_label, fontsize=10)
    ax.set_title("Target vs decoy: m/z × ion mobility × intensity", fontsize=11)
    ax.legend(markerscale=2, fontsize=8, loc="upper left")
    ax.tick_params(labelsize=8)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, filename), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Subsystem 16: Isotope-envelope CCS consistency
# ---------------------------------------------------------------------------


def plot_isotope_ccs_consistency(
    features_df: pd.DataFrame,
    output_path: str,
    n_profiles: int = 12,
    seed: int = 42,
) -> None:
    """
    Four-panel diagnostic for the isotope-envelope CCS consistency features.

    Real singly-charged isotopologues of one molecule share a CCS; a chimeric
    envelope (isobaric mass coincidence) does not.  ``isotope_ccs_spread`` is the
    CCS analogue of IsoMobil's IPMV, and these panels show whether it separates
    targets from decoys without being an intensity proxy.

    A. Spread distribution by class (targets vs decoys, rows with >= 2 peaks),
       titled with the target-vs-decoy AUC — the headline discrimination view.
    B. Envelope CCS profiles for up to ``n_profiles`` features: observed CCS at
       isotopologue index 0/1/2.  Flat = consistent envelope, sloped/jagged =
       chimera.  A co-located decoy on the same feature is overlaid when present.
    C. ``isotope_ccs_spread`` vs log10(I_0), coloured by class — the feature must
       not be a mere intensity proxy.
    D. ``isotope_ccs_n_peaks`` distribution (0-3) split by class.

    Silently skips when the columns are absent (feature-list mode, TSF, or no TIMS
    dimension).  Saved to ``output_path``.
    """
    if "isotope_ccs_spread" not in features_df.columns:
        return
    if "isotope_ccs_n_peaks" not in features_df.columns:
        return

    feat = features_df.reset_index(drop=True)
    spread = pd.to_numeric(feat["isotope_ccs_spread"], errors="coerce").values
    n_peaks = pd.to_numeric(feat["isotope_ccs_n_peaks"], errors="coerce").values
    is_decoy = (
        feat.get("is_decoy", pd.Series(False, index=feat.index))
        .fillna(False).astype(bool).values
    )
    if not np.isfinite(spread).any():
        logger.debug("plot_isotope_ccs_consistency: no finite spread values, skipping")
        return

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    T_COLOR, D_COLOR = "steelblue", "tomato"

    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.25)

    # ---- Panel A: spread distribution by class ----
    ax = fig.add_subplot(gs[0, 0])
    fin = np.isfinite(spread)
    t_sp, d_sp = spread[fin & ~is_decoy], spread[fin & is_decoy]
    auc = float("nan")
    if len(t_sp) >= 5 and len(d_sp) >= 5:
        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score(is_decoy[fin].astype(int), spread[fin]))
    if fin.any():
        hi = float(np.percentile(spread[fin], 99))
        bins = np.linspace(0.0, max(hi, 1e-6), 50)
        if len(t_sp):
            ax.hist(t_sp, bins=bins, alpha=0.55, color=T_COLOR, density=True,
                    label=f"Target (n={len(t_sp)})")
        if len(d_sp):
            ax.hist(d_sp, bins=bins, alpha=0.55, color=D_COLOR, density=True,
                    label=f"Decoy (n={len(d_sp)})")
    _auc_str = "n/a" if not np.isfinite(auc) else f"{auc:.3f}"
    ax.set_xlabel("isotope_ccs_spread (Å²)", fontsize=10)
    ax.set_ylabel("density", fontsize=10)
    ax.set_title(
        f"A. Envelope CCS spread by class (n_peaks ≥ 2)\ntarget-vs-decoy AUC = {_auc_str}",
        fontsize=10,
    )
    ax.legend(fontsize=8)
    ax.tick_params(labelsize=8)

    # ---- Panel B: envelope CCS profiles ----
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.axis("off")
    ccs_cols = [f"isotope_ccs_m{k}" for k in (0, 1, 2)]
    if all(c in feat.columns for c in ccs_cols) and "feature_mz" in feat.columns:
        prof = feat[ccs_cols].apply(pd.to_numeric, errors="coerce").values
        fmz = pd.to_numeric(feat["feature_mz"], errors="coerce").values
        # Prefer features whose target has a measurable (>= 2 peak) envelope.
        cand = np.where(np.isfinite(spread) & ~is_decoy)[0]
        if len(cand) == 0:
            cand = np.where(np.isfinite(spread))[0]
        if len(cand):
            rng = np.random.default_rng(seed)
            pick = cand if len(cand) <= n_profiles else rng.choice(
                cand, size=n_profiles, replace=False
            )
            pick = pick[np.argsort(fmz[pick])]
            ncol = 4
            nrow = int(np.ceil(len(pick) / ncol))
            inner = gridspec.GridSpecFromSubplotSpec(
                nrow, ncol, subplot_spec=gs[0, 1], hspace=0.75, wspace=0.45
            )
            for j, i in enumerate(pick):
                sub = fig.add_subplot(inner[j // ncol, j % ncol])
                sub.plot([0, 1, 2], prof[i], "o-", color=T_COLOR, ms=4, lw=1.2)
                lab = f"{fmz[i]:.3f}\nT {spread[i]:.1f}"
                # Co-located decoy on the identical feature (mz_shuffle-style nulls).
                co = np.where((fmz == fmz[i]) & is_decoy)[0]
                if len(co):
                    c = co[0]
                    sub.plot([0, 1, 2], prof[c], "s--", color=D_COLOR, ms=4, lw=1.2)
                    lab += f" | D {spread[c]:.1f}"
                sub.set_title(lab, fontsize=6)
                sub.set_xticks([0, 1, 2])
                sub.tick_params(labelsize=5)
            fig.text(
                0.74, 0.955,
                "B. Envelope CCS profiles (x = isotopologue k, y = observed CCS Å²)",
                fontsize=10, ha="center",
            )

    # ---- Panel C: spread vs log10(I_0) ----
    ax = fig.add_subplot(gs[1, 0])
    if "isotope_ccs_int_m0" in feat.columns:
        i0 = pd.to_numeric(feat["isotope_ccs_int_m0"], errors="coerce").values
        m = np.isfinite(spread) & np.isfinite(i0) & (i0 > 0)
        if m.any():
            li0 = np.log10(i0[m])
            sp_m, dec_m = spread[m], is_decoy[m]
            ax.scatter(li0[~dec_m], sp_m[~dec_m], s=7, alpha=0.35, color=T_COLOR,
                       linewidths=0, label="Target")
            ax.scatter(li0[dec_m], sp_m[dec_m], s=7, alpha=0.35, color=D_COLOR,
                       linewidths=0, label="Decoy")
            r = (float(np.corrcoef(li0, sp_m)[0, 1])
                 if len(li0) > 1 and np.std(li0) > 0 and np.std(sp_m) > 0
                 else float("nan"))
            ax.set_title(
                f"C. Spread vs M0 intensity (r = {r:.3f})\nnot an intensity proxy if |r| is small",
                fontsize=10,
            )
            ax.legend(fontsize=8, markerscale=2)
    else:
        ax.set_title("C. Spread vs M0 intensity (isotope_ccs_int_m0 absent)", fontsize=10)
    ax.set_xlabel("log₁₀(I₀)", fontsize=10)
    ax.set_ylabel("isotope_ccs_spread (Å²)", fontsize=10)
    ax.tick_params(labelsize=8)

    # ---- Panel D: peak-count distribution ----
    ax = fig.add_subplot(gs[1, 1])
    levels = [0, 1, 2, 3]
    width = 0.38
    x = np.arange(len(levels))
    for off, mask, color, name in [
        (-width / 2, ~is_decoy, T_COLOR, "Target"),
        (+width / 2, is_decoy, D_COLOR, "Decoy"),
    ]:
        vals = n_peaks[mask & np.isfinite(n_peaks)]
        counts = [int((vals == lv).sum()) for lv in levels]
        frac = np.array(counts, dtype=float) / max(len(vals), 1)
        ax.bar(x + off, frac, width=width, color=color, alpha=0.75,
               label=f"{name} (n={len(vals)})")
    ax.set_xticks(x)
    ax.set_xticklabels([str(lv) for lv in levels])
    ax.set_xlabel("isotope_ccs_n_peaks", fontsize=10)
    ax.set_ylabel("fraction of class", fontsize=10)
    ax.set_title("D. Envelope peak-count distribution", fontsize=10)
    ax.legend(fontsize=8)
    ax.tick_params(labelsize=8)

    fig.suptitle("Isotope-envelope CCS consistency (IsoMobil-style IPMV in CCS units)",
                 fontsize=12)
    _save_and_close(fig, output_path)


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
    pep_method: str = "gaussian",
    importances_r1: np.ndarray | None = None,
    importances_r2: np.ndarray | None = None,
    importance_names: list[str] | None = None,
    importance_names_r2: list[str] | None = None,
    structure_coefs_r1: np.ndarray | None = None,
    structure_names_r1: list[str] | None = None,
    structure_coefs_r2: np.ndarray | None = None,
    structure_names_r2: list[str] | None = None,
    debug_dir: str = "debug",
    n_subset: int = 50,
    seed: int = 42,
    gt_peptides: list[str] | None = None,
    storey_pi0_val: float | None = None,
    ccs_tol_pct: float | None = None,
    single_round: bool = False,
    region_debug: dict | None = None,
) -> None:
    """
    Generate all debug figures and save them under ``debug_dir``.

    When ``single_round`` is True (rescore was run with ``single_round``, so no
    round-2 retrain occurred), the score figures relabel the round-2/final
    panels as "final (winners)" rather than "R2", and the round-2 feature
    importance panel is suppressed by the caller (``importances_r2=None``). The
    final score still lives in the ``*_score_r2`` column (it equals the R1 score
    on winners); only the labelling changes.

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

    # Build feature_mz → reweighted_q_value mapping once for co-feature ranking.
    # Used by plot_ion_image_colocalization to rank co-features by match quality.
    # Falls back to q_value when reweighted_q_value is NaN.
    _feat_al = features_df.reset_index(drop=True)
    _res_al = result_df.reset_index(drop=True)
    _winner_arr = (
        _res_al.get("is_tdc_winner", pd.Series(False, index=_res_al.index))
        .fillna(False).astype(bool).values
    )
    _rw_q_arr = pd.to_numeric(
        _res_al.get("reweighted_q_value", pd.Series(float("nan"), index=_res_al.index)),
        errors="coerce",
    ).values
    _q_fb_arr = pd.to_numeric(
        _res_al.get("q_value", pd.Series(float("nan"), index=_res_al.index)),
        errors="coerce",
    ).values
    feature_qvals: dict[float, float] = {}
    feature_peptides: dict[float, str] = {}
    if "feature_mz" in _feat_al.columns:
        _fmz_arr = _feat_al["feature_mz"].values
        _pep_arr = _feat_al["peptide"].values if "peptide" in _feat_al.columns else None
        for _wi in np.where(_winner_arr)[0]:
            _mz = float(_fmz_arr[_wi])
            _q = float(_rw_q_arr[_wi]) if np.isfinite(_rw_q_arr[_wi]) else float(_q_fb_arr[_wi])
            feature_qvals[_mz] = _q
            if _pep_arr is not None:
                feature_peptides[_mz] = str(_pep_arr[_wi])

    if ion_images is not None:
        try:
            # Build a full (unsampled) set of FDR ≤ 5% winners for ion images.
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
            _id_mask = _is_winner & (_rw_q <= 0.05)
            if not _id_mask.any() and "pep" in res_aligned.columns:
                _pep = pd.to_numeric(res_aligned["pep"], errors="coerce")
                _winner_idx = np.where(_is_winner.values)[0]
                _finite = np.isfinite(_pep.values[_winner_idx])
                if _finite.any():
                    _ranked = np.argsort(_pep.values[_winner_idx[_finite]])
                    _top5 = _winner_idx[np.where(_finite)[0][_ranked[:5]]]
                    _id_mask = pd.Series(False, index=res_aligned.index)
                    _id_mask.iloc[_top5] = True
            _id_idx = np.where(_id_mask.values)[0].tolist()
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
            id_subset["_total"] = len(feat_aligned)

            # Combine ID rows + sampled R1/L rows, then collapse to ONE row per
            # protein. Every ion-image colocalization figure already shows the
            # whole protein (precursor + all same-protein features + protein
            # mean), so per-peptide figures of the same protein only differ in
            # which feature is highlighted and the panel order — no new
            # information. Keep the best (lowest reweighted_q_value) peptide as
            # the protein's representative; targets and decoys are separate
            # proteins (DECOY_ prefix), so each yields its own figure.
            subset_for_images = pd.concat(
                [id_subset, subset[subset["_group"].isin(["R1", "L"])].copy()],
                ignore_index=True,
            )
            if "protein" in subset_for_images.columns and "reweighted_q_value" in subset_for_images.columns:
                subset_for_images = (
                    subset_for_images
                    .sort_values("reweighted_q_value", ascending=True, na_position="last")
                    .drop_duplicates(subset="protein", keep="first")
                    .reset_index(drop=True)
                )
            # Rank proteins by q-value, then cap the figure count: if more than
            # 300 proteins, subsample to 200 (reproducible) so the set stays
            # manageable.
            subset_for_images = subset_for_images.sort_values(
                "reweighted_q_value", ascending=True, na_position="last"
            ).reset_index(drop=True)
            _PROT_VIZ_CAP, _PROT_VIZ_TARGET = 300, 200
            _n_prot_total = len(subset_for_images)
            if _n_prot_total > _PROT_VIZ_CAP:
                _rng = np.random.default_rng(seed)
                _keep = sorted(
                    _rng.choice(_n_prot_total, size=_PROT_VIZ_TARGET, replace=False).tolist()
                )
                subset_for_images = subset_for_images.iloc[_keep].reset_index(drop=True)
                logger.info(
                    "Ion image colocalization: %d proteins exceed %d; subsampled to %d for visualization",
                    _n_prot_total, _PROT_VIZ_CAP, _PROT_VIZ_TARGET,
                )
            subset_for_images["_rank"] = np.arange(1, len(subset_for_images) + 1)
            _n_id_prot = int((subset_for_images["_group"] == "ID").sum()) if "_group" in subset_for_images.columns else 0
            logger.info(
                "Ion image colocalization: one figure per protein — %d proteins "
                "(%d with an ID at ≤5%% FDR)",
                len(subset_for_images), _n_id_prot,
            )
            plot_ion_image_colocalization(
                subset_for_images, features_df, ion_images, ion_image_mzs,
                out_dir=os.path.join(debug_dir, "ion_images"),
                feature_qvals=feature_qvals,
                feature_peptides=feature_peptides,
            )
            logger.info("Ion image colocalization figures saved to %s/ion_images/", debug_dir)
        except Exception as exc:
            logger.warning("Ion image colocalization figures failed: %s", exc)

        try:
            plot_ion_image_pearson_distribution(
                features_df, result_df, ion_images, ion_image_mzs,
                out_dir=debug_dir,
            )
            logger.info(
                "Ion image Pearson distribution saved to %s/ion_image_pearson_distribution.png",
                debug_dir,
            )
        except Exception as exc:
            logger.warning("Ion image Pearson distribution failed: %s", exc)

        try:
            plot_protein_spatial_coherence(
                features_df, result_df, ion_images, ion_image_mzs,
                out_dir=debug_dir,
            )
            logger.info(
                "Protein spatial coherence saved to %s/protein_spatial_coherence.png",
                debug_dir,
            )
        except Exception as exc:
            logger.warning("Protein spatial coherence failed: %s", exc)

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
            single_round=single_round,
        )
        logger.info("Feature distribution figures saved to %s/feature_distributions/", debug_dir)
    except Exception as exc:
        logger.warning("Feature distribution figures failed: %s", exc)

    try:
        plot_ccs_scatter(
            features_df, result_df,
            out_dir=debug_dir,
            gt_peptides=gt_peptides,
            ccs_tol_pct=ccs_tol_pct,
        )
        if "im2deep_observed_ccs" in features_df.columns:
            logger.info("CCS scatter saved to %s/ccs_scatter.png", debug_dir)
    except Exception as exc:
        logger.warning("CCS scatter failed: %s", exc)

    try:
        plot_isotope_ccs_consistency(
            features_df,
            output_path=os.path.join(debug_dir, "isotope_ccs_consistency.png"),
        )
        if "isotope_ccs_spread" in features_df.columns:
            logger.info(
                "Isotope-envelope CCS consistency saved to %s/isotope_ccs_consistency.png",
                debug_dir,
            )
    except Exception as exc:
        logger.warning("Isotope-envelope CCS consistency figure failed: %s", exc)

    try:
        plot_mz_mobility_intensity_scatter(features_df, out_dir=debug_dir)
        ccs_present = any(c in features_df.columns for c in ("im2deep_observed_ccs", "im2deep_predicted_ccs"))
        if ccs_present:
            logger.info(
                "m/z × mobility × intensity scatter saved to %s/mz_mobility_intensity_scatter.png",
                debug_dir,
            )
    except Exception as exc:
        logger.warning("m/z × mobility × intensity scatter failed: %s", exc)

    try:
        plot_ids_vs_fdr(result_df, out_dir=debug_dir, pi0=storey_pi0_val)
        logger.info("IDs vs FDR curve saved to %s/ids_vs_fdr.png", debug_dir)
    except Exception as exc:
        logger.warning("IDs vs FDR curve failed: %s", exc)

    try:
        plot_protein_colocalization_by_group(
            features_df, result_df,
            out_dir=debug_dir,
        )
        if any(col in features_df.columns for col, _ in _COLOC_COLS):
            logger.info(
                "Protein colocalization by group saved to %s/protein_colocalization_by_group.png",
                debug_dir,
            )
    except Exception as exc:
        logger.warning("Protein colocalization by group plot failed: %s", exc)

    if region_debug:
        try:
            plot_region_colocalization(
                features_df, region_debug,
                ion_image_shape=(ion_images.shape[1], ion_images.shape[2]) if ion_images is not None else None,
                out_dir=debug_dir,
            )
            logger.info(
                "Region colocalization viz saved to %s/region_segmentation.png + region_profiles.png",
                debug_dir,
            )
        except Exception as exc:
            logger.warning("Region colocalization plot failed: %s", exc)
        if ion_images is not None and ion_image_mzs is not None:
            try:
                plot_region_ion_images(
                    features_df, region_debug, ion_images, ion_image_mzs,
                    out_dir=os.path.join(debug_dir, "region_ion_images"),
                )
                logger.info(
                    "Region ion-image panels saved to %s/region_ion_images/",
                    debug_dir,
                )
            except Exception as exc:
                logger.warning("Region ion-image panels failed: %s", exc)

    try:
        plot_target_decoy_mz_distribution(
            features_df, result_df,
            out_dir=debug_dir,
        )
        logger.info("T/D m/z distribution saved to %s/target_decoy_mz_distribution.png", debug_dir)
    except Exception as exc:
        logger.warning("T/D m/z distribution plot failed: %s", exc)

    try:
        plot_candidate_competition(
            features_df, result_df,
            out_dir=debug_dir,
            ccs_tol_pct=ccs_tol_pct,
        )
        logger.info("Candidate competition saved to %s/candidate_competition.png", debug_dir)
    except Exception as exc:
        logger.warning("Candidate competition plot failed: %s", exc)

    try:
        plot_score_pp(
            features_df, result_df,
            out_dir=debug_dir,
            pi0=storey_pi0_val,
            single_round=single_round,
        )
        logger.info("Score PP plot saved to %s/score_pp_plot.png", debug_dir)
    except Exception as exc:
        logger.warning("Score PP plot failed: %s", exc)

    try:
        plot_pep_mixture(
            result_df,
            out_dir=debug_dir,
            model_name=model_name,
            pep_method=pep_method,
            single_round=single_round,
        )
        logger.info("PEP mixture plot saved to %s/pep_mixture.png", debug_dir)
    except Exception as exc:
        logger.warning("PEP mixture plot failed: %s", exc)

    try:
        plot_score_distributions(
            features_df, result_df,
            out_dir=debug_dir,
            single_round=single_round,
        )
        logger.info("Score distributions saved to %s/score_distributions.png", debug_dir)
    except Exception as exc:
        logger.warning("Score distributions failed: %s", exc)

    if importances_r1 is not None or importances_r2 is not None:
        imp_names = importance_names or feature_names or []
        try:
            plot_feature_importance(
                imp_names, importances_r1, importances_r2,
                out_dir=os.path.join(debug_dir, "feature_importance"),
                model_name=model_name,
                names_r2=importance_names_r2,
                structure_coefs_r1=structure_coefs_r1,
                structure_names_r1=structure_names_r1,
                structure_coefs_r2=structure_coefs_r2,
                structure_names_r2=structure_names_r2,
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
                            feature_qvals=feature_qvals,
                            feature_peptides=feature_peptides,
                        )
                    except Exception as exc:
                        logger.warning("GT ion image figures failed: %s", exc)
        except Exception as exc:
            logger.warning("GT debug figures failed: %s", exc)
