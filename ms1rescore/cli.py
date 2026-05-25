"""Command-line interface for ms1rescore."""

import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd

from ms1rescore import __version__

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MALDI data loading
# ---------------------------------------------------------------------------


def _read_feature_mzs(path: str) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """
    Load feature m/z values from a plain text file or a SCiLS Lab CSV export.

    Plain text: one m/z value per line, no header. Returns (mzs, None, None).
    SCiLS CSV: semicolon-delimited; lines starting with '#' are comments;
               the first non-comment line is a header (first column = 'm/z').
               If a 'CCS [Å²]' column is present it is returned as the second
               element. If an 'Intensity' column is present (e.g. the SCiLS
               'Intensity [Regions]' column) it is returned as the third element.
    """
    with open(path) as fh:
        lines = [ln.rstrip("\n") for ln in fh]

    data_lines = [ln for ln in lines if not ln.startswith("#") and ln.strip()]
    if not data_lines:
        raise ValueError(f"No data lines found in {path!r}")

    if ";" in data_lines[0]:
        # SCiLS-style: first non-comment line is header, rest are data rows.
        header = data_lines[0].split(";")
        rows = data_lines[1:]
        mzs = np.array([float(row.split(";")[0]) for row in rows if row.strip()], dtype=np.float64)

        def _extract_col(col_idx: int) -> np.ndarray:
            vals: list[float] = []
            for row in rows:
                if row.strip():
                    parts = row.split(";")
                    try:
                        vals.append(float(parts[col_idx]))
                    except (IndexError, ValueError):
                        vals.append(np.nan)
            return np.array(vals, dtype=np.float64)

        # Find CCS column
        ccs_col_idx = None
        for i, col in enumerate(header):
            if "CCS" in col and "Å" in col:  # Å = U+00C5
                ccs_col_idx = i
                break
        ccs: np.ndarray | None = _extract_col(ccs_col_idx) if ccs_col_idx is not None else None

        # Find intensity column — prefer 'Intensity [Regions]', accept any 'Intensity' column
        # that is not an interval-width or CCS column.
        intensity_col_idx = None
        for i, col in enumerate(header):
            col_lower = col.lower()
            if "intensity" in col_lower and "interval" not in col_lower and "width" not in col_lower:
                intensity_col_idx = i
                break
        intensities: np.ndarray | None = (
            _extract_col(intensity_col_idx) if intensity_col_idx is not None else None
        )
    else:
        mzs = np.array([float(ln) for ln in data_lines], dtype=np.float64)
        ccs = None
        intensities = None

    return mzs, ccs, intensities


def _load_maldi(
    npz_path: str | None,
    mzs_path: str | None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray | None, dict | None]:
    """
    Load MALDI feature data from disk.

    Parameters
    ----------
    npz_path
        NumPy NPZ file with ``"mzs"`` key (required) and optional ``"images"``
        key (3D ion image array).
    mzs_path
        Plain text file with one m/z value per line (no header), or a SCiLS
        Lab CSV export (semicolon-delimited) with optional ``CCS [Å²]`` column.

    Returns
    -------
    (maldi_mzs, ion_images, ion_image_mzs, ccs, extra_ion_images, maldi_intensities)
        ``ion_images``, ``ion_image_mzs``, and ``extra_ion_images`` are ``None``
        when loading from a plain text file or when the NPZ has no ``"images"`` key.
        ``ccs`` is ``None`` unless a SCiLS CSV with a ``CCS [Å²]`` column was
        supplied as ``mzs_path``.
        ``maldi_intensities`` is ``None`` unless a SCiLS CSV with an ``Intensity``
        column (e.g. ``Intensity [Regions]``) was supplied as ``mzs_path``.
    """
    if npz_path is not None:
        logger.info(f"Loading MALDI data from NPZ: {npz_path}")
        data = np.load(npz_path)
        if "mzs" not in data:
            logger.error(
                f"NPZ file {npz_path!r} has no 'mzs' key. Found: {list(data.keys())}"
            )
            sys.exit(1)
        mzs = data["mzs"]
        images = data["images"] if "images" in data else None
        image_mzs = mzs if images is not None else None
        _extra_keys = ("m1", "m2", "na", "k", "chca")
        extra_ion_images: dict | None = (
            {k: data[f"extra_{k}"] for k in _extra_keys if f"extra_{k}" in data}
            or None
        )
        logger.info(
            f"  {len(mzs)} MALDI features"
            + (f", ion images {images.shape}" if images is not None else ", no ion images")
            + (f", extra images: {list(extra_ion_images)}" if extra_ion_images else "")
        )
        return mzs, images, image_mzs, None, extra_ion_images, None

    logger.info(f"Loading MALDI m/z values from text file: {mzs_path}")
    try:
        mzs, ccs, intensities = _read_feature_mzs(mzs_path)
    except Exception as exc:
        logger.error(f"Could not read {mzs_path!r}: {exc}")
        sys.exit(1)
    if mzs.ndim != 1:
        logger.error(
            f"Expected a 1D array of m/z values in {mzs_path!r}, "
            f"got shape {mzs.shape}. Ensure one value per line."
        )
        sys.exit(1)
    _int_msg = ", intensities loaded from CSV" if intensities is not None else ""
    logger.info(f"  {len(mzs)} MALDI features, no ion images{_int_msg}")
    return mzs, None, None, ccs, None, intensities


# ---------------------------------------------------------------------------
# Digest parameter inference from LC-MS/MS identifications
# ---------------------------------------------------------------------------


def _infer_digest_params(
    lcms_ids,
    missed_cleavages_override: int | None,
    min_length_override: int | None,
    max_length_override: int | None,
) -> tuple[int, int, int]:
    """
    Infer missed_cleavages, min_length, and max_length from LC-MS/MS peptides.

    Override values take priority. Falls back to standard defaults
    (2 / 7 / 30) if the peptide table is empty.
    """
    seqs = lcms_ids.peptides["sequence"] if len(lcms_ids.peptides) > 0 else None

    if seqs is not None and len(seqs) > 0:
        inferred_min = int(seqs.str.len().min())
        inferred_max = int(seqs.str.len().max())

        def _count_mc(seq: str) -> int:
            return sum(1 for aa in seq[:-1] if aa in "KR")

        inferred_mc = int(seqs.apply(_count_mc).max())
    else:
        inferred_min, inferred_max, inferred_mc = 7, 30, 2

    min_length = (
        min_length_override if min_length_override is not None else inferred_min
    )
    max_length = (
        max_length_override if max_length_override is not None else inferred_max
    )
    missed_cleavages = (
        missed_cleavages_override
        if missed_cleavages_override is not None
        else inferred_mc
    )

    logger.info(
        f"Digest parameters: min_length={min_length}, max_length={max_length}, "
        f"missed_cleavages={missed_cleavages}"
        + (" (inferred from LC-MS/MS IDs)" if seqs is not None else " (defaults)")
    )
    return min_length, max_length, missed_cleavages


# ---------------------------------------------------------------------------
# Output writing
# ---------------------------------------------------------------------------


def _write_results(
    result,
    output_dir: str,
) -> None:
    """Write rescoring results to TSV files in ``output_dir``.

    Writes all candidates (targets and decoys) with q-value annotation.
    No hard filtering is applied — downstream consumers can filter by
    ``reweighted_q_value``, ``is_tdc_winner``, and ``is_decoy`` as needed.
    """
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "ms1rescore_matches.tsv")
    result.to_csv(out_path, sep="\t", index=False)
    n_winners = result.get("is_tdc_winner", result["is_decoy"].apply(lambda x: not x)).sum()
    logger.info(f"  Wrote {len(result)} candidates ({n_winners} TDC winners) → {out_path}")
    # Filter per-feature winners filtered by q-value
    if "is_tdc_winner" in result.columns and "reweighted_q_value" in result.columns:
        winners = result[result["is_tdc_winner"] & (result["reweighted_q_value"] <= 0.01)]
        peptides_out = os.path.join(output_dir, "ms1rescore_peptides.tsv")
        cols = [c for c in ["feature_idx", "feature_mz", "peptide", "protein", "reweighted_q_value"] if c in winners.columns]
        peptides = winners[cols].drop_duplicates().sort_values("reweighted_q_value")
        peptides.to_csv(peptides_out, sep="\t", index=False)
        logger.info(f"  Wrote {len(peptides)} peptide-level winners → {peptides_out}")

# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ms1rescore",
        description=(
            "Symmetric target-decoy rescoring for MALDI-MSI MS1 data. "
            "Matches MALDI features to an in-silico tryptic digest and "
            "rescores candidates using MALDI-intrinsic features, with "
            "optional LC-MS/MS evidence as a Bayesian prior."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--version", action="version", version=f"ms1rescore {__version__}"
    )

    parser.add_argument(
        "-c",
        "--config-file",
        default=None,
        metavar="PATH",
        help=(
            "JSON or TOML configuration file. Values here override defaults but "
            "are overridden by explicit CLI arguments."
        ),
    )

    # --- Required inputs ---
    req = parser.add_argument_group("required inputs")
    req.add_argument(
        "--fasta",
        "-f",
        required=False,
        default=None,
        metavar="PATH",
        help=(
            "Protein FASTA file (forward sequences only; decoys are generated). "
            "Required when --digest is specified. Ignored otherwise."
        ),
    )
    req.add_argument(
        "--digest",
        action="store_true",
        default=False,
        help=(
            "Digest the provided --fasta to create additional candidates beyond "
            "the LC-MS/MS identified peptides. Requires --fasta. Without this flag, "
            "only LC-MS/MS identified peptides (--lcms-peptides or --msf) are used "
            "as candidates and --fasta is not needed."
        ),
    )
    req.add_argument(
        "--extra-fasta",
        metavar="PATH",
        default=None,
        help=(
            "Additional FASTA file whose proteins are always included in the "
            "candidate database (e.g. contaminants, spike-ins). Works with both "
            "Strategy A and C. Peptides already present in the primary database "
            "are not duplicated."
        ),
    )
    req.add_argument(
        "--mzml",
        "-l",
        required=False,
        default=None,
        action="append",
        metavar="PATH",
        help=(
            "LC-MS/MS mzML file path. Repeat for multiple files: -l a.mzML -l b.mzML. "
            "Optional: omit when no LC-MS/MS mzML is available. XIC, spectral angle, "
            "and RT residual features will not be computed, but ID-derived prior "
            "features (from --lcms-peptides) are still used."
        ),
    )

    # --- MALDI input (mutually exclusive, one required) ---
    maldi_group = parser.add_argument_group("MALDI input (one required)")
    maldi_exc = maldi_group.add_mutually_exclusive_group(required=False)
    maldi_exc.add_argument(
        "--maldi-npz",
        metavar="PATH",
        help=(
            "NumPy NPZ file with a 'mzs' key (1D float64, m/z values) and an "
            "optional 'images' key (3D float, ion images of shape "
            "(n_features, height, width)). This is the standard format produced "
            "by the MALDI data extraction pipeline."
        ),
    )
    maldi_exc.add_argument(
        "--maldi-mzs",
        metavar="PATH",
        help=(
            "Plain text file with one MALDI feature m/z value per line "
            "(no header). Ion images will not be available."
        ),
    )
    maldi_exc.add_argument(
        "--maldi-raw",
        metavar="PATH",
        help=(
            "Bruker .d directory (TSF format, e.g. from timsTOF fleX). "
            "Features are detected automatically by binning centroided peaks "
            "across all pixels. Ion images and spatial features are computed "
            "and optionally saved (see --save-npz and --save-spatial)."
        ),
    )
    maldi_exc.add_argument(
        "--maldi-imzml",
        metavar="PATH",
        help=(
            "imzML file (.imzML + .ibd). SCiLS Lab-style interval-based "
            "feature extraction is performed automatically. Ion images and "
            "spatial features are reconstructed from the interval intensity "
            "matrix. Use --maldi-d instead when the raw Bruker .d directory "
            "is available to obtain full adduct/isotope extra images."
        ),
    )
    maldi_exc.add_argument(
        "--maldi-d",
        metavar="PATH",
        help=(
            "Bruker .d directory (preferred raw path; provides full ion image "
            "extraction including adduct and isotopologue extra images). Use "
            "instead of --maldi-imzml when the raw data is available. "
            "Functionally equivalent to --maldi-raw."
        ),
    )

    maldi_group.add_argument(
        "--feature-mzs",
        metavar="PATH",
        default=None,
        help=(
            "Pre-computed feature m/z values to use with --maldi-raw, skipping "
            "automatic feature detection (step 1) but still extracting ion images "
            "and spatial features. Accepts a plain text file (one m/z per line) or "
            "a SCiLS Lab CSV export (semicolon-delimited, '#' comment lines, first "
            "data column = m/z)."
        ),
    )

    # --- Raw extraction parameters ---
    raw_grp = parser.add_argument_group(
        "raw MALDI extraction (--maldi-raw only)",
        description=(
            "Parameters for feature detection and ion image extraction when "
            "starting from a raw Bruker .d directory."
        ),
    )
    raw_grp.add_argument(
        "--ppm-bin",
        type=float,
        default=None,
        metavar="FLOAT",
        help="Peak-binning tolerance for feature detection (ppm). Default: 5.0.",
    )
    raw_grp.add_argument(
        "--extraction-ppm",
        type=float,
        default=None,
        metavar="FLOAT",
        help=(
            "m/z window for raw ion image extraction (ppm). Controls which raw "
            "data points contribute to each ion image. Should be slightly wider "
            "than the instrument's typical peak width. Default: 25.0."
        ),
    )
    raw_grp.add_argument(
        "--matching-ppm",
        type=float,
        default=None,
        metavar="FLOAT",
        help=(
            "m/z window for candidate matching (ppm). Applied when linking "
            "peptide candidates to detected MALDI features. Default: 20.0."
        ),
    )
    raw_grp.add_argument(
        "--min-fraction",
        type=float,
        default=None,
        metavar="FLOAT",
        help=(
            "Minimum fraction of pixels a peak must be detected in to be "
            "kept as a feature. Default: 0.01 (1%%)."
        ),
    )
    raw_grp.add_argument(
        "--peak-prominence",
        type=float,
        default=None,
        metavar="FLOAT",
        help=(
            "Minimum peak prominence for SCiLS-style feature detection on profile "
            "data, as a fraction of the mean-spectrum maximum. Lower values detect "
            "more (weaker) features; higher values are more conservative. "
            "Default: 0.01. Typical range: 0.001–0.05."
        ),
    )
    raw_grp.add_argument(
        "--smoothing-window",
        type=int,
        default=None,
        metavar="INT",
        help=(
            "Savitzky-Golay smoothing window length (odd integer ≥ 3) applied to "
            "the mean spectrum before peak detection (profile mode only). "
            "Larger values smooth more but can shift peak apices. Default: 11."
        ),
    )
    raw_grp.add_argument(
        "--smoothing-polyorder",
        type=int,
        default=None,
        metavar="INT",
        help=(
            "Savitzky-Golay polynomial order for mean-spectrum smoothing "
            "(must be < --smoothing-window). Default: 2."
        ),
    )
    raw_grp.add_argument(
        "--interval-ppm-tolerance",
        type=float,
        default=None,
        metavar="FLOAT",
        help=(
            "Fallback interval half-width (ppm) used when no valley flanks a "
            "detected peak in the mean spectrum (profile mode only). Default: 5.0."
        ),
    )
    raw_grp.add_argument(
        "--min-interval-width-ppm",
        type=float,
        default=None,
        metavar="FLOAT",
        help=(
            "Minimum interval full-width (ppm). Intervals narrower than this are "
            "symmetrically expanded around the apex (profile mode only). Default: 1.0."
        ),
    )
    raw_grp.add_argument(
        "--normalize-rms",
        action="store_true",
        help=(
            "RMS-normalize each pixel spectrum before mean spectrum accumulation "
            "(profile mode only). Takes priority over the default TIC normalization. "
            "Matches the SCiLS Lab default normalization."
        ),
    )
    raw_grp.add_argument(
        "--baseline-correction",
        action="store_true",
        help=(
            "Apply rolling-minimum baseline subtraction to the mean spectrum before "
            "peak detection (profile mode only)."
        ),
    )
    raw_grp.add_argument(
        "--baseline-window-ppm",
        type=float,
        default=None,
        metavar="FLOAT",
        help="Half-width (ppm) of the rolling-minimum baseline window. Default: 500.0.",
    )
    raw_grp.add_argument(
        "--calibrant-mzs",
        type=float,
        nargs="*",
        default=None,
        metavar="MZ",
        help=(
            "Theoretical m/z values of internal calibrants (e.g. trypsin autolysis "
            "peaks). When provided, detected apices are used to fit a linear ppm "
            "correction and all intervals are recalibrated (profile mode only). "
            "Example: --calibrant-mzs 842.51 870.54 1045.56"
        ),
    )
    raw_grp.add_argument(
        "--calibrant-tol-ppm",
        type=float,
        default=None,
        metavar="FLOAT",
        help="Search window (ppm) for matching detected apices to calibrant m/z. Default: 200.0.",
    )
    raw_grp.add_argument(
        "--deisotope",
        action="store_true",
        help=(
            "Remove isotope satellite peaks using ms_deisotope after interval "
            "detection, retaining only monoisotopic peaks (profile mode only)."
        ),
    )
    raw_grp.add_argument(
        "--deisotope-error-ppm",
        type=float,
        default=None,
        metavar="FLOAT",
        help="PPM error tolerance for isotope envelope fitting. Default: 15.0.",
    )
    raw_grp.add_argument(
        "--deisotope-min-score",
        type=float,
        default=None,
        metavar="FLOAT",
        help="Minimum MSDeconV fit score to accept an isotope envelope. Default: 10.0.",
    )
    raw_grp.add_argument(
        "--deisotope-averagine",
        default=None,
        choices=["peptide", "glycopeptide", "glycan", "heparin"],
        metavar="MODEL",
        help="Averagine model for isotope envelope prediction. Default: peptide.",
    )
    raw_grp.add_argument(
        "--deisotope-scorer",
        default=None,
        choices=["MSDeconVFitter", "PenalizedMSDeconVFitter"],
        metavar="SCORER",
        help="ms_deisotope scoring function. Default: MSDeconVFitter.",
    )
    raw_grp.add_argument(
        "--deisotope-charge-range",
        type=int,
        nargs=2,
        default=None,
        metavar=("MIN", "MAX"),
        help="Charge range for deconvolution. Default: 1 1 (MALDI [M+H]+).",
    )
    raw_grp.add_argument(
        "--filter-mass-defect",
        action="store_true",
        help=(
            "Apply Senko-plot peptide corridor mass defect filter after interval "
            "detection (profile mode only). Removes lipids and matrix clusters."
        ),
    )
    raw_grp.add_argument(
        "--mass-defect-halfwidth",
        type=float,
        default=None,
        metavar="FLOAT",
        help=(
            "Half-width of the mass defect corridor. Default 0.5 passes all peaks "
            "(effectively disabled). Use 0.15–0.20 for a meaningful peptide filter."
        ),
    )
    raw_grp.add_argument(
        "--picking-height",
        type=float,
        default=None,
        metavar="FLOAT",
        help=(
            "Picking height for apex m/z centroid refinement (mMass-style). "
            "The apex is reported as the midpoint of the two interpolated "
            "crossing points at this fraction of the peak maximum. "
            "Default 0.75 matches the mMass 75%% setting. Use 0.0 to disable "
            "(raw smoothed-spectrum apex)."
        ),
    )
    raw_grp.add_argument(
        "--local-prominence-window-da",
        type=float,
        default=None,
        metavar="FLOAT",
        help=(
            "Half-width in Da of the sliding-window local maximum used as the "
            "reference for the peak height threshold. When > 0, the threshold "
            "for each peak is peak_prominence × local_max(±window) instead of "
            "peak_prominence × global_max. This reduces the effective threshold "
            "in low-signal m/z regions (e.g. >1600 Da) where genuine peptide "
            "peaks would otherwise fall below the global threshold. "
            "Default 0 (global max, disabled). Suggested value: 200."
        ),
    )
    raw_grp.add_argument(
        "--save-npz",
        metavar="PATH",
        help=(
            "Save extracted features and ion images as an NPZ file to this "
            "path, so subsequent runs can use --maldi-npz instead of "
            "re-extracting from raw data."
        ),
    )
    raw_grp.add_argument(
        "--save-spatial",
        metavar="PATH",
        help=(
            "Save the computed spatial features TSV to this path "
            "(columns: feature_mz, n_pixels_detected, fraction_detected, "
            "mean_intensity, intensity_p90, intensity_sum, "
            "spatial_autocorrelation, intensity_cv)."
        ),
    )

    # --- Candidate generation ---
    cand = parser.add_argument_group("candidate generation")
    cand.add_argument(
        "--ppm-tolerance",
        type=float,
        default=None,
        metavar="FLOAT",
        help="Mass tolerance for MALDI-to-database matching (ppm).",
    )
    cand.add_argument(
        "--missed-cleavages",
        type=int,
        default=None,
        metavar="INT",
        help=(
            "Maximum missed cleavages for in-silico digest. When "
            "--lcms-peptides is provided (Strategy C), inferred from the "
            "maximum number of internal K/R in identified sequences. "
            "Default for Strategy A (full FASTA): 2."
        ),
    )
    cand.add_argument(
        "--min-length",
        type=int,
        default=None,
        metavar="INT",
        help=(
            "Minimum peptide length. Inferred from LC-MS/MS IDs when "
            "--lcms-peptides is provided. Default for Strategy A: 7."
        ),
    )
    cand.add_argument(
        "--max-length",
        type=int,
        default=None,
        metavar="INT",
        help=(
            "Maximum peptide length. Inferred from LC-MS/MS IDs when "
            "--lcms-peptides is provided. Default for Strategy A: 30."
        ),
    )
    cand.add_argument(
        "--decoy-method",
        choices=("shuffle", "mz_shift", "balanced_shuffle"),
        default=None,
        help=(
            "Decoy generation strategy. 'shuffle' (default): K/R-preserving protein "
            "shuffle, standard target-decoy competition. 'mz_shift': observation-space "
            "decoys — each target peptide generates a shifted m/z query "
            "(delta_min..delta_max Da away) that is matched against MALDI features. "
            "'balanced_shuffle': iterative K/R-preserving protein shuffle with MALDI-match "
            "filtering — only shuffled peptides that match a MALDI feature are kept, "
            "subsampled to target_ratio * N_target. Ensures ~1:1 T:D even when the "
            "MALDI feature list is sparse."
        ),
    )
    cand.add_argument(
        "--mz-shift-delta-min",
        type=float,
        default=None,
        metavar="FLOAT",
        help="mz_shift only: minimum absolute m/z shift in Da (default 5.0).",
    )
    cand.add_argument(
        "--mz-shift-delta-max",
        type=float,
        default=None,
        metavar="FLOAT",
        help="mz_shift only: maximum absolute m/z shift in Da (default 20.0).",
    )
    cand.add_argument(
        "--mz-shift-snap-tolerance-ppm",
        type=float,
        default=None,
        metavar="FLOAT",
        help=(
            "mz_shift only: maximum ppm distance between the shifted query and the "
            "nearest MALDI feature for the snap to be accepted (default 50.0). "
            "Increase for sparse feature lists."
        ),
    )
    cand.add_argument(
        "--max-shuffle-rounds",
        type=int,
        default=None,
        metavar="INT",
        help=(
            "balanced_shuffle only: maximum number of shuffle rounds to attempt when "
            "collecting decoy candidates (default 50). Increase if T:D ratio is low "
            "due to sparse MALDI features."
        ),
    )
    cand.add_argument(
        "--decoy-target-ratio",
        type=float,
        default=None,
        metavar="FLOAT",
        help=(
            "balanced_shuffle only: target T:D candidate ratio (default 1.0). "
            "The function collects up to int(ratio * N_target) decoy candidates."
        ),
    )

    # --- Rescoring ---
    rescore_grp = parser.add_argument_group("rescoring")
    rescore_grp.add_argument(
        "--model",
        choices=("svm", "catboost", "lda", "qda"),
        default=None,
        help=(
            "Rescoring backend. 'lda': sklearn LDA with median imputation and "
            "standardization; no extra dependencies (default). 'qda': sklearn QDA "
            "(QuadraticDiscriminantAnalysis, reg_param=0.1); same structure as LDA. "
            "'svm': mokapot PercolatorModel trained on MALDI-intrinsic features. "
            "'catboost': semi-supervised CatBoostRanker with pseudo-label iteration "
            "(requires pip install ms1rescore[catboost])."
        ),
    )
    rescore_grp.add_argument(
        "--train-fdr",
        type=float,
        default=None,
        metavar="FLOAT",
        help=(
            "FDR threshold used for: (1) SVM model training; "
            "(2) best-feature seed initialization and pseudo-label iteration "
            "threshold in LDA/QDA backends (default 0.01)."
        ),
    )
    rescore_grp.add_argument(
        "--max-iter",
        type=int,
        default=None,
        metavar="INT",
        help="Maximum pseudo-label iterations for LDA/QDA backends (default 5).",
    )
    rescore_grp.add_argument(
        "--init-ppm-threshold",
        type=float,
        default=None,
        metavar="FLOAT",
        help="CatBoost only: ppm_error_abs threshold for the initial positive seed.",
    )
    rescore_grp.add_argument(
        "--init-isotope-threshold",
        type=float,
        default=None,
        metavar="FLOAT",
        help="CatBoost only: theo_isotope_cosine threshold for the initial positive seed.",
    )
    rescore_grp.add_argument(
        "--n-interaction-features",
        type=int,
        default=None,
        metavar="INT",
        help=(
            "LDA only: number of top-importance R1 features to expand with pairwise "
            "interaction terms (PolynomialFeatures degree=2) before R2 training. "
            "Set to 0 to disable. Default: 5."
        ),
    )
    rescore_grp.add_argument(
        "--only-main-features",
        action="store_true",
        help=(
            "Replace MALDI_INTRINSIC_FEATURES with a reduced set of ~19 "
            "non-collinear representative features (MAIN_FEATURES). "
            "Removes redundancy within collinear groups (ppm, isotope_theo, "
            "sequence_comp, etc.) before training. Disabled by default."
        ),
    )
    rescore_grp.add_argument(
        "--storey-pi0",
        action="store_true",
        help=(
            "Apply Storey pi0 correction to R2 TDC q-values. "
            "Estimates the null fraction among target winners from the R2 score "
            "distribution and multiplies raw q-values by pi0 (≤ 1.0). "
            "Adds storey_q_value and storey_reweighted_q_value columns to the "
            "output. Disabled by default."
        ),
    )
    rescore_grp.add_argument(
        "--lda-r2-median-filter",
        action="store_true",
        help=(
            "LDA only: before R2 training, drop features whose |R1 importance| is "
            "below the median across all R1 features. Disabled by default."
        ),
    )
    rescore_grp.add_argument(
        "--use-protein-level-feats",
        action="store_true",
        help=(
            "Include protein-level features (protein_n_features, protein_coverage, "
            "protein_rank, protein_best_ratio, protein_colocalization_*) in the "
            "rescoring model. These features aggregate signal across all candidates "
            "sharing a protein, which can break the TDC null model symmetry. "
            "Disabled by default."
        ),
    )
    rescore_grp.add_argument(
        "--n-debug",
        type=int,
        default=None,
        metavar="INT",
        help="Number of candidates to sample for per-candidate debug figures (default 50).",
    )
    rescore_grp.add_argument(
        "--debug-seed",
        type=int,
        default=None,
        metavar="INT",
        help="Random seed for debug candidate sampling (default 42).",
    )
    rescore_grp.add_argument(
        "--debug-gt",
        metavar="PATH",
        default=None,
        help=(
            "Path to a plain-text file with one ground-truth peptide sequence per line. "
            "If --verbose is set, debug figures are generated for each GT peptide "
            "found among the candidates (prefixed GT_). Peptides absent from the "
            "candidate set produce a 'not a candidate' placeholder figure. "
            "Ignored when --verbose is not set."
        ),
    )
    rescore_grp.add_argument(
        "--features-preset",
        choices=("all", "main"),
        default=None,
        help=(
            "'all' (default): use MALDI_INTRINSIC_FEATURES. 'main': use the "
            "reduced MAIN_FEATURES set. Overridden by --only-main-features."
        ),
    )
    rescore_grp.add_argument(
        "--features-exclude",
        nargs="*",
        default=None,
        metavar="FEATURE",
        help=(
            "Space-separated list of feature names to exclude from the ranker. "
            "Example: --features-exclude peptide_length n_proline. "
            "Useful for ablation studies without editing source code."
        ),
    )
    rescore_grp.add_argument(
        "--pseudo-label-max-iter",
        type=int,
        default=None,
        metavar="INT",
        help="Maximum pseudo-label iterations for LDA/QDA/CatBoost (default 5).",
    )
    rescore_grp.add_argument(
        "--pseudo-label-fdr",
        type=float,
        default=None,
        metavar="FLOAT",
        help="q-value threshold for pseudo-label expansion (default 0.10).",
    )
    rescore_grp.add_argument(
        "--r1-seed-percentile",
        type=float,
        default=None,
        metavar="FLOAT",
        help=(
            "Fallback top-N ppm percentile for initial positive seed when the "
            "ppm/isotope threshold yields zero positives (default 0.10 = top 10%%)."
        ),
    )
    rescore_grp.add_argument(
        "--catboost-iterations",
        type=int,
        default=None,
        metavar="INT",
        help="Number of boosting rounds for CatBoostRanker (default 500).",
    )
    rescore_grp.add_argument(
        "--mokapot-max-iter",
        type=int,
        default=None,
        metavar="INT",
        help="Maximum iterations for mokapot PercolatorModel (default 10).",
    )

    # --- Strategy C: LC-MS/MS-guided candidates ---
    strat_c = parser.add_argument_group(
        "Strategy C — LC-MS/MS-guided candidates (optional)",
        description=(
            "When --lcms-peptides is provided, candidates are generated by "
            "digesting only the identified proteins and adding directly "
            "identified peptides, rather than the full FASTA (Strategy A). "
            "Min/max length and missed cleavages are inferred from the "
            "identified sequences unless overridden."
        ),
    )
    strat_c.add_argument(
        "--lcms-peptides",
        metavar="PATH",
        help="Peptide-level LC-MS/MS results file. Activates Strategy C.",
    )
    strat_c.add_argument(
        "--lcms-proteins",
        metavar="PATH",
        help=(
            "Protein-level results file (optional). Proteins are derived "
            "from the peptide table when omitted."
        ),
    )
    strat_c.add_argument(
        "--lcms-psms",
        metavar="PATH",
        help="PSM-level file for RT and intensity aggregation (optional).",
    )
    strat_c.add_argument(
        "--lcms-id-format",
        choices=("percolator", "mzidentml", "psm_utils", "msf"),
        default=None,
        help=(
            "Format of the LC-MS/MS identification files. "
            "Use 'msf' to read directly from a ProteomeDiscoverer .msf file "
            "(the same file passed to --msf can be reused)."
        ),
    )
    strat_c.add_argument(
        "--psm-utils-reader",
        metavar="READER",
        default=None,
        help=(
            "psm_utils reader to use when --lcms-id-format is 'psm_utils'. "
            "Accepts a filetype key (e.g. 'maxquant', 'tsv', 'fragpipe') or "
            "a reader class name (e.g. 'MSMSReader', 'TSVReader'). "
            "When omitted, the reader is inferred from the file extension."
        ),
    )
    strat_c.add_argument(
        "--protein-fdr",
        type=float,
        default=None,
        metavar="FLOAT",
        help="Protein FDR threshold for Strategy C protein filtering.",
    )
    strat_c.add_argument(
        "--peptide-fdr",
        type=float,
        default=None,
        metavar="FLOAT",
        help="Peptide FDR threshold for Strategy C candidate inclusion.",
    )

    # --- Optional extras ---
    extras = parser.add_argument_group("optional extras")
    extras.add_argument(
        "--im2deep-calibration",
        choices=["linear", "spline", "finetune"],
        default=None,
        metavar="METHOD",
        help=(
            "CCS calibration strategy for IM2Deep predictions when observed CCS "
            "values are provided (via --feature-mzs or --maldi-imzml). "
            "'linear' applies a global additive shift (default); "
            "'spline' fits a piecewise spline for non-linear bias correction; "
            "'finetune' adapts the neural network weights to the observed MALDI CCS "
            "via transfer learning (requires ≥ 100 single-candidate calibration peptides)."
        ),
    )
    extras.add_argument(
        "--spatial-features",
        metavar="PATH",
        help=(
            "Pre-computed per-feature spatial statistics TSV "
            "(fraction_detected, intensity_cv, spatial_autocorrelation, etc.)."
        ),
    )
    extras.add_argument(
        "--msf",
        metavar="PATH",
        help="ProteomeDiscoverer .msf file for DeepLC retention-time finetuning.",
    )
    # --- Output ---
    out_grp = parser.add_argument_group("output")
    out_grp.add_argument(
        "--output-dir",
        "-o",
        default=None,
        metavar="PATH",
        help=(
            "Output directory. Written files: ms1rescore_psms.tsv (always), "
            "ms1rescore_peptides.tsv and ms1rescore_proteins.tsv (SVM only)."
        ),
    )
    out_grp.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable DEBUG-level logging (default: INFO).",
    )

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    import json as _json
    from argparse import Namespace as _Namespace
    from pathlib import Path as _Path
    from ms1rescore.config_parser import parse_configurations

    parser = build_parser()
    args = parser.parse_args()

    # --- Cascade config: defaults → config file → CLI args ---
    _config_sources = []
    if getattr(args, "config_file", None):
        _config_sources.append(args.config_file)

    # store_true flags default to False (not None) in argparse, so we can't
    # distinguish "not given" from "explicitly False". Convert False → None for
    # these attrs so none_overrides_value=False lets the config file win when
    # the flag is absent.
    _STORE_TRUE_ATTRS = frozenset({
        "verbose", "storey_pi0", "lda_r2_median_filter",
        "only_main_features", "use_protein_level_feats",
    })

    # Only pass top-level configurable params (not file paths or extraction params)
    # through the cascade; extraction params are handled separately below.
    _TOP_LEVEL_ATTRS = (
        "model", "train_fdr", "n_interaction_features", "storey_pi0",
        "lda_r2_median_filter", "only_main_features", "use_protein_level_feats",
        "n_debug", "debug_seed", "verbose", "output_dir",
        "ppm_tolerance", "missed_cleavages", "min_length", "max_length",
        "decoy_method", "mz_shift_delta_min", "mz_shift_delta_max",
        "mz_shift_snap_tolerance_ppm", "max_shuffle_rounds", "decoy_target_ratio",
        "protein_fdr", "peptide_fdr", "lcms_id_format",
        "im2deep_calibration", "init_ppm_threshold", "init_isotope_threshold",
        "features_preset", "features_exclude",
        "pseudo_label_max_iter", "pseudo_label_fdr", "r1_seed_percentile",
        "catboost_iterations", "mokapot_max_iter", "max_iter",
        # file paths
        "fasta", "extra_fasta", "mzml",
        "maldi_npz", "maldi_mzs", "maldi_raw", "maldi_imzml", "maldi_d",
        "feature_mzs", "save_npz", "save_spatial", "spatial_features",
        "lcms_peptides", "lcms_proteins", "lcms_psms", "msf",
        "debug_gt", "psm_utils_reader",
    )
    _top_ns = _Namespace(**{
        k: (True if getattr(args, k, False) else None)
           if k in _STORE_TRUE_ATTRS
           else getattr(args, k, None)
        for k in _TOP_LEVEL_ATTRS
    })
    _config_sources.append(_top_ns)
    _ms1cfg = parse_configurations(_config_sources)["ms1rescore"]

    # Extraction params: config defaults overridden by non-None CLI args.
    _extraction = dict(_ms1cfg.get("maldi_extraction", {}))
    _EXTRACTION_SCALAR_ATTRS = (
        "ppm_bin", "extraction_ppm", "matching_ppm", "min_fraction",
        "peak_prominence", "smoothing_window", "smoothing_polyorder",
        "interval_ppm_tolerance", "min_interval_width_ppm", "baseline_window_ppm",
        "calibrant_tol_ppm", "deisotope_error_ppm", "deisotope_min_score",
        "deisotope_averagine", "deisotope_scorer", "deisotope_charge_range",
        "mass_defect_halfwidth", "picking_height", "local_prominence_window_da",
        "calibrant_mzs",
    )
    for _attr in _EXTRACTION_SCALAR_ATTRS:
        _val = getattr(args, _attr, None)
        if _val is not None:
            _extraction[_attr] = _val
    for _bkey in ("normalize_rms", "baseline_correction", "deisotope", "filter_mass_defect"):
        if getattr(args, _bkey, False):
            _extraction[_bkey] = True

    # Convenience aliases from config
    output_dir = _ms1cfg["output_dir"]
    verbose = _ms1cfg["verbose"]

    # Write full merged config to output dir for reproducibility
    os.makedirs(output_dir, exist_ok=True)
    _Path(output_dir, ".full_config.json").write_text(
        _json.dumps({"ms1rescore": _ms1cfg}, indent=2, default=str)
    )

    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )
    # Third-party loggers that emit excessive DEBUG noise regardless of user intent.
    for _noisy in ("numba", "numba.core", "imzy", "koyo",
                   "matplotlib", "matplotlib.font_manager", "matplotlib.pyplot",
                   "matplotlib.backends", "PIL"):
        logging.getLogger(_noisy).setLevel(logging.WARNING)

    # --- Validate argument combinations ---
    if args.digest and not _ms1cfg.get("fasta"):
        parser.error("--digest requires --fasta.")

    # Validate mutually exclusive MALDI inputs (argparse enforces CLI; check config too)
    _maldi_input_keys = ("maldi_npz", "maldi_mzs", "maldi_raw", "maldi_imzml", "maldi_d")
    _active_maldi = [k for k in _maldi_input_keys if _ms1cfg.get(k)]
    if len(_active_maldi) > 1:
        parser.error(
            f"Only one MALDI input source may be specified; got: {_active_maldi}"
        )
    if len(_active_maldi) == 0:
        parser.error(
            "No MALDI input specified. Provide one of: --maldi-npz, --maldi-mzs, "
            "--maldi-raw, --maldi-imzml, --maldi-d (or set the equivalent key in "
            "the config file)."
        )

    lcms_id_source = _ms1cfg.get("lcms_peptides")
    if not args.digest and not lcms_id_source and not _ms1cfg.get("msf"):
        parser.error(
            "No candidate source: provide --lcms-peptides (or --msf) to use "
            "LC-MS/MS identified peptides, or add --digest with --fasta to "
            "perform an in-silico digest."
        )

    # --- Load MALDI data ---
    spatial_features = None
    maldi_envelopes = None
    _ccs_arr: np.ndarray | None = None
    _ccs_source_mzs: np.ndarray | None = None  # mzs aligned with _ccs_arr; may differ from maldi_mzs
    _mzs_intensities: np.ndarray | None = None  # per-feature intensity from SCiLS CSV
    _feature_mzs_intensities: np.ndarray | None = None  # set only in --maldi-raw/--maldi-d block

    _maldi_raw_path: str | None = _ms1cfg.get("maldi_raw") or _ms1cfg.get("maldi_d")
    _maldi_imzml_path: str | None = _ms1cfg.get("maldi_imzml")
    _feature_mzs_path: str | None = _ms1cfg.get("feature_mzs")
    if _maldi_raw_path:
        from ms1rescore.maldi_extraction import extract_maldi_data

        logger.info(
            "MALDI features detected from raw data (detect_features). "
            "LC-MS/MS identifications will be used for candidate generation and "
            "prior features only, not for feature selection."
        )
        precomputed_mzs = None
        if _feature_mzs_path:
            logger.info(f"Loading pre-computed feature m/z values from {_feature_mzs_path}")
            try:
                precomputed_mzs, _ccs_arr, _feature_mzs_intensities = _read_feature_mzs(_feature_mzs_path)
                _ccs_source_mzs = precomputed_mzs
            except Exception as exc:
                logger.error(f"Could not read --feature-mzs {_feature_mzs_path!r}: {exc}")
                sys.exit(1)
            logger.info(f"  {len(precomputed_mzs)} features loaded (skipping detection)")

        logger.info(f"Extracting MALDI features from raw data: {_maldi_raw_path}")
        maldi_mzs, ion_images, extra_ion_images, spatial_features, maldi_envelopes = extract_maldi_data(
            _maldi_raw_path,
            feature_mzs=precomputed_mzs,
            ppm_bin=_extraction["ppm_bin"],
            extraction_ppm=_extraction["extraction_ppm"],
            matching_ppm=_extraction["matching_ppm"],
            min_fraction=_extraction["min_fraction"],
            peak_prominence=_extraction["peak_prominence"],
            smoothing_window=_extraction["smoothing_window"],
            smoothing_polyorder=_extraction["smoothing_polyorder"],
            ppm_tolerance=_extraction["interval_ppm_tolerance"],
            min_interval_width_ppm=_extraction["min_interval_width_ppm"],
            normalize_rms=_extraction["normalize_rms"],
            baseline_correction=_extraction["baseline_correction"],
            baseline_window_ppm=_extraction["baseline_window_ppm"],
            calibrant_mzs=_extraction["calibrant_mzs"],
            calibrant_tol_ppm=_extraction["calibrant_tol_ppm"],
            deisotope=_extraction["deisotope"],
            deisotope_averagine=_extraction["deisotope_averagine"],
            deisotope_scorer=_extraction["deisotope_scorer"],
            deisotope_min_score=_extraction["deisotope_min_score"],
            deisotope_charge_range=tuple(_extraction["deisotope_charge_range"]),
            deisotope_error_ppm=_extraction["deisotope_error_ppm"],
            filter_mass_defect=_extraction["filter_mass_defect"],
            mass_defect_halfwidth=_extraction["mass_defect_halfwidth"],
            picking_height=_extraction["picking_height"],
            local_prominence_window_da=_extraction["local_prominence_window_da"],
            output_npz=_ms1cfg.get("save_npz"),
            output_spatial_tsv=_ms1cfg.get("save_spatial"),
            output_dir=output_dir,
            verbose=verbose,
        )
        ion_image_mzs = maldi_mzs if ion_images is not None else None
        logger.info(
            f"  {len(maldi_mzs)} features extracted"
            + (f", ion image shape: {ion_images.shape[1:]}" if ion_images is not None else "")
        )
    elif _maldi_imzml_path:
        from ms1rescore.maldi_imzml import (
            SCiLSConfig, extract_scils_features,
            reconstruct_ion_images_from_intervals, build_envelopes_from_intervals,
        )

        logger.info(
            "MALDI features extracted from imzML data (SCiLS Lab-style interval extraction). "
            "Ion images reconstructed from interval intensity matrix."
        )
        logger.info(f"Extracting MALDI features from imzML: {_maldi_imzml_path}")
        cfg = SCiLSConfig(
            min_pixel_fraction=_extraction["min_fraction"],
            peak_prominence=_extraction["peak_prominence"],
            smoothing_window=_extraction["smoothing_window"],
            smoothing_polyorder=_extraction["smoothing_polyorder"],
            ppm_tolerance=_extraction["interval_ppm_tolerance"],
            min_interval_width_ppm=_extraction["min_interval_width_ppm"],
            normalize_rms=_extraction["normalize_rms"],
            baseline_correction=_extraction["baseline_correction"],
            baseline_window_ppm=_extraction["baseline_window_ppm"],
            calibrant_mzs=_extraction.get("calibrant_mzs") or [],
            calibrant_tol_ppm=_extraction["calibrant_tol_ppm"],
            deisotope=_extraction["deisotope"],
            deisotope_averagine=_extraction["deisotope_averagine"],
            deisotope_scorer=_extraction["deisotope_scorer"],
            deisotope_min_score=_extraction["deisotope_min_score"],
            deisotope_charge_range=tuple(_extraction["deisotope_charge_range"]),
            deisotope_error_ppm=_extraction["deisotope_error_ppm"],
            filter_mass_defect=_extraction["filter_mass_defect"],
            mass_defect_halfwidth=_extraction["mass_defect_halfwidth"],
            picking_height=_extraction["picking_height"],
            local_prominence_window_da=_extraction["local_prominence_window_da"],
        )
        intervals, intensity_matrix, pixel_coords, mean_1_over_k0 = extract_scils_features(
            _maldi_imzml_path,
            config=cfg,
            output_dir=output_dir,
            visualize=False,
        )
        maldi_mzs = np.array([apex for _, _, apex in intervals])

        # Reconstruct 3D ion images from the flat interval intensity matrix
        ion_images = reconstruct_ion_images_from_intervals(
            intensity_matrix, pixel_coords, len(intervals)
        )
        ion_image_mzs = maldi_mzs if len(intervals) > 0 else None
        extra_ion_images = None  # adduct images unavailable from pre-integrated intervals

        # Compute spatial features from reconstructed ion images
        if len(intervals) > 0:
            from ms1rescore.maldi_extraction import compute_spatial_features as _csf
            spatial_features = _csf(ion_images, maldi_mzs, len(pixel_coords))

        # Build approximate isotope envelopes from interval mean intensities
        maldi_envelopes = build_envelopes_from_intervals(intervals, intensity_matrix)

        logger.info(
            f"  {len(maldi_mzs)} intervals extracted"
            + (f", ion images {ion_images.shape[1:]}" if len(intervals) > 0 else "")
        )
        if mean_1_over_k0 is not None and len(mean_1_over_k0) == len(maldi_mzs):
            from ms1rescore.maldi_imzml import one_over_k0_to_ccs
            _ccs_arr = one_over_k0_to_ccs(mean_1_over_k0, maldi_mzs)
            logger.info("  Converted mean 1/K0 to CCS using Mason-Schamp equation")
    else:
        maldi_mzs, ion_images, ion_image_mzs, _ccs_arr, extra_ion_images, _mzs_intensities = _load_maldi(
            _ms1cfg.get("maldi_npz"), _ms1cfg.get("maldi_mzs")
        )

    # --- Optional spatial features (explicit file overrides extracted ones) ---
    if _ms1cfg.get("spatial_features"):
        logger.info(f"Loading spatial features from {_ms1cfg['spatial_features']}")
        spatial_features = pd.read_csv(_ms1cfg["spatial_features"], sep="\t")

    # --- Resolve Strategy C source ---
    # If --lcms-peptides is not given but --msf is, use the MSF for Strategy C.
    lcms_peptides_path = _ms1cfg.get("lcms_peptides")
    lcms_id_format = _ms1cfg["lcms_id_format"]
    if lcms_peptides_path is None and _ms1cfg.get("msf") is not None:
        lcms_peptides_path = _ms1cfg.get("msf")
        lcms_id_format = "msf"
        logger.info(
            f"No --lcms-peptides provided; using --msf ({_ms1cfg['msf']}) "
            f"as Strategy C ID source (format='msf')."
        )

    # --- Resolve digest parameters ---
    if lcms_peptides_path:
        from ms1rescore.lcms_ids import parse_lcms_ids

        logger.info("Parsing LC-MS/MS identifications for Strategy C...")
        lcms_ids = parse_lcms_ids(
            proteins_path=_ms1cfg.get("lcms_proteins"),
            peptides_path=lcms_peptides_path,
            psms_path=_ms1cfg.get("lcms_psms"),
            protein_fdr=_ms1cfg["protein_fdr"],
            peptide_fdr=_ms1cfg["peptide_fdr"],
            format=lcms_id_format,
            psm_utils_reader=_ms1cfg.get("psm_utils_reader"),
        )
        if verbose:
            logger.debug("Writing parsed LC-MS/MS IDs to debug_lcms_ids.tsv")
            lcms_ids.peptides.to_csv(
                f"{output_dir}/4_debug_lcms_ids.tsv", sep="\t", index=False
            )
        min_length, max_length, missed_cleavages = _infer_digest_params(
            lcms_ids,
            missed_cleavages_override=_ms1cfg["missed_cleavages"],
            min_length_override=_ms1cfg["min_length"],
            max_length_override=_ms1cfg["max_length"],
        )
    else:
        min_length = _ms1cfg["min_length"]
        max_length = _ms1cfg["max_length"]
        missed_cleavages = _ms1cfg["missed_cleavages"]

    logger.info(
        f"Parameters extracted: min_length={min_length}, "
        f"max_length={max_length}, missed_cleavages={missed_cleavages}"
    )

    # --- Build observed CCS dict from loaded CCS array ---
    # _ccs_source_mzs is set when the CCS array comes from a file whose m/z list
    # may differ from maldi_mzs (e.g. --maldi-raw + --feature-mzs: extract_maldi_data
    # can drop zero-signal features, making maldi_mzs shorter than precomputed_mzs).
    # Build a m/z→CCS lookup and re-index into maldi_mzs to handle this case.
    observed_ccs: dict | None = None
    if _ccs_arr is not None:
        _ref_mzs = _ccs_source_mzs if _ccs_source_mzs is not None else maldi_mzs
        if _ref_mzs is not None and len(_ccs_arr) == len(_ref_mzs):
            _mz_to_ccs = {
                float(mz): float(v)
                for mz, v in zip(_ref_mzs, _ccs_arr)
                if np.isfinite(v)
            }
            observed_ccs = {
                idx: _mz_to_ccs[float(mz)]
                for idx, mz in enumerate(maldi_mzs)
                if float(mz) in _mz_to_ccs
            }
            if observed_ccs:
                logger.info(
                    f"  CCS values loaded for {len(observed_ccs)}/{len(maldi_mzs)} features"
                )
            else:
                logger.warning(
                    "  CCS array found but no m/z values matched maldi_mzs; "
                    "CCS features will be skipped"
                )

    # --- Align --feature-mzs intensities to maldi_mzs ---
    # When --maldi-raw + --feature-mzs is used, maldi_mzs may be shorter than
    # precomputed_mzs (extract_maldi_data can drop zero-signal features).
    # Re-index using a m/z lookup, the same way CCS is handled above.
    if _feature_mzs_intensities is not None and _ccs_source_mzs is not None:
        _mz_to_intensity = {
            float(mz): float(v)
            for mz, v in zip(_ccs_source_mzs, _feature_mzs_intensities)
            if np.isfinite(v)
        }
        _aligned = np.array(
            [_mz_to_intensity.get(float(mz), np.nan) for mz in maldi_mzs],
            dtype=np.float64,
        )
        n_matched = int(np.isfinite(_aligned).sum())
        if n_matched > 0:
            _mzs_intensities = _aligned
            logger.info(
                f"  SCiLS intensities aligned for {n_matched}/{len(maldi_mzs)} features"
            )
        else:
            logger.warning(
                "  --feature-mzs intensity column found but no m/z values matched "
                "maldi_mzs; intensity feature will fall back to raw extraction"
            )

    # --- Load GT peptides (only relevant when debug is enabled) ---
    gt_peptides: list[str] | None = None
    _debug_gt_path = _ms1cfg.get("debug_gt")
    if verbose and _debug_gt_path:
        try:
            with open(_debug_gt_path) as _fh:
                gt_peptides = [line.strip() for line in _fh if line.strip()]
            logger.info("GT peptides loaded: %d from %s", len(gt_peptides), _debug_gt_path)
        except Exception as _exc:
            logger.warning("Could not read --debug-gt file %s: %s", _debug_gt_path, _exc)

    # --- Run pipeline ---
    from ms1rescore.pipeline import rescore

    logger.info("Starting ms1rescore pipeline...")
    _, result_df, _ = rescore(
        fasta_path=_ms1cfg.get("fasta"),
        maldi_mzs=maldi_mzs,
        mzml_paths=_ms1cfg.get("mzml") or [],
        ion_images=ion_images,
        ion_image_mzs=ion_image_mzs,
        extra_ion_images=extra_ion_images,
        spatial_features=spatial_features,
        maldi_envelopes=maldi_envelopes,
        msf_path=args.msf,
        ppm_tolerance=_ms1cfg["ppm_tolerance"],
        train_fdr=_ms1cfg["train_fdr"],
        missed_cleavages=missed_cleavages,
        min_length=min_length,
        max_length=max_length,
        model=_ms1cfg["model"],
        init_ppm_threshold=_ms1cfg["init_ppm_threshold"],
        init_isotope_threshold=_ms1cfg["init_isotope_threshold"],
        n_interaction_features=_ms1cfg["n_interaction_features"],
        lda_r2_median_filter=_ms1cfg["lda_r2_median_filter"],
        storey_pi0=_ms1cfg["storey_pi0"],
        only_main_features=_ms1cfg["only_main_features"],
        lcms_proteins_path=_ms1cfg.get("lcms_proteins"),
        lcms_peptides_path=lcms_peptides_path,
        lcms_psms_path=_ms1cfg.get("lcms_psms"),
        lcms_id_format=lcms_id_format,
        psm_utils_reader=_ms1cfg.get("psm_utils_reader"),
        protein_fdr=_ms1cfg["protein_fdr"],
        peptide_fdr=_ms1cfg["peptide_fdr"],
        extra_fasta_path=_ms1cfg.get("extra_fasta"),
        use_protein_level_features=_ms1cfg["use_protein_level_feats"],
        verbose=verbose,
        output_dir=output_dir,
        debug_dir=os.path.join(output_dir, "debug") if verbose else None,
        n_debug=_ms1cfg["n_debug"],
        debug_seed=_ms1cfg["debug_seed"],
        observed_ccs_per_feature=observed_ccs,
        im2deep_calibration=_ms1cfg["im2deep_calibration"],
        im2deep_kwargs=_ms1cfg.get("im2deep"),
        digest=args.digest,
        gt_peptides=gt_peptides,
        maldi_intensities=_mzs_intensities,
        decoy_method=_ms1cfg["decoy_method"],
        mz_shift_delta_min=_ms1cfg["mz_shift_delta_min"],
        mz_shift_delta_max=_ms1cfg["mz_shift_delta_max"],
        mz_shift_snap_tolerance_ppm=_ms1cfg["mz_shift_snap_tolerance_ppm"],
        max_shuffle_rounds=_ms1cfg["max_shuffle_rounds"],
        target_ratio=_ms1cfg["decoy_target_ratio"],
        features_preset=_ms1cfg["features_preset"],
        features_exclude=_ms1cfg["features_exclude"],
        pseudo_label_max_iter=_ms1cfg["pseudo_label_max_iter"],
        pseudo_label_fdr=_ms1cfg["pseudo_label_fdr"],
        r1_seed_percentile=_ms1cfg["r1_seed_percentile"],
        catboost_iterations=_ms1cfg["catboost_iterations"],
        mokapot_max_iter=_ms1cfg["mokapot_max_iter"],
        max_iter=_ms1cfg["max_iter"],
    )

    # --- Write results ---
    logger.info(f"Writing results to {os.path.abspath(output_dir)}")
    if verbose:
        logger.debug("Writing complete result DataFrame to debug_result_df.tsv")
        result_df.to_csv(f"{output_dir}/5_debug_result_df.tsv", sep="\t", index=False)
    _write_results(result_df, output_dir)
    if gt_peptides and "is_tdc_winner" in result_df.columns:
        gt_set = set(gt_peptides)
        winners = result_df[result_df["is_tdc_winner"] & ~result_df["is_decoy"].astype(bool)]
        n_gt_winners = winners.drop_duplicates(subset=["peptide"])["peptide"].isin(gt_set).sum()
        logger.info(
            "%d/%d GT peptides are round-2 (feature-level) winners.",
            n_gt_winners, len(gt_set),
        )
        winners_fdr = winners[winners['q_value'] < 0.01]
        n_gt_winners_fdr = winners_fdr.drop_duplicates(subset=["peptide"])["peptide"].isin(gt_set).sum()
        logger.info(
            "%d/%d GT peptides are round-2 winners at 1%% FDR.",
            n_gt_winners_fdr, len(gt_set),
        )
        winners_fdr_5 = winners[winners['q_value'] < 0.05]
        n_gt_winners_fdr_5 = winners_fdr_5.drop_duplicates(subset=["peptide"])["peptide"].isin(gt_set).sum()
        logger.debug(
            "%d/%d GT peptides are round-2 winners at 5%% FDR.",
            n_gt_winners_fdr_5, len(gt_set),
        )
    logger.info("Done.")


if __name__ == "__main__":
    main()
