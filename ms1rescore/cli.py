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


def _load_maldi(
    npz_path: str | None,
    mzs_path: str | None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """
    Load MALDI feature data from disk.

    Parameters
    ----------
    npz_path
        NumPy NPZ file with ``"mzs"`` key (required) and optional ``"images"``
        key (3D ion image array).
    mzs_path
        Plain text file with one m/z value per line (no header).

    Returns
    -------
    (maldi_mzs, ion_images, ion_image_mzs)
        ``ion_images`` and ``ion_image_mzs`` are ``None`` when loading from a
        plain text file or when the NPZ has no ``"images"`` key.
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
        logger.info(
            f"  {len(mzs)} MALDI features"
            + (
                f", ion images {images.shape}"
                if images is not None
                else ", no ion images"
            )
        )
        return mzs, images, image_mzs

    logger.info(f"Loading MALDI m/z values from text file: {mzs_path}")
    try:
        mzs = np.loadtxt(mzs_path)
    except Exception as exc:
        logger.error(f"Could not read {mzs_path!r}: {exc}")
        sys.exit(1)
    if mzs.ndim != 1:
        logger.error(
            f"Expected a 1D array of m/z values in {mzs_path!r}, "
            f"got shape {mzs.shape}. Ensure one value per line."
        )
        sys.exit(1)
    logger.info(f"  {len(mzs)} MALDI features, no ion images")
    return mzs, None, None


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
    ``q_value``, ``is_tdc_winner``, and ``is_decoy`` as needed.
    """
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "ms1rescore_psms.tsv")
    result.to_csv(out_path, sep="\t", index=False)
    n_winners = result.get("is_tdc_winner", result["is_decoy"].apply(lambda x: not x)).sum()
    logger.info(f"  Wrote {len(result)} candidates ({n_winners} TDC winners) → {out_path}")
    # Filter per-feature winners filtered by q-value
    if "is_tdc_winner" in result.columns and "q_value" in result.columns:
        winners = result[result["is_tdc_winner"] & (result["q_value"] <= 0.01)]
        peptides_out = os.path.join(output_dir, "ms1rescore_peptides.tsv")
        peptides = winners[["feature_idx", "feature_mz", "peptide", "q_value"]].drop_duplicates().sort_values("q_value")
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

    # --- Required inputs ---
    req = parser.add_argument_group("required inputs")
    req.add_argument(
        "--fasta",
        "-f",
        required=True,
        metavar="PATH",
        help="Protein FASTA file (forward sequences only; decoys are generated).",
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
        required=True,
        action="append",
        metavar="PATH",
        help="LC-MS/MS mzML file path. Repeat for multiple files: -l a.mzML -l b.mzML",
    )

    # --- MALDI input (mutually exclusive, one required) ---
    maldi_group = parser.add_argument_group("MALDI input (one required)")
    maldi_exc = maldi_group.add_mutually_exclusive_group(required=True)
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
            "spatial features are computed and optionally saved."
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
        default=5.0,
        metavar="FLOAT",
        help="Peak-binning tolerance for feature detection (ppm). Default: 5.0.",
    )
    raw_grp.add_argument(
        "--extraction-ppm",
        type=float,
        default=25.0,
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
        default=20.0,
        metavar="FLOAT",
        help=(
            "m/z window for candidate matching (ppm). Applied when linking "
            "peptide candidates to detected MALDI features. Default: 20.0."
        ),
    )
    raw_grp.add_argument(
        "--min-fraction",
        type=float,
        default=0.01,
        metavar="FLOAT",
        help=(
            "Minimum fraction of pixels a peak must be detected in to be "
            "kept as a feature. Default: 0.01 (1%%)."
        ),
    )
    raw_grp.add_argument(
        "--peak-prominence",
        type=float,
        default=0.01,
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
        default=11,
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
        default=2,
        metavar="INT",
        help=(
            "Savitzky-Golay polynomial order for mean-spectrum smoothing "
            "(must be < --smoothing-window). Default: 2."
        ),
    )
    raw_grp.add_argument(
        "--interval-ppm-tolerance",
        type=float,
        default=10.0,
        metavar="FLOAT",
        help=(
            "Fallback interval half-width (ppm) used when no valley flanks a "
            "detected peak in the mean spectrum (profile mode only). Default: 5.0."
        ),
    )
    raw_grp.add_argument(
        "--min-interval-width-ppm",
        type=float,
        default=2.0,
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
        default=500.0,
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
        default=200.0,
        metavar="FLOAT",
        help="Search window (ppm) for matching detected apices to calibrant m/z. Default: 200.0.",
    )
    raw_grp.add_argument(
        "--deisotope",
        action="store_true",
        help=(
            "Remove M+1 and M+2 isotope peaks (z=1 spacing, 1.003355 Da) after "
            "interval detection, retaining only monoisotopic peaks (profile mode only)."
        ),
    )
    raw_grp.add_argument(
        "--deisotope-tol-da",
        type=float,
        default=0.15,
        metavar="FLOAT",
        help="Tolerance (Da) around the 1.003355 Da isotope spacing. Default: 0.15.",
    )
    raw_grp.add_argument(
        "--deisotope-min-fold",
        type=float,
        default=0.67,
        metavar="FLOAT",
        help=(
            "Averagine-relative intensity guard for deisotoping. A peak is removed as "
            "M+1 only if M0 >= M+1 × (1/lambda) × FLOAT, where lambda = 0.000509 × M0 "
            "(averagine Poisson parameter) and 1/lambda is the expected M0/M+1 ratio. "
            "Default 0.67: M0 must be at least 67%% of the averagine-expected ratio "
            "(e.g. >= 1.35× M+1 for a 975 Da compound). This prevents false-positive "
            "deisotoping of two unrelated peptides ~1 Da apart."
        ),
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
        default=0.5,
        metavar="FLOAT",
        help=(
            "Half-width of the mass defect corridor. Default 0.5 passes all peaks "
            "(effectively disabled). Use 0.15–0.20 for a meaningful peptide filter."
        ),
    )
    raw_grp.add_argument(
        "--picking-height",
        type=float,
        default=0.75,
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
        default=0.0,
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
        default=20.0,
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

    # --- Rescoring ---
    rescore_grp = parser.add_argument_group("rescoring")
    rescore_grp.add_argument(
        "--model",
        choices=("svm", "catboost", "generative"),
        default="svm",
        help=(
            "Rescoring backend. 'svm': mokapot PercolatorModel trained on "
            "MALDI-intrinsic features (default). 'catboost': semi-supervised "
            "CatBoostRanker with pseudo-label iteration (requires "
            "pip install ms1rescore[catboost]). 'generative': probabilistic "
            "generative scorer, no training required."
        ),
    )
    rescore_grp.add_argument(
        "--train-fdr",
        type=float,
        default=0.01,
        metavar="FLOAT",
        help="FDR threshold for SVM model training.",
    )
    rescore_grp.add_argument(
        "--init-ppm-threshold",
        type=float,
        default=2.0,
        metavar="FLOAT",
        help="CatBoost only: ppm_error_abs threshold for the initial positive seed.",
    )
    rescore_grp.add_argument(
        "--init-isotope-threshold",
        type=float,
        default=0.7,
        metavar="FLOAT",
        help="CatBoost only: theo_isotope_cosine threshold for the initial positive seed.",
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
        default="percolator",
        help=(
            "Format of the LC-MS/MS identification files. "
            "Use 'msf' to read directly from a ProteomeDiscoverer .msf file "
            "(the same file passed to --msf can be reused)."
        ),
    )
    strat_c.add_argument(
        "--protein-fdr",
        type=float,
        default=0.01,
        metavar="FLOAT",
        help="Protein FDR threshold for Strategy C protein filtering.",
    )
    strat_c.add_argument(
        "--peptide-fdr",
        type=float,
        default=0.01,
        metavar="FLOAT",
        help="Peptide FDR threshold for Strategy C candidate inclusion.",
    )

    # --- Optional extras ---
    extras = parser.add_argument_group("optional extras")
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
    extras.add_argument(
        "--cache-dir",
        metavar="PATH",
        help=(
            "Directory for caching intermediate results "
            "(MS2PIP predictions, DeepLC predictions, LC-MS/MS data)."
        ),
    )

    # --- Output ---
    out_grp = parser.add_argument_group("output")
    out_grp.add_argument(
        "--output-dir",
        "-o",
        default=".",
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
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )
    # numba JIT compilation emits thousands of DEBUG lines via its own loggers;
    # keep them silent unless the user explicitly wants them.
    for _noisy in ("numba", "numba.core", "imzy", "koyo"):
        logging.getLogger(_noisy).setLevel(logging.WARNING)

    # --- Load MALDI data ---
    spatial_features = None

    if args.maldi_raw:
        from ms1rescore.maldi_extraction import extract_maldi_data

        logger.info(
            "MALDI features detected from raw data (detect_features). "
            "LC-MS/MS identifications will be used for candidate generation and "
            "prior features only, not for feature selection."
        )
        logger.info(f"Extracting MALDI features from raw data: {args.maldi_raw}")
        maldi_mzs, ion_images, spatial_features = extract_maldi_data(
            args.maldi_raw,
            ppm_bin=args.ppm_bin,
            extraction_ppm=args.extraction_ppm,
            matching_ppm=args.matching_ppm,
            min_fraction=args.min_fraction,
            peak_prominence=args.peak_prominence,
            smoothing_window=args.smoothing_window,
            smoothing_polyorder=args.smoothing_polyorder,
            ppm_tolerance=args.interval_ppm_tolerance,
            min_interval_width_ppm=args.min_interval_width_ppm,
            normalize_rms=args.normalize_rms,
            baseline_correction=args.baseline_correction,
            baseline_window_ppm=args.baseline_window_ppm,
            calibrant_mzs=args.calibrant_mzs,
            calibrant_tol_ppm=args.calibrant_tol_ppm,
            deisotope=args.deisotope,
            deisotope_tol_da=args.deisotope_tol_da,
            deisotope_min_fold=args.deisotope_min_fold,
            filter_mass_defect=args.filter_mass_defect,
            mass_defect_halfwidth=args.mass_defect_halfwidth,
            picking_height=args.picking_height,
            local_prominence_window_da=args.local_prominence_window_da,
            output_npz=args.save_npz,
            output_spatial_tsv=args.save_spatial,
            output_dir=args.output_dir,
            verbose=args.verbose,
        )
        ion_image_mzs = maldi_mzs if ion_images is not None else None
        logger.info(
            f"  {len(maldi_mzs)} features extracted"
            + (f", ion image shape: {ion_images.shape[1:]}" if ion_images is not None else "")
        )
    elif args.maldi_imzml:
        from ms1rescore.maldi_imzml import SCiLSConfig, extract_scils_features

        logger.info(
            "MALDI features extracted from imzML data (SCiLS Lab-style interval extraction). "
            "LC-MS/MS identifications will be used for candidate generation and "
            "prior features only, not for feature selection."
        )
        logger.info(f"Extracting MALDI features from imzML: {args.maldi_imzml}")
        cfg = SCiLSConfig(
            min_pixel_fraction=args.min_fraction,
            peak_prominence=args.peak_prominence,
            smoothing_window=args.smoothing_window,
            smoothing_polyorder=args.smoothing_polyorder,
            ppm_tolerance=args.interval_ppm_tolerance,
            min_interval_width_ppm=args.min_interval_width_ppm,
            normalize_rms=args.normalize_rms,
            baseline_correction=args.baseline_correction,
            baseline_window_ppm=args.baseline_window_ppm,
            calibrant_mzs=args.calibrant_mzs or [],
            calibrant_tol_ppm=args.calibrant_tol_ppm,
            deisotope=args.deisotope,
            deisotope_tol_da=args.deisotope_tol_da,
            deisotope_min_fold=args.deisotope_min_fold,
            filter_mass_defect=args.filter_mass_defect,
            mass_defect_halfwidth=args.mass_defect_halfwidth,
            picking_height=args.picking_height,
            local_prominence_window_da=args.local_prominence_window_da,
        )
        intervals, intensity_matrix, pixel_coords = extract_scils_features(
            args.maldi_imzml,
            config=cfg,
            output_dir=args.output_dir,
            visualize=False,
        )
        maldi_mzs = np.array([apex for _, _, apex in intervals])
        ion_images = None
        ion_image_mzs = None
        spatial_features = None
        logger.info(f"  {len(maldi_mzs)} intervals extracted")
    else:
        maldi_mzs, ion_images, ion_image_mzs = _load_maldi(
            args.maldi_npz, args.maldi_mzs
        )

    # --- Optional spatial features (explicit file overrides extracted ones) ---
    if args.spatial_features:
        logger.info(f"Loading spatial features from {args.spatial_features}")
        spatial_features = pd.read_csv(args.spatial_features, sep="\t")

    # --- Resolve Strategy C source ---
    # If --lcms-peptides is not given but --msf is, use the MSF for Strategy C.
    lcms_peptides_path = args.lcms_peptides
    lcms_id_format = args.lcms_id_format
    if lcms_peptides_path is None and args.msf is not None:
        lcms_peptides_path = args.msf
        lcms_id_format = "msf"
        logger.info(
            f"No --lcms-peptides provided; using --msf ({args.msf}) "
            f"as Strategy C ID source (format='msf')."
        )

    # --- Resolve digest parameters ---
    if lcms_peptides_path:
        from ms1rescore.lcms_ids import parse_lcms_ids

        logger.info("Parsing LC-MS/MS identifications for Strategy C...")
        lcms_ids = parse_lcms_ids(
            proteins_path=args.lcms_proteins,
            peptides_path=lcms_peptides_path,
            psms_path=args.lcms_psms,
            protein_fdr=args.protein_fdr,
            peptide_fdr=args.peptide_fdr,
            format=lcms_id_format,
        )
        if args.verbose:
            logger.debug("Writing parsed LC-MS/MS IDs to debug_lcms_ids.tsv")
            lcms_ids.peptides.to_csv(
                f"{args.output_dir}/4_debug_lcms_ids.tsv", sep="\t", index=False
            )
        min_length, max_length, missed_cleavages = _infer_digest_params(
            lcms_ids,
            missed_cleavages_override=args.missed_cleavages,
            min_length_override=args.min_length,
            max_length_override=args.max_length,
        )
    else:
        min_length = args.min_length if args.min_length is not None else 7
        max_length = args.max_length if args.max_length is not None else 30
        missed_cleavages = (
            args.missed_cleavages if args.missed_cleavages is not None else 2
        )

    logger.info(
        f"Parameters extracted: min_length={min_length}, "
        f"max_length={max_length}, missed_cleavages={missed_cleavages}"
    )

    # --- Run pipeline ---
    from ms1rescore.pipeline import rescore

    logger.info("Starting ms1rescore pipeline...")
    psm_list, result_df, feature_names = rescore(
        fasta_path=args.fasta,
        maldi_mzs=maldi_mzs,
        mzml_paths=args.mzml,
        ion_images=ion_images,
        ion_image_mzs=ion_image_mzs,
        spatial_features=spatial_features,
        msf_path=args.msf,
        ppm_tolerance=args.ppm_tolerance,
        train_fdr=args.train_fdr,
        cache_dir=args.cache_dir,
        missed_cleavages=missed_cleavages,
        min_length=min_length,
        max_length=max_length,
        model=args.model,
        init_ppm_threshold=args.init_ppm_threshold,
        init_isotope_threshold=args.init_isotope_threshold,
        lcms_proteins_path=args.lcms_proteins,
        lcms_peptides_path=lcms_peptides_path,
        lcms_psms_path=args.lcms_psms,
        lcms_id_format=lcms_id_format,
        protein_fdr=args.protein_fdr,
        peptide_fdr=args.peptide_fdr,
        extra_fasta_path=args.extra_fasta,
        verbose=args.verbose,
        output_dir=args.output_dir,
    )

    # --- Write results ---
    logger.info(f"Writing results to {os.path.abspath(args.output_dir)}")
    if args.verbose:
        logger.debug("Writing complete result DataFrame to debug_result_df.tsv")
        result_df.to_csv(f"{args.output_dir}/5_debug_result_df.tsv", sep="\t", index=False)
    _write_results(result_df, args.output_dir)
    logger.info("Done.")


if __name__ == "__main__":
    main()
