"""
Extract MALDI-MSI features from raw Bruker .d/TSF data via imzy.

Three steps:
  1. detect_features   — find consensus m/z peaks across all pixels
  2. extract_ion_images — extract a 3D ion image array for those peaks
  3. compute_spatial_features — fraction_detected, mean_intensity, CV, Moran's I

All three are orchestrated by extract_maldi_data(), which is the public entry point.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Feature detection
# ---------------------------------------------------------------------------


def detect_features(
    reader,
    ppm_bin: float = 5.0,
    min_fraction: float = 0.01,
) -> np.ndarray:
    """
    Detect consensus m/z features by histogram-binning all centroided peaks.

    Parameters
    ----------
    reader
        An imzy reader object (e.g. TSFReader).  Must support
        ``reader.n_pixels`` and ``reader.spectra_iter(silent=False)``.
    ppm_bin
        Width of each histogram bin in ppm.  Peaks within one bin are
        merged into a single feature.  Default 5.0 ppm.
    min_fraction
        Minimum fraction of pixels a peak must be detected in to survive
        as a feature.  Default 0.01 (1 %).

    Returns
    -------
    np.ndarray
        Sorted 1D float64 array of feature m/z values.
    """
    n_pixels = reader.n_pixels
    min_count = max(1, int(min_fraction * n_pixels))

    # --- Pass 1: collect all peaks into parallel arrays ---
    mzs_list: list[np.ndarray] = []
    ints_list: list[np.ndarray] = []
    px_list: list[np.ndarray] = []

    for px_idx, (mzs, ints) in enumerate(reader.spectra_iter(silent=False)):
        if len(mzs) == 0:
            continue
        mzs_list.append(mzs.astype(np.float64))
        ints_list.append(ints.astype(np.float32))
        px_list.append(np.full(len(mzs), px_idx, dtype=np.int32))

    if not mzs_list:
        logger.warning("No peaks found in any pixel — returning empty feature list.")
        return np.array([], dtype=np.float64)

    all_mzs = np.concatenate(mzs_list)
    all_ints = np.concatenate(ints_list)
    all_pxs = np.concatenate(px_list)

    logger.info(
        f"  Collected {len(all_mzs):,} peaks from {n_pixels:,} pixels "
        f"({len(all_mzs)/n_pixels:.0f} peaks/pixel average)"
    )

    # --- Pass 2: sort by m/z and greedy-bin ---
    order = np.argsort(all_mzs, kind="stable")
    all_mzs = all_mzs[order]
    all_ints = all_ints[order]
    all_pxs = all_pxs[order]

    feature_mzs: list[float] = []
    n = len(all_mzs)
    i = 0
    while i < n:
        anchor = all_mzs[i]
        j = i + 1
        # extend group while within ppm_bin of the anchor (first peak in group)
        while j < n and (all_mzs[j] - anchor) / anchor * 1e6 <= ppm_bin:
            j += 1

        n_unique = len(np.unique(all_pxs[i:j]))
        if n_unique >= min_count:
            weights = all_ints[i:j].astype(np.float64)
            w_sum = weights.sum()
            if w_sum > 0:
                feat_mz = float(np.dot(all_mzs[i:j], weights) / w_sum)
            else:
                feat_mz = float(all_mzs[i:j].mean())
            feature_mzs.append(feat_mz)

        i = j

    result = np.array(feature_mzs, dtype=np.float64)
    logger.info(
        f"  {len(result)} features detected "
        f"(ppm_bin={ppm_bin}, min_fraction={min_fraction})"
    )
    return result


# ---------------------------------------------------------------------------
# Ion image extraction
# ---------------------------------------------------------------------------


def extract_ion_images(
    reader,
    feature_mzs: np.ndarray,
    ppm: float = 20.0,
) -> np.ndarray:
    """
    Extract a 3D ion image array for the given feature m/z values.

    Uses ``reader.get_ion_images()`` which makes a single streaming pass
    over all pixels and returns shape ``(n_features, height, width)``.

    Parameters
    ----------
    reader
        An imzy reader (TSFReader / IMZMLReader / etc.).
    feature_mzs
        1D array of feature m/z values.
    ppm
        Extraction tolerance in ppm.  Default 20.0.

    Returns
    -------
    np.ndarray
        Shape ``(n_features, height, width)``, dtype float32.
        Pixels with no detected peak are 0.0.
    """
    logger.info(
        f"  Extracting ion images for {len(feature_mzs)} features " f"at ±{ppm} ppm..."
    )
    # fill_value=0.0 so absent peaks become 0, not NaN
    images = reader.get_ion_images(
        np.asarray(feature_mzs, dtype=np.float64),
        ppm=ppm,
        fill_value=0.0,
        silent=False,
    )
    logger.info(f"  Ion image array: shape={images.shape}, dtype={images.dtype}")
    return images.astype(np.float32)


# ---------------------------------------------------------------------------
# Spatial feature computation
# ---------------------------------------------------------------------------


def _morans_i(image: np.ndarray) -> float:
    """
    Compute Moran's I spatial autocorrelation for a 2D ion image.

    Uses a queen's contiguity (8-neighbour) weight matrix implemented via
    scipy.ndimage.convolve — O(pixels) rather than O(pixels^2).
    """
    from scipy.ndimage import convolve

    n = float(image.size)
    mean_val = image.mean()
    dev = image.astype(np.float64) - mean_val

    kernel = np.ones((3, 3), dtype=np.float64)
    kernel[1, 1] = 0.0

    neighbor_dev_sum = convolve(dev, kernel, mode="constant", cval=0.0)
    numerator = float((dev * neighbor_dev_sum).sum())

    w_sum = float(
        convolve(
            np.ones_like(image, dtype=np.float64), kernel, mode="constant", cval=0.0
        ).sum()
    )
    denominator = float((dev**2).sum())

    if denominator < 1e-12 or w_sum == 0.0:
        return 0.0
    return (n / w_sum) * (numerator / denominator)


def compute_spatial_features(
    ion_images: np.ndarray,
    feature_mzs: np.ndarray,
    n_pixels_total: int,
) -> pd.DataFrame:
    """
    Compute per-feature spatial statistics from the 3D ion image array.

    Produces the same columns as ``spatial_features_tsf.csv``:
    ``feature_mz``, ``n_pixels_detected``, ``fraction_detected``,
    ``mean_intensity``, ``spatial_autocorrelation``, ``intensity_cv``.

    Parameters
    ----------
    ion_images
        Shape ``(n_features, height, width)``, dtype float32.
    feature_mzs
        1D array of feature m/z values aligned with ``ion_images``.
    n_pixels_total
        Total number of measured pixels (``reader.n_pixels``); used for
        ``fraction_detected`` so that unmeasured image area (zeros outside
        the imaging region) is excluded from the denominator.
    """
    records = []
    for i, mz in enumerate(feature_mzs):
        img = ion_images[i]
        flat = img.flatten()
        detected = flat[flat > 0.0]

        n_det = int(len(detected))
        frac = n_det / n_pixels_total if n_pixels_total > 0 else 0.0
        mean_int = float(detected.mean()) if n_det > 0 else 0.0
        cv = float(detected.std() / mean_int) if mean_int > 0 else 0.0
        p90 = float(np.percentile(detected, 90)) if n_det > 0 else 0.0
        total = float(detected.sum()) if n_det > 0 else 0.0
        mi = _morans_i(img)

        records.append(
            {
                "feature_mz": float(mz),
                "n_pixels_detected": n_det,
                "fraction_detected": frac,
                "mean_intensity": mean_int,
                "intensity_p90": p90,
                "intensity_sum": total,
                "spatial_autocorrelation": mi,
                "intensity_cv": cv,
            }
        )

    df = pd.DataFrame(records)
    logger.info(
        f"  Spatial features computed for {len(df)} features "
        f"(mean fraction_detected={df['fraction_detected'].mean():.3f})"
    )
    return df


# ---------------------------------------------------------------------------
# LC-MS/MS guided feature m/z computation
# ---------------------------------------------------------------------------


_PROTON = 1.007276


def _find_col(df: "pd.DataFrame", *candidates: str) -> "str | None":
    """Return the first column whose lowercased name contains one of the candidates."""
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in lower:
            return lower[cand]
    for cand in candidates:
        for col_lower, col_orig in lower.items():
            if cand in col_lower:
                return col_orig
    return None


def _deduplicate_mzs(mzs: np.ndarray, merge_ppm: float = 1.0) -> np.ndarray:
    """Merge m/z values within ``merge_ppm`` of each other (take group mean)."""
    mzs = np.sort(mzs)
    groups: list[list[float]] = [[float(mzs[0])]]
    for mz in mzs[1:]:
        if abs(mz - groups[-1][-1]) / groups[-1][-1] * 1e6 < merge_ppm:
            groups[-1].append(float(mz))
        else:
            groups.append([float(mz)])
    return np.array([np.mean(g) for g in groups], dtype=np.float64)


def _features_from_lcms_file_diagnostic(
    path: str,
    format: str = "percolator",
    peptide_fdr: float = 0.01,
    mz_min: float = 750.0,
    mz_max: float = 2900.0,
) -> np.ndarray:
    """
    DIAGNOSTIC / LEGACY MODE ONLY. Do not use for rescoring.

    Defines MALDI features from LC-MS/MS-identified peptide m/z values.
    This is circular for rescoring: it pre-selects features that are
    guaranteed to match LC-MS/MS candidates, making the rescoring trivial
    and producing results that do not generalise to unidentified MALDI features.

    Use detect_features() for all rescoring workflows. LC-MS/MS identifications
    should inform candidate generation (Strategy C in candidates.py) and prior
    features (LCMS_PRIOR_FEATURES), not the MALDI feature set itself.

    Reads pre-computed masses directly from the file so that peptide
    modifications are correctly accounted for.  m/z values are deduplicated
    with a 1 ppm tolerance and filtered to the MALDI acquisition range.

    Parameters
    ----------
    path
        Path to the LC-MS/MS ID file.
    format
        File format: ``"percolator"`` (TSV), ``"msf"`` (ProteomeDiscoverer
        ``.msf`` SQLite), or ``"mzidentml"``.
    peptide_fdr
        Peptide-level FDR threshold; rows above this threshold are ignored.
    mz_min, mz_max
        MALDI acquisition m/z range; features outside are dropped.

    Returns
    -------
    np.ndarray
        Sorted 1D float64 array of unique [M+H]+ m/z values.
    """
    mh_mzs: np.ndarray

    if format == "percolator":
        df = pd.read_csv(path, sep="\t")
        qcol = _find_col(df, "q-value", "qvalue", "q_value", "percolatorqvalue")
        if qcol is not None:
            df = df[df[qcol].astype(float) <= peptide_fdr]

        mzcol = _find_col(df, "mh_mz", "mh+", "mhovermass")
        masscol = _find_col(df, "mass") if mzcol is None else None

        if mzcol is not None:
            mh_mzs = df[mzcol].dropna().to_numpy(dtype=np.float64)
        elif masscol is not None:
            mh_mzs = df[masscol].dropna().to_numpy(dtype=np.float64) + _PROTON
        else:
            raise ValueError(
                f"Could not find MH_mz or Mass column in {path!r}. "
                f"Columns present: {list(df.columns)}"
            )

    elif format == "msf":
        import sqlite3

        conn = sqlite3.connect(path)
        try:
            rows = conn.execute(
                "SELECT Mass FROM TargetPsms WHERE PercolatorqValue <= ? AND Mass IS NOT NULL",
                (peptide_fdr,),
            ).fetchall()
        finally:
            conn.close()
        if not rows:
            raise ValueError(f"No PSMs at FDR {peptide_fdr} in {path!r}")
        mh_mzs = np.array([r[0] for r in rows], dtype=np.float64) + _PROTON

    elif format == "mzidentml":
        try:
            from pyteomics import mzid as _mzid
        except ImportError as exc:
            raise ImportError("pyteomics required for mzIdentML") from exc
        mh_list = []
        with _mzid.MzIdentML(path) as mzid:
            for sir in mzid:
                for sii in sir.get("SpectrumIdentificationItem", []):
                    qval = np.nan
                    for cv in sii.get("cvParam", []):
                        if cv.get("accession") == "MS:1002354":
                            qval = float(cv.get("value", np.nan))
                    if not np.isnan(qval) and qval <= peptide_fdr:
                        mz = sii.get("experimentalMassToCharge", np.nan)
                        charge = sii.get("chargeState", 1) or 1
                        if not np.isnan(mz):
                            mh_list.append(float(mz) * charge - (charge - 1) * _PROTON)
        mh_mzs = np.array(mh_list, dtype=np.float64)

    else:
        raise ValueError(
            f"Unknown format {format!r} for features_from_lcms_file. "
            "Use 'percolator', 'msf', or 'mzidentml'."
        )

    mh_mzs = _deduplicate_mzs(mh_mzs, merge_ppm=1.0)
    mask = (mh_mzs >= mz_min) & (mh_mzs <= mz_max)
    result = mh_mzs[mask]
    logger.info(
        f"  {len(result)} LC-MS/MS-guided features in [{mz_min:.0f}, {mz_max:.0f}] Da "
        f"({int((~mask).sum())} outside range dropped, {len(mh_mzs)} unique before filter)"
    )
    return result


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------


def extract_maldi_data(
    d_path: str,
    ppm_bin: float = 5.0,
    extraction_ppm: float = 25.0,
    matching_ppm: float = 20.0,
    min_fraction: float = 0.01,
    feature_mzs: np.ndarray | None = None,
    images_path: str | None = None,
    image_batch_size: int = 100,
    output_npz: str | None = None,
    output_spatial_tsv: str | None = None,
    output_dir: str | None = None,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Extract MALDI features, ion images, and spatial statistics from a raw
    Bruker ``.d`` directory.

    Features are always detected from raw centroided spectra via
    ``detect_features``.  Pass ``feature_mzs`` only to reuse a cached feature
    set from a previous run; do not populate it from LC-MS/MS identifications
    (that is circular — see ``_features_from_lcms_file_diagnostic``).
    Features with zero MALDI signal are removed after extraction.

    By default ion images are loaded fully into RAM (fast, single pass).
    For datasets where the full ``(n_features, H, W)`` float32 array does not
    fit in RAM, set ``images_path`` to a file path: images are then written
    directly to a memory-mapped file in batches of ``image_batch_size``
    features, capping peak RAM at ``image_batch_size × H × W × 4`` bytes.
    The returned ``ion_images`` is a ``np.memmap`` that is transparent to all
    downstream code (colocalization, spatial features, NPZ saving).

    Parameters
    ----------
    d_path
        Path to Bruker ``.d`` directory.
    ppm_bin
        Peak-binning tolerance for feature detection (ppm).  Ignored when
        ``feature_mzs`` is provided.
    extraction_ppm
        m/z window for raw ion image extraction.  Controls which raw data
        points contribute to each ion image.  Should be slightly wider than
        the instrument's typical peak width in imaging mode to avoid cutting
        off peak tails.  Default 25.0 ppm.
    matching_ppm
        m/z window for candidate matching.  Applied to the detected feature
        centroid m/z when linking peptide candidates to MALDI features.
        Should reflect the accuracy of the centroid estimate.  Default 20.0 ppm.
        This value is stored in the returned ``spatial_df`` as metadata but is
        not used internally — pass it to ``match_to_maldi_features``.
    min_fraction
        Minimum fraction of pixels a peak must be detected in.  Ignored when
        ``feature_mzs`` is provided.
    feature_mzs
        If given, use these m/z values instead of running ``detect_features``.
        Intended for reuse of a cached feature set from a previous run, not
        for LC-MS/MS guided feature selection.  Must be a sorted 1D float64
        array.
    images_path
        If given, write ion images to this path as a float32 memmap instead
        of allocating in RAM.  Enables extraction of datasets that would
        otherwise OOM.
    image_batch_size
        Features per batch when ``images_path`` is set.  Default 100.
    output_npz
        If given, save ``{mzs, images, x_coords, y_coords}`` to this path.
    output_spatial_tsv
        If given, save the spatial features DataFrame to this TSV path.

    Returns
    -------
    (feature_mzs, ion_images, spatial_df)
        ``feature_mzs`` — 1D float64, shape ``(n_features,)``
        ``ion_images``  — float32 array or memmap, shape ``(n_features, H, W)``
        ``spatial_df``  — DataFrame with per-feature spatial statistics
    """
    try:
        import imzy
    except ImportError as exc:
        raise ImportError(
            "imzy is required for raw MALDI data extraction. "
            "Install with: pip install ms1rescore[maldi]"
        ) from exc

    logger.info(f"Opening MALDI dataset: {d_path}")
    with imzy.get_reader(d_path) as reader:
        logger.info(
            f"  {reader.n_pixels:,} pixels, image shape {reader.image_shape}, "
            f"m/z range {reader.mz_min:.1f}–{reader.mz_max:.1f}"
        )

        if feature_mzs is not None:
            logger.info(
                f"Step 1/3: Using {len(feature_mzs)} provided feature m/z values "
                f"(skipping detection)."
            )
        else:
            logger.info("Step 1/3: Detecting features...")
            feature_mzs = detect_features(
                reader, ppm_bin=ppm_bin, min_fraction=min_fraction
            )

        if len(feature_mzs) == 0:
            raise ValueError(
                f"No features for {d_path!r}. "
                "If using detect_features, try lowering min_fraction. "
                "If using feature_mzs, ensure the sequences are valid and in range."
            )

        x_coords = reader.x_coordinates
        y_coords = reader.y_coordinates

        logger.info("Step 2/3: Extracting ion images...")
        if images_path is None:
            # Default: single pass, full array in RAM.
            ion_images = extract_ion_images(reader, feature_mzs, ppm=extraction_ppm)
            logger.info("Step 3/3: Computing spatial features...")
            spatial_df = compute_spatial_features(
                ion_images, feature_mzs, reader.n_pixels
            )
        else:
            # Memory-efficient: write to disk in batches, never hold full array in RAM.
            n_features = len(feature_mzs)
            height, width = reader.image_shape
            ion_images = np.memmap(
                images_path,
                dtype=np.float32,
                mode="w+",
                shape=(n_features, height, width),
            )
            logger.info(
                f"  Memmap {n_features} × {height} × {width} float32 → {images_path}"
            )
            spatial_chunks: list[pd.DataFrame] = []
            for batch_start in range(0, n_features, image_batch_size):
                batch_end = min(batch_start + image_batch_size, n_features)
                batch_mzs = feature_mzs[batch_start:batch_end]
                batch_images = reader.get_ion_images(
                    np.asarray(batch_mzs, dtype=np.float64),
                    ppm=extraction_ppm,
                    fill_value=0.0,
                    silent=True,
                ).astype(np.float32)
                ion_images[batch_start:batch_end] = batch_images
                logger.info("Step 3/3: Computing spatial features...")
                spatial_chunks.append(
                    compute_spatial_features(batch_images, batch_mzs, reader.n_pixels)
                )
                del batch_images
            ion_images.flush()
            spatial_df = pd.concat(spatial_chunks, ignore_index=True)

    # Drop features with zero MALDI signal (no pixels detected).
    # This is especially important when feature_mzs come from LC-MS/MS IDs,
    # where some peptide masses may have no corresponding MALDI signal.
    detected_mask = spatial_df["n_pixels_detected"].to_numpy() > 0
    if not detected_mask.all():
        n_removed = int((~detected_mask).sum())
        logger.info(
            f"  Dropping {n_removed} features with zero MALDI signal "
            f"({detected_mask.sum()} features retained)."
        )
        feature_mzs = feature_mzs[detected_mask]
        ion_images = ion_images[detected_mask]
        spatial_df = spatial_df[detected_mask].reset_index(drop=True)

    if output_npz is not None:
        npz_path = f"{output_dir}/{output_npz}" if output_dir else output_npz
        np.savez_compressed(
            npz_path,
            mzs=feature_mzs,
            images=ion_images,
            x_coords=x_coords,
            y_coords=y_coords,
        )
        logger.info(f"  Saved NPZ → {npz_path}")

    if output_spatial_tsv is not None:
        tsv_path = (
            f"{output_dir}/{output_spatial_tsv}" if output_dir else output_spatial_tsv
        )
        spatial_df.to_csv(tsv_path, sep="\t", index=False)
        logger.info(f"  Saved spatial features → {tsv_path}")

    return feature_mzs, ion_images, spatial_df
