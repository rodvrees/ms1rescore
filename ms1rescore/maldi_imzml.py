"""
SCiLS Lab-style m/z interval extraction for imzML data.

Algorithm:
  1. Build a mean spectrum across all pixels on a common m/z grid.
  2. Detect intervals on the mean spectrum (SG smooth → find_peaks →
     valley-to-valley boundaries, ppm fallback if no clear valley).
  3. Integrate each pixel over each interval (sum or apex).
  4. Filter intervals by minimum intensity, minimum pixel fraction, and
     minimum interval width.

Public entry point: extract_scils_features()
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.signal import find_peaks, savgol_filter

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class SCiLSConfig:
    ppm_tolerance: float = 10.0          # fallback interval half-width (ppm)
    smoothing_window: int = 7
    smoothing_polyorder: int = 2
    peak_prominence: float = 0.01        # min peak prominence as fraction of mean-spectrum max
    min_intensity: float = 0.0           # min mean intensity to keep interval
    min_pixel_fraction: float = 0.01     # min fraction of pixels with signal
    min_interval_width_ppm: float = 2.0  # minimum interval full-width (ppm)
    normalize_tic: bool = True           # TIC-normalize per pixel before integration
    use_apex: bool = False               # False = sum intensities; True = apex intensity
    mz_grid_resolution: float = 0.001   # Da, for mean spectrum accumulation


# ---------------------------------------------------------------------------
# imzML mode detection
# ---------------------------------------------------------------------------


def _detect_imzml_mode(parser) -> str:
    mode = getattr(parser, "spectrum_mode", None)
    if mode in ("profile", "centroid"):
        return mode
    if len(parser.coordinates) == 0:
        return "centroid"
    mzs, _ = parser.getspectrum(0)
    mzs = np.asarray(mzs, dtype=np.float64)
    if len(mzs) < 2:
        return "centroid"
    median_gap = float(np.median(np.diff(mzs)))
    # Profile data has sub-Da spacing; centroid data has gaps of many Da
    return "profile" if median_gap < 0.01 else "centroid"


# ---------------------------------------------------------------------------
# Mean spectrum construction
# ---------------------------------------------------------------------------


def _build_mean_spectrum(
    parser, config: SCiLSConfig
) -> tuple[np.ndarray, np.ndarray]:
    """Return (mz_grid, mean_intensities) accumulated over all pixels.

    Uses a fixed-resolution m/z grid. Accumulates via np.add.at so cost is
    O(n_peaks) per pixel rather than O(n_grid * log n_spectrum).

    For aligned profile data (all pixels share the same m/z axis length and
    first/last m/z), a fast vectorised sum path is used instead.
    """
    n_pixels = len(parser.coordinates)
    if n_pixels == 0:
        return np.array([]), np.array([])

    # Determine m/z range from first pixel
    mzs0, _ = parser.getspectrum(0)
    mzs0 = np.asarray(mzs0, dtype=np.float64)
    mz_min = float(mzs0[0]) if len(mzs0) > 0 else 100.0
    mz_max = float(mzs0[-1]) if len(mzs0) > 0 else 4000.0

    # Check for aligned profile: all spectra same length and same m/z bounds
    aligned = False
    if len(mzs0) > 1 and n_pixels > 1:
        mzs1, _ = parser.getspectrum(min(1, n_pixels - 1))
        mzs1 = np.asarray(mzs1, dtype=np.float64)
        aligned = (
            len(mzs1) == len(mzs0)
            and abs(float(mzs1[0]) - mz_min) < 1e-6
            and abs(float(mzs1[-1]) - mz_max) < 1e-6
        )

    if aligned:
        return _build_mean_spectrum_aligned(parser, mzs0, n_pixels)

    # Scan all pixels for true m/z range
    for i in range(n_pixels):
        mzs, _ = parser.getspectrum(i)
        mzs = np.asarray(mzs, dtype=np.float64)
        if len(mzs) == 0:
            continue
        mz_min = min(mz_min, float(mzs[0]))
        mz_max = max(mz_max, float(mzs[-1]))

    res = config.mz_grid_resolution
    n_bins = int(np.ceil((mz_max - mz_min) / res)) + 1
    grid_sum = np.zeros(n_bins, dtype=np.float64)
    mz_grid = mz_min + np.arange(n_bins, dtype=np.float64) * res

    for i in range(n_pixels):
        mzs, ints = parser.getspectrum(i)
        mzs = np.asarray(mzs, dtype=np.float64)
        ints = np.asarray(ints, dtype=np.float64)
        if len(mzs) == 0:
            continue
        indices = np.round((mzs - mz_min) / res).astype(np.intp)
        valid = (indices >= 0) & (indices < n_bins)
        np.add.at(grid_sum, indices[valid], ints[valid])

    mean_ints = grid_sum / n_pixels
    return mz_grid, mean_ints.astype(np.float32)


def _build_mean_spectrum_aligned(
    parser, mzs0: np.ndarray, n_pixels: int
) -> tuple[np.ndarray, np.ndarray]:
    """Fast path: all spectra share the same m/z axis — just sum intensity arrays."""
    acc = np.zeros(len(mzs0), dtype=np.float64)
    for i in range(n_pixels):
        _, ints = parser.getspectrum(i)
        acc += np.asarray(ints, dtype=np.float64)
    return mzs0, (acc / n_pixels).astype(np.float32)


# ---------------------------------------------------------------------------
# Interval detection on the mean spectrum
# ---------------------------------------------------------------------------


def _detect_intervals(
    mz_grid: np.ndarray,
    mean_ints: np.ndarray,
    config: SCiLSConfig,
) -> list[tuple[float, float, float]]:
    """Return list of (mz_start, mz_end, mz_apex) intervals from mean spectrum.

    Valley-to-valley: for each detected peak, left boundary = deepest valley
    between this peak and the previous peak (or spectrum edge), right boundary
    = deepest valley between this peak and the next peak (or spectrum edge).
    Falls back to a fixed ppm half-width when no flanking valley is available
    (e.g. isolated peak at the spectrum edge).
    """
    if len(mz_grid) < config.smoothing_window:
        return []

    smoothed = savgol_filter(
        mean_ints.astype(np.float64),
        config.smoothing_window,
        config.smoothing_polyorder,
    )
    smoothed = np.maximum(smoothed, 0.0)

    max_int = float(smoothed.max())
    prominence = config.peak_prominence * max_int if max_int > 0 else 0.0

    # Detect peaks on the mean spectrum
    peaks, _ = find_peaks(smoothed, prominence=prominence)
    if len(peaks) == 0:
        return []

    # Detect valleys (local minima) as peaks in the negated signal
    valleys, _ = find_peaks(-smoothed)
    valley_set = set(valleys.tolist())

    intervals: list[tuple[float, float, float]] = []
    for k, pk in enumerate(peaks):
        mz_apex = float(mz_grid[pk])

        # Left boundary: largest valley index strictly less than pk
        left_vals = valleys[valleys < pk]
        if len(left_vals) > 0:
            li = int(left_vals[-1])
        else:
            # Fallback: ppm half-width from apex
            half_da = mz_apex * config.ppm_tolerance * 1e-6
            li = int(np.searchsorted(mz_grid, mz_apex - half_da))
            li = max(li, 0)

        # Right boundary: smallest valley index strictly greater than pk
        right_vals = valleys[valleys > pk]
        if len(right_vals) > 0:
            ri = int(right_vals[0])
        else:
            half_da = mz_apex * config.ppm_tolerance * 1e-6
            ri = int(np.searchsorted(mz_grid, mz_apex + half_da, side="right")) - 1
            ri = min(ri, len(mz_grid) - 1)

        mz_start = float(mz_grid[li])
        mz_end = float(mz_grid[ri])

        # Filter by minimum interval width
        width_ppm = (mz_end - mz_start) / mz_apex * 1e6
        if width_ppm < config.min_interval_width_ppm:
            # Expand symmetrically to meet minimum width
            half_needed = mz_apex * config.min_interval_width_ppm * 0.5e-6
            mz_start = mz_apex - half_needed
            mz_end = mz_apex + half_needed

        intervals.append((mz_start, mz_end, mz_apex))

    return intervals


# ---------------------------------------------------------------------------
# Per-pixel integration
# ---------------------------------------------------------------------------


def _integrate_pixel(
    mzs: np.ndarray,
    ints: np.ndarray,
    intervals: list[tuple[float, float, float]],
    config: SCiLSConfig,
) -> np.ndarray:
    """Return a float32 array of length n_intervals for one pixel.

    TIC normalization is applied to the raw intensities before interval
    integration so that the resulting interval intensities are comparable
    across pixels regardless of total ion load. Normalizing before rather
    than after integration is correct: if we normalized after, the sum of
    intensities in different intervals would still reflect total ion load
    differences within each interval.
    """
    result = np.zeros(len(intervals), dtype=np.float32)
    if len(mzs) == 0 or len(ints) == 0:
        return result

    ints = ints.astype(np.float32)

    if config.normalize_tic:
        tic = float(np.sum(ints))
        if tic > 0.0:
            ints = ints / tic

    mzs = mzs.astype(np.float64)

    for j, (mz_lo, mz_hi, mz_apex) in enumerate(intervals):
        lo = int(np.searchsorted(mzs, mz_lo, side="left"))
        hi = int(np.searchsorted(mzs, mz_hi, side="right"))
        if lo >= hi:
            continue
        seg = ints[lo:hi]
        if config.use_apex:
            result[j] = float(seg.max())
        else:
            result[j] = float(seg.sum())

    return result


# ---------------------------------------------------------------------------
# Interval filtering
# ---------------------------------------------------------------------------


def _filter_intervals(
    intensity_matrix: np.ndarray,
    intervals: list[tuple[float, float, float]],
    config: SCiLSConfig,
    n_pixels: int,
) -> np.ndarray:
    """Return a boolean keep-mask over intervals.

    An interval is kept when:
    - Its mean intensity across all pixels exceeds config.min_intensity.
    - The fraction of pixels with non-zero signal meets config.min_pixel_fraction.
    """
    mean_int = intensity_matrix.mean(axis=0)
    pixel_frac = (intensity_matrix > 0).sum(axis=0) / max(n_pixels, 1)

    keep = (mean_int >= config.min_intensity) & (
        pixel_frac >= config.min_pixel_fraction
    )
    return keep


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def _visualize_scils(
    mz_grid: np.ndarray,
    mean_ints: np.ndarray,
    intervals: list[tuple[float, float, float]],
    intensity_matrix: np.ndarray,
    pixel_coords: list[tuple[int, int]],
    output_dir: str,
    top_n: int = 9,
) -> None:
    """Save 4 diagnostic PNG files to output_dir."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)

    apices = np.array([iv[2] for iv in intervals])

    # 1. Mean spectrum with interval shading
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(mz_grid, mean_ints, lw=0.5, color="steelblue", label="mean spectrum")
    for mz_lo, mz_hi, _ in intervals:
        ax.axvspan(mz_lo, mz_hi, alpha=0.15, color="orange")
    ax.set_xlabel("m/z")
    ax.set_ylabel("Mean intensity")
    ax.set_title(f"Mean spectrum — {len(intervals)} intervals")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "scils_mean_spectrum.png"), dpi=150)
    plt.close(fig)

    # 2. Interval m/z histogram
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(apices, bins=min(100, len(apices)), color="steelblue", edgecolor="none")
    ax.set_xlabel("Interval apex m/z")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of interval apex m/z values")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "scils_interval_mz_histogram.png"), dpi=150)
    plt.close(fig)

    # 3. Per-interval signal statistics
    mean_int = intensity_matrix.mean(axis=0)
    pixel_frac = (intensity_matrix > 0).sum(axis=0) / max(intensity_matrix.shape[0], 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(pixel_frac, bins=50, color="steelblue", edgecolor="none")
    axes[0].set_xlabel("Pixel fraction detected")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Pixel fraction per interval")

    axes[1].hist(np.log10(mean_int + 1e-12), bins=50, color="darkorange", edgecolor="none")
    axes[1].set_xlabel("log10(mean interval intensity)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Mean intensity per interval")

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "scils_interval_stats.png"), dpi=150)
    plt.close(fig)

    # 4. Ion image mosaic for top_n highest mean-intensity intervals
    if intensity_matrix.shape[0] == 0 or len(pixel_coords) == 0:
        return

    xs = [p[0] for p in pixel_coords]
    ys = [p[1] for p in pixel_coords]
    W = max(xs) + 1
    H = max(ys) + 1

    order = np.argsort(mean_int)[::-1]
    top_idx = order[: min(top_n, len(order))]

    n_cols = min(3, len(top_idx))
    n_rows = int(np.ceil(len(top_idx) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    axes = np.array(axes).reshape(-1)

    for plot_k, feat_k in enumerate(top_idx):
        img = np.zeros((H, W), dtype=np.float32)
        col = intensity_matrix[:, feat_k]
        for pix_i, (px, py) in enumerate(pixel_coords):
            img[py, px] = col[pix_i]

        vmax = float(np.percentile(img[img > 0], 99)) if (img > 0).any() else 1.0
        ax = axes[plot_k]
        ax.imshow(
            img ** 0.5,
            cmap="hot",
            vmin=0,
            vmax=vmax ** 0.5,
            aspect="auto",
            interpolation="nearest",
        )
        ax.set_title(f"m/z {apices[feat_k]:.3f}", fontsize=8)
        ax.axis("off")

    for ax in axes[len(top_idx):]:
        ax.axis("off")

    fig.suptitle(f"Top {len(top_idx)} intervals by mean intensity (γ=0.5)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "scils_ion_image_mosaic.png"), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def extract_scils_features(
    imzml_path: str,
    config: Optional[SCiLSConfig] = None,
    output_dir: Optional[str] = None,
    visualize: bool = False,
) -> tuple[list[tuple[float, float, float]], np.ndarray, list[tuple[int, int]]]:
    """Extract SCiLS Lab-style interval features from an imzML dataset.

    Parameters
    ----------
    imzml_path:
        Path to the .imzML file (.ibd must be in the same directory).
    config:
        SCiLSConfig instance. Defaults are used when None.
    output_dir:
        Directory for output files and (if visualize=True) diagnostic plots.
    visualize:
        When True, saves 4 diagnostic PNG files to output_dir.

    Returns
    -------
    intervals : list of (mz_start, mz_end, mz_apex)
        One tuple per kept interval.
    intensity_matrix : np.ndarray, shape (n_pixels, n_intervals), float32
        Per-pixel integrated intensities.
    pixel_coords : list of (x, y)
        0-based pixel coordinates, same row order as intensity_matrix.
    """
    from pyimzml.ImzMLParser import ImzMLParser

    if config is None:
        config = SCiLSConfig()

    with ImzMLParser(imzml_path) as parser:
        n_pixels = len(parser.coordinates)
        mode = _detect_imzml_mode(parser)
        logger.info(
            f"imzML: {n_pixels} pixels, mode={mode}, "
            f"file={os.path.basename(imzml_path)}"
        )

        # Pass 1: build mean spectrum
        logger.info("Pass 1: building mean spectrum…")
        mz_grid, mean_ints = _build_mean_spectrum(parser, config)

        if len(mz_grid) == 0:
            logger.warning("Empty imzML dataset — no spectra found.")
            return [], np.zeros((0, 0), dtype=np.float32), []

        # Detect intervals from mean spectrum
        intervals = _detect_intervals(mz_grid, mean_ints, config)
        logger.info(f"  {len(intervals)} intervals detected from mean spectrum")

        if len(intervals) == 0:
            return [], np.zeros((n_pixels, 0), dtype=np.float32), []

        # Pass 2: integrate each pixel
        logger.info("Pass 2: integrating pixels over intervals…")
        intensity_matrix = np.zeros((n_pixels, len(intervals)), dtype=np.float32)
        pixel_coords: list[tuple[int, int]] = []

        for i, (x, y, _z) in enumerate(parser.coordinates):
            pixel_coords.append((x - 1, y - 1))  # 1-based → 0-based
            mzs, ints = parser.getspectrum(i)
            mzs = np.asarray(mzs, dtype=np.float64)
            ints = np.asarray(ints, dtype=np.float64)
            intensity_matrix[i] = _integrate_pixel(mzs, ints, intervals, config)

    # Filter intervals
    keep = _filter_intervals(intensity_matrix, intervals, config, n_pixels)
    n_kept = int(keep.sum())
    logger.info(
        f"  {n_kept}/{len(intervals)} intervals kept after filtering "
        f"(min_intensity={config.min_intensity}, "
        f"min_pixel_fraction={config.min_pixel_fraction})"
    )

    intervals = [iv for iv, k in zip(intervals, keep) if k]
    intensity_matrix = intensity_matrix[:, keep]

    if visualize and output_dir is not None:
        _visualize_scils(
            mz_grid, mean_ints, intervals, intensity_matrix, pixel_coords,
            output_dir=output_dir,
        )

    return intervals, intensity_matrix, pixel_coords
