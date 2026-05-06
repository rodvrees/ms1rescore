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
    normalize_rms: bool = False          # RMS-normalize per pixel (alternative to TIC; takes priority)
    use_apex: bool = False               # False = sum intensities; True = apex intensity
    mz_grid_resolution: float = 0.001   # Da, for mean spectrum accumulation
    # Baseline correction (applied to mean spectrum before peak detection)
    baseline_correction: bool = False
    baseline_window_ppm: float = 500.0   # rolling-minimum window half-width (ppm)
    # Internal standard recalibration (applied to detected apices)
    calibrant_mzs: list = field(default_factory=list)  # theoretical m/z of calibrants
    calibrant_tol_ppm: float = 200.0     # search window to find each calibrant
    # Deisotoping (remove M+1/M+2 peaks at z=1 spacing)
    deisotope: bool = False
    deisotope_tol_da: float = 0.15       # tolerance around 1.003355 Da
    deisotope_min_fold: float = 0.67     # fraction of averagine-expected M0/M+1 ratio; lower = more permissive
    # Senko mass defect filter (peptide corridor)
    filter_mass_defect: bool = False
    mass_defect_halfwidth: float = 0.5   # half-width of corridor; 0.5 = all pass
    # Picking height centroid (mMass-style apex refinement)
    picking_height: float = 0.75         # fraction of apex intensity; 0.0 = disabled (raw apex)
    # Local adaptive prominence (sliding-window local max as reference instead of global max)
    local_prominence_window_da: float = 0.0  # 0 = global max reference; >0 = window half-width in Da


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
        return _build_mean_spectrum_aligned(parser, mzs0, n_pixels, config)

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
        if config.normalize_rms:
            rms = float(np.sqrt(np.mean(ints ** 2)))
            if rms > 0.0:
                ints = ints / rms
        elif config.normalize_tic:
            tic = float(ints.sum())
            if tic > 0.0:
                ints = ints / tic
        indices = np.round((mzs - mz_min) / res).astype(np.intp)
        valid = (indices >= 0) & (indices < n_bins)
        np.add.at(grid_sum, indices[valid], ints[valid])

    mean_ints = grid_sum / n_pixels
    return mz_grid, mean_ints.astype(np.float32)


def _build_mean_spectrum_aligned(
    parser, mzs0: np.ndarray, n_pixels: int, config: SCiLSConfig
) -> tuple[np.ndarray, np.ndarray]:
    """Fast path: all spectra share the same m/z axis — just sum intensity arrays."""
    acc = np.zeros(len(mzs0), dtype=np.float64)
    for i in range(n_pixels):
        _, ints = parser.getspectrum(i)
        ints = np.asarray(ints, dtype=np.float64)
        if config.normalize_rms:
            rms = float(np.sqrt(np.mean(ints ** 2)))
            if rms > 0.0:
                ints = ints / rms
        elif config.normalize_tic:
            tic = float(ints.sum())
            if tic > 0.0:
                ints = ints / tic
        acc += ints
    return mzs0, (acc / n_pixels).astype(np.float32)


# ---------------------------------------------------------------------------
# Pre-processing helpers for mean spectrum
# ---------------------------------------------------------------------------


def _subtract_baseline(
    mz_grid: np.ndarray, ints: np.ndarray, window_ppm: float
) -> np.ndarray:
    """Rolling-minimum baseline subtraction on the mean spectrum.

    Uses a fixed window width (in points) computed at the median m/z — a valid
    approximation for the typical 750–2900 Da MALDI range at 500 ppm.
    """
    from scipy.ndimage import gaussian_filter1d, minimum_filter1d

    resolution = float(mz_grid[1] - mz_grid[0])
    median_mz = float(np.median(mz_grid))
    half_pts = max(1, int(round(median_mz * window_ppm * 1e-6 / resolution)))
    ints_f64 = ints.astype(np.float64)
    baseline = minimum_filter1d(ints_f64, size=2 * half_pts + 1)
    baseline_smooth = gaussian_filter1d(baseline, sigma=float(half_pts))
    return np.maximum(ints_f64 - baseline_smooth, 0.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Post-detection interval processing helpers
# ---------------------------------------------------------------------------


def _recalibrate_intervals(
    intervals: list[tuple[float, float, float]],
    calibrant_mzs: list[float],
    tol_ppm: float,
) -> list[tuple[float, float, float]]:
    """Linear ppm recalibration using internal standard m/z values.

    Finds each calibrant among the detected apices (within tol_ppm), fits a
    linear model of ppm_offset vs m/z, and applies the correction to all
    interval apices and boundaries. Requires ≥1 calibrant to be found.
    """
    if not intervals or not calibrant_mzs:
        return intervals

    apices = np.array([iv[2] for iv in intervals])
    found_theo: list[float] = []
    found_obs: list[float] = []

    for theo in calibrant_mzs:
        diffs_ppm = np.abs(apices - theo) / theo * 1e6
        idx = int(np.argmin(diffs_ppm))
        if diffs_ppm[idx] <= tol_ppm:
            found_theo.append(theo)
            found_obs.append(float(apices[idx]))

    if len(found_theo) == 0:
        logger.warning("Recalibration: no calibrants found within %.0f ppm — skipping.", tol_ppm)
        return intervals

    offsets_ppm = [(obs - theo) / theo * 1e6 for obs, theo in zip(found_obs, found_theo)]

    if len(found_theo) >= 2:
        coeffs = np.polyfit(found_theo, offsets_ppm, deg=1)
        logger.info(
            "Recalibration: %d/%d calibrants found; linear fit slope=%.4g intercept=%.4g ppm",
            len(found_theo), len(calibrant_mzs), coeffs[0], coeffs[1],
        )
    else:
        # Single calibrant: constant shift
        coeffs = np.array([0.0, offsets_ppm[0]])
        logger.info(
            "Recalibration: 1/%d calibrant found; constant shift=%.3g ppm",
            len(calibrant_mzs), offsets_ppm[0],
        )

    recal: list[tuple[float, float, float]] = []
    for mz_lo, mz_hi, mz_apex in intervals:
        correction_ppm = float(np.polyval(coeffs, mz_apex))
        scale = 1.0 / (1.0 + correction_ppm * 1e-6)
        recal.append((mz_lo * scale, mz_hi * scale, mz_apex * scale))
    return recal


def _deisotope_intervals(
    intervals: list[tuple[float, float, float]],
    tol_da: float,
    apex_intensities: np.ndarray | None = None,
    min_fold: float = 0.7,
) -> list[tuple[float, float, float]]:
    """Remove M+1 and M+2 isotope peaks (z=1, spacing 1.003355 Da).

    Sorts by apex ascending. A peak i is flagged as an isotope of peak j when:
      - |apex_i − apex_j − k × 1.003355| < tol_da  (k = 1 or 2), AND
      - intensity_j >= intensity_i × (1/lambda) × min_fold

    where lambda = 0.000509 × mz_j (averagine Poisson parameter).

    Two physically motivated guards prevent false-positive deisotoping:

    1. **k-specific expected ratio**: for k=1 the expected M0/M+1 ratio is
       1/lambda; for k=2 the expected M0/M+2 ratio is 2/lambda². The M+2
       threshold is always much higher than the M+1 threshold, preventing
       false removal of an unrelated peptide that coincidentally falls ~2 Da
       above a more intense peak.

    2. **Absolute minimum of 1.0**: the threshold is floored at 1.0, meaning
       M0 must always be at least as intense as M+k. When M+k > M0 in the
       mean spectrum (obs_ratio < 1) the pair is physically inconsistent with
       a true isotope pattern — likely two unrelated peptides of similar
       intensity — so removal is skipped. This also prevents false-positive
       deisotoping at high masses (>~1400 Da) where the averagine-expected
       M0/M+1 ratio drops below 1.43 and the guard becomes overly permissive.

    ``min_fold`` (default 0.7) scales the averagine-expected ratio. When
    apex_intensities is None the intensity guard is skipped entirely.
    """
    NEUTRON = 1.003355
    AVERAGINE_SLOPE = 0.000509  # lambda = slope × M0_mz
    if not intervals:
        return intervals

    order = np.argsort([iv[2] for iv in intervals])
    sorted_ivs = [intervals[k] for k in order]
    apices = np.array([iv[2] for iv in sorted_ivs])

    if apex_intensities is not None:
        sorted_ints = np.asarray(apex_intensities, dtype=np.float64)[order]
    else:
        sorted_ints = None

    is_isotope = np.zeros(len(apices), dtype=bool)
    for i in range(1, len(apices)):
        for k in (1, 2):
            target = apices[i] - k * NEUTRON
            diffs = np.abs(apices[:i] - target)
            best_j = int(np.argmin(diffs))
            if diffs[best_j] <= tol_da:
                if sorted_ints is None:
                    is_isotope[i] = True
                    break
                lam = AVERAGINE_SLOPE * apices[best_j]
                if lam <= 0:
                    expected_ratio = 2.0 if k == 1 else 8.0
                elif k == 1:
                    # M0/M+1 from averagine Poisson: E[M0]/E[M+1] = 1/lambda
                    expected_ratio = 1.0 / lam
                else:
                    # M0/M+2 from averagine Poisson: E[M0]/E[M+2] = 2/lambda^2
                    # Much higher than 1/lambda — prevents false-positive k=2 flagging
                    expected_ratio = 2.0 / (lam ** 2)
                # Floor at 1.0: M0 must always be at least as intense as M+k.
                # If M+k > M0 in the mean spectrum (obs_ratio < 1) the pair is
                # inconsistent with a true isotope pattern — skip removal.
                threshold = max(expected_ratio * min_fold, 1.0)
                if sorted_ints[best_j] >= sorted_ints[i] * threshold:
                    is_isotope[i] = True
                    break

    return [iv for iv, flag in zip(sorted_ivs, is_isotope) if not flag]


def _filter_mass_defect(
    intervals: list[tuple[float, float, float]],
    halfwidth: float,
) -> list[tuple[float, float, float]]:
    """Senko-plot peptide corridor mass defect filter.

    Uses floor-based mass defect (always in [0, 1)) and an averagine-derived
    corridor centred on:
      expected_defect = 0.000509 × floor(neutral_mass)

    The slope +0.000509 comes from the averagine residue composition
    (C4.9 H7.8 N1.3 O1.5 S0.04 per 111 Da); it is positive because H atoms
    dominate the defect accumulation. Using round() instead of floor() flips
    the sign for neutrals with fractional part > 0.5 and produces a bimodal
    distribution that breaks the corridor test.

    With halfwidth=0.5 (default), all peaks pass — effectively disabled.
    Use halfwidth≈0.25 to filter lipids and matrix clusters while retaining
    all tryptic peptides in the 800–2000 Da range.
    """
    PROTON = 1.007276
    kept: list[tuple[float, float, float]] = []
    for mz_lo, mz_hi, mz_apex in intervals:
        neutral = mz_apex - PROTON
        nominal = int(np.floor(neutral))
        defect = neutral - nominal          # always in [0, 1)
        expected = 0.000509 * nominal
        if abs(defect - expected) <= halfwidth:
            kept.append((mz_lo, mz_hi, mz_apex))
    return kept


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

    working = mean_ints
    if config.baseline_correction:
        working = _subtract_baseline(mz_grid, working, config.baseline_window_ppm)

    smoothed = savgol_filter(
        working.astype(np.float64),
        config.smoothing_window,
        config.smoothing_polyorder,
    )
    smoothed = np.maximum(smoothed, 0.0)

    max_int = float(smoothed.max())
    if max_int == 0.0:
        return []

    # Detect peaks on the mean spectrum.
    # Use absolute height threshold (matches the paper's "4% relative intensity
    # threshold" = peaks above 4% of the base peak), not scipy topographic
    # prominence.  Topographic prominence can reject genuine peaks that are
    # surrounded by moderate neighbours even when their absolute height clearly
    # exceeds the threshold.
    if config.local_prominence_window_da > 0.0:
        # Adaptive threshold: 4% of the local max within a sliding window.
        # Reduces the effective threshold in low-signal regions (e.g. >1600 Da).
        from scipy.ndimage import maximum_filter1d as _mf1d
        resolution = float(mz_grid[1] - mz_grid[0]) if len(mz_grid) > 1 else config.mz_grid_resolution
        half_pts = max(1, int(round(config.local_prominence_window_da / resolution)))
        local_max = _mf1d(smoothed, size=2 * half_pts + 1)
        height_thresholds = config.peak_prominence * local_max
        peaks_all, _ = find_peaks(smoothed)
        peaks = peaks_all[smoothed[peaks_all] >= height_thresholds[peaks_all]]
    else:
        global_threshold = config.peak_prominence * max_int
        peaks, _ = find_peaks(smoothed, height=global_threshold)
    if len(peaks) == 0:
        return []

    # Detect valleys (local minima) as peaks in the negated signal
    valleys, _ = find_peaks(-smoothed)

    intervals: list[tuple[float, float, float]] = []
    for pk in peaks:
        # --- Apex m/z: raw grid point or picking-height centroid ---
        if config.picking_height > 0.0:
            threshold = config.picking_height * smoothed[pk]
            # Left crossing: walk left until below threshold, then interpolate
            li_cross = pk - 1
            while li_cross > 0 and smoothed[li_cross] >= threshold:
                li_cross -= 1
            if smoothed[li_cross + 1] > smoothed[li_cross]:
                t = (threshold - smoothed[li_cross]) / (smoothed[li_cross + 1] - smoothed[li_cross])
                left_mz = mz_grid[li_cross] + t * (mz_grid[li_cross + 1] - mz_grid[li_cross])
            else:
                left_mz = float(mz_grid[li_cross])
            # Right crossing: walk right until below threshold, then interpolate
            ri_cross = pk + 1
            while ri_cross < len(smoothed) - 1 and smoothed[ri_cross] >= threshold:
                ri_cross += 1
            if smoothed[ri_cross] < smoothed[ri_cross - 1]:
                t = (threshold - smoothed[ri_cross - 1]) / (smoothed[ri_cross] - smoothed[ri_cross - 1])
                right_mz = mz_grid[ri_cross - 1] + t * (mz_grid[ri_cross] - mz_grid[ri_cross - 1])
            else:
                right_mz = float(mz_grid[ri_cross])
            mz_apex = (left_mz + right_mz) / 2.0
        else:
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

    if config.normalize_rms:
        rms = float(np.sqrt(np.mean(ints.astype(np.float64) ** 2)))
        if rms > 0.0:
            ints = ints / rms
    elif config.normalize_tic:
        tic = float(np.sum(ints))
        if tic > 0.0:
            ints = ints / tic

    mzs = mzs.astype(np.float64)

    for j, (mz_lo, mz_hi, _) in enumerate(intervals):
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

        if config.calibrant_mzs:
            intervals = _recalibrate_intervals(intervals, config.calibrant_mzs, config.calibrant_tol_ppm)
            logger.info(f"  {len(intervals)} intervals after recalibration")

        if config.deisotope:
            n_before = len(intervals)
            apex_ints = np.array(
                [float(mean_ints[np.argmin(np.abs(mz_grid - iv[2]))]) for iv in intervals],
                dtype=np.float64,
            )
            intervals = _deisotope_intervals(intervals, config.deisotope_tol_da, apex_ints, config.deisotope_min_fold)
            logger.info(f"  {len(intervals)}/{n_before} intervals after deisotoping")

        if config.filter_mass_defect:
            n_before = len(intervals)
            intervals = _filter_mass_defect(intervals, config.mass_defect_halfwidth)
            logger.info(f"  {len(intervals)}/{n_before} intervals after mass defect filter")

        if len(intervals) == 0:
            return [], np.zeros((n_pixels, 0), dtype=np.float32), []

        # Pass 2: integrate each pixel
        logger.info("Pass 2: integrating pixels over intervals…")
        intensity_matrix = np.zeros((n_pixels, len(intervals)), dtype=np.float32)
        pixel_coords: list[tuple[int, int]] = []

        for i, (x, y, *_) in enumerate(parser.coordinates):
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
