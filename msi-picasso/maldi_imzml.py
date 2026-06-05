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
    # Deisotoping via ms_deisotope (remove isotope satellite peaks)
    deisotope: bool = False
    deisotope_averagine: str = "peptide"          # averagine model: peptide, glycopeptide, glycan, heparin
    deisotope_scorer: str = "MSDeconVFitter"      # MSDeconVFitter or PenalizedMSDeconVFitter
    deisotope_min_score: float = 10.0             # minimum MSDeconV fit score to accept an envelope
    deisotope_charge_range: tuple = field(default_factory=lambda: (1, 1))  # MALDI is predominantly [M+H]+
    deisotope_error_ppm: float = 15.0             # ppm error tolerance for isotope envelope fitting
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


def _merge_duplicate_intervals(
    intervals: list[tuple[float, float, float]],
    apex_intensities: np.ndarray,
    tol_da: float = 0.001,
) -> tuple[list[tuple[float, float, float]], np.ndarray]:
    """Merge near-identical interval apices within tol_da into a single interval.

    Intervals are sorted ascending by apex. Consecutive intervals whose weighted-mean
    apex separation is <= tol_da are merged: boundaries are expanded, apex is the
    intensity-weighted mean, intensity is the cluster maximum.
    """
    if not intervals:
        return intervals, apex_intensities

    order = np.argsort([iv[2] for iv in intervals])
    sorted_ivs = [intervals[k] for k in order]
    sorted_ints = np.asarray(apex_intensities, dtype=np.float64)[order]

    merged_ivs: list[tuple[float, float, float]] = []
    merged_ints: list[float] = []

    cur_lo, cur_hi, cur_apex = sorted_ivs[0]
    cur_int = float(sorted_ints[0])
    sum_int = cur_int
    weighted_apex = cur_apex * cur_int
    max_int = cur_int

    for i in range(1, len(sorted_ivs)):
        mz_lo, mz_hi, mz_apex = sorted_ivs[i]
        i_int = float(sorted_ints[i])
        cluster_apex = weighted_apex / sum_int if sum_int > 0 else cur_apex
        if (mz_apex - cluster_apex) <= tol_da:
            cur_lo = min(cur_lo, mz_lo)
            cur_hi = max(cur_hi, mz_hi)
            sum_int += i_int
            weighted_apex += mz_apex * i_int
            if i_int > max_int:
                max_int = i_int
        else:
            merged_ivs.append((cur_lo, cur_hi, weighted_apex / sum_int if sum_int > 0 else cur_apex))
            merged_ints.append(max_int)
            cur_lo, cur_hi, cur_apex = mz_lo, mz_hi, mz_apex
            cur_int = i_int
            sum_int = i_int
            weighted_apex = mz_apex * i_int
            max_int = i_int

    merged_ivs.append((cur_lo, cur_hi, weighted_apex / sum_int if sum_int > 0 else cur_apex))
    merged_ints.append(max_int)
    return merged_ivs, np.array(merged_ints, dtype=np.float64)


def _deisotope_intervals(
    intervals: list[tuple[float, float, float]],
    apex_intensities: np.ndarray,
    averagine: str = "peptide",
    scorer: str = "MSDeconVFitter",
    min_score: float = 10.0,
    charge_range: tuple = (1, 1),
    error_ppm: float = 15.0,
    return_diagnostics: bool = False,
) -> list[tuple[float, float, float]] | tuple[list[tuple[float, float, float]], list[dict]]:
    """Remove isotope satellite peaks using ms_deisotope.

    Feeds the interval apex positions and intensities as a peak list into
    ms_deisotope's AveraginePeakDependenceGraphDeconvoluter. Peaks identified
    as M+1, M+2, … of a higher-scoring monoisotopic peak are removed.
    Peaks that cannot be fitted into any isotope envelope are kept (conservative:
    no false-positive removal of isolated signals).

    Behavioral differences vs the previous custom implementation:
    - Error tolerance is ppm-based (error_ppm) rather than Da-based. At 1000 Da,
      15 ppm ≈ 0.015 Da (stricter than the old 0.15 Da default).
    - Filtering is based on MSDeconV fit score, not hand-coded k-specific averagine
      ratio guards. A fitted envelope where M+k > M0 scores poorly and will not
      exceed min_score, giving the same effect as the old "absolute floor at 1.0".
    - Spatial ion-image similarity is no longer used.
    - charge_range defaults to (1, 1): MALDI data is predominantly [M+H]+.
    """
    from ms_deisotope import deconvolute_peaks
    import ms_deisotope.averagine as avg_module
    from ms_deisotope.scoring import MSDeconVFitter, PenalizedMSDeconVFitter

    if not intervals:
        if return_diagnostics:
            return [], []
        return []

    _scorer_map = {
        "MSDeconVFitter": MSDeconVFitter(min_score),
        "PenalizedMSDeconVFitter": PenalizedMSDeconVFitter(min_score),
    }
    if scorer not in _scorer_map:
        raise ValueError(f"Unknown deisotope scorer {scorer!r}. Choose from: {list(_scorer_map)}")

    averagine_model = getattr(avg_module, averagine, None)
    if averagine_model is None:
        raise ValueError(f"Unknown averagine model {averagine!r}. Choose from: peptide, glycopeptide, glycan, heparin")

    # Scale intensities to [0, 1000] range before passing to ms_deisotope.
    # ms_deisotope has an internal minimum absolute intensity threshold (~9 counts);
    # mean spectra from RMS-normalized MALDI data are in the 0.5–2.0 range and
    # would all fall below that floor, causing deconvolute_peaks to return nothing.
    # Scaling preserves all intensity ratios (MSDeconV scoring is ratio-based).
    max_int = float(np.max(apex_intensities)) if len(apex_intensities) else 1.0
    scale = 1000.0 / max_int if max_int > 0 else 1.0
    peaks = [(iv[2], float(apex_intensities[i]) * scale) for i, iv in enumerate(intervals)]

    result = deconvolute_peaks(
        peaks,
        charge_range=charge_range,
        error_tolerance=error_ppm * 1e-6,
        averagine=averagine_model,
        scorer=_scorer_map[scorer],
        use_quick_charge=True,
    )

    # Build a set of observed input m/z values for phantom-M+1 guard below.
    observed_input_mzs = {iv[2] for iv in intervals}

    # Collect m/z values of isotope satellites (envelope positions > 0).
    # Peaks not assigned to any envelope are kept (isolated monoisotopics).
    # Two guards prevent false-positive satellite removal:
    #
    # 1. M+1 > M0: skip envelopes where the M+1 position is more intense than M0.
    #    MSDeconV scoring is scale-dependent and can accept physically inconsistent
    #    envelopes after intensity rescaling; we enforce the ratio check explicitly.
    #
    # 2. Phantom M+1: skip removal of M+2 (k≥2) when the M+1 position is a phantom
    #    peak not present in the input list. Without an observed M+1, a M0/M+2
    #    connection likely represents two unrelated peptides ~2 Da apart.
    satellite_mzs: set[float] = set()
    parent_score: dict[float, float] = {}   # satellite_mz → MSDeconV score of its parent
    parent_charge: dict[float, int] = {}
    for dp in result.peak_set:
        env = dp.envelope
        if len(env) >= 2 and env[1].intensity > env[0].intensity:
            continue  # guard 1: M+1 > M0
        # Check if M+1 is an observed peak (not a phantom inserted by ms_deisotope).
        m1_observed = len(env) < 2 or any(
            abs(env[1].mz - obs) < error_ppm * 1e-6 * env[1].mz + 0.001
            for obs in observed_input_mzs
        )
        for k, ep in enumerate(env):
            if k == 0:
                continue
            if k >= 2 and not m1_observed:
                continue  # guard 2: skip M+2 if M+1 is phantom
            satellite_mzs.add(ep.mz)
            parent_score[ep.mz] = dp.score
            parent_charge[ep.mz] = dp.charge

    # Match satellites back to intervals with 1 mDa tolerance.
    # ms_deisotope returns the same float objects from the input peak list,
    # so exact equality usually holds; the 1 mDa guard covers edge cases.
    def _is_satellite(apex: float) -> bool:
        return any(abs(apex - s) < 0.001 for s in satellite_mzs)

    kept = [iv for iv in intervals if not _is_satellite(iv[2])]

    # Secondary pass: catch M+1 peaks that survived ms_deisotope but whose M0
    # is present and more intense. In heterogeneous single-cell MALDI-MSI, the
    # mean-spectrum M0/M+1 ratio is compressed toward 1:1 relative to the
    # averagine prediction (cell-to-cell variation inflates the mean M+1), so
    # ms_deisotope often declines to form an envelope even for genuine satellites.
    # A simple intensity floor (M0 > M+1) is sufficient for the mean spectrum:
    # if the M+1 is genuinely a different peptide, it will typically be at least
    # as intense as M0, so the floor leaves it untouched.
    NEUTRON_MASS = 1.003355
    apex_int_by_mz = {iv[2]: float(apex_intensities[i]) for i, iv in enumerate(intervals)}
    kept_mz_set = {iv[2] for iv in kept}
    ratio_satellite_mzs: set[float] = set()
    for iv in kept:
        mz = iv[2]
        mz_int = apex_int_by_mz.get(mz, 0.0)
        tol = error_ppm * 1e-6 * mz + 0.001
        m0_target = mz - NEUTRON_MASS
        for m0_mz in kept_mz_set:
            if abs(m0_mz - m0_target) <= tol and apex_int_by_mz.get(m0_mz, 0.0) > mz_int:
                ratio_satellite_mzs.add(mz)
                break
    kept = [iv for iv in kept if iv[2] not in ratio_satellite_mzs]

    if not return_diagnostics:
        return kept

    # Build per-interval diagnostic records
    diagnostics: list[dict] = []
    for iv in intervals:
        apex = iv[2]
        sat = next((s for s in satellite_mzs if abs(apex - s) < 0.001), None)
        if sat is not None:
            removed, score, charge = True, parent_score.get(sat), parent_charge.get(sat)
        elif apex in ratio_satellite_mzs:
            removed, score, charge = True, None, None  # removed by secondary ratio pass
        else:
            removed, score, charge = False, None, None
        diagnostics.append({"apex_mz": apex, "removed": removed, "score": score, "charge": charge})
    return kept, diagnostics


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
# Ion mobility helpers
# ---------------------------------------------------------------------------


def one_over_k0_to_ccs(
    one_over_k0: np.ndarray | float,
    mz: np.ndarray | float,
    charge: int = 1,
    mass_gas: float = 28.013,
    temp: float = 31.85,
    t_diff: float = 273.15,
) -> np.ndarray | float:
    """Convert reduced ion mobility (1/K0, Vs/cm²) to CCS (Å²).

    Uses the same Mason-Schamp formula and default constants as im2deep.utils.im2ccs
    (N2 drift gas, 31.85 °C). Adapted from theGreatHerrLebert/ionmob.
    """
    # Same formula as im2deep.utils.im2ccs (SUMMARY_CONSTANT = 18509.8632163405)
    _SUMMARY_CONSTANT = 18509.8632163405
    one_over_k0 = np.asarray(one_over_k0, dtype=np.float64)
    mz = np.asarray(mz, dtype=np.float64)
    reduced_mass = (mz * charge * mass_gas) / (mz * charge + mass_gas)
    return (_SUMMARY_CONSTANT * charge) / (np.sqrt(reduced_mass * (temp + t_diff)) / one_over_k0)


def _parse_mobility_offsets(
    imzml_path: str,
) -> tuple[np.ndarray, np.ndarray, str] | None:
    """Parse per-spectrum mobility array offsets from imzML XML.

    Returns (offsets, lengths, dtype_str) or None if no mobilityArray group found.
    dtype_str is 'float32' or 'float64'.
    """
    import xml.etree.ElementTree as ET

    def _strip_ns(tag: str) -> str:
        return tag.split("}")[-1] if "}" in tag else tag

    tree = ET.parse(imzml_path)
    root = tree.getroot()

    # Find the referenceableParamGroup used for mobility arrays.
    mobility_group_id: str | None = None
    mobility_dtype = "float64"

    for elem in root.iter():
        if _strip_ns(elem.tag) == "referenceableParamGroup":
            group_id = elem.get("id", "")
            if "mobility" in group_id.lower():
                mobility_group_id = group_id
                for child in elem:
                    acc = child.get("accession", "")
                    if acc == "MS:1000521":   # 32-bit float
                        mobility_dtype = "float32"
                    elif acc == "MS:1000523": # 64-bit float
                        mobility_dtype = "float64"
                break

    if mobility_group_id is None:
        return None

    offsets: list[int] = []
    lengths: list[int] = []

    for spectrum_elem in root.iter():
        if _strip_ns(spectrum_elem.tag) != "spectrum":
            continue

        for bda in spectrum_elem:
            if _strip_ns(bda.tag) != "binaryDataArray":
                continue

            is_mobility = False
            offset: int | None = None
            length: int | None = None

            for child in bda:
                child_tag = _strip_ns(child.tag)
                if child_tag == "referenceableParamGroupRef":
                    if child.get("ref", "") == mobility_group_id:
                        is_mobility = True
                elif child_tag == "cvParam":
                    acc = child.get("accession", "")
                    val = child.get("value")
                    if acc == "IMS:1000102" and val is not None:
                        offset = int(val)
                    elif acc == "IMS:1000103" and val is not None:
                        length = int(val)

            if is_mobility and offset is not None and length is not None:
                offsets.append(offset)
                lengths.append(length)
                break  # only one mobility array per spectrum

    if not offsets:
        return None

    return np.array(offsets, dtype=np.int64), np.array(lengths, dtype=np.int64), mobility_dtype


# ---------------------------------------------------------------------------
# Ion image reconstruction helpers
# ---------------------------------------------------------------------------


def reconstruct_ion_images_from_intervals(
    intensity_matrix: np.ndarray,
    pixel_coords: list[tuple[int, int]],
    n_intervals: int,
) -> np.ndarray:
    """Reconstruct a (n_intervals, H, W) ion image array from the interval matrix.

    Parameters
    ----------
    intensity_matrix
        Shape (n_pixels, n_intervals), float32.  Each row is one pixel; each
        column is one SCiLS interval.
    pixel_coords
        0-based (x, y) tuples, same row order as intensity_matrix.
    n_intervals
        Number of intervals (= intensity_matrix.shape[1]).

    Returns
    -------
    np.ndarray
        Shape (n_intervals, H, W), float32.
    """
    if n_intervals == 0 or len(pixel_coords) == 0:
        return np.zeros((0, 1, 1), dtype=np.float32)

    coords = np.asarray(pixel_coords, dtype=np.int32)
    xs = coords[:, 0]
    ys = coords[:, 1]
    W = int(xs.max()) + 1
    H = int(ys.max()) + 1

    ion_images = np.zeros((n_intervals, H, W), dtype=np.float32)
    # Vectorised scatter: intensity_matrix.T has shape (n_intervals, n_pixels)
    ion_images[:, ys, xs] = intensity_matrix.T
    return ion_images


def build_envelopes_from_intervals(
    intervals: list[tuple[float, float, float]],
    intensity_matrix: np.ndarray,
) -> dict[float, list[float]]:
    """Build approximate maldi_envelopes from SCiLS interval data.

    For each interval apex m0, searches for M+1 and M+2 intervals within 0.5 Da
    and records their mean intensities.  Intervals not found within tolerance get
    mean intensity 0.0.

    Parameters
    ----------
    intervals
        List of (mz_start, mz_end, mz_apex) tuples.
    intensity_matrix
        Shape (n_pixels, n_intervals), float32.

    Returns
    -------
    dict mapping float(mz_apex) → [m0_mean, m1_mean, m2_mean].
    """
    NEUTRON = 1.003355
    if not intervals:
        return {}

    apices = np.array([iv[2] for iv in intervals])
    mean_ints = intensity_matrix.mean(axis=0)  # (n_intervals,)

    # Sort apices for efficient nearest-neighbour search
    order = np.argsort(apices)
    sorted_apices = apices[order]
    sorted_means = mean_ints[order]

    def _find_nearest_mean(targets: np.ndarray, tol: float = 0.5) -> np.ndarray:
        idx = np.searchsorted(sorted_apices, targets)
        idx = np.clip(idx, 0, len(sorted_apices) - 1)
        idx_prev = np.clip(idx - 1, 0, len(sorted_apices) - 1)
        diff_curr = np.abs(sorted_apices[idx] - targets)
        diff_prev = np.abs(sorted_apices[idx_prev] - targets)
        best_idx = np.where(diff_curr <= diff_prev, idx, idx_prev)
        best_diff = np.minimum(diff_curr, diff_prev)
        return np.where(best_diff < tol, sorted_means[best_idx], 0.0)

    m1_means = _find_nearest_mean(apices + NEUTRON)
    m2_means = _find_nearest_mean(apices + 2 * NEUTRON)

    return {
        float(apex): [float(m0_mean), float(m1_mean), float(m2_mean)]
        for apex, m0_mean, m1_mean, m2_mean in zip(apices, mean_ints, m1_means, m2_means)
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def extract_scils_features(
    imzml_path: str,
    config: Optional[SCiLSConfig] = None,
    output_dir: Optional[str] = None,
    visualize: bool = False,
) -> tuple[list[tuple[float, float, float]], np.ndarray, list[tuple[int, int]], np.ndarray | None]:
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
    mean_1_over_k0 : np.ndarray or None
        Intensity-weighted mean 1/K0 per interval (Vs/cm²), or None when
        no mobility binary arrays are present in the imzML file.
    """
    # Patch pyimzml to tolerate CV params with no value attribute in
    # mobilityArray referenceableParamGroups (pyimzml 1.5.5 crash).
    import pyimzml.ontology.ontology as _pyo
    _orig_convert = _pyo.convert_xml_value
    _pyo.convert_xml_value = lambda dtype, v: None if v is None else _orig_convert(dtype, v)

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
            return [], np.zeros((0, 0), dtype=np.float32), [], None

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
            intervals, apex_ints = _merge_duplicate_intervals(intervals, apex_ints)
            intervals = _deisotope_intervals(
                intervals, apex_ints,
                averagine=config.deisotope_averagine,
                scorer=config.deisotope_scorer,
                min_score=config.deisotope_min_score,
                charge_range=config.deisotope_charge_range,
                error_ppm=config.deisotope_error_ppm,
            )
            logger.info(f"  {len(intervals)}/{n_before} intervals after deisotoping")

        if config.filter_mass_defect:
            n_before = len(intervals)
            intervals = _filter_mass_defect(intervals, config.mass_defect_halfwidth)
            logger.info(f"  {len(intervals)}/{n_before} intervals after mass defect filter")

        if len(intervals) == 0:
            return [], np.zeros((n_pixels, 0), dtype=np.float32), [], None

        # Check for per-spectrum mobility arrays before Pass 2
        ibd_path = imzml_path.rsplit(".", 1)[0] + ".ibd"
        mob_offsets_result = _parse_mobility_offsets(imzml_path)
        has_mobility = (
            mob_offsets_result is not None
            and len(mob_offsets_result[0]) == n_pixels
            and os.path.exists(ibd_path)
        )

        if has_mobility:
            mob_offsets, mob_lengths, mob_dtype = mob_offsets_result
            mob_weight_sum = np.zeros(len(intervals), dtype=np.float64)
            mob_int_sum = np.zeros(len(intervals), dtype=np.float64)
            logger.info("  Mobility arrays detected — accumulating 1/K0 during Pass 2")

        # Pass 2: integrate each pixel
        logger.info("Pass 2: integrating pixels over intervals…")
        intensity_matrix = np.zeros((n_pixels, len(intervals)), dtype=np.float32)
        pixel_coords: list[tuple[int, int]] = []

        ibd_fh = open(ibd_path, "rb") if has_mobility else None
        try:
            for i, (x, y, *_) in enumerate(parser.coordinates):
                pixel_coords.append((x - 1, y - 1))  # 1-based → 0-based
                mzs, ints = parser.getspectrum(i)
                mzs = np.asarray(mzs, dtype=np.float64)
                ints = np.asarray(ints, dtype=np.float64)
                intensity_matrix[i] = _integrate_pixel(mzs, ints, intervals, config)

                if has_mobility and ibd_fh is not None:
                    n_pts = int(mob_lengths[i])
                    itemsize = 4 if mob_dtype == "float32" else 8
                    ibd_fh.seek(int(mob_offsets[i]))
                    mob = np.frombuffer(ibd_fh.read(n_pts * itemsize), dtype=mob_dtype).astype(np.float64)
                    for j, (mz_lo, mz_hi, _) in enumerate(intervals):
                        lo = int(np.searchsorted(mzs, mz_lo, side="left"))
                        hi = int(np.searchsorted(mzs, mz_hi, side="right"))
                        if lo < hi and lo < len(mob):
                            hi_m = min(hi, len(mob))
                            seg_ints = ints[lo:hi_m]
                            seg_mob = mob[lo:hi_m]
                            mob_weight_sum[j] += float((seg_ints * seg_mob).sum())
                            mob_int_sum[j] += float(seg_ints.sum())
        finally:
            if ibd_fh is not None:
                ibd_fh.close()

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

    if has_mobility:
        mean_1_over_k0_full = np.where(
            mob_int_sum > 0, mob_weight_sum / mob_int_sum, np.nan
        )
        mean_1_over_k0: np.ndarray | None = mean_1_over_k0_full[keep]
    else:
        mean_1_over_k0 = None

    if visualize and output_dir is not None:
        _visualize_scils(
            mz_grid, mean_ints, intervals, intensity_matrix, pixel_coords,
            output_dir=output_dir,
        )

    return intervals, intensity_matrix, pixel_coords, mean_1_over_k0
