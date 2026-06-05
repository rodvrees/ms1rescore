"""
Extract MALDI-MSI features from raw Bruker .d/TSF data via imzy.

Three steps:
  1. detect_features   — find consensus m/z peaks across all pixels
  2. extract_ion_images — extract a 3D ion image array for those peaks
  3. compute_spatial_features — fraction_detected, mean_intensity, CV, Moran's I

All three are orchestrated by extract_maldi_data(), which is the public entry point.
"""

import logging
import os

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
    Detect consensus m/z features by per-pixel logarithmic histogram binning.

    Uses O(n_bins) memory regardless of the total number of peaks across all
    pixels.  Each pixel's peaks are accumulated into fine bins (ppm_bin/4 wide)
    in a single streaming pass.  A greedy merge step then re-applies the
    ppm_bin grouping rule on the (small) set of non-empty bin centroids,
    matching the semantics of the original collect-then-sort algorithm while
    avoiding the large intermediate peak arrays.

    Parameters
    ----------
    reader
        An imzy reader object.  Must support ``reader.n_pixels``,
        ``reader.mz_min``, ``reader.mz_max``, and
        ``reader.spectra_iter(silent=False)``.
    ppm_bin
        Feature grouping tolerance in ppm.  Default 5.0.
    min_fraction
        Minimum fraction of pixels a feature must be detected in.
        Default 0.01 (1 %).

    Returns
    -------
    np.ndarray
        Sorted 1D float64 array of feature m/z values.
    """
    n_pixels = reader.n_pixels
    min_count = max(1, int(min_fraction * n_pixels))

    mz_ref = float(getattr(reader, "mz_min", 100.0))
    if mz_ref <= 0:
        mz_ref = 100.0
    mz_top = float(getattr(reader, "mz_max", 4000.0))

    # Fine bins at ppm_bin/4 resolution: each real feature spans ≤4 fine bins,
    # preventing the original ppm_bin-wide bins from splitting features at edges.
    fine_ppm = ppm_bin / 4.0
    fine_log_width = np.log1p(fine_ppm * 1e-6)
    n_bins = int(np.ceil(np.log(mz_top / mz_ref) / fine_log_width)) + 2
    log_mz_ref = np.log(mz_ref)

    # Three arrays at O(n_bins × 8 bytes) — ~26 MB for a typical MALDI range.
    bin_pixel_count = np.zeros(n_bins, dtype=np.int32)
    bin_intensity_sum = np.zeros(n_bins, dtype=np.float64)
    bin_mz_int_sum = np.zeros(n_bins, dtype=np.float64)

    n_total_peaks = 0
    for _px, (mzs, ints) in enumerate(reader.spectra_iter(silent=False)):
        if len(mzs) == 0:
            continue
        mzs_f = np.asarray(mzs, dtype=np.float64)
        ints_f = np.asarray(ints, dtype=np.float64)
        valid = mzs_f > 0
        if not valid.all():
            mzs_f = mzs_f[valid]
            ints_f = ints_f[valid]
        if len(mzs_f) == 0:
            continue

        n_total_peaks += len(mzs_f)
        bin_idx = np.floor((np.log(mzs_f) - log_mz_ref) / fine_log_width).astype(np.int32)
        np.clip(bin_idx, 0, n_bins - 1, out=bin_idx)

        # unique_bins: distinct bins hit this pixel; inverse: maps each peak → position in unique_bins.
        # bincount on inverse is O(n_peaks), not O(n_bins) — avoids 1M-element array per pixel.
        unique_bins, inverse = np.unique(bin_idx, return_inverse=True)
        n_uniq = len(unique_bins)

        bin_pixel_count[unique_bins] += 1
        bin_intensity_sum[unique_bins] += np.bincount(inverse, weights=ints_f, minlength=n_uniq)
        bin_mz_int_sum[unique_bins] += np.bincount(inverse, weights=mzs_f * ints_f, minlength=n_uniq)

    logger.info(
        f"  Processed {n_total_peaks:,} peaks from {n_pixels:,} pixels "
        f"({n_total_peaks / max(n_pixels, 1):.0f} peaks/pixel average)"
    )

    # Candidate bins: any bin that received at least one pixel contribution.
    cand_idx = np.where(bin_pixel_count > 0)[0]
    if len(cand_idx) == 0:
        logger.warning("No peaks found in any pixel — returning empty feature list.")
        return np.array([], dtype=np.float64)

    cand_int = bin_intensity_sum[cand_idx]
    cand_mz_int = bin_mz_int_sum[cand_idx]
    cand_cnt = bin_pixel_count[cand_idx]
    cand_mzs = np.where(
        cand_int > 0,
        cand_mz_int / cand_int,
        mz_ref * np.exp((cand_idx + 0.5) * fine_log_width),
    )

    # Greedy merge: group candidate-bin centroids within ppm_bin of each other
    # (same rule as the original algorithm).  Centroided spectra have at most
    # one peak per feature per pixel, so pixel counts sum without double-counting.
    feature_mzs: list[float] = []
    n = len(cand_mzs)
    i = 0
    while i < n:
        anchor = cand_mzs[i]
        j = i + 1
        while j < n and (cand_mzs[j] - anchor) / anchor * 1e6 <= ppm_bin:
            j += 1

        if int(cand_cnt[i:j].sum()) >= min_count:
            grp_int = cand_int[i:j].sum()
            grp_mz_int = cand_mz_int[i:j].sum()
            feature_mzs.append(
                float(grp_mz_int / grp_int) if grp_int > 0 else float(cand_mzs[i:j].mean())
            )
        i = j

    result = np.array(feature_mzs, dtype=np.float64)
    logger.info(
        f"  {len(result)} features detected "
        f"(ppm_bin={ppm_bin}, min_fraction={min_fraction})"
    )
    return result


# ---------------------------------------------------------------------------
# Profile mean spectrum (for SCiLS-style feature detection)
# ---------------------------------------------------------------------------


def _build_profile_mean_spectrum(reader, normalize_rms: bool = False, normalize_tic: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Return (mz_grid, mean_intensities) for aligned profile data.

    All pixels must share the same m/z axis (verified upstream via
    reader.is_centroid == False and imzy's aligned profile guarantee).
    """
    mz_grid = None
    acc = None
    count = 0
    for mzs, ints in reader.spectra_iter(silent=False):
        mzs = np.asarray(mzs, dtype=np.float64)
        ints = np.asarray(ints, dtype=np.float64)
        if normalize_rms:
            rms = float(np.sqrt(np.mean(ints ** 2)))
            if rms > 0.0:
                ints = ints / rms
        elif normalize_tic:
            tic = float(ints.sum())
            if tic > 0.0:
                ints = ints / tic
        if mz_grid is None:
            mz_grid = mzs
            acc = ints.copy()
        else:
            acc += ints
        count += 1
    if mz_grid is None or count == 0:
        return np.array([]), np.array([])
    return mz_grid, (acc / count).astype(np.float32)


# ---------------------------------------------------------------------------
# Ion image extraction
# ---------------------------------------------------------------------------


def _extract_centroid_fast(reader, feature_mzs: np.ndarray, ppm: float) -> np.ndarray | None:
    """
    Fast ion image extraction for Bruker TSF centroid data.

    The standard imzy path calls ``tsf_read_line_spectrum_v2`` (read raw
    indices) **and** ``tsf_index_to_mz`` (convert to m/z) for every pixel —
    two DLL round-trips each.  On a network filesystem at ~100 ms/call that
    is ~3 hours for 49 K pixels.

    This function pre-converts the feature m/z windows to raw spectral index
    windows **once** using the reference frame's calibration, then only calls
    ``tsf_read_line_spectrum_v2`` per pixel — halving the DLL round-trips.
    Peak-to-feature assignment is done with a vectorised binary search +
    ``np.bincount`` (no Python inner loop over features).

    The calibration approximation (reference frame for all pixels) introduces
    < 5 ppm systematic offset — well within the extraction window.

    Returns ``None`` if the reader does not expose the required DLL handles
    (non-Bruker readers fall back to ``get_ion_images``).
    """
    from ctypes import POINTER, c_double, c_float

    if not (
        hasattr(reader, "dll")
        and hasattr(reader, "handle")
        and hasattr(reader, "mz_to_index")
        and reader.is_centroid
    ):
        return None

    n_features = len(feature_mzs)
    n_pixels = reader.n_pixels

    ppm_factor = ppm * 1e-6
    feat_mz = np.asarray(feature_mzs, dtype=np.float64)

    # Sort features by m/z so idx_min / idx_max are monotone for searchsorted.
    sort_order = np.argsort(feat_mz)
    feat_mz_s = feat_mz[sort_order]
    mzs_min = feat_mz_s * (1.0 - ppm_factor)
    mzs_max = feat_mz_s * (1.0 + ppm_factor)

    # Convert m/z windows → raw spectral index windows using reference frame (once).
    frame_indices = getattr(reader, "frame_indices", None)
    ref_frame = int(frame_indices[0]) if frame_indices is not None else 1
    idx_min = np.asarray(reader.mz_to_index(ref_frame, mzs_min), dtype=np.float64)
    idx_max = np.asarray(reader.mz_to_index(ref_frame, mzs_max), dtype=np.float64)

    # Inverse permutation: sorted-feature index → original feature index.
    inv_sort = np.empty(n_features, dtype=np.intp)
    inv_sort[sort_order] = np.arange(n_features, dtype=np.intp)

    output = np.zeros((n_pixels, n_features), dtype=np.float32)

    dll = reader.dll
    handle = reader.handle
    buf_size = max(4096, getattr(reader, "buffer_size_centroid", 1024))
    index_buf = np.empty(buf_size, dtype=np.float64)
    intensity_buf = np.empty(buf_size, dtype=np.float32)

    for px_i in range(n_pixels):
        frame_id = int(frame_indices[px_i]) if frame_indices is not None else px_i + 1

        while True:
            required = dll.tsf_read_line_spectrum_v2(
                handle,
                frame_id,
                index_buf.ctypes.data_as(POINTER(c_double)),
                intensity_buf.ctypes.data_as(POINTER(c_float)),
                buf_size,
            )
            if required < 0:
                break
            if required > buf_size:
                buf_size = required
                index_buf = np.empty(buf_size, dtype=np.float64)
                intensity_buf = np.empty(buf_size, dtype=np.float32)
            else:
                break

        if required <= 0:
            continue

        raw_idx = index_buf[:required]
        raw_int = intensity_buf[:required]

        # For each peak find the first feature whose idx_max >= peak_index.
        pf = np.searchsorted(idx_max, raw_idx, side="left")
        valid = (pf < n_features) & (raw_idx >= idx_min[pf])
        if not valid.any():
            continue

        orig = inv_sort[pf[valid]]
        output[px_i] += np.bincount(
            orig,
            weights=raw_int[valid].astype(np.float64),
            minlength=n_features,
        ).astype(np.float32)

    return reader.reshape_batch(output)


def _extract_profile_fast(reader, feature_mzs: np.ndarray, ppm: float) -> np.ndarray | None:
    """
    Fast vectorized ion image extraction for profile-mode MALDI data.

    Profile data has a **fixed m/z axis** shared across all pixels.  imzy's
    default path wraps 1,400 numpy index arrays into a ``numba.typed.List``
    on every one of the 49 K pixels (49 K × 1,400 typed-list entries), which
    dominates runtime.

    This function pre-computes the start/end bin indices for every feature
    window **once** on the shared m/z axis, then per pixel uses a cumulative-
    sum trick to accumulate all feature intensities in a single O(n_mz_points)
    pass — no inner Python loop over features, no numba typed-list overhead.

    Per pixel work:
      ``cumsum = np.cumsum(y)``              # O(n_mz_points), vectorised
      ``out[px] = cumsum[hi] - cumsum[lo]``  # O(n_features), vectorised

    Returns ``None`` when the reader reports centroid data (use
    ``_extract_centroid_fast`` instead) or when ``get_spectrum`` is
    unavailable.
    """
    if reader.is_centroid:
        return None

    try:
        mz_axis, _ = reader.get_spectrum(0)
    except Exception:
        return None

    mz_axis = np.asarray(mz_axis, dtype=np.float64)
    ppm_factor = ppm * 1e-6
    feat_mz = np.asarray(feature_mzs, dtype=np.float64)
    mz_min = feat_mz * (1.0 - ppm_factor)
    mz_max = feat_mz * (1.0 + ppm_factor)

    # Window bounds into the fixed profile m/z axis — computed once.
    lo = np.searchsorted(mz_axis, mz_min, side="left")
    hi = np.searchsorted(mz_axis, mz_max, side="right")

    n_pixels = reader.n_pixels
    n_features = len(feature_mzs)
    output = np.zeros((n_pixels, n_features), dtype=np.float32)
    cs = np.empty(len(mz_axis) + 1, dtype=np.float64)
    cs[0] = 0.0

    for px_i, (_, ints) in enumerate(reader.spectra_iter(silent=False)):
        if len(ints) == 0:
            continue
        np.cumsum(np.asarray(ints, dtype=np.float64), out=cs[1:])
        output[px_i] = (cs[hi] - cs[lo]).astype(np.float32)

    return reader.reshape_batch(output)


def _extract_profile_fast_multi(
    reader,
    feat_mzs_list: list[np.ndarray],
    ppm: float,
) -> list[np.ndarray] | None:
    """
    Single-pass profile extraction for multiple feature sets.

    Instead of 6 separate ``spectra_iter()`` calls (main + m1/m2/na/k/chca),
    all feature windows are concatenated, the Rust ``accumulate_profile_chunk``
    function accumulates intensities in parallel per chunk, and the flat output
    is split back into per-set 3-D arrays.

    Returns a list of ``(n_features_i, H, W)`` float32 arrays in the same
    order as ``feat_mzs_list``, or ``None`` if profile mode is unavailable.
    """
    if reader.is_centroid:
        return None
    try:
        mz_axis, _ = reader.get_spectrum(0)
    except Exception:
        return None

    try:
        from ms1rescore_rs import accumulate_profile_chunk as _rust_accum
        _use_rust = True
    except ImportError:
        _use_rust = False

    mz_axis = np.asarray(mz_axis, dtype=np.float64)
    n_mz = len(mz_axis)
    ppm_factor = ppm * 1e-6

    # Build lo/hi index arrays for every feature set and concatenate.
    all_lo: list[np.ndarray] = []
    all_hi: list[np.ndarray] = []
    set_sizes: list[int] = []
    for feat_mzs in feat_mzs_list:
        mzs = np.asarray(feat_mzs, dtype=np.float64)
        lo = np.searchsorted(mz_axis, mzs * (1.0 - ppm_factor), side="left")
        hi = np.searchsorted(mz_axis, mzs * (1.0 + ppm_factor), side="right")
        all_lo.append(lo)
        all_hi.append(hi)
        set_sizes.append(len(feat_mzs))

    cat_lo = np.concatenate(all_lo).tolist()
    cat_hi = np.concatenate(all_hi).tolist()
    n_total_feats = sum(set_sizes)

    n_pixels = reader.n_pixels
    output_flat = np.zeros((n_pixels, n_total_feats), dtype=np.float32)

    CHUNK = 512
    px_buf: list[np.ndarray] = []
    px_start = 0

    def _flush(buf: list[np.ndarray], start: int) -> None:
        chunk = np.ascontiguousarray(np.stack(buf, axis=0))  # (chunk, n_mz)
        if _use_rust:
            flat = _rust_accum(chunk, cat_lo, cat_hi)
            output_flat[start : start + len(buf)] = flat.reshape(len(buf), n_total_feats)
        else:
            cs = np.empty((len(buf), n_mz + 1), dtype=np.float64)
            cs[:, 0] = 0.0
            np.cumsum(chunk.astype(np.float64), axis=1, out=cs[:, 1:])
            for fi in range(n_total_feats):
                output_flat[start : start + len(buf), fi] = (
                    cs[:, cat_hi[fi]] - cs[:, cat_lo[fi]]
                )

    for px_i, (_, ints) in enumerate(reader.spectra_iter(silent=False)):
        arr = np.asarray(ints, dtype=np.float32)
        if len(arr) < n_mz:
            arr = np.pad(arr, (0, n_mz - len(arr)))
        elif len(arr) > n_mz:
            arr = arr[:n_mz]
        px_buf.append(arr)
        if len(px_buf) == CHUNK:
            _flush(px_buf, px_start)
            px_start += CHUNK
            px_buf = []

    if px_buf:
        _flush(px_buf, px_start)

    # Split flat output back into per-set 3-D arrays.
    results: list[np.ndarray] = []
    col = 0
    for i, size in enumerate(set_sizes):
        flat_set = output_flat[:, col : col + size]  # (n_pixels, n_features_i)
        results.append(reader.reshape_batch(flat_set))
        col += size

    return results


def extract_ion_images(
    reader,
    feature_mzs: np.ndarray,
    ppm: float = 20.0,
) -> np.ndarray:
    """
    Extract a 3D ion image array for the given feature m/z values.

    Tries fast extraction paths first, then falls back to imzy's
    ``get_ion_images()``:

    1. **Bruker TSF centroid** (``_extract_centroid_fast``): skips the
       per-pixel ``tsf_index_to_mz`` DLL call entirely; uses pre-converted
       raw index windows and ``np.bincount`` accumulation.
    2. **Profile mode** (``_extract_profile_fast``): replaces imzy's
       per-pixel ``numba.typed.List`` construction with a vectorised
       cumulative-sum trick on the fixed profile m/z axis.
    3. **Fallback**: ``reader.get_ion_images()`` — single streaming pass
       via imzy.

    Returns shape ``(n_features, height, width)``, dtype float32.
    """
    n = len(feature_mzs)
    logger.info(f"  Extracting ion images for {n} features at ±{ppm} ppm...")

    images = _extract_centroid_fast(reader, feature_mzs, ppm)
    if images is not None:
        logger.debug("  Used fast centroid extraction (skipped per-pixel tsf_index_to_mz).")
    else:
        images = _extract_profile_fast(reader, feature_mzs, ppm)
        if images is not None:
            logger.debug("  Used fast profile extraction (vectorised cumsum, no numba typed-list).")
        else:
            images = reader.get_ion_images(
                np.asarray(feature_mzs, dtype=np.float64),
                ppm=ppm,
                fill_value=0.0,
                silent=False,
            ).astype(np.float32)

    logger.info(f"  Ion image array: shape={images.shape}, dtype={images.dtype}")
    return images.astype(np.float32)


# ---------------------------------------------------------------------------
# Isotope envelope mean extraction (single streaming pass)
# ---------------------------------------------------------------------------


def _collect_pixel_spectra(reader) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Stream all pixels and return flat CSR arrays for Rust consumption.

    Returns
    -------
    flat_mzs : np.ndarray, float64
        All pixel m/z values concatenated.
    flat_ints : np.ndarray, float32
        All pixel intensities concatenated.
    pixel_offsets : list[int]
        CSR-style offsets; pixel i spans ``[offsets[i], offsets[i+1])``.
    """
    n_pixels = reader.n_pixels

    # Profile fast path: all pixels share a fixed m/z axis.
    # Pre-allocate flat arrays to avoid per-pixel list appends + final concatenate.
    if not reader.is_centroid:
        try:
            mz_axis, _ = reader.get_spectrum(0)
            mz_axis = np.asarray(mz_axis, dtype=np.float64)
            n_mz = len(mz_axis)
            flat_mzs = np.tile(mz_axis, n_pixels)
            flat_ints = np.empty(n_pixels * n_mz, dtype=np.float32)
            pixel_offsets = list(range(0, (n_pixels + 1) * n_mz, n_mz))
            for i, (_, ints) in enumerate(reader.spectra_iter(silent=True)):
                flat_ints[i * n_mz : (i + 1) * n_mz] = np.asarray(ints, dtype=np.float32)
            return flat_mzs, flat_ints, pixel_offsets
        except Exception:
            pass  # fall through to generic path

    # Centroid path: variable-length spectra, pre-allocated growing buffer.
    # Initial capacity: ~1000 peaks/pixel; doubles on overflow.
    cap = n_pixels * 1000
    flat_mzs = np.empty(cap, dtype=np.float64)
    flat_ints = np.empty(cap, dtype=np.float32)
    pixel_offsets = [0] * (n_pixels + 1)
    pos = 0
    for i, (mzs, ints) in enumerate(reader.spectra_iter(silent=True)):
        mz_arr = np.asarray(mzs, dtype=np.float64)
        int_arr = np.asarray(ints, dtype=np.float32)
        n = len(mz_arr)
        if pos + n > cap:
            cap = max(pos + n, cap * 2)
            new_mzs = np.empty(cap, dtype=np.float64)
            new_ints = np.empty(cap, dtype=np.float32)
            new_mzs[:pos] = flat_mzs[:pos]
            new_ints[:pos] = flat_ints[:pos]
            flat_mzs = new_mzs
            flat_ints = new_ints
        flat_mzs[pos : pos + n] = mz_arr
        flat_ints[pos : pos + n] = int_arr
        pos += n
        pixel_offsets[i + 1] = pos
    return flat_mzs[:pos].copy(), flat_ints[:pos].copy(), pixel_offsets


def _compute_isotope_means_python(
    flat_mzs: np.ndarray,
    flat_ints: np.ndarray,
    pixel_offsets: list[int],
    target_mzs: np.ndarray,
    ppm_tolerance: float,
) -> np.ndarray:
    """Pure-Python fallback for compute_maldi_isotope_means (no Rust required).

    Uses window-sum aggregation to match the RAM path (_extract_centroid_fast
    uses np.bincount weighted sum) and the fixed Rust path.
    O(n_pixels * n_targets) with small constant — acceptable for Rust-absent envs.
    """
    n_targets = len(target_mzs)
    n_pixels = len(pixel_offsets) - 1
    sums = np.zeros(n_targets, dtype=np.float64)
    tols = target_mzs * ppm_tolerance * 1e-6
    lo_bounds = target_mzs - tols
    hi_bounds = target_mzs + tols

    for px in range(n_pixels):
        lo = pixel_offsets[px]
        hi = pixel_offsets[px + 1]
        if lo == hi:
            continue
        mzs = flat_mzs[lo:hi]
        ints = flat_ints[lo:hi].astype(np.float64)

        # Window-sum: for each target find all peaks within [mz-tol, mz+tol]
        # and sum their intensities.
        lo_idxs = np.searchsorted(mzs, lo_bounds, side="left")
        hi_idxs = np.searchsorted(mzs, hi_bounds, side="right")
        for ti in range(n_targets):
            if lo_idxs[ti] < hi_idxs[ti]:
                sums[ti] += ints[lo_idxs[ti]:hi_idxs[ti]].sum()

    return sums / max(n_pixels, 1)


def compute_isotope_envelope_means(
    reader,
    feature_mzs: np.ndarray,
    extraction_ppm: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute per-feature mean intensities for M0, M+1, and M+2 in one pass.

    Streams all pixels exactly once (via ``reader.spectra_iter``), builds flat
    CSR arrays, and dispatches to the Rust ``compute_maldi_isotope_means``
    function if available — falling back to vectorised Python otherwise.

    Returns
    -------
    m0_means, m1_means, m2_means : np.ndarray, shape (n_features,), float32
        Spatial-mean intensity for each isotope peak.
    """
    _NEUTRON = 1.003355
    n_feat = len(feature_mzs)

    logger.info("  Streaming pixel spectra for M+1/M+2 isotope envelope extraction...")
    flat_mzs, flat_ints, pixel_offsets = _collect_pixel_spectra(reader)
    n_pixels = len(pixel_offsets) - 1
    logger.debug(f"    Collected {n_pixels} pixels, {len(flat_mzs):,} total peaks")

    # M0: computed from the flat arrays directly (avoids relying on ion_images).
    # M+1 and M+2: extracted together in the same Rust call.
    all_targets = np.concatenate([
        feature_mzs,
        feature_mzs + _NEUTRON,
        feature_mzs + 2 * _NEUTRON,
    ])  # shape (3 * n_feat,)

    try:
        from ms1rescore_rs import compute_maldi_isotope_means as _rs_means
        means = np.asarray(
            _rs_means(flat_mzs, flat_ints, pixel_offsets, all_targets.tolist(), extraction_ppm),
            dtype=np.float64,
        )
        logger.debug("  compute_maldi_isotope_means: used Rust implementation")
    except (ImportError, AttributeError):
        means = _compute_isotope_means_python(
            flat_mzs, flat_ints, pixel_offsets, all_targets, extraction_ppm
        )
        logger.debug("  compute_maldi_isotope_means: used Python fallback")

    # Free the large flat arrays as soon as Rust is done.
    del flat_mzs, flat_ints

    m0_means = means[:n_feat].astype(np.float32)
    m1_means = means[n_feat:2 * n_feat].astype(np.float32)
    m2_means = means[2 * n_feat:].astype(np.float32)
    return m0_means, m1_means, m2_means


# ---------------------------------------------------------------------------
# Spatial feature computation
# ---------------------------------------------------------------------------


def _queen_w_sum(H: int, W: int) -> float:
    """Total queen's-contiguity weight for an H×W grid with zero boundary."""
    ones = np.ones((H, W), dtype=np.float64)
    pad = np.pad(ones, 1, mode="constant", constant_values=0.0)
    nsum = (
        pad[0:H, 0:W] + pad[0:H, 1:W+1] + pad[0:H, 2:W+2]
        + pad[1:H+1, 0:W] + pad[1:H+1, 2:W+2]
        + pad[2:H+2, 0:W] + pad[2:H+2, 1:W+1] + pad[2:H+2, 2:W+2]
    )
    return float(nsum.sum())


def _compute_chunk(
    chunk_images: np.ndarray,
    chunk_mzs: np.ndarray,
    n_pixels_total: int,
) -> pd.DataFrame:
    """
    Compute spatial statistics for a contiguous batch of features.

    Module-level so it is picklable by ``ProcessPoolExecutor``.
    Each worker calls this independently on its slice of ``ion_images``.

    Parameters
    ----------
    chunk_images
        Shape ``(n_chunk, H, W)``, dtype float32.
    chunk_mzs
        1D float64 array of m/z values aligned with ``chunk_images``.
    n_pixels_total
        Total measured pixels used for ``fraction_detected`` denominator.
    """
    n_features, H, W = chunk_images.shape
    n_img = H * W

    flat32 = chunk_images.reshape(n_features, n_img)
    if flat32.dtype != np.float32:
        flat32 = flat32.astype(np.float32)

    # --- Intensity statistics ---
    n_det = (flat32 > 0).sum(axis=1)
    intensity_sum = flat32.sum(axis=1, dtype=np.float64)
    sum_sq = np.einsum("ij,ij->i", flat32, flat32, dtype=np.float64)
    mean_int = np.where(n_det > 0, intensity_sum / n_det, 0.0)
    var = np.where(n_det > 0, sum_sq / n_det - mean_int ** 2, 0.0)
    cv = np.where(mean_int > 0, np.sqrt(np.maximum(var, 0.0)) / mean_int, 0.0)
    frac = n_det / n_pixels_total if n_pixels_total > 0 else np.zeros(n_features)

    # --- p90 of nonzero pixels ---
    p90_idx = np.clip(n_img - n_det + (0.9 * n_det).astype(int), 0, n_img - 1)
    if n_det.max() > 0:
        sorted32 = np.sort(flat32, axis=1)
        p90 = np.where(
            n_det > 0,
            sorted32[np.arange(n_features), p90_idx].astype(np.float64),
            0.0,
        )
        del sorted32
    else:
        p90 = np.zeros(n_features, dtype=np.float64)

    # --- Moran's I via batched zero-padded neighbour sum (no scipy) ---
    imgs_f = flat32.reshape(n_features, H, W)
    dev32 = imgs_f - imgs_f.mean(axis=(1, 2), keepdims=True)

    pad = np.pad(dev32, ((0, 0), (1, 1), (1, 1)), mode="constant", constant_values=0.0)
    neighbor_dev = pad[:, 0:H, 0:W].copy()
    neighbor_dev += pad[:, 0:H, 1:W+1]
    neighbor_dev += pad[:, 0:H, 2:W+2]
    neighbor_dev += pad[:, 1:H+1, 0:W]
    neighbor_dev += pad[:, 1:H+1, 2:W+2]
    neighbor_dev += pad[:, 2:H+2, 0:W]
    neighbor_dev += pad[:, 2:H+2, 1:W+1]
    neighbor_dev += pad[:, 2:H+2, 2:W+2]
    del pad

    numerators = (dev32 * neighbor_dev).sum(axis=(1, 2), dtype=np.float64)
    denominators = (dev32 * dev32).sum(axis=(1, 2), dtype=np.float64)
    del neighbor_dev, dev32

    w_sum = _queen_w_sum(H, W)
    morans_i = np.where(
        denominators > 1e-12,
        (n_img / w_sum) * numerators / denominators,
        0.0,
    )

    # --- Spatial Shannon entropy of the per-feature intensity distribution ---
    safe_sum = np.maximum(intensity_sum, 1e-12)[:, None]
    p = flat32.astype(np.float64) / safe_sum
    with np.errstate(divide="ignore", invalid="ignore"):
        spatial_entropy = -np.where(p > 0, p * np.log(p), 0.0).sum(axis=1)
    spatial_entropy = np.where(intensity_sum > 0, spatial_entropy, 0.0)

    log_mean_intensity = np.log1p(mean_int)

    return pd.DataFrame({
        "feature_mz": chunk_mzs.astype(np.float64),
        "n_pixels_detected": n_det.astype(int),
        "fraction_detected": frac,
        "mean_intensity": mean_int,
        "log_mean_intensity": log_mean_intensity,
        "intensity_p90": p90,
        "intensity_sum": intensity_sum,
        "spatial_autocorrelation": morans_i,
        "spatial_entropy": spatial_entropy,
        "intensity_cv": cv,
    })


def compute_spatial_features(
    ion_images: np.ndarray,
    feature_mzs: np.ndarray,
    n_pixels_total: int,
    n_workers: int | None = None,
) -> pd.DataFrame:
    """
    Compute per-feature spatial statistics from the 3D ion image array.

    Features are split across ``n_workers`` threads (default: all CPU cores).
    NumPy releases the GIL during array operations, so threads run in parallel
    without pickling overhead.  Each thread runs ``_compute_chunk`` on a
    contiguous slice of ``ion_images`` (shared memory — no copying).

    Parameters
    ----------
    ion_images
        Shape ``(n_features, height, width)``, dtype float32.
    feature_mzs
        1D array of feature m/z values aligned with ``ion_images``.
    n_pixels_total
        Total number of measured pixels (``reader.n_pixels``).
    n_workers
        Number of worker threads.  ``None`` → ``os.cpu_count()``.
    """
    import os
    from concurrent.futures import ThreadPoolExecutor

    n_features = len(feature_mzs)
    if n_workers is None:
        n_workers = os.cpu_count() or 1
    n_workers = min(n_workers, n_features)

    if n_workers <= 1:
        df = _compute_chunk(ion_images, np.asarray(feature_mzs), n_pixels_total)
        logger.info(
            f"  Spatial features computed for {len(df)} features "
            f"(mean fraction_detected={df['fraction_detected'].mean():.3f})"
        )
        return df

    # Split feature axis into equal chunks — one per thread.
    splits = np.array_split(np.arange(n_features), n_workers)
    chunks = [
        (ion_images[idx], np.asarray(feature_mzs)[idx], n_pixels_total)
        for idx in splits
        if len(idx) > 0
    ]
    logger.info(
        f"  Computing spatial features for {n_features} features "
        f"across {len(chunks)} threads..."
    )

    with ThreadPoolExecutor(max_workers=len(chunks)) as pool:
        futures = [
            pool.submit(_compute_chunk, imgs, mzs, n_pix)
            for imgs, mzs, n_pix in chunks
        ]
        dfs = [f.result() for f in futures]

    df = pd.concat(dfs, ignore_index=True)
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

# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------


def extract_maldi_data(
    d_path: str,
    ppm_bin: float = 5.0,
    extraction_ppm: float = 25.0,
    matching_ppm: float = 20.0,
    min_fraction: float = 0.01,
    peak_prominence: float = 0.01,
    smoothing_window: int = 7,
    smoothing_polyorder: int = 2,
    ppm_tolerance: float = 10.0,
    min_interval_width_ppm: float = 2.0,
    normalize_rms: bool = False,
    baseline_correction: bool = False,
    baseline_window_ppm: float = 500.0,
    calibrant_mzs: list | None = None,
    calibrant_tol_ppm: float = 200.0,
    deisotope: bool = False,
    deisotope_averagine: str = "peptide",
    deisotope_scorer: str = "MSDeconVFitter",
    deisotope_min_score: float = 10.0,
    deisotope_charge_range: tuple = (1, 1),
    deisotope_error_ppm: float = 15.0,
    filter_mass_defect: bool = False,
    mass_defect_halfwidth: float = 0.5,
    picking_height: float = 0.75,
    local_prominence_window_da: float = 0.0,
    feature_mzs: np.ndarray | None = None,
    images_path: str | None = None,
    image_batch_size: int = 100,
    output_npz: str | None = None,
    output_spatial_tsv: str | None = None,
    output_dir: str | None = None,
    verbose: bool = False,
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
            "Install with: pip install MSI-PICASSO[maldi]"
        ) from exc

    logger.info(f"Opening MALDI dataset: {d_path}")
    reader = imzy.get_reader(d_path)
    logger.info(
        f"  {reader.n_pixels:,} pixels, image shape {reader.image_shape}, "
        f"m/z range {reader.mz_min:.1f}–{reader.mz_max:.1f}"
    )

    if feature_mzs is not None:
        logger.info(
            f"Step 1/3: Using {len(feature_mzs)} provided feature m/z values "
            f"(skipping detection)."
        )
    elif not reader.is_centroid:
        logger.info(
            "Step 1/3: Profile data detected — building mean spectrum for "
            "SCiLS-style interval detection..."
        )
        from msi_picasso.maldi_imzml import (
            SCiLSConfig, _detect_intervals,
            _recalibrate_intervals, _merge_duplicate_intervals,
            _deisotope_intervals, _filter_mass_defect,
        )

        scils_cfg = SCiLSConfig(
            min_pixel_fraction=min_fraction,
            peak_prominence=peak_prominence,
            smoothing_window=smoothing_window,
            smoothing_polyorder=smoothing_polyorder,
            ppm_tolerance=ppm_tolerance,
            min_interval_width_ppm=min_interval_width_ppm,
            normalize_rms=normalize_rms,
            baseline_correction=baseline_correction,
            baseline_window_ppm=baseline_window_ppm,
            calibrant_mzs=calibrant_mzs or [],
            calibrant_tol_ppm=calibrant_tol_ppm,
            deisotope=deisotope,
            deisotope_averagine=deisotope_averagine,
            deisotope_scorer=deisotope_scorer,
            deisotope_min_score=deisotope_min_score,
            deisotope_charge_range=deisotope_charge_range,
            deisotope_error_ppm=deisotope_error_ppm,
            filter_mass_defect=filter_mass_defect,
            mass_defect_halfwidth=mass_defect_halfwidth,
            picking_height=picking_height,
            local_prominence_window_da=local_prominence_window_da,
        )
        mz_grid, mean_ints = _build_profile_mean_spectrum(
            reader, normalize_rms=normalize_rms, normalize_tic=not normalize_rms
        )
        intervals = _detect_intervals(mz_grid, mean_ints, scils_cfg)
        if calibrant_mzs:
            intervals = _recalibrate_intervals(intervals, calibrant_mzs, calibrant_tol_ppm)
        if deisotope:
            apex_ints = np.array(
                [float(mean_ints[np.argmin(np.abs(mz_grid - iv[2]))]) for iv in intervals],
                dtype=np.float64,
            )
            intervals, apex_ints = _merge_duplicate_intervals(intervals, apex_ints)
            intervals = _deisotope_intervals(
                intervals, apex_ints,
                averagine=deisotope_averagine,
                scorer=deisotope_scorer,
                min_score=deisotope_min_score,
                charge_range=deisotope_charge_range,
                error_ppm=deisotope_error_ppm,
            )
        if filter_mass_defect:
            intervals = _filter_mass_defect(intervals, mass_defect_halfwidth)
        feature_mzs = np.array(
            [iv[2] for iv in intervals], dtype=np.float64
        )
        logger.info(f"  {len(feature_mzs)} intervals detected from mean spectrum")
        if verbose:
            logger.info(f"  Detected feature m/z values:\n  {feature_mzs}")
            if output_dir:
                features_txt = os.path.join(output_dir, "1_detected_features.txt")
                np.savetxt(features_txt, feature_mzs, fmt="%.6f")
                logger.info(f"  Saved detected features → {features_txt}")
    else:
        logger.info("Step 1/3: Detecting features...")
        feature_mzs = detect_features(
            reader, ppm_bin=ppm_bin, min_fraction=min_fraction
        )
        if verbose:
            logger.info(f"  Detected feature m/z values:\n  {feature_mzs}")
            # Save detected features to a text file for debugging.
            if output_dir:
                features_txt = os.path.join(output_dir, "1_detected_features.txt")
                np.savetxt(features_txt, feature_mzs, fmt="%.6f")
                logger.info(f"  Saved detected features → {features_txt}")

    if len(feature_mzs) == 0:
        raise ValueError(
            f"No features for {d_path!r}. "
            "If using detect_features, try lowering min_fraction. "
            "If using feature_mzs, ensure the sequences are valid and in range."
        )

    x_coords = reader.x_coordinates
    y_coords = reader.y_coordinates

    logger.info("Step 2/3: Extracting ion images...")
    _NEUTRON = 1.003355
    _ADDUCT_DELTAS = {"na": 21.9819, "k": 37.9559, "chca": 171.0320}
    _extra_keys = ["m1", "m2", "na", "k", "chca"]
    _extra_deltas = [
        _NEUTRON, 2.0 * _NEUTRON,
        _ADDUCT_DELTAS["na"], _ADDUCT_DELTAS["k"], _ADDUCT_DELTAS["chca"],
    ]
    _extra_raw: dict | None = None

    if images_path is None:
        logger.debug("  No images_path given, extracting full ion image array in RAM.")
        # Attempt a single spectra_iter() pass for all 6 feature sets (profile mode).
        _all_feat_mzs = [feature_mzs] + [feature_mzs + d for d in _extra_deltas]
        _multi = _extract_profile_fast_multi(reader, _all_feat_mzs, ppm=extraction_ppm)
        if _multi is not None:
            logger.debug(
                "  Used fast profile multi-extraction (single spectra_iter pass, Rust rayon)."
            )
            ion_images = _multi[0]
            _extra_raw = {k: _multi[i + 1] for i, k in enumerate(_extra_keys)}
        else:
            ion_images = extract_ion_images(reader, feature_mzs, ppm=extraction_ppm)

        if verbose:
            logger.info(f"  Ion images shape: {ion_images.shape}, dtype: {ion_images.dtype}")
            if output_dir:
                images_npy = os.path.join(output_dir, "2_ion_images.npy")
                np.save(images_npy, ion_images)
                logger.info(f"  Saved ion images → {images_npy}")

        logger.info("Step 3/3: Computing spatial features...")
        spatial_df = compute_spatial_features(
            ion_images, feature_mzs, reader.n_pixels
        )
        if verbose:
            logger.info(f"  Spatial features DataFrame:\n{spatial_df.head()}")
            if output_dir:
                spatial_csv = os.path.join(output_dir, "3_spatial_features.csv")
                spatial_df.to_csv(spatial_csv, index=False)
                logger.info(f"  Saved spatial features → {spatial_csv}")
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
        if _extra_raw is not None:
            _extra_raw = {k: v[detected_mask] for k, v in _extra_raw.items()}
        spatial_df = spatial_df[detected_mask].reset_index(drop=True)

    # Extract M+1, M+2, and adduct ion images for spatial colocalization features (E1/E2).
    # These peaks are typically absent from the feature list (monoisotopic-only detection),
    # so extracting images at their exact m/z positions is the only way to compute
    # isotopologue and adduct colocalization features.
    # Only done for the RAM path; the memmap path (very large datasets) skips this.
    if images_path is None:
        if _extra_raw is not None:
            # Already extracted in the single-pass profile extraction above.
            extra_ion_images: dict | None = _extra_raw
        else:
            logger.info("  Extracting M+1/M+2 and adduct isotopologue images for colocalization...")
            extra_ion_images = {
                "m1":   extract_ion_images(reader, feature_mzs + _NEUTRON,               ppm=extraction_ppm),
                "m2":   extract_ion_images(reader, feature_mzs + 2.0 * _NEUTRON,         ppm=extraction_ppm),
                "na":   extract_ion_images(reader, feature_mzs + _ADDUCT_DELTAS["na"],   ppm=extraction_ppm),
                "k":    extract_ion_images(reader, feature_mzs + _ADDUCT_DELTAS["k"],    ppm=extraction_ppm),
                "chca": extract_ion_images(reader, feature_mzs + _ADDUCT_DELTAS["chca"], ppm=extraction_ppm),
            }
    else:
        extra_ion_images = None

    if output_npz is not None:
        if output_dir and not os.path.isabs(str(output_npz)):
            npz_path = os.path.join(output_dir, output_npz)
        else:
            npz_path = output_npz
        os.makedirs(os.path.dirname(os.path.abspath(npz_path)), exist_ok=True)
        save_kwargs: dict = dict(
            mzs=feature_mzs,
            images=np.asarray(ion_images),
            x_coords=x_coords,
            y_coords=y_coords,
        )
        if extra_ion_images is not None:
            for key, arr in extra_ion_images.items():
                save_kwargs[f"extra_{key}"] = np.asarray(arr)
        np.savez_compressed(npz_path, **save_kwargs)
        logger.info(f"  Saved NPZ → {npz_path}")

    if output_spatial_tsv is not None:
        if output_dir and not os.path.isabs(str(output_spatial_tsv)):
            tsv_path = os.path.join(output_dir, output_spatial_tsv)
        else:
            tsv_path = output_spatial_tsv
        os.makedirs(os.path.dirname(os.path.abspath(tsv_path)), exist_ok=True)
        spatial_df.to_csv(tsv_path, sep="\t", index=False)
        logger.info(f"  Saved spatial features → {tsv_path}")

    # --- Compute MALDI isotope envelopes (M0/M+1/M+2 mean spatial intensity) ---
    logger.info("Computing MALDI isotope envelopes...")
    if extra_ion_images is not None:
        # RAM path: ion_images and m1/m2 images are already in memory — compute
        # spatial means directly without a second streaming pass.
        # sum / n_pixels matches the divisor used by compute_maldi_isotope_means.
        n_px = reader.n_pixels
        m0_means = (ion_images.sum(axis=(1, 2)) / n_px).astype(np.float32)
        m1_means = (extra_ion_images["m1"].sum(axis=(1, 2)) / n_px).astype(np.float32)
        m2_means = (extra_ion_images["m2"].sum(axis=(1, 2)) / n_px).astype(np.float32)
        logger.debug("  Computed envelope means from in-memory ion images (no extra streaming pass).")
    else:
        # Memmap path: images not fully in RAM; stream pixels once via Rust.
        m0_means, m1_means, m2_means = compute_isotope_envelope_means(
            reader, feature_mzs, extraction_ppm
        )
    maldi_envelopes = {
        float(mz): [float(m0), float(m1), float(m2)]
        for mz, m0, m1, m2 in zip(feature_mzs, m0_means, m1_means, m2_means)
        if m0 > 0
    }
    logger.info(f"  {len(maldi_envelopes)}/{len(feature_mzs)} features with M0 signal for envelope scoring")

    return feature_mzs, ion_images, extra_ion_images, spatial_df, maldi_envelopes
