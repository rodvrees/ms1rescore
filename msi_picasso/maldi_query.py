"""Raw MALDI querying: derive ion images directly from candidate m/z values.

In the standard pipeline the MALDI feature list is detected (or supplied) first
and candidates are matched against it.  Raw-query mode inverts this: candidates
are generated first, and their m/z values become the query grid for direct ion
image extraction from the raw Bruker ``.d`` data.  This guarantees an ion image
exists for every candidate (including decoys that land in empty m/z space, which
then yield genuine zero-signal images).

``query_raw_maldi`` returns the same 5-tuple as
``maldi_extraction.extract_maldi_data`` so all downstream code (spatial features,
colocalization, isotope envelopes, the feature generator, and the pipeline) is
unchanged.

``extract_observed_feature_stats_raw`` is a separate companion that reads the raw
``.d`` via ``alphatims`` and returns, per candidate m/z, the observed peak centroid
m/z (for a symmetric mass-accuracy ``ppm_error``) and the observed CCS (from the
ion-mobility 1/K0).  It is separate from ``query_raw_maldi`` because ``imzy`` (used
for the ion images) exposes neither the per-peak m/z centroid nor mobility, so the
``.d`` must be opened a second time with ``alphatims``.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _weighted_mean_in_windows(
    peak_mzs: np.ndarray,
    peak_ints: np.ndarray,
    values: np.ndarray,
    query_mzs: np.ndarray,
    ppm: float,
) -> np.ndarray:
    """Intensity-weighted mean of ``values`` per query m/z window.

    For each query m/z ``q`` the window is ``[q*(1-ppm), q*(1+ppm)]``; every peak
    falling inside contributes ``intensity`` to the weight and
    ``intensity * value`` to the numerator, returning ``Σ(int·value)/Σ(int)``.
    Used with ``values = mobility`` (mean 1/K0) and with ``values = peak m/z``
    (the intensity-weighted observed peak centroid m/z).  Returns an array
    aligned with ``query_mzs``; entries with no signal are ``NaN``.

    Pure / vectorised — unit-testable without alphatims or a real ``.d``.  A peak
    may fall in more than one window when query m/z are closer than the ppm
    tolerance, in which case it contributes to each.
    """
    query_mzs = np.asarray(query_mzs, dtype=np.float64)
    n = len(query_mzs)
    wsum = np.zeros(n, dtype=np.float64)
    isum = np.zeros(n, dtype=np.float64)
    if len(peak_mzs) == 0 or n == 0:
        return np.full(n, np.nan)

    peak_mzs = np.asarray(peak_mzs, dtype=np.float64)
    peak_ints = np.asarray(peak_ints, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    ppm_f = ppm * 1e-6

    # peak mz is in window of query q  <=>  q in [mz/(1+ppm), mz/(1-ppm)]
    lo = np.searchsorted(query_mzs, peak_mzs / (1.0 + ppm_f), side="left")
    hi = np.searchsorted(query_mzs, peak_mzs / (1.0 - ppm_f), side="right")
    counts = np.clip(hi - lo, 0, None).astype(np.int64)
    total = int(counts.sum())
    if total == 0:
        return np.full(n, np.nan)

    # Expand (peak, window) pairs without a Python loop.
    peak_rep = np.repeat(np.arange(len(peak_mzs)), counts)
    starts = np.cumsum(counts) - counts
    within = np.arange(total) - np.repeat(starts, counts)
    qidx = np.repeat(lo, counts) + within

    np.add.at(wsum, qidx, peak_ints[peak_rep] * values[peak_rep])
    np.add.at(isum, qidx, peak_ints[peak_rep])
    out = np.full(n, np.nan)
    nz = isum > 0
    out[nz] = wsum[nz] / isum[nz]
    return out


# Backwards-compatible alias: mean 1/K0 is just a weighted mean with values = mobility.
def _weighted_mean_inv_k0(peak_mzs, peak_ints, peak_mob, query_mzs, ppm):
    return _weighted_mean_in_windows(peak_mzs, peak_ints, peak_mob, query_mzs, ppm)


def extract_observed_feature_stats_raw(
    d_path: str,
    query_mzs: np.ndarray,
    extraction_ppm: float = 25.0,
    charge: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Observed per-candidate ``(CCS, peak-centroid m/z)`` from the raw Bruker ``.d``.

    A single ``alphatims`` pass collects every peak inside a ``query_mzs`` window
    (±``extraction_ppm``) across all MALDI pixels, then computes two
    intensity-weighted means per window:

    - **observed centroid m/z** ``Σ(int·mz)/Σ(int)`` — the observed peak position,
      used to recompute a symmetric mass-accuracy ``ppm_error`` in raw-query mode.
      Needs only m/z + intensity, so it is available even without a TIMS dimension.
    - **observed CCS (Å²)** — from the intensity-weighted mean 1/K0 converted via
      ``maldi_imzml.one_over_k0_to_ccs`` (N2 drift gas, charge 1 for [M+H]+).
      ``NaN`` when the dataset has no ion-mobility dimension.

    Returns ``(ccs, centroid_mz)``, each aligned 1:1 with ``query_mzs`` with
    ``NaN`` where no signal.  Both are all-``NaN`` (with a warning) when
    ``alphatims`` is unavailable.
    """
    query_mzs = np.asarray(query_mzs, dtype=np.float64)
    n = len(query_mzs)
    nan_result = np.full(n, np.nan)
    if n == 0:
        return nan_result, nan_result

    try:
        import alphatims.bruker as atb
    except ImportError:
        logger.warning(
            "alphatims not installed — cannot read observed CCS/peak centroids from %s; "
            "CCS/mobility features and raw-query ppm will be skipped. "
            "Install with: pip install alphatims",
            d_path,
        )
        return nan_result, nan_result

    logger.info("Extracting observed peak centroids + CCS from raw .d via alphatims: %s", d_path)
    tims = atb.TimsTOF(str(d_path))

    scan_max = int(tims.scan_max_index)
    mob_arr = np.asarray(tims.mobility_values, dtype=np.float64)
    has_mobility = scan_max > 1 and mob_arr.size > 1
    if not has_mobility:
        logger.warning(
            "Dataset at %s has no ion-mobility dimension (scan_max=%d); observed CCS "
            "unavailable (peak-centroid ppm still computed).",
            d_path, scan_max,
        )

    push_indptr = np.asarray(tims.push_indptr, dtype=np.int64)
    tof_idx_np = np.asarray(tims.tof_indices, dtype=np.int64)
    intensity_np = np.asarray(tims.intensity_values, dtype=np.float64)
    mz_arr_np = np.asarray(tims.mz_values, dtype=np.float64)  # per-TOF-bin m/z
    tof_max_idx = int(tims.tof_max_index)

    # Mark every TOF bin inside any query window (typically 1-3% of bins).
    ppm_f = extraction_ppm * 1e-6
    relevant_tof = np.zeros(tof_max_idx, dtype=np.bool_)
    for qmz in query_mzs:
        lo = int(np.searchsorted(mz_arr_np, float(qmz * (1.0 - ppm_f)), "left"))
        hi = int(np.searchsorted(mz_arr_np, float(qmz * (1.0 + ppm_f)), "right"))
        if lo < hi:
            relevant_tof[lo:hi] = True

    n_peaks = len(tof_idx_np)
    coll_mz: list[np.ndarray] = []
    coll_int: list[np.ndarray] = []
    coll_scan: list[np.ndarray] = []

    # Stream raw peaks in chunks; keep only peaks in a query window.
    _CHUNK = 50_000_000
    for c0 in range(0, n_peaks, _CHUNK):
        c1 = min(c0 + _CHUNK, n_peaks)
        tof_c = tof_idx_np[c0:c1]
        mask = relevant_tof[tof_c]
        if not mask.any():
            continue
        raw_c = np.arange(c0, c1, dtype=np.int64)[mask]
        coll_mz.append(mz_arr_np[tof_c[mask]])
        coll_int.append(intensity_np[raw_c])
        if has_mobility:
            push = np.searchsorted(push_indptr, raw_c, side="right") - 1
            coll_scan.append((push % scan_max).astype(np.int64))

    if not coll_mz:
        logger.warning(
            "No peaks fell inside any candidate m/z window in %s; "
            "observed CCS and peak centroids all-NaN.",
            d_path,
        )
        return nan_result, nan_result

    peak_mzs = np.concatenate(coll_mz)
    peak_ints = np.concatenate(coll_int)

    # Observed peak centroid m/z (needs only m/z + intensity).
    centroid_mz = _weighted_mean_in_windows(
        peak_mzs, peak_ints, peak_mzs, query_mzs, extraction_ppm
    )

    if has_mobility:
        peak_mob = mob_arr[np.concatenate(coll_scan)]
        mean_inv_k0 = _weighted_mean_in_windows(
            peak_mzs, peak_ints, peak_mob, query_mzs, extraction_ppm
        )
        from msi_picasso.maldi_imzml import one_over_k0_to_ccs

        ccs = np.asarray(
            one_over_k0_to_ccs(mean_inv_k0, query_mzs, charge=charge), dtype=np.float64
        )
    else:
        ccs = nan_result

    logger.info(
        "  Observed peak centroid for %d/%d features; observed CCS for %d/%d.",
        int(np.isfinite(centroid_mz).sum()), n, int(np.isfinite(ccs).sum()), n,
    )
    return ccs, centroid_mz


def query_raw_maldi(
    d_path: str,
    query_mzs: np.ndarray,
    extraction_ppm: float = 25.0,
    compute_spatial: bool = True,
    extra_images: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict | None, pd.DataFrame, dict]:
    """
    Extract ion images directly at the supplied candidate-derived m/z values.

    Parameters
    ----------
    d_path
        Path to the raw Bruker ``.d`` directory.
    query_mzs
        Sorted, unique, NaN-free [M+H]+ m/z values to query.  Derived from
        ``candidates_df["feature_mz"].dropna().unique()`` (sorted).  For
        ``mz_shift`` decoys this is the *shifted* m/z (the off-target anchor),
        which is exactly the m/z whose ion image the decoy should receive.
    extraction_ppm
        m/z half-window for ion image assembly (default 25.0 ppm).
    compute_spatial
        Compute per-feature spatial statistics (always required downstream;
        kept for API symmetry with the plan).
    extra_images
        Extract M+1/M+2 and Na/K/CHCA adduct ion images for colocalization
        features.  Set ``False`` to skip (returns ``extra_ion_images=None``).

    Returns
    -------
    (feature_mzs, ion_images, extra_ion_images, spatial_df, maldi_envelopes)
        Identical schema to ``extract_maldi_data``.  ``feature_mzs`` equals the
        sorted ``query_mzs`` (no zero-signal features are dropped).
    """
    query_mzs = np.asarray(query_mzs, dtype=np.float64)
    # These invariants are load-bearing: the binary-search matching downstream
    # assumes a sorted grid, and a NaN query m/z would corrupt extraction.
    assert query_mzs.ndim == 1, "query_mzs must be a 1D array"
    assert not np.any(np.isnan(query_mzs)), "query_mzs must not contain NaN"
    assert np.all(np.diff(query_mzs) >= 0), "query_mzs must be sorted"

    from msi_picasso.maldi_extraction import extract_maldi_data

    logger.info(
        "Raw-query MALDI extraction: %d candidate-derived m/z values at ±%.1f ppm",
        len(query_mzs), extraction_ppm,
    )
    feature_mzs, ion_images, extra_ion_images, spatial_df, maldi_envelopes, _pixel_coords = extract_maldi_data(
        d_path,
        extraction_ppm=extraction_ppm,
        feature_mzs=query_mzs,
        drop_zero_signal=False,  # keep every candidate m/z, incl. zero-signal decoys
    )

    if not extra_images:
        extra_ion_images = None

    return feature_mzs, ion_images, extra_ion_images, spatial_df, maldi_envelopes
