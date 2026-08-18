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

from msi_picasso.utils import NEUTRON

logger = logging.getLogger(__name__)


def _weighted_mean_in_windows(
    peak_mzs: np.ndarray,
    peak_ints: np.ndarray,
    values: np.ndarray,
    query_mzs: np.ndarray,
    ppm: float,
    return_weight: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Intensity-weighted mean of ``values`` per query m/z window.

    For each query m/z ``q`` the window is ``[q*(1-ppm), q*(1+ppm)]``; every peak
    falling inside contributes ``intensity`` to the weight and
    ``intensity * value`` to the numerator, returning ``Σ(int·value)/Σ(int)``.
    Used with ``values = mobility`` (mean 1/K0) and with ``values = peak m/z``
    (the intensity-weighted observed peak centroid m/z).  Returns an array
    aligned with ``query_mzs``; entries with no signal are ``NaN``.

    With ``return_weight=True`` the total in-window intensity ``Σint`` is returned
    alongside the mean as ``(mean, intensity_sum)`` — used by the isotope-envelope
    CCS consistency feature, which needs both the observed CCS and the integrated
    intensity of each envelope peak.

    Pure / vectorised — unit-testable without alphatims or a real ``.d``.  A peak
    may fall in more than one window when query m/z are closer than the ppm
    tolerance, in which case it contributes to each.
    """
    query_mzs = np.asarray(query_mzs, dtype=np.float64)
    n = len(query_mzs)
    wsum = np.zeros(n, dtype=np.float64)
    isum = np.zeros(n, dtype=np.float64)
    if len(peak_mzs) == 0 or n == 0:
        return (np.full(n, np.nan), isum) if return_weight else np.full(n, np.nan)

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
        return (np.full(n, np.nan), isum) if return_weight else np.full(n, np.nan)

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
    return (out, isum) if return_weight else out


# Backwards-compatible alias: mean 1/K0 is just a weighted mean with values = mobility.
def _weighted_mean_inv_k0(peak_mzs, peak_ints, peak_mob, query_mzs, ppm):
    return _weighted_mean_in_windows(peak_mzs, peak_ints, peak_mob, query_mzs, ppm)


_MOB_QUALITY_COLS = (
    "mob_2d_concentration",
    "mob_k0_spread",
    "mob_mz_spread_ppm",
    "mob_peak_snr",
)


def _peak_quality_in_windows(
    peak_mzs: np.ndarray,
    peak_ints: np.ndarray,
    peak_mob: np.ndarray,
    query_mzs: np.ndarray,
    window_ppm: float,
    k0_tol: float,
    core_ppm: float = 5.0,
) -> dict[str, np.ndarray]:
    """Intrinsic peak-quality descriptors in the joint (m/z, intensity, 1/K0) space.

    For each query m/z ``q`` the outer window is ``[q*(1-window_ppm), q*(1+window_ppm)]``.
    From the intensity-weighted peaks inside it, four per-window descriptors of "how
    clean / compact / intense is the observed ion here" are computed (all
    intensity-weighted, so the intensity axis enters as the weight):

    - ``mob_2d_concentration`` — Σintensity inside a tight box (±``core_ppm`` in m/z,
      ±``k0_tol`` in 1/K0) around the intensity-weighted (m/z, 1/K0) centroid, divided
      by the total in-window intensity.  ∈ [0, 1]; high = one compact 2D blob.
    - ``mob_k0_spread`` — intensity-weighted std of 1/K0 (V·s/cm²); low = a single tight
      mobility species rather than a smear of co-eluting ions.
    - ``mob_mz_spread_ppm`` — intensity-weighted std of m/z, in ppm; low = sharp peak.
    - ``mob_peak_snr`` — ``log10(band / off-band)`` where ``band`` is the intensity within
      ±``k0_tol`` of the 1/K0 centroid and ``off-band`` is the rest of the window; high =
      the mobility peak dominates the chemical background at this m/z.

    Returns a dict of arrays aligned 1:1 with ``query_mzs``; windows with no signal are
    ``NaN`` (left for the worst-case ``FEATURE_NAN_FILL`` at scoring).  Note the peak pool
    is collected at ``extraction_ppm`` upstream, so an effective window wider than that
    sees only the extracted peaks.  Pure / vectorised — unit-testable without alphatims.
    """
    query_mzs = np.asarray(query_mzs, dtype=np.float64)
    n = len(query_mzs)
    out = {c: np.full(n, np.nan) for c in _MOB_QUALITY_COLS}
    if len(peak_mzs) == 0 or n == 0:
        return out

    peak_mzs = np.asarray(peak_mzs, dtype=np.float64)
    peak_ints = np.asarray(peak_ints, dtype=np.float64)
    peak_mob = np.asarray(peak_mob, dtype=np.float64)
    ppm_f = window_ppm * 1e-6

    # Same peak→window expansion as _weighted_mean_in_windows.
    lo = np.searchsorted(query_mzs, peak_mzs / (1.0 + ppm_f), side="left")
    hi = np.searchsorted(query_mzs, peak_mzs / (1.0 - ppm_f), side="right")
    counts = np.clip(hi - lo, 0, None).astype(np.int64)
    total = int(counts.sum())
    if total == 0:
        return out

    peak_rep = np.repeat(np.arange(len(peak_mzs)), counts)
    starts = np.cumsum(counts) - counts
    within = np.arange(total) - np.repeat(starts, counts)
    qidx = np.repeat(lo, counts) + within

    w = peak_ints[peak_rep]
    mz = peak_mzs[peak_rep]
    k0 = peak_mob[peak_rep]

    isum = np.zeros(n)
    s_mz = np.zeros(n)
    s_mz2 = np.zeros(n)
    s_k0 = np.zeros(n)
    s_k02 = np.zeros(n)
    np.add.at(isum, qidx, w)
    np.add.at(s_mz, qidx, w * mz)
    np.add.at(s_mz2, qidx, w * mz * mz)
    np.add.at(s_k0, qidx, w * k0)
    np.add.at(s_k02, qidx, w * k0 * k0)

    nz = isum > 0
    mz_mean = np.full(n, np.nan)
    k0_mean = np.full(n, np.nan)
    mz_mean[nz] = s_mz[nz] / isum[nz]
    k0_mean[nz] = s_k0[nz] / isum[nz]
    # variance = E[x²] − E[x]²  (clip tiny negatives from float rounding)
    mz_var = np.clip(s_mz2[nz] / isum[nz] - mz_mean[nz] ** 2, 0.0, None)
    k0_var = np.clip(s_k02[nz] / isum[nz] - k0_mean[nz] ** 2, 0.0, None)
    out["mob_mz_spread_ppm"][nz] = np.sqrt(mz_var) / mz_mean[nz] * 1e6
    out["mob_k0_spread"][nz] = np.sqrt(k0_var)

    # Per-pair membership around the intensity-weighted centroid.  qidx only indexes
    # windows that received peaks, so mz_mean[qidx]/k0_mean[qidx] are finite.
    in_k0_band = np.abs(k0 - k0_mean[qidx]) <= k0_tol
    in_mz_core = np.abs(mz - mz_mean[qidx]) <= mz_mean[qidx] * core_ppm * 1e-6
    in_core = in_k0_band & in_mz_core

    band = np.zeros(n)
    core = np.zeros(n)
    np.add.at(band, qidx, w * in_k0_band)
    np.add.at(core, qidx, w * in_core)

    _eps = 1e-12
    out["mob_2d_concentration"][nz] = core[nz] / isum[nz]
    off = isum - band
    out["mob_peak_snr"][nz] = np.log10((band[nz] + _eps) / (off[nz] + _eps))
    return out


def extract_observed_feature_stats_raw(
    d_path: str,
    query_mzs: np.ndarray,
    extraction_ppm: float = 25.0,
    charge: int = 1,
    mob_quality_window_ppm: float = 25.0,
    mob_quality_k0_tol: float = 0.02,
) -> tuple[np.ndarray, np.ndarray, dict | None, dict | None]:
    """Observed per-candidate ``(CCS, peak-centroid m/z, peak_quality, envelope)`` from the raw ``.d``.

    A single ``alphatims`` pass collects every peak inside a ``query_mzs`` window
    (±``extraction_ppm``) across all MALDI pixels, then computes two
    intensity-weighted means per window:

    - **observed centroid m/z** ``Σ(int·mz)/Σ(int)`` — the observed peak position,
      used to recompute a symmetric mass-accuracy ``ppm_error`` in raw-query mode.
      Needs only m/z + intensity, so it is available even without a TIMS dimension.
    - **observed CCS (Å²)** — from the intensity-weighted mean 1/K0 converted via
      ``maldi_imzml.one_over_k0_to_ccs`` (N2 drift gas, charge 1 for [M+H]+).
      ``NaN`` when the dataset has no ion-mobility dimension.

    Additionally, when the dataset has ion mobility:

    - ``peak_quality`` is a dict of four intrinsic joint-space peak-quality descriptors
      (see :func:`_peak_quality_in_windows`), each aligned 1:1 with ``query_mzs``.
    - ``envelope`` is a dict with keys ``ccs_m0``/``ccs_m1``/``ccs_m2`` and
      ``int_m0``/``int_m1``/``int_m2``, holding the observed CCS and integrated
      intensity of the three singly-charged isotopologue positions
      ``query_mz + k * NEUTRON``.  Real isotopologues of one molecule share a CCS;
      a chimeric envelope (isobaric mass coincidence) does not, so the spread of
      these three CCS values is the isotope-envelope CCS-consistency feature (the
      CCS analogue of IsoMobil's IPMV).  Uses only observed peaks — no predicted
      CCS enters, so there is no m/z-baseline leak.

    Both are ``None`` without a TIMS dimension or when ``alphatims`` is unavailable.

    Returns ``(ccs, centroid_mz, peak_quality, envelope)``, the arrays aligned 1:1 with
    ``query_mzs`` with ``NaN`` where no signal.  ``ccs``/``centroid_mz`` are all-``NaN``
    (with a warning) when ``alphatims`` is unavailable.
    """
    query_mzs = np.asarray(query_mzs, dtype=np.float64)
    n = len(query_mzs)
    nan_result = np.full(n, np.nan)
    if n == 0:
        return nan_result, nan_result, None, None

    try:
        import alphatims.bruker as atb
    except ImportError:
        logger.warning(
            "alphatims not installed — cannot read observed CCS/peak centroids from %s; "
            "CCS/mobility features and raw-query ppm will be skipped. "
            "Install with: pip install alphatims",
            d_path,
        )
        return nan_result, nan_result, None, None

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

    # Mark every TOF bin inside any query window (typically 1-3% of bins).  With ion
    # mobility the M+1/M+2 isotopologue windows are marked too, so the same single pass
    # feeds the isotope-envelope CCS consistency feature.  The windows are ppm-wide and
    # ~1 Da apart, so they never overlap and the M0 statistics are unaffected.
    ppm_f = extraction_ppm * 1e-6
    envelope_ks = (0, 1, 2) if has_mobility else (0,)
    relevant_tof = np.zeros(tof_max_idx, dtype=np.bool_)
    for k in envelope_ks:
        for qmz in query_mzs + k * NEUTRON:
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
        return nan_result, nan_result, None, None

    peak_mzs = np.concatenate(coll_mz)
    peak_ints = np.concatenate(coll_int)

    # Observed peak centroid m/z (needs only m/z + intensity).
    centroid_mz = _weighted_mean_in_windows(
        peak_mzs, peak_ints, peak_mzs, query_mzs, extraction_ppm
    )

    peak_quality = None
    envelope = None
    if has_mobility:
        peak_mob = mob_arr[np.concatenate(coll_scan)]
        from msi_picasso.maldi_imzml import one_over_k0_to_ccs

        # Observed CCS + integrated intensity at each singly-charged isotopologue
        # position.  k == 0 is the M0 anchor, so `ccs` is just the k == 0 slice.
        envelope = {}
        for k in envelope_ks:
            mz_k = query_mzs + k * NEUTRON
            mean_inv_k0, int_k = _weighted_mean_in_windows(
                peak_mzs, peak_ints, peak_mob, mz_k, extraction_ppm, return_weight=True
            )
            envelope[f"ccs_m{k}"] = np.asarray(
                one_over_k0_to_ccs(mean_inv_k0, mz_k, charge=charge), dtype=np.float64
            )
            envelope[f"int_m{k}"] = int_k
        ccs = envelope["ccs_m0"]
        # Intrinsic joint (m/z, intensity, 1/K0) peak-quality descriptors.
        peak_quality = _peak_quality_in_windows(
            peak_mzs, peak_ints, peak_mob, query_mzs,
            window_ppm=mob_quality_window_ppm, k0_tol=mob_quality_k0_tol,
        )
    else:
        ccs = nan_result

    logger.info(
        "  Observed peak centroid for %d/%d features; observed CCS for %d/%d.",
        int(np.isfinite(centroid_mz).sum()), n, int(np.isfinite(ccs).sum()), n,
    )
    if envelope is not None:
        logger.info(
            "  Isotope-envelope observed CCS: M0 %d/%d, M+1 %d/%d, M+2 %d/%d features.",
            *[x for k in (0, 1, 2)
              for x in (int(np.isfinite(envelope[f"ccs_m{k}"]).sum()), n)],
        )
    return ccs, centroid_mz, peak_quality, envelope


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
