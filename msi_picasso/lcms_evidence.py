"""
Raw LC-MS/MS evidence extraction for symmetric target-decoy rescoring.

All feature computation functions take (peptide, precursor_mz, charge, lcms_data)
and return feature values. No function takes an is_decoy parameter.
"""

import logging
import os
import warnings
from dataclasses import dataclass, field

warnings.filterwarnings("ignore", category=FutureWarning, module="alphatims")

import numpy as np
import pandas as pd

from msi_picasso.utils import (
    NEUTRON,
    PROTON,
    composition_from_sequence,
    cosine_similarity,
    mz_to_mass,
    spectral_angle,
    theoretical_isotope_distribution,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LCMSData: pre-loaded and indexed LC-MS/MS data
# ---------------------------------------------------------------------------


@dataclass
class LCMSData:
    """Pre-loaded and indexed LC-MS/MS data from mzML files."""

    # MS1 data
    ms1_rts: np.ndarray = field(default_factory=lambda: np.array([]))
    ms1_mz_arrays: list[np.ndarray] = field(default_factory=list)
    ms1_int_arrays: list[np.ndarray] = field(default_factory=list)

    # MS2 data
    ms2_precursor_mz: np.ndarray = field(default_factory=lambda: np.array([]))
    ms2_precursor_charge: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=int)
    )
    ms2_precursor_rt: np.ndarray = field(default_factory=lambda: np.array([]))
    ms2_mz_arrays: list[np.ndarray] = field(default_factory=list)
    ms2_int_arrays: list[np.ndarray] = field(default_factory=list)

    # Precomputed index for fast MS2 lookup by precursor m/z
    ms2_mz_sort_idx: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))

    def build_index(self):
        """Build sorted index for fast MS2 precursor m/z lookup."""
        if len(self.ms2_precursor_mz) > 0:
            self.ms2_mz_sort_idx = np.argsort(self.ms2_precursor_mz)


def load_lcms_data(
    mzml_paths: list[str],
) -> LCMSData:
    """
    Load MS1 and MS2 scans from mzML or Bruker .d files into indexed arrays.

    When a path ends with ``.d``, ``load_lcms_data_from_d`` is used (requires
    alphatims).  Otherwise the path is treated as mzML (pyteomics).  Mixed
    lists (some mzML, some .d) are not supported — use one format per run.

    Parameters
    ----------
    mzml_paths
        Paths to mzML files or a single Bruker .d folder.
    """
    # Route Bruker .d files to the alphatims loader
    if len(mzml_paths) == 1 and mzml_paths[0].rstrip("/\\").endswith(".d"):
        return load_lcms_data_from_d(mzml_paths[0])

    from pyteomics import mzml

    ms1_rts = []
    ms1_mz_arrays = []
    ms1_int_arrays = []
    ms2_precursor_mz = []
    ms2_precursor_charge = []
    ms2_precursor_rt = []
    ms2_mz_arrays = []
    ms2_int_arrays = []

    for path in mzml_paths:
        logger.info(f"Loading {path}...")
        with mzml.MzML(path) as reader:
            for spectrum in reader:
                ms_level = spectrum.get("ms level", 0)
                rt = (
                    spectrum.get("scanList", {})
                    .get("scan", [{}])[0]
                    .get("scan start time", 0.0)
                )
                # pyteomics returns a unitfloat; normalize to minutes
                if getattr(rt, "unit_info", "minute") == "second":
                    rt = float(rt) / 60.0
                else:
                    rt = float(rt)

                if ms_level == 1:
                    mz_arr = spectrum.get("m/z array", np.array([]))
                    int_arr = spectrum.get("intensity array", np.array([]))
                    if len(mz_arr) > 0:
                        ms1_rts.append(rt)
                        ms1_mz_arrays.append(mz_arr.astype(np.float64))
                        ms1_int_arrays.append(int_arr.astype(np.float64))

                elif ms_level == 2:
                    precursor_list = spectrum.get("precursorList", {}).get(
                        "precursor", []
                    )
                    if not precursor_list:
                        continue
                    ion_list = (
                        precursor_list[0]
                        .get("selectedIonList", {})
                        .get("selectedIon", [{}])
                    )
                    if not ion_list:
                        continue

                    prec_mz = ion_list[0].get("selected ion m/z", 0.0)
                    prec_charge = int(ion_list[0].get("charge state", 2))
                    if prec_mz <= 0:
                        continue

                    mz_arr = spectrum.get("m/z array", np.array([]))
                    int_arr = spectrum.get("intensity array", np.array([]))
                    if len(mz_arr) == 0:
                        continue

                    ms2_precursor_mz.append(prec_mz)
                    ms2_precursor_charge.append(prec_charge)
                    ms2_precursor_rt.append(rt)
                    ms2_mz_arrays.append(mz_arr.astype(np.float64))
                    ms2_int_arrays.append(int_arr.astype(np.float64))

    data = LCMSData(
        ms1_rts=np.array(ms1_rts),
        ms1_mz_arrays=ms1_mz_arrays,
        ms1_int_arrays=ms1_int_arrays,
        ms2_precursor_mz=np.array(ms2_precursor_mz),
        ms2_precursor_charge=np.array(ms2_precursor_charge, dtype=int),
        ms2_precursor_rt=np.array(ms2_precursor_rt),
        ms2_mz_arrays=ms2_mz_arrays,
        ms2_int_arrays=ms2_int_arrays,
    )
    data.build_index()

    logger.info(
        f"Loaded {len(data.ms1_rts)} MS1 scans, {len(data.ms2_precursor_mz)} MS2 scans "
        f"from {len(mzml_paths)} files"
    )

    return data


def load_lcms_data_from_d(
    d_path: str,
) -> LCMSData:
    """
    Load MS1 and MS2 scan data from a Bruker timsTOF .d folder using alphatims.

    Supports ddaPASEF and diaPASEF acquisition modes.  MS1 frames are summed
    across the mobility dimension (all scan ranges) to produce conventional
    1-D spectra.  MS2 precursor spectra are extracted via
    ``TimsTOF.index_precursors()`` (vectorised; no per-precursor Python loop).

    Requires: ``pip install alphatims``
    """

    try:
        import alphatims.bruker as atb
    except ImportError as exc:
        raise ImportError(
            "alphatims is required to read Bruker .d files. "
            "Install with: pip install alphatims"
        ) from exc

    logger.info(f"Loading timsTOF data from {d_path} ...")
    tims = atb.TimsTOF(d_path)
    logger.info(f"  Acquisition mode: {tims.acquisition_mode}")

    # --- MS1: one summed spectrum per MS1 frame ---
    ms1_frame_ids = tims.frames.loc[tims.frames["MsMsType"] == 0, "Id"].values
    logger.info(f"  Extracting {len(ms1_frame_ids)} MS1 frames...")
    ms1_rts = []
    ms1_mz_arrays = []
    ms1_int_arrays = []
    for fid in ms1_frame_ids:
        fd = tims[int(fid), :, :]
        if len(fd) == 0:
            continue
        ms1_rts.append(tims.rt_values[fid] / 60.0)  # alphatims rt_values is in seconds
        mz_vals = fd["mz_values"].values.astype(np.float64)
        int_vals = fd["intensity_values"].values.astype(np.float64)
        # alphatims returns peaks interleaved by mobility scan, not sorted by m/z.
        # Sort so that searchsorted-based XIC and envelope extraction works correctly.
        sort_idx = np.argsort(mz_vals, kind="stable")
        ms1_mz_arrays.append(mz_vals[sort_idx])
        ms1_int_arrays.append(int_vals[sort_idx])

    # --- MS2: vectorised spectrum extraction via index_precursors() ---
    prec_df = tims.precursors
    if prec_df is not None and len(prec_df) > 0:
        logger.info(f"  Extracting {len(prec_df)} MS2 precursor spectra...")
        indptr, tof_indices, int_values = tims.index_precursors()

        prec_ids = prec_df["Id"].values.astype(int)
        starts = indptr[prec_ids]
        ends = indptr[prec_ids + 1]
        has_peaks = ends > starts

        ms2_precursor_mz = np.where(
            prec_df["MonoisotopicMz"].values > 0,
            prec_df["MonoisotopicMz"].values,
            prec_df["AverageMz"].values,
        )[has_peaks]
        ms2_precursor_charge = np.where(
            prec_df["Charge"].values > 0,
            prec_df["Charge"].values,
            2,
        )[has_peaks].astype(int)
        parent_frame_ids = prec_df["Parent"].values.astype(int)[has_peaks]
        ms2_precursor_rt = tims.rt_values[parent_frame_ids] / 60.0  # alphatims rt_values is in seconds

        ms2_mz_arrays = [
            tims.mz_values[tof_indices[s:e]].astype(np.float64)
            for s, e in zip(starts[has_peaks], ends[has_peaks])
        ]
        ms2_int_arrays = [
            int_values[s:e].astype(np.float64)
            for s, e in zip(starts[has_peaks], ends[has_peaks])
        ]
    else:
        logger.warning("  No precursors found — MS2 spectral features will not be computed.")
        ms2_precursor_mz = np.array([])
        ms2_precursor_charge = np.array([], dtype=int)
        ms2_precursor_rt = np.array([])
        ms2_mz_arrays = []
        ms2_int_arrays = []

    result = LCMSData(
        ms1_rts=np.array(ms1_rts),
        ms1_mz_arrays=ms1_mz_arrays,
        ms1_int_arrays=ms1_int_arrays,
        ms2_precursor_mz=ms2_precursor_mz,
        ms2_precursor_charge=ms2_precursor_charge,
        ms2_precursor_rt=ms2_precursor_rt,
        ms2_mz_arrays=ms2_mz_arrays,
        ms2_int_arrays=ms2_int_arrays,
    )
    result.build_index()

    logger.info(
        f"  Loaded {len(result.ms1_rts)} MS1 frames, "
        f"{len(result.ms2_precursor_mz)} MS2 precursors from {d_path}"
    )

    return result


# ---------------------------------------------------------------------------
# MS2 spectral matching
# ---------------------------------------------------------------------------


def _find_matching_ms2_scans(
    neutral_mass: float,
    lcms_data: LCMSData,
    ppm_tolerance: float = 20.0,
) -> list[int]:
    """
    Find MS2 scans whose precursor matches the candidate by neutral mass.

    Compares neutral masses (not m/z) so that charge-1 MALDI features match
    charge-2/3 LC-MS/MS precursors for the same peptide.
    """
    if not hasattr(lcms_data, "_ms2_neutral_mass"):
        # Compute and cache neutral masses from MS2 precursor m/z and charge
        lcms_data._ms2_neutral_mass = (
            lcms_data.ms2_precursor_mz * lcms_data.ms2_precursor_charge
            - lcms_data.ms2_precursor_charge * PROTON
        )
        lcms_data._ms2_mass_sort_idx = np.argsort(lcms_data._ms2_neutral_mass)

    sorted_mass = lcms_data._ms2_neutral_mass[lcms_data._ms2_mass_sort_idx]
    tol = neutral_mass * ppm_tolerance / 1e6
    lo = np.searchsorted(sorted_mass, neutral_mass - tol, side="left")
    hi = np.searchsorted(sorted_mass, neutral_mass + tol, side="right")
    return lcms_data._ms2_mass_sort_idx[lo:hi].tolist()


def _match_and_score_spectrum(
    pred_mz: np.ndarray,
    pred_int: np.ndarray,
    obs_mz: np.ndarray,
    obs_int: np.ndarray,
    fragment_tol_da: float = 0.02,
) -> float:
    """
    Compute spectral angle between predicted and observed MS2 spectrum.

    For each predicted fragment ion, finds the closest observed peak within
    tolerance. Builds matched intensity vectors and computes spectral angle.
    """
    if len(pred_mz) == 0 or len(obs_mz) == 0:
        return 0.0

    matched_pred = []
    matched_obs = []

    for j in range(len(pred_mz)):
        target = pred_mz[j]
        idx = np.searchsorted(obs_mz, target)

        best_dist = fragment_tol_da + 1
        best_int = 0.0
        for k in [idx - 1, idx]:
            if 0 <= k < len(obs_mz):
                dist = abs(obs_mz[k] - target)
                if dist < best_dist:
                    best_dist = dist
                    best_int = obs_int[k]

        if best_dist <= fragment_tol_da:
            matched_pred.append(pred_int[j])
            matched_obs.append(best_int)

    if len(matched_pred) < 3:
        return 0.0

    return spectral_angle(np.array(matched_pred), np.array(matched_obs))


def get_ms2pip_predictions(
    peptide_charge_pairs: list[tuple[str, int]],
    model: str = "timsTOF2024",
) -> dict[tuple[str, int], tuple[np.ndarray, np.ndarray]]:
    """
    Batch predict MS2 spectra for specific (peptide, charge) pairs using MS2PIP.

    Parameters
    ----------
    peptide_charge_pairs
        List of (peptide_sequence, charge_state) tuples to predict.

    Returns dict mapping (peptide, charge) → (mz_array, intensity_array)
    where mz_array and intensity_array are concatenated b+y ions sorted by m/z.
    """
    from ms2pip.core import predict_batch
    from psm_utils import PSM, PSMList, Peptidoform

    # Group by charge for efficient batching
    by_charge: dict[int, list[str]] = {}
    for pep, charge in peptide_charge_pairs:
        by_charge.setdefault(charge, []).append(pep)

    cache = {}
    for charge, peptides in sorted(by_charge.items()):
        psm_list = PSMList(
            psm_list=[
                PSM(peptidoform=Peptidoform(f"{pep}/{charge}"), spectrum_id=f"{i}")
                for i, pep in enumerate(peptides)
            ]
        )

        logger.info(
            f"Running MS2PIP for {len(peptides)} peptides at charge {charge}..."
        )
        results = predict_batch(psm_list, model=model, processes=20)

        for r in results:
            pep = peptides[r.psm_index]
            if r.predicted_intensity is None or r.theoretical_mz is None:
                continue

            mz_b = r.theoretical_mz.get("b", np.array([]))
            mz_y = r.theoretical_mz.get("y", np.array([]))
            int_b = r.predicted_intensity.get("b", np.array([]))
            int_y = r.predicted_intensity.get("y", np.array([]))

            all_mz = np.concatenate([mz_b, mz_y])
            all_int = np.concatenate([int_b, int_y])

            # MS2PIP returns log2 intensities (can be negative). Convert to
            # linear scale and normalise to [0, 1] so that spectral angle
            # comparisons against raw observed intensities are meaningful.
            all_int = np.exp2(all_int)
            max_int = all_int.max()
            if max_int > 0:
                all_int = all_int / max_int

            sort_order = np.argsort(all_mz)
            cache[(pep, charge)] = (all_mz[sort_order], all_int[sort_order])

    logger.info(f"MS2PIP: {len(cache)} predictions total")
    return cache


# ---------------------------------------------------------------------------
# XIC extraction
# ---------------------------------------------------------------------------


def _extract_xic(
    target_mz: float,
    lcms_data: LCMSData,
    ppm_tolerance: float = 20.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract ion chromatogram for a target m/z across all MS1 scans.

    Returns (rts, intensities) for scans where a matching peak was found.
    """
    tol = target_mz * ppm_tolerance / 1e6
    rts = []
    intensities = []

    for scan_idx in range(len(lcms_data.ms1_rts)):
        mz_arr = lcms_data.ms1_mz_arrays[scan_idx]
        int_arr = lcms_data.ms1_int_arrays[scan_idx]

        lo = np.searchsorted(mz_arr, target_mz - tol, side="left")
        hi = np.searchsorted(mz_arr, target_mz + tol, side="right")

        if lo < hi:
            # Sum all peaks in the window: accumulates signal across ion mobility scans
            # (timsTOF frames contain multiple mobility scans at the same m/z; summing
            # gives the total intensity at this m/z, equivalent to a conventional
            # 2-D LC-MS projection). For mzML centroid data this equals the single peak.
            rts.append(lcms_data.ms1_rts[scan_idx])
            intensities.append(float(int_arr[lo:hi].sum()))

    return np.array(rts), np.array(intensities)


def extract_all_xics(
    unique_mzs: np.ndarray,
    lcms_data: LCMSData,
    ppm_tolerance: float = 20.0,
) -> dict[float, tuple[np.ndarray, np.ndarray]]:
    """
    Extract XICs for all unique m/z values at once.

    Uses Rust (ms1rescore_rs) if available, else falls back to Python.
    Returns dict mapping m/z → (rts, intensities).
    """
    logger.info(f"Extracting XICs for {len(unique_mzs)} unique m/z values...")

    try:
        from ms1rescore_rs import extract_xics_batch

        mz_list = unique_mzs.tolist() if hasattr(unique_mzs, "tolist") else list(unique_mzs)
        results = extract_xics_batch(
            lcms_data.ms1_rts.tolist(),
            [arr.tolist() for arr in lcms_data.ms1_mz_arrays],
            [arr.tolist() for arr in lcms_data.ms1_int_arrays],
            mz_list,
            ppm_tolerance,
        )
        xics = {}
        for mz, (rts, ints) in zip(mz_list, results):
            xics[mz] = (np.array(rts), np.array(ints))
        logger.info("  (used Rust backend)")
        return xics
    except ImportError:
        pass

    # Python fallback
    xics = {}
    for mz in unique_mzs:
        xics[mz] = _extract_xic(mz, lcms_data, ppm_tolerance)
    return xics


# ---------------------------------------------------------------------------
# DeepLC RT predictions
# ---------------------------------------------------------------------------


def _run_deeplc_finetune(
    psm_list,
    epochs: int,
    lr: float,
    patience: int,
):
    """Shared helper: seed, thread-limit, call finetune(), return model."""
    import random as _random

    import torch as _torch
    from deeplc.core import finetune
    from threadpoolctl import threadpool_limits

    _torch.manual_seed(42)
    np.random.seed(42)
    _random.seed(42)
    _orig_threads = _torch.get_num_threads()
    _torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        model = finetune(
            psm_list,
            train_kwargs={"epochs": epochs, "learning_rate": lr, "patience": patience},
        )
    _torch.set_num_threads(_orig_threads)
    return model


def finetune_deeplc(
    msf_path: str,
    epochs: int = 40,
    lr: float = 0.001,
    patience: int = 10,
):
    """
    Finetune DeepLC on high-confidence PSMs from a PD .msf file.

    RetentionTime in TargetPsms is in minutes (PD convention); no conversion
    needed. Uses only observed retention times — no target/decoy labels.
    """
    import sqlite3

    from psm_utils import PSM, PSMList, Peptidoform

    conn = sqlite3.connect(msf_path)
    df = pd.read_sql_query(
        """
        SELECT DISTINCT
            Sequence AS peptide,
            RetentionTime AS rt
        FROM TargetPsms
        WHERE PercolatorqValue <= 0.01
          AND RetentionTime IS NOT NULL
        """,
        conn,
    )
    conn.close()

    if len(df) < 50:
        logger.warning(f"Only {len(df)} PSMs for DeepLC finetuning — using default model")
        return None

    logger.info(
        f"Finetuning DeepLC on {len(df)} PSMs "
        f"(epochs={epochs}, lr={lr}, patience={patience})…"
    )
    psm_list = PSMList(
        psm_list=[
            PSM(
                peptidoform=Peptidoform(f"{row['peptide']}/2"),
                spectrum_id=f"cal_{i}",
                retention_time=float(row["rt"]),
            )
            for i, (_, row) in enumerate(df.iterrows())
        ]
    )
    return _run_deeplc_finetune(psm_list, epochs=epochs, lr=lr, patience=patience)


def finetune_deeplc_from_df(
    rt_df: "pd.DataFrame",
    epochs: int = 40,
    lr: float = 0.001,
    patience: int = 10,
):
    """
    Finetune DeepLC on a DataFrame with columns ``sequence`` and ``rt_mean``
    (retention time in **minutes**).

    Used when an MSF file is not available (e.g. FragPipe output).
    ``rt_df`` is typically ``lcms_ids.peptides[["sequence", "rt_mean"]].dropna()``.
    """
    from psm_utils import PSM, PSMList, Peptidoform

    df = rt_df[["sequence", "rt_mean"]].dropna(subset=["rt_mean"])

    if len(df) < 50:
        logger.warning(
            f"Only {len(df)} peptides with RT for DeepLC finetuning — using default model"
        )
        return None

    logger.info(
        f"Finetuning DeepLC on {len(df)} peptides "
        f"(epochs={epochs}, lr={lr}, patience={patience})…"
    )
    psm_list = PSMList(
        psm_list=[
            PSM(
                peptidoform=Peptidoform(f"{row['sequence']}/2"),
                spectrum_id=f"cal_{i}",
                retention_time=float(row["rt_mean"]),
            )
            for i, (_, row) in enumerate(df.iterrows())
        ]
    )
    return _run_deeplc_finetune(psm_list, epochs=epochs, lr=lr, patience=patience)


def get_deeplc_predictions(
    unique_peptides: list[str],
    model=None,
) -> dict[str, float]:
    """
    Batch predict retention times for all unique peptides.

    Returns dict mapping peptide → predicted_rt.
    """
    from deeplc.core import predict
    from psm_utils import PSM, PSMList, Peptidoform

    logger.info(f"Predicting RT for {len(unique_peptides)} peptides with DeepLC...")
    psm_list = PSMList(
        psm_list=[
            PSM(peptidoform=Peptidoform(f"{pep}/2"), spectrum_id=f"pred_{i}")
            for i, pep in enumerate(unique_peptides)
        ]
    )

    predicted_rts = predict(psm_list, model=model)
    return dict(zip(unique_peptides, predicted_rts))


# ---------------------------------------------------------------------------
# MS1 isotope envelope extraction from XIC best scan
# ---------------------------------------------------------------------------


def _extract_ms1_envelope(
    target_mz: float,
    scan_idx: int,
    lcms_data: LCMSData,
    charge: int = 1,
    n_peaks: int = 3,
    ppm_tolerance: float = 20.0,
    normalize: bool = True,
) -> np.ndarray:
    """
    Extract isotope envelope [M0, M+1, M+2, ...] from an MS1 scan.

    Returns normalized intensities by default. Pass normalize=False to get raw
    summed counts (needed when accumulating across multiple scans before normalizing).
    Zeros if peaks not found.
    """
    mz_arr = lcms_data.ms1_mz_arrays[scan_idx]
    int_arr = lcms_data.ms1_int_arrays[scan_idx]
    spacing = NEUTRON / charge

    intensities = np.zeros(n_peaks)
    for k in range(n_peaks):
        expected_mz = target_mz + k * spacing
        tol = expected_mz * ppm_tolerance / 1e6

        lo = np.searchsorted(mz_arr, expected_mz - tol, side="left")
        hi = np.searchsorted(mz_arr, expected_mz + tol, side="right")

        if lo < hi:
            intensities[k] = float(int_arr[lo:hi].sum())

    if normalize:
        total = intensities.sum()
        if total > 0:
            intensities /= total
    return intensities


# ---------------------------------------------------------------------------
# Main evidence computation
# ---------------------------------------------------------------------------


def compute_all_lcms_evidence(
    candidates_df: pd.DataFrame,
    lcms_data: LCMSData,
    ms2pip_cache: dict,
    deeplc_cache: dict | None = None,
    maldi_envelopes: dict | None = None,
    ppm_tolerance: float = 20.0,
    fragment_tol_da: float = 0.02,
    rt_window_min: float = 0.0,
) -> dict[int, dict[str, float]]:
    """
    Compute LC-MS/MS evidence features for all candidates.

    MS1 features are DeepLC-anchored: the predicted RT for each peptide is used
    to locate the nearest MS1 scan (or a window of scans when rt_window_min > 0),
    then signal, SNR, and isotope features are extracted at the precursor m/z.
    This is fully symmetric — targets and decoys receive identical treatment;
    no is_decoy branching occurs anywhere.

    rt_window_min
        When > 0, sum signal across all MS1 scans within ±rt_window_min minutes
        of the predicted RT instead of using only the single nearest scan. Falls
        back to nearest scan if no scans fall within the window. Typical value:
        0.5–2.0 min.

    MS2 features (spectral angle, n_matches) use neutral-mass matching of MS2
    scans and are pre-computed once per unique feature m/z.
    """
    unique_mzs = candidates_df["feature_mz"].unique()

    # Pre-compute MS2 scan indices per feature m/z (shared across all candidates)
    logger.info(f"  Pre-computing MS2 matches for {len(unique_mzs)} features...")
    from msi_picasso.utils import mz_to_mass
    feature_ms2_scans: dict[float, list[int]] = {}
    for mz in unique_mzs:
        neutral_mass = mz_to_mass(mz, charge=1)
        feature_ms2_scans[mz] = _find_matching_ms2_scans(neutral_mass, lcms_data, ppm_tolerance)

    n_with_ms2 = sum(1 for v in feature_ms2_scans.values() if v)
    logger.info(f"  {n_with_ms2}/{len(unique_mzs)} features have MS2 matches")

    # Per-candidate computation
    logger.info(f"  Computing per-candidate evidence for {len(candidates_df)} candidates...")
    evidence: dict[int, dict[str, float]] = {}
    n_total = len(candidates_df)

    peptides = candidates_df["peptide"].values
    feature_mzs = candidates_df["feature_mz"].values
    has_comp = "n_C" in candidates_df.columns
    if has_comp:
        n_C_arr = candidates_df["n_C"].values
        n_H_arr = candidates_df["n_H"].values
        n_N_arr = candidates_df["n_N"].values
        n_O_arr = candidates_df["n_O"].values
        n_S_arr = candidates_df["n_S"].values

    has_ms1 = len(lcms_data.ms1_rts) > 0

    # Cache peptide → nearest MS1 scan index (same peptide → same DeepLC RT → same scan)
    peptide_scan_cache: dict[str, int] = {}

    for i in range(n_total):
        if (i + 1) % 50000 == 0:
            logger.info(f"    {i+1}/{n_total}")

        mz = feature_mzs[i]
        peptide = peptides[i]
        ms2_scan_indices = feature_ms2_scans[mz]

        # Pre-fetch predicted RT so MS2 scans can be restricted to the DeepLC RT window.
        predicted_rt = deeplc_cache.get(peptide) if deeplc_cache is not None else None

        # Apply DeepLC RT filter to MS2 scans when calibration is available.
        if rt_window_min > 0 and predicted_rt is not None:
            ms2_scan_indices = [
                idx for idx in ms2_scan_indices
                if abs(float(lcms_data.ms2_precursor_rt[idx]) - predicted_rt) <= rt_window_min
            ]

        result: dict[str, float] = {
            "lcms_ms2_n_matches": float(len(ms2_scan_indices)),
            "lcms_ms2_spectral_angle": 0.0,
            "lcms_ms1_intensity": 0.0,
            "lcms_ms1_snr": 0.0,
            "lcms_ms1_isotope_cosine": np.nan,
            "theo_m1_ratio_diff_lcms": np.nan,
            "theo_m2_ratio_diff_lcms": np.nan,
            "log_theo_m1_ratio_diff_lcms": np.nan,
            "log_theo_m2_ratio_diff_lcms": np.nan,
            "isotope_envelope_cosine": np.nan,
            "isotope_envelope_pearson": np.nan,
            "isotope_envelope_mse": np.nan,
            "log_isotope_m1_ratio_diff": np.nan,
            "log_isotope_m2_ratio_diff": np.nan,
            "isotope_n_matched": 0.0,
            "lcms_ms1_apex_rt_delta": np.nan,
            "lcms_ms1_frac_apex_signal": 0.0,
            "lcms_ms1_n_scans_with_signal": 0.0,
            "lcms_ms2_rt_delta": np.nan,
        }

        # --- MS2 spectral angle ---
        # NaN = no MS2PIP prediction available (no information).
        # 0.0 = prediction available but spectral angle genuinely low (<3 matched fragments).
        best_angle = 0.0
        best_charge = 1  # fallback; updated to the charge of the best-SA scan
        best_sa_rt = np.nan
        has_prediction = False
        for scan_idx in ms2_scan_indices:
            scan_charge = int(lcms_data.ms2_precursor_charge[scan_idx])
            pred = ms2pip_cache.get((peptide, scan_charge))
            if pred is None:
                continue
            has_prediction = True
            angle = _match_and_score_spectrum(
                pred[0], pred[1],
                lcms_data.ms2_mz_arrays[scan_idx],
                lcms_data.ms2_int_arrays[scan_idx],
                fragment_tol_da,
            )
            if angle > best_angle:
                best_angle = angle
                best_charge = scan_charge
                best_sa_rt = float(lcms_data.ms2_precursor_rt[scan_idx])
        result["lcms_ms2_spectral_angle"] = best_angle if has_prediction else np.nan
        if np.isfinite(best_sa_rt) and predicted_rt is not None:
            result["lcms_ms2_rt_delta"] = abs(best_sa_rt - predicted_rt)

        # --- DeepLC-anchored MS1 features ---
        if deeplc_cache is None or not has_ms1:
            evidence[candidates_df.index[i]] = result
            continue

        if predicted_rt is None:
            evidence[candidates_df.index[i]] = result
            continue

        # MS1 scan selection (cached per peptide: same sequence → same predicted RT)
        if peptide not in peptide_scan_cache:
            if rt_window_min > 0:
                mask = np.abs(lcms_data.ms1_rts - predicted_rt) <= rt_window_min
                idxs = np.where(mask)[0]
                if len(idxs) == 0:
                    idxs = np.array([int(np.argmin(np.abs(lcms_data.ms1_rts - predicted_rt)))])
            else:
                idxs = np.array([int(np.argmin(np.abs(lcms_data.ms1_rts - predicted_rt)))])
            peptide_scan_cache[peptide] = idxs
        scan_indices = peptide_scan_cache[peptide]

        # Signal: sum of peaks in ±ppm window across all selected scans
        tol = mz * ppm_tolerance / 1e6
        signal = 0.0
        bg_chunks: list[np.ndarray] = []
        for scan_idx in scan_indices:
            mz_arr = lcms_data.ms1_mz_arrays[scan_idx]
            int_arr = lcms_data.ms1_int_arrays[scan_idx]
            sig_lo = np.searchsorted(mz_arr, mz - tol, side="left")
            sig_hi = np.searchsorted(mz_arr, mz + tol, side="right")
            if sig_lo < sig_hi:
                signal += float(int_arr[sig_lo:sig_hi].sum())
            bg_tol = mz * 500.0 / 1e6
            bg_lo = np.searchsorted(mz_arr, mz - bg_tol, side="left")
            bg_hi = np.searchsorted(mz_arr, mz + bg_tol, side="right")
            if sig_lo > bg_lo or sig_hi < bg_hi:
                bg_chunks.append(np.concatenate([int_arr[bg_lo:sig_lo], int_arr[sig_hi:bg_hi]]))

        result["lcms_ms1_intensity"] = float(np.log1p(signal))

        # SNR: log10(signal / background). When the local ±500 ppm window has no
        # non-zero peaks, the background is effectively zero — this is a clean
        # region and the signal stands out clearly, so use log10(signal) as a
        # sentinel rather than collapsing it to 0 alongside the no-signal case.
        if signal > 0:
            background = 0.0
            if bg_chunks:
                bg_vals = np.concatenate(bg_chunks)
                bg_nonzero = bg_vals[bg_vals > 0]
                if len(bg_nonzero) > 0:
                    background = float(np.median(bg_nonzero))
            if background > 0:
                result["lcms_ms1_snr"] = float(np.log10(signal / background))
            else:
                result["lcms_ms1_snr"] = float(np.log10(signal))

        # Isotope envelope: accumulate raw counts across selected scans, then normalize.
        # Extract at the LC-MS/MS precursor m/z and charge rather than the MALDI [M+H]+
        # m/z, because LC-MS/MS peptides are detected at charge 2+ and the envelope
        # spacing is NEUTRON/charge at that m/z.
        if not has_comp:
            evidence[candidates_df.index[i]] = result
            continue

        neutral_mass = mz - PROTON
        lc_mz = (neutral_mass + best_charge * PROTON) / best_charge

        # F1-F3: apex window features (only when rt_window_min > 0, i.e. DeepLC calibrated).
        # scan_indices already contains all MS1 scans within ±rt_window_min of predicted_rt.
        if rt_window_min > 0 and len(scan_indices) > 0:
            tol_apex = lc_mz * ppm_tolerance / 1e6
            win_signals = np.array([
                float(
                    lcms_data.ms1_int_arrays[wi][
                        np.searchsorted(lcms_data.ms1_mz_arrays[wi], lc_mz - tol_apex, side="left"):
                        np.searchsorted(lcms_data.ms1_mz_arrays[wi], lc_mz + tol_apex, side="right")
                    ].sum()
                )
                for wi in scan_indices
            ])
            result["lcms_ms1_n_scans_with_signal"] = float((win_signals > 0).sum())
            max_sig = float(win_signals.max())
            if max_sig > 0:
                apex_local_idx = int(np.argmax(win_signals))
                apex_scan_idx = scan_indices[apex_local_idx]
                result["lcms_ms1_apex_rt_delta"] = abs(
                    float(lcms_data.ms1_rts[apex_scan_idx]) - predicted_rt
                )
                # Fraction of apex signal present at the DeepLC-anchored (nearest) scan
                anchor_scan_idx = scan_indices[
                    int(np.argmin(np.abs(lcms_data.ms1_rts[scan_indices] - predicted_rt)))
                ]
                anchor_mz_arr = lcms_data.ms1_mz_arrays[anchor_scan_idx]
                anchor_int_arr = lcms_data.ms1_int_arrays[anchor_scan_idx]
                lo = np.searchsorted(anchor_mz_arr, lc_mz - tol_apex, side="left")
                hi = np.searchsorted(anchor_mz_arr, lc_mz + tol_apex, side="right")
                anchor_sig = float(anchor_int_arr[lo:hi].sum()) if lo < hi else 0.0
                result["lcms_ms1_frac_apex_signal"] = anchor_sig / max_sig

        raw_env = np.zeros(3)
        for scan_idx in scan_indices:
            raw_env += _extract_ms1_envelope(lc_mz, scan_idx, lcms_data, charge=best_charge, n_peaks=3, normalize=False)
        env_total = raw_env.sum()
        env = raw_env / env_total if env_total > 0 else raw_env
        if env.sum() > 0:
            nc = int(n_C_arr[i]); nh = int(n_H_arr[i]); nn = int(n_N_arr[i])
            no_val = int(n_O_arr[i]); ns = int(n_S_arr[i])
            theo_env = theoretical_isotope_distribution(nc, nh, nn, no_val, ns, n_peaks=3)

            result["lcms_ms1_isotope_cosine"] = float(cosine_similarity(env, theo_env))
            if theo_env[0] > 0 and env[0] > 0:
                m1_diff = abs(env[1] / env[0] - theo_env[1] / theo_env[0])
                m2_diff = abs(env[2] / env[0] - theo_env[2] / theo_env[0])
                result["theo_m1_ratio_diff_lcms"] = float(m1_diff)
                result["theo_m2_ratio_diff_lcms"] = float(m2_diff)
                result["log_theo_m1_ratio_diff_lcms"] = float(np.log1p(m1_diff))
                result["log_theo_m2_ratio_diff_lcms"] = float(np.log1p(m2_diff))

            # MALDI vs LC-MS/MS envelope comparison (if MALDI envelopes available)
            if maldi_envelopes is not None:
                maldi_env = maldi_envelopes.get(mz)
                if maldi_env is not None:
                    k = min(len(maldi_env), len(env))
                    if k >= 2:
                        a = np.array(maldi_env[:k], dtype=np.float64)
                        b = env[:k].astype(np.float64)
                        matched = int(np.sum((a > 0) & (b > 0)))
                        result["isotope_n_matched"] = float(matched)
                        na_norm = np.linalg.norm(a)
                        nb_norm = np.linalg.norm(b)
                        if na_norm > 0 and nb_norm > 0:
                            result["isotope_envelope_cosine"] = float(
                                np.dot(a, b) / (na_norm * nb_norm)
                            )
                        # Pearson r (numpy, no scipy dependency)
                        a_c = a - a.mean(); b_c = b - b.mean()
                        denom = np.sqrt((a_c**2).sum() * (b_c**2).sum())
                        if denom > 0:
                            result["isotope_envelope_pearson"] = float(
                                np.dot(a_c, b_c) / denom
                            )
                        a_sum = a.sum()
                        a_n = a / a_sum if a_sum > 0 else a
                        result["isotope_envelope_mse"] = float(np.mean((a_n - b) ** 2))
                        if a[0] > 0 and b[0] > 0:
                            result["log_isotope_m1_ratio_diff"] = float(
                                np.log1p(abs(a[1] / a[0] - b[1] / b[0]))
                            )
                            if k >= 3:
                                result["log_isotope_m2_ratio_diff"] = float(
                                    np.log1p(abs(a[2] / a[0] - b[2] / b[0]))
                                )

        evidence[candidates_df.index[i]] = result

    logger.info(f"Computed LC-MS/MS evidence for {len(evidence)} candidates")
    return evidence
