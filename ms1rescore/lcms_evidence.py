"""
Raw LC-MS/MS evidence extraction for symmetric target-decoy rescoring.

All feature computation functions take (peptide, precursor_mz, charge, lcms_data)
and return feature values. No function takes an is_decoy parameter.
"""

import logging
import os
import pickle
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ms1rescore.utils import (
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
    cache_path: str | None = None,
) -> LCMSData:
    """
    Load MS1 and MS2 scans from mzML files into indexed arrays.

    Parameters
    ----------
    mzml_paths
        Paths to mzML files.
    cache_path
        If provided, cache loaded data to this pickle file for fast reloading.
    """
    if cache_path and os.path.exists(cache_path):
        logger.info(f"Loading cached LC-MS/MS data from {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

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

    if cache_path:
        logger.info(f"Caching LC-MS/MS data to {cache_path}")
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    return data


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
    model: str = "HCD",
    cache_path: str | None = None,
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
    if cache_path and os.path.exists(cache_path):
        logger.info(f"Loading cached MS2PIP predictions from {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

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

            sort_order = np.argsort(all_mz)
            cache[(pep, charge)] = (all_mz[sort_order], all_int[sort_order])

    logger.info(f"MS2PIP: {len(cache)} predictions total")

    if cache_path:
        logger.info(f"Caching MS2PIP predictions to {cache_path}")
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)

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
            # Take the highest intensity peak within tolerance
            best_idx = lo + np.argmax(int_arr[lo:hi])
            rts.append(lcms_data.ms1_rts[scan_idx])
            intensities.append(int_arr[best_idx])

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


def finetune_deeplc(
    msf_path: str,
    cache_path: str | None = None,
):
    """
    Finetune DeepLC on high-confidence PSMs from PD .msf file.

    Uses only observed retention times for calibration — no target/decoy labels
    are used in the finetuned model.
    """
    import torch

    if cache_path and os.path.exists(cache_path):
        logger.info(f"Loading cached DeepLC model from {cache_path}")
        return torch.load(cache_path, weights_only=False)

    import sqlite3

    from deeplc.core import finetune
    from psm_utils import PSM, PSMList, Peptidoform

    conn = sqlite3.connect(msf_path)
    # Use high-confidence target PSMs with observed RT
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
        logger.warning(
            f"Only {len(df)} PSMs for DeepLC finetuning — using default model"
        )
        return None

    logger.info(f"Finetuning DeepLC on {len(df)} PSMs...")
    psm_list = PSMList(
        psm_list=[
            PSM(
                peptidoform=Peptidoform(f"{row['peptide']}/2"),
                spectrum_id=f"cal_{i}",
                retention_time=row["rt"],
            )
            for i, (_, row) in enumerate(df.iterrows())
        ]
    )

    model = finetune(psm_list)

    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        torch.save(model, cache_path)
        logger.info(f"Saved finetuned DeepLC model to {cache_path}")

    return model


def get_deeplc_predictions(
    unique_peptides: list[str],
    model=None,
    cache_path: str | None = None,
) -> dict[str, float]:
    """
    Batch predict retention times for all unique peptides.

    Returns dict mapping peptide → predicted_rt.
    """
    if cache_path and os.path.exists(cache_path):
        logger.info(f"Loading cached DeepLC predictions from {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

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
    cache = dict(zip(unique_peptides, predicted_rts))

    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)

    return cache


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
) -> np.ndarray:
    """
    Extract isotope envelope [M0, M+1, M+2, ...] from an MS1 scan.

    Returns normalized intensities. Zeros if peaks not found.
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
            # Closest to expected m/z
            dists = np.abs(mz_arr[lo:hi] - expected_mz)
            best = lo + np.argmin(dists)
            intensities[k] = int_arr[best]

    total = intensities.sum()
    if total > 0:
        intensities /= total
    return intensities


# ---------------------------------------------------------------------------
# Main evidence computation
# ---------------------------------------------------------------------------


def compute_lcms_evidence(
    peptide: str,
    precursor_mz: float,
    lcms_data: LCMSData,
    ms2pip_cache: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]],
    deeplc_cache: dict[str, float] | None = None,
    xic_cache: dict[float, tuple[np.ndarray, np.ndarray]] | None = None,
    ppm_tolerance: float = 20.0,
    fragment_tol_da: float = 0.02,
) -> dict[str, float]:
    """
    Compute all LC-MS/MS evidence features for a single candidate.

    This function takes NO is_decoy parameter — symmetry enforced by API design.

    Parameters
    ----------
    peptide
        Peptide sequence (plain amino acids).
    precursor_mz
        Theoretical [M+H]+ m/z for this candidate.
    lcms_data
        Pre-loaded LC-MS/MS data.
    ms2pip_cache
        MS2PIP predictions: (peptide, charge) → (mz_array, int_array).
    deeplc_cache
        DeepLC RT predictions: peptide → predicted_rt.
    xic_cache
        Pre-computed XICs: mz → (rts, intensities).
    ppm_tolerance
        Mass tolerance in ppm.
    fragment_tol_da
        Fragment matching tolerance in Da.

    Returns
    -------
    dict with feature values.
    """
    result = {
        "lcms_ms2_spectral_angle": 0.0,
        "lcms_ms2_n_matches": 0,
        "lcms_xic_max_intensity": 0.0,
        "lcms_xic_n_scans": 0,
        "lcms_xic_snr": 0.0,
        "lcms_rt_residual": np.nan,
        "lcms_ms1_isotope_cosine": np.nan,
    }

    # --- MS2 spectral matching (compare by neutral mass, not m/z) ---
    neutral_mass = mz_to_mass(precursor_mz, charge=1)  # MALDI [M+H]+
    matching_scans = _find_matching_ms2_scans(neutral_mass, lcms_data, ppm_tolerance)
    result["lcms_ms2_n_matches"] = len(matching_scans)

    best_angle = 0.0
    for scan_idx in matching_scans:
        scan_charge = int(lcms_data.ms2_precursor_charge[scan_idx])
        pred = ms2pip_cache.get((peptide, scan_charge))
        if pred is None:
            continue
        pred_mz, pred_int = pred
        obs_mz = lcms_data.ms2_mz_arrays[scan_idx]
        obs_int = lcms_data.ms2_int_arrays[scan_idx]

        angle = _match_and_score_spectrum(
            pred_mz, pred_int, obs_mz, obs_int, fragment_tol_da
        )
        if angle > best_angle:
            best_angle = angle

    result["lcms_ms2_spectral_angle"] = best_angle

    # --- XIC features ---
    if xic_cache is not None and precursor_mz in xic_cache:
        xic_rts, xic_ints = xic_cache[precursor_mz]
    else:
        xic_rts, xic_ints = _extract_xic(precursor_mz, lcms_data, ppm_tolerance)

    if len(xic_ints) > 0:
        result["lcms_xic_max_intensity"] = float(np.log1p(xic_ints.max()))

        nonzero = xic_ints[xic_ints > 0]
        if len(nonzero) > 0:
            noise = np.percentile(nonzero, 5)
            result["lcms_xic_n_scans"] = int((xic_ints > 0.1 * xic_ints.max()).sum())
            result["lcms_xic_snr"] = float(xic_ints.max() / noise) if noise > 0 else 0.0

            # Best scan for RT and isotope extraction
            best_xic_idx = np.argmax(xic_ints)
            best_xic_rt = xic_rts[best_xic_idx]

            # --- RT residual ---
            if deeplc_cache is not None and peptide in deeplc_cache:
                predicted_rt = deeplc_cache[peptide]
                result["lcms_rt_residual"] = float(abs(predicted_rt - best_xic_rt))

            # --- MS1 isotope cosine ---
            # Find the MS1 scan index closest to best XIC RT
            best_ms1_idx = np.argmin(np.abs(lcms_data.ms1_rts - best_xic_rt))
            observed_env = _extract_ms1_envelope(
                precursor_mz, best_ms1_idx, lcms_data, charge=1, n_peaks=3
            )
            if observed_env.sum() > 0:
                comp = composition_from_sequence(peptide)
                theo_env = theoretical_isotope_distribution(
                    comp["C"], comp["H"], comp["N"], comp["O"], comp["S"], n_peaks=3
                )
                result["lcms_ms1_isotope_cosine"] = float(
                    cosine_similarity(observed_env, theo_env)
                )

    return result


def compute_all_lcms_evidence(
    candidates_df: pd.DataFrame,
    lcms_data: LCMSData,
    ms2pip_cache: dict,
    deeplc_cache: dict | None = None,
    ppm_tolerance: float = 20.0,
    fragment_tol_da: float = 0.02,
) -> dict[int, dict[str, float]]:
    """
    Compute LC-MS/MS evidence features for all candidates.

    Optimized: pre-computes per-feature data (XIC, MS2 scan matches, best MS1 scan)
    once per unique m/z, then only peptide-specific work (spectral angle, RT residual,
    isotope cosine) is done per candidate.
    """
    unique_mzs = candidates_df["feature_mz"].unique()
    xic_charges = [1, 2, 3, 4]

    # --- Pre-compute per-feature (shared across all candidates at same m/z) ---
    # For XIC: search at charge 1-4 m/z for the same neutral mass, take best signal.
    logger.info(f"  Pre-computing per-feature data for {len(unique_mzs)} features...")

    # Build all m/z values to extract XICs for (each feature × each charge)
    from ms1rescore.utils import mass_to_mz
    all_xic_mzs = []
    mz_to_feature = {}  # (xic_mz, charge) → feature_mz
    for mz in unique_mzs:
        neutral_mass = mz_to_mass(mz, charge=1)
        for c in xic_charges:
            xic_mz = mass_to_mz(neutral_mass, c)
            all_xic_mzs.append(xic_mz)
            mz_to_feature[(xic_mz, c)] = mz

    xic_cache = extract_all_xics(np.array(all_xic_mzs), lcms_data, ppm_tolerance)

    feature_data = {}  # feature_mz → dict with shared XIC features + MS2 scan indices
    for mz in unique_mzs:
        neutral_mass = mz_to_mass(mz, charge=1)

        # Find the best XIC across all charge states
        best_max_int = 0.0
        best_rts = np.array([])
        best_ints = np.array([])
        best_charge = 1
        for c in xic_charges:
            xic_mz = mass_to_mz(neutral_mass, c)
            rts, ints = xic_cache.get(xic_mz, (np.array([]), np.array([])))
            if len(ints) > 0 and ints.max() > best_max_int:
                best_max_int = ints.max()
                best_rts = rts
                best_ints = ints
                best_charge = c

        fd = {
            "lcms_xic_max_intensity": 0.0,
            "lcms_xic_n_scans": 0,
            "lcms_xic_snr": 0.0,
            "best_xic_rt": None,
            "best_ms1_idx": None,
            "best_xic_charge": best_charge,
            "best_xic_mz": mass_to_mz(neutral_mass, best_charge),
            "ms2_scan_indices": _find_matching_ms2_scans(
                neutral_mass, lcms_data, ppm_tolerance
            ),
        }

        if len(best_ints) > 0 and best_ints.max() > 0:
            fd["lcms_xic_max_intensity"] = float(np.log1p(best_ints.max()))
            nonzero = best_ints[best_ints > 0]
            if len(nonzero) > 0:
                noise = np.percentile(nonzero, 5)
                fd["lcms_xic_n_scans"] = int((best_ints > 0.1 * best_ints.max()).sum())
                fd["lcms_xic_snr"] = float(best_ints.max() / noise) if noise > 0 else 0.0
                best_idx = np.argmax(best_ints)
                fd["best_xic_rt"] = float(best_rts[best_idx])
                fd["best_ms1_idx"] = int(
                    np.argmin(np.abs(lcms_data.ms1_rts - best_rts[best_idx]))
                )

        feature_data[mz] = fd

    n_with_xic = sum(1 for fd in feature_data.values() if fd["best_xic_rt"] is not None)
    logger.info(f"  {n_with_xic}/{len(unique_mzs)} features have XIC signal (searched charges {xic_charges})")

    # --- Per-candidate: only peptide-specific work ---
    logger.info(f"  Computing per-candidate evidence for {len(candidates_df)} candidates...")
    evidence = {}
    n_total = len(candidates_df)

    # Use itertuples for speed
    peptides = candidates_df["peptide"].values
    feature_mzs = candidates_df["feature_mz"].values
    n_C = candidates_df["n_C"].values if "n_C" in candidates_df.columns else None

    for i in range(n_total):
        if (i + 1) % 50000 == 0:
            logger.info(f"    {i+1}/{n_total}")

        mz = feature_mzs[i]
        peptide = peptides[i]
        fd = feature_data[mz]

        result = {
            "lcms_xic_max_intensity": fd["lcms_xic_max_intensity"],
            "lcms_xic_n_scans": fd["lcms_xic_n_scans"],
            "lcms_xic_snr": fd["lcms_xic_snr"],
            "lcms_xic_best_charge": fd["best_xic_charge"],
            "lcms_ms2_n_matches": len(fd["ms2_scan_indices"]),
            "lcms_ms2_spectral_angle": 0.0,
            "lcms_rt_residual": np.nan,
            "lcms_ms1_isotope_cosine": np.nan,
        }

        # Spectral angle (peptide-specific: depends on MS2PIP prediction)
        best_angle = 0.0
        for scan_idx in fd["ms2_scan_indices"]:
            scan_charge = int(lcms_data.ms2_precursor_charge[scan_idx])
            pred = ms2pip_cache.get((peptide, scan_charge))
            if pred is None:
                continue
            angle = _match_and_score_spectrum(
                pred[0], pred[1],
                lcms_data.ms2_mz_arrays[scan_idx],
                lcms_data.ms2_int_arrays[scan_idx],
                fragment_tol_da,
            )
            if angle > best_angle:
                best_angle = angle
        result["lcms_ms2_spectral_angle"] = best_angle

        # RT residual (peptide-specific)
        if fd["best_xic_rt"] is not None and deeplc_cache is not None:
            predicted_rt = deeplc_cache.get(peptide)
            if predicted_rt is not None:
                result["lcms_rt_residual"] = float(abs(predicted_rt - fd["best_xic_rt"]))

        # MS1 isotope cosine (peptide-specific: depends on composition)
        # Use the charge and m/z at which the best XIC was found
        if fd["best_ms1_idx"] is not None and n_C is not None:
            observed_env = _extract_ms1_envelope(
                fd["best_xic_mz"], fd["best_ms1_idx"], lcms_data,
                charge=fd["best_xic_charge"], n_peaks=3,
            )
            if observed_env.sum() > 0:
                nc, nh, nn, no_val, ns = (
                    int(n_C[i]),
                    int(candidates_df["n_H"].values[i]),
                    int(candidates_df["n_N"].values[i]),
                    int(candidates_df["n_O"].values[i]),
                    int(candidates_df["n_S"].values[i]),
                )
                theo_env = theoretical_isotope_distribution(nc, nh, nn, no_val, ns, n_peaks=3)
                result["lcms_ms1_isotope_cosine"] = float(
                    cosine_similarity(observed_env, theo_env)
                )

        evidence[candidates_df.index[i]] = result

    logger.info(f"Computed LC-MS/MS evidence for {len(evidence)} candidates")
    return evidence
