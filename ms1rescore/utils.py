"""Shared utilities: isotope distributions, spectral angle, mass calculations."""

from functools import lru_cache

import numpy as np
from brainpy import isotopic_variants

NEUTRON = 1.003355
PROTON = 1.007276

# Request at least this many peaks from brainpy so the normalization denominator
# captures essentially all isotope signal before truncating to n_peaks.
_NORM_NPEAKS = 6


@lru_cache(maxsize=None)
def theoretical_isotope_distribution(
    n_C: int,
    n_H: int,
    n_N: int,
    n_O: int,
    n_S: int,
    n_peaks: int = 4,
) -> np.ndarray:
    """
    Compute theoretical isotope distribution using brainpy (Mercury algorithm).

    Returns distribution [M0, M1, M2, ...] normalized over all peaks returned
    by brainpy (full-spectrum norm), then truncated to n_peaks.

    Results are cached by composition tuple — O(unique compositions) calls
    rather than O(n_candidates) when used in a vectorized loop.
    """
    composition = {"C": n_C, "H": n_H, "N": n_N, "O": n_O, "S": n_S}
    # charge only shifts the .mz axis; .intensity is charge-independent, so charge=0 is fine
    peaks = isotopic_variants(composition, npeaks=max(n_peaks, _NORM_NPEAKS), charge=0)
    intensities = np.array([p.intensity for p in peaks], dtype=float)
    total = intensities.sum()
    if total < 1e-12:
        return np.zeros(n_peaks, dtype=float)
    intensities /= total
    if len(intensities) < n_peaks:
        intensities = np.pad(intensities, (0, n_peaks - len(intensities)))
    return intensities[:n_peaks]


def composition_from_sequence(peptide: str) -> dict[str, int]:
    """Get elemental composition {C, H, N, O, S} for a peptide sequence."""
    from pyteomics.mass import Composition

    comp = Composition(sequence=peptide)
    return {
        "C": comp.get("C", 0),
        "H": comp.get("H", 0),
        "N": comp.get("N", 0),
        "O": comp.get("O", 0),
        "S": comp.get("S", 0),
    }


def averagine_composition(mass: float) -> dict[str, int]:
    """Compute averagine elemental composition for a given mass."""
    return {
        "C": int(round(mass * 0.04443)),
        "H": int(round(mass * 0.06981)),
        "N": int(round(mass * 0.01221)),
        "O": int(round(mass * 0.01329)),
        "S": int(round(mass * 0.00037)),
    }


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors. Returns 0 if either is zero."""
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


def spectral_angle(a: np.ndarray, b: np.ndarray) -> float:
    """Spectral angle: 1 - arccos(cosine) / pi. Range [0, 1], 1 = identical."""
    cos = cosine_similarity(a, b)
    cos = np.clip(cos, -1.0, 1.0)
    return float(1.0 - np.arccos(cos) / np.pi)


def mz_to_mass(mz: float, charge: int) -> float:
    """Convert m/z to neutral mass."""
    return mz * charge - charge * PROTON


def mass_to_mz(mass: float, charge: int) -> float:
    """Convert neutral mass to m/z."""
    return (mass + charge * PROTON) / charge


def ppm_error(observed_mz: float, theoretical_mz: float) -> float:
    """Compute ppm error."""
    return abs(observed_mz - theoretical_mz) / theoretical_mz * 1e6
