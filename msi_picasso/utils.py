"""Shared utilities: isotope distributions, spectral angle, mass calculations."""

from functools import lru_cache

import numpy as np
from brainpy import isotopic_variants

NEUTRON = 1.003355
PROTON = 1.007276

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

    Returns distribution [M0, M1, M2, ...] truncated to n_peaks, then
    normalized sum-to-1 over the truncated peaks so downstream comparisons
    against sum-to-1 observed envelopes (used by chi² and KL) are unbiased.

    Results are cached by composition tuple — O(unique compositions) calls
    rather than O(n_candidates) when used in a vectorized loop.
    """
    composition = {"C": n_C, "H": n_H, "N": n_N, "O": n_O, "S": n_S}
    # charge only shifts the .mz axis; .intensity is charge-independent, so charge=0 is fine
    peaks = isotopic_variants(composition, npeaks=n_peaks, charge=0)
    intensities = np.array([p.intensity for p in peaks], dtype=float)
    if len(intensities) < n_peaks:
        intensities = np.pad(intensities, (0, n_peaks - len(intensities)))
    intensities = intensities[:n_peaks]
    total = intensities.sum()
    if total < 1e-12:
        return np.zeros(n_peaks, dtype=float)
    intensities /= total
    return intensities


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


AVERAGINE_C = 0.04443
AVERAGINE_H = 0.06981
AVERAGINE_N = 0.01221
AVERAGINE_O = 0.01329
AVERAGINE_S = 0.00037


def averagine_composition(mass: float) -> dict[str, int]:
    """Compute averagine elemental composition for a given mass."""
    return {
        "C": int(round(mass * AVERAGINE_C)),
        "H": int(round(mass * AVERAGINE_H)),
        "N": int(round(mass * AVERAGINE_N)),
        "O": int(round(mass * AVERAGINE_O)),
        "S": int(round(mass * AVERAGINE_S)),
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