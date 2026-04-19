"""Shared utilities: isotope distributions, spectral angle, mass calculations."""

from math import exp, factorial

import numpy as np

NEUTRON = 1.003355
PROTON = 1.007276


def theoretical_isotope_distribution(
    n_C: int,
    n_H: int,
    n_N: int,
    n_O: int,
    n_S: int,
    n_peaks: int = 4,
) -> np.ndarray:
    """
    Compute theoretical isotope distribution using the Poisson approximation.

    Returns normalized distribution [M0, M1, M2, ...].
    """
    lam = (
        n_C * 0.01109
        + n_H * 0.000115
        + n_N * 0.00364
        + n_O * 0.00205
        + n_S * 0.04493
    )
    dist = np.array([exp(-lam) * lam**k / factorial(k) for k in range(n_peaks)])
    total = dist.sum()
    return dist / total if total > 0 else dist


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
