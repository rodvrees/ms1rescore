"""Envelope estimator.

Measured on the amyloid TMA: the plain spatial mean of the M+1 channel is pulled
toward whatever unrelated ion shares the M+1 window, which pinned the observed
M+1/M0 ratio near 0.95 at every mass, against an averagine theory running 0.50
(m/z 800) to 1.02 (m/z 2000).
"""
import numpy as np

from msi_picasso.maldi_extraction import _weighted_isotope_channel


def _synthetic(n_px=2000, ratio=0.55, seed=0):
    """One feature: peptide on part of the section, interferent everywhere else."""
    rng = np.random.default_rng(seed)
    on = rng.random(n_px) < 0.3
    m0 = np.zeros(n_px)
    m0[on] = rng.gamma(2.0, 50.0, on.sum())
    interferent = rng.gamma(2.0, 40.0, n_px)  # unrelated ion in the M+1 window
    m1 = ratio * m0 + interferent
    return m0.reshape(1, n_px), m1.reshape(1, n_px), interferent


def test_weighted_channel_beats_plain_mean():
    ratio = 0.55
    m0, m1, _ = _synthetic(ratio=ratio)
    m0_mean = m0.mean(axis=1)

    weighted = _weighted_isotope_channel(m0, m1, m0_mean) / m0_mean
    plain = m1.mean(axis=1) / m0_mean

    # The plain mean is dragged far above the true ratio by the interferent;
    # weighting by M0 stays much closer to it.
    assert plain[0] > ratio * 2
    assert abs(weighted[0] - ratio) < abs(plain[0] - ratio)


def test_weighted_channel_exact_without_interference():
    rng = np.random.default_rng(1)
    m0 = rng.gamma(2.0, 50.0, (3, 500))
    ratio = np.array([0.4, 0.8, 1.2])
    m1 = m0 * ratio[:, None]
    m0_mean = m0.mean(axis=1)

    got = _weighted_isotope_channel(m0, m1, m0_mean) / m0_mean
    assert np.allclose(got, ratio, atol=1e-5)


def test_weighted_channel_zero_signal_is_zero():
    m0 = np.zeros((2, 100))
    m1 = np.ones((2, 100))
    assert np.all(_weighted_isotope_channel(m0, m1, m0.mean(axis=1)) == 0.0)
