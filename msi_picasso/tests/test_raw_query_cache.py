"""Round-trip and key-sensitivity checks for the on-disk raw-query stats cache."""

import numpy as np

from msi_picasso.maldi_query import (
    _load_obs_stats_cache,
    _obs_stats_cache_path,
    _save_obs_stats_cache,
)


def test_roundtrip_with_peak_quality(tmp_path):
    mzs = np.array([800.0, 1000.5, 1200.25])
    path = _obs_stats_cache_path(str(tmp_path), "/data/x.d", mzs, 25.0, 1, 10.0, 0.02)
    ccs = np.array([300.0, np.nan, 420.5])
    centroid = np.array([800.001, 1000.4, np.nan])
    pq = {"mob_peak_snr": np.array([1.0, 2.0, 3.0]), "mob_k0_spread": np.zeros(3)}

    assert _load_obs_stats_cache(path) is None  # cold
    _save_obs_stats_cache(path, ccs, centroid, pq)

    got_ccs, got_centroid, got_pq = _load_obs_stats_cache(path)
    np.testing.assert_array_equal(got_ccs, ccs)  # NaN-preserving
    np.testing.assert_array_equal(got_centroid, centroid)
    assert set(got_pq) == set(pq)
    for k in pq:
        np.testing.assert_array_equal(got_pq[k], pq[k])


def test_roundtrip_without_peak_quality(tmp_path):
    """No TIMS dimension -> peak_quality is None, and must come back as None, not {}."""
    mzs = np.array([900.0])
    path = _obs_stats_cache_path(str(tmp_path), "/data/x.d", mzs, 25.0, 1, 10.0, 0.02)
    _save_obs_stats_cache(path, np.array([np.nan]), np.array([900.1]), None)
    assert _load_obs_stats_cache(path)[2] is None


def test_key_changes_with_grid_and_params(tmp_path):
    """A different candidate set or window must miss, never silently reuse stale stats."""
    d = str(tmp_path)
    base = _obs_stats_cache_path(d, "/data/x.d", np.array([800.0, 900.0]), 25.0, 1, 10.0, 0.02)
    same = _obs_stats_cache_path(d, "/data/x.d", np.array([800.0, 900.0]), 25.0, 1, 10.0, 0.02)
    assert base == same

    # one m/z moved by 1 mDa -> different decoy set -> must not collide
    assert base != _obs_stats_cache_path(
        d, "/data/x.d", np.array([800.0, 900.001]), 25.0, 1, 10.0, 0.02)
    # extra candidate
    assert base != _obs_stats_cache_path(
        d, "/data/x.d", np.array([800.0, 900.0, 1000.0]), 25.0, 1, 10.0, 0.02)
    # different .d
    assert base != _obs_stats_cache_path(
        d, "/data/y.d", np.array([800.0, 900.0]), 25.0, 1, 10.0, 0.02)
    # different extraction window
    assert base != _obs_stats_cache_path(
        d, "/data/x.d", np.array([800.0, 900.0]), 10.0, 1, 10.0, 0.02)
    # different mobility-quality tolerance
    assert base != _obs_stats_cache_path(
        d, "/data/x.d", np.array([800.0, 900.0]), 25.0, 1, 10.0, 0.05)


def test_corrupt_cache_recomputes(tmp_path):
    path = tmp_path / "obs_stats_deadbeef.npz"
    path.write_bytes(b"not an npz")
    assert _load_obs_stats_cache(str(path)) is None
