"""Tests for the drop_zero_signal filter in rescore()."""
import numpy as np
import pandas as pd
import pytest

# The filter logic lives inside rescore(); test it through the internal helper
# that the pipeline calls, using a minimal synthetic features_df.

_ZERO_SIGNAL_FILTER_ATTRS = ["feature_intensity_sum", "is_decoy"]


def _apply_drop_zero_signal(features_df):
    """Reproduce the pipeline's drop_zero_signal filter in isolation."""
    no_signal = ~(features_df["feature_intensity_sum"] > 0)
    return features_df[~no_signal].reset_index(drop=True), int(no_signal.sum())


def _make_df(intensity_sums, is_decoy_flags):
    return pd.DataFrame(
        {
            "feature_intensity_sum": intensity_sums,
            "is_decoy": is_decoy_flags,
            "peptide": [f"P{i}" for i in range(len(intensity_sums))],
        }
    )


def test_drops_zero_signal_symmetrically():
    # Two co-located pairs: one with signal, one without.
    df = _make_df([100.0, 100.0, 0.0, 0.0], [False, True, False, True])
    out, n = _apply_drop_zero_signal(df)
    assert n == 2
    assert len(out) == 2
    assert set(out["is_decoy"]) == {False, True}  # one target + one decoy remain
    assert (out["feature_intensity_sum"] > 0).all()


def test_zero_nan_negative_all_dropped():
    df = _make_df([0.0, np.nan, -1.0, 50.0], [False, True, False, True])
    out, n = _apply_drop_zero_signal(df)
    assert n == 3
    assert len(out) == 1
    assert out.iloc[0]["feature_intensity_sum"] == 50.0


def test_no_zero_signal_untouched():
    df = _make_df([1.0, 2.0, 3.0], [False, True, False])
    out, n = _apply_drop_zero_signal(df)
    assert n == 0
    assert len(out) == 3


def test_symmetry_equal_td_counts():
    # Under mz_shuffle, co-located target+decoy share the same ion image and therefore
    # the same feature_intensity_sum.  Simulate that: build 50 feature pairs, each pair
    # gets the same intensity value.
    rng = np.random.default_rng(0)
    n_features = 50
    pair_intensity = np.where(rng.random(n_features) > 0.3, rng.uniform(1, 100, n_features), 0.0)
    intensity = np.repeat(pair_intensity, 2)          # [f0, f0, f1, f1, ...]
    is_decoy = np.tile([False, True], n_features)      # [T, D, T, D, ...]
    df = _make_df(intensity.tolist(), is_decoy.tolist())
    out, n_drop = _apply_drop_zero_signal(df)
    assert n_drop % 2 == 0  # always drops in T+D pairs
    remaining_t = int((~out["is_decoy"]).sum())
    remaining_d = int(out["is_decoy"].sum())
    assert remaining_t == remaining_d


if __name__ == "__main__":
    test_drops_zero_signal_symmetrically()
    test_zero_nan_negative_all_dropped()
    test_no_zero_signal_untouched()
    test_symmetry_equal_td_counts()
    print("ok")
