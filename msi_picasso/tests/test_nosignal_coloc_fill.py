import numpy as np
import pandas as pd

from msi_picasso.pipeline import _fill_nosignal_coloc_worst_case


def test_nosignal_coloc_worst_case_fill():
    # Two co-located target/decoy pairs: one signal-bearing, one zero-signal.
    df = pd.DataFrame(
        {
            "is_decoy": [False, True, False, True],
            "feature_intensity_sum": [100.0, 100.0, 0.0, 0.0],  # pair 2 = no signal
            "protein_colocalization": [0.8, 0.3, np.nan, np.nan],
            "protein_colocalization_n_partners": [2, 2, np.nan, np.nan],  # count, untouched
        }
    )
    out = _fill_nosignal_coloc_worst_case(df.copy())

    # Zero-signal rows filled with the pooled finite min (worst in-distribution), not median.
    worst = 0.3
    assert out.loc[2, "protein_colocalization"] == worst
    assert out.loc[3, "protein_colocalization"] == worst
    # Symmetric: target and decoy at the no-signal pair get the identical fill.
    assert out.loc[2, "protein_colocalization"] == out.loc[3, "protein_colocalization"]
    # Signal-bearing rows untouched.
    assert out.loc[0, "protein_colocalization"] == 0.8
    assert out.loc[1, "protein_colocalization"] == 0.3
    # _n_partners (a count) is never worst-filled; left NaN for the median imputer.
    assert np.isnan(out.loc[2, "protein_colocalization_n_partners"])


def test_nopartner_nan_left_for_imputer():
    # A single-feature protein has signal but NaN coloc (no within-protein partner).
    # That is "coloc undefined", not "no evidence" -> must NOT be worst-filled.
    df = pd.DataFrame(
        {
            "feature_intensity_sum": [100.0, 100.0],
            "protein_colocalization": [0.5, np.nan],  # row 1 = signal-bearing singleton
        }
    )
    out = _fill_nosignal_coloc_worst_case(df.copy())
    assert np.isnan(out.loc[1, "protein_colocalization"])


if __name__ == "__main__":
    test_nosignal_coloc_worst_case_fill()
    test_nopartner_nan_left_for_imputer()
    print("ok")
