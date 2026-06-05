# MSI-PICASSO — feature correctness audit

Audit date: 2026-05-18 (bugs `kendrick_mass_defect`, `log_mean_intensity`, `spatial_entropy`, `charge_proxy` fixed 2026-05-18). Backend exercised: LDA + balanced_shuffle. Reference run: `results/balanced_shuffle/` (3,452 candidates; 951 target / 2,501 decoy). Visualisations in [`audit_plots/`](../audit_plots/). All file references are relative to `MSI-PICASSO/MSI-PICASSO/`.

## Executive summary

This pass audited every feature in `MALDI_INTRINSIC_FEATURES`, `LCMS_PRIOR_FEATURES`, `SPATIAL_PRIOR_FEATURES`, and `PROTEIN_LEVEL_FEATURES` against first-principles formulas. Four bugs were fixed on 2026-05-18 (see "Fixed in this audit" below). The remaining open items:

1. **`has_oxidized_met` is actually `has_methionine`** (`maldi_features.py:1005`) — the docstring acknowledges it is a *susceptibility proxy*, but the column name implies actual oxidation detection. Plain-sequence candidates never carry `M[Oxidation]` annotations, so the literal definition is unreachable. Open: rename or wire up modification detection.

### Reference

- **Plot index**: per-feature PNGs in [`audit_plots/<feature>.png`](../audit_plots/).
- **One-page overview**: [`audit_plots/summary_grid.png`](../audit_plots/summary_grid.png) — KDEs (target vs decoy) for every audited feature.

---

## Summary table — features audited CORRECT

| Feature | File / function | Plot |
|---|---|---|
| `ppm_error`, `ppm_error_abs` | `candidates.py:match_to_maldi_features` L196,216-217 | [ppm_error_abs.png](../audit_plots/ppm_error_abs.png) |
| `ppm_rank`, `ppm_best_ratio` | `maldi_features.py:compute_mass_accuracy_features` L130-132 | [ppm_rank_best_ratio.png](../audit_plots/ppm_rank_best_ratio.png) |
| `ppm_error_calibrated_z` | `maldi_features.py:compute_calibrated_ppm_features` L596-662 | (only when pixel_coords supplied) |
| `mass_defect_residual` | `maldi_features.py:compute_mass_defect_features` L542-571 | [mass_defect_residual.png](../audit_plots/mass_defect_residual.png) |
| `chca_cluster_distance_ppm` | `maldi_features.py:compute_chca_cluster_features` L578-589 | [chca_cluster_distance_ppm.png](../audit_plots/chca_cluster_distance_ppm.png) |
| `n_candidates`, `log_n_candidates` | `candidates.py:232`, `maldi_features.py:138` | [n_candidates.png](../audit_plots/n_candidates.png) |
| `peptide_length` | `maldi_features.py:165` | [peptide_length.png](../audit_plots/peptide_length.png) |
| `n_missed_cleavages` | `features.rs:76-85` + Python fallback | [n_missed_cleavages.png](../audit_plots/n_missed_cleavages.png) |
| `has_modifications` | `maldi_features.py:172` (hardcoded 0 for plain seqs) | [has_modifications.png](../audit_plots/has_modifications.png) |
| `nterm_basic` | `maldi_features.py:1000` | [nterm_basic.png](../audit_plots/nterm_basic.png) |
| `peptide_pi` | `maldi_features.py:_compute_pi*` + Rust `features.rs:43` | [peptide_pi.png](../audit_plots/peptide_pi.png) |
| `nterm_pyroglu_risk` | `maldi_features.py:1001,1008` | [nterm_pyroglu_risk.png](../audit_plots/nterm_pyroglu_risk.png) |
| `acidic_residue_density` | `maldi_features.py:1009` | [acidic_residue_density.png](../audit_plots/acidic_residue_density.png) |
| `has_cys`, `n_proline`, `n_tryptophan`, `n_tyrosine` | `maldi_features.py:1006-1011` | [simple_residue_counts.png](../audit_plots/simple_residue_counts.png) |
| `n_arginine`, `n_basic_residues`, `n_phenylalanine`, `n_aromatic` | `maldi_features.py:246-250` | [simple_residue_counts.png](../audit_plots/simple_residue_counts.png) |
| `gravy_score` | `maldi_features.py:238-244` + Rust | [gravy_score.png](../audit_plots/gravy_score.png) |
| `theo_isotope_cosine` | `maldi_features.py:504-512` | [theo_isotope_cosine.png](../audit_plots/theo_isotope_cosine.png) |
| `theo_isotope_chi2` | `maldi_features.py:513-516` | [theo_isotope_chi2.png](../audit_plots/theo_isotope_chi2.png) |
| `theo_isotope_kl` | `maldi_features.py:517-522` | [theo_isotope_kl.png](../audit_plots/theo_isotope_kl.png) |
| `theo_m1_ratio_diff`, `theo_m2_ratio_diff` | `maldi_features.py:523-525` | [theo_m1_ratio_diff.png](../audit_plots/theo_m1_ratio_diff.png), [theo_m2_ratio_diff.png](../audit_plots/theo_m2_ratio_diff.png) |
| `theo_has_sulfur` | `maldi_features.py:458` | [theo_has_sulfur.png](../audit_plots/theo_has_sulfur.png) |
| `averagine_deviation`, `averagine_deviation_sulfur` | `maldi_features.py:464-492` | [averagine_deviation.png](../audit_plots/averagine_deviation.png), [averagine_deviation_sulfur.png](../audit_plots/averagine_deviation_sulfur.png) |
| `monoisotopic_confidence` | `maldi_features.py:461-462` — `M0/(M0+M+1)` (2-peak); see design note in UNCLEAR→resolved section | [monoisotopic_confidence.png](../audit_plots/monoisotopic_confidence.png) |
| `log_maldi_intensity*` | `maldi_features.py:176-193` | [log_maldi_intensity.png](../audit_plots/log_maldi_intensity.png), [log_maldi_intensity_p90.png](../audit_plots/log_maldi_intensity_p90.png), [log_maldi_intensity_sum.png](../audit_plots/log_maldi_intensity_sum.png) |
| `lcms_ms2_spectral_angle` (sentinel) | `lcms_evidence.py:849-869` + `utils.spectral_angle` | [lcms_ms2_spectral_angle.png](../audit_plots/lcms_ms2_spectral_angle.png) |
| `lcms_ms2_n_matches` | `lcms_evidence.py:833` | [lcms_ms2_n_matches.png](../audit_plots/lcms_ms2_n_matches.png) |
| `lcms_ms1_intensity` | `lcms_evidence.py:893-910` | [lcms_ms1_intensity.png](../audit_plots/lcms_ms1_intensity.png) |
| `lcms_ms1_snr` (M5 fix) | `lcms_evidence.py:904-926` | [lcms_ms1_snr.png](../audit_plots/lcms_ms1_snr.png) |
| `lcms_ms1_isotope_cosine` (M6 fix: charge-aware) | `lcms_evidence.py:852-955` | [lcms_ms1_isotope_cosine.png](../audit_plots/lcms_ms1_isotope_cosine.png) |
| `theo_m{1,2}_ratio_diff_lcms` | `lcms_evidence.py:948-955` | [theo_m1_ratio_diff_lcms.png](../audit_plots/theo_m1_ratio_diff_lcms.png), [theo_m2_ratio_diff_lcms.png](../audit_plots/theo_m2_ratio_diff_lcms.png) |
| `isotope_envelope_{cosine,pearson,mse}` | `lcms_evidence.py:967-980` | [isotope_envelope_cosine.png](../audit_plots/isotope_envelope_cosine.png), [isotope_envelope_pearson.png](../audit_plots/isotope_envelope_pearson.png), [isotope_envelope_mse.png](../audit_plots/isotope_envelope_mse.png) |
| `isotope_m{1,2}_ratio_diff`, `isotope_n_matched` | `lcms_evidence.py:981-988`, `965` | [isotope_m1_ratio_diff.png](../audit_plots/isotope_m1_ratio_diff.png), [isotope_m2_ratio_diff.png](../audit_plots/isotope_m2_ratio_diff.png), [isotope_n_matched.png](../audit_plots/isotope_n_matched.png) |
| `lcms_ccs_delta`, `lcms_ccs_abs_pct` | `maldi_features.py:compute_lcms_ccs_features` L904-949 | [lcms_ccs_delta.png](../audit_plots/lcms_ccs_delta.png), [lcms_ccs_abs_pct.png](../audit_plots/lcms_ccs_abs_pct.png) |
| `spatial_autocorrelation` (Moran's I) | `maldi_features.py:1222-1271` + `maldi_extraction.py:713-737` | [spatial_autocorrelation.png](../audit_plots/spatial_autocorrelation.png) |
| `spatial_morans_i`, `spatial_gearys_c` | `maldi_features.py:1263-1271` | [spatial_morans_i.png](../audit_plots/spatial_morans_i.png), [spatial_gearys_c.png](../audit_plots/spatial_gearys_c.png) |
| `fraction_detected`, `intensity_cv` | `maldi_extraction.py:_compute_chunk` L692-708 | [fraction_detected.png](../audit_plots/fraction_detected.png), [intensity_cv.png](../audit_plots/intensity_cv.png) |
| `isotope_image_colocalization_{m1,m2,mean}` | `maldi_features.py:1032-1107` | [isotope_image_colocalization_m1.png](../audit_plots/isotope_image_colocalization_m1.png), [_m2](../audit_plots/isotope_image_colocalization_m2.png), [_mean](../audit_plots/isotope_image_colocalization_mean.png) |
| `adduct_colocalization_{na,k,chca}` | `maldi_features.py:1114-1188` | [adduct_colocalization_na.png](../audit_plots/adduct_colocalization_na.png), [_k](../audit_plots/adduct_colocalization_k.png), [_chca](../audit_plots/adduct_colocalization_chca.png) |
| `protein_colocalization*`, `_pearson_r_matrix` | `maldi_features.py:362-435`, `293-336` | [protein_colocalization.png](../audit_plots/protein_colocalization.png) |
| `protein_n_features`, `log_protein_n_features`, `protein_rank`, `protein_best_ratio` | `maldi_features.py:142-160` | [protein_n_features.png](../audit_plots/protein_n_features.png) etc. |
| `_find_matching_ms2_scans` (neutral mass) | `lcms_evidence.py:274-297` | (no plot — matches `mz_to_mass`) |
| `generative_score*` (when computed) | `probabilistic_scorer.py:136-258` | (not in this LDA run; skipped) |
| `im2deep_*` | `maldi_features.py:725-897` | [im2deep_delta_ccs.png](../audit_plots/im2deep_delta_ccs.png), [_abs_delta_ccs_pct](../audit_plots/im2deep_abs_delta_ccs_pct.png), [_zscore](../audit_plots/im2deep_ccs_zscore.png), [_rank](../audit_plots/im2deep_ccs_rank.png) |

---

## Pre-existing audit items (separate from feature correctness)

| Tag | Issue | Status |
|---|---|---|
| H4 | R1 TDC is computed over all candidates, not per-feature winners | Open |
| L2 | All-K/R / single non-K/R peptides → decoy == target | Fixed 2026-05-18 |
| L6 | Mokapot CV at low n | Informational |

---

## Visualisation reference

The full per-feature gallery is in `/home/robbe/MALDI_MSI_score/audit_plots/`. The one-page summary grid combining target-vs-decoy KDEs for every feature is at [`audit_plots/summary_grid.png`](../audit_plots/summary_grid.png).
