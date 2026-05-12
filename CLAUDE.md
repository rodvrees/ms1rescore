# ms1rescore — CLAUDE.md

## Purpose

`ms1rescore` is a symmetric target-decoy rescoring package for MALDI-MSI MS1 data. It takes MALDI feature m/z values, a protein FASTA, and optional LC-MS/MS mzML files, then produces FDR-controlled peptide identifications via mokapot (SVM-based semi-supervised rescoring).

**Why it was built this way:** The prior approach (ms2rescore "Approach B") used ProteomeDiscoverer (PD) output for candidates and features. This introduced label leakage — `lcms_xcorr` (a PD search engine score) had AUC 0.993 and was a near-perfect surrogate for the PD target/decoy label, making rescoring trivial but invalid. This package replaces that with:
- Candidates from in-silico tryptic digest of forward + reversed FASTA (no PD)
- All LC-MS/MS features computed from raw mzML (no PD-derived scores)
- Strict symmetric design: **no feature computation function takes `is_decoy` as a parameter**

---

## Repository layout

```
ms1rescore/
├── pyproject.toml              # Package metadata and dependencies
├── CLAUDE.md                   # This file
├── ms1rescore/                 # Python package
│   ├── __init__.py             # __version__ = "0.1.0"
│   ├── utils.py                # Shared math utilities
│   ├── candidates.py           # FASTA digest + MALDI m/z matching
│   ├── lcms_ids.py             # Parse LC-MS/MS IDs for Strategy C candidate generation
│   ├── lcms_evidence.py        # LC-MS/MS feature extraction
│   ├── maldi_extraction.py     # Raw MALDI extraction: feature detection, ion images, spatial features
│   ├── maldi_features.py       # MALDI-side rescoring features
│   ├── feature_generator.py    # Orchestration + PSMList construction
│   ├── pipeline.py             # End-to-end pipeline function
│   └── tests/                  # Unit tests (pytest; testpaths configured in pyproject.toml)
│       ├── fixtures/           # Static test fixtures (e.g. test_maldi.mgf)
│       ├── test_deisotoping.py
│       ├── test_evidence_score.py
│       ├── test_isotope_distribution.py
│       ├── test_lcms_ids.py
│       └── test_maldi_extraction.py
└── ms1rescore-rs/              # Rust extension (PyO3 + rayon)
    ├── Cargo.toml
    └── src/
        ├── lib.rs              # PyO3 module definition
        ├── digest.rs           # Peptide mass + composition + m/z matching
        ├── xic.rs              # Parallel XIC extraction
        ├── spectral.rs         # Spectral angle computation
        └── isotope.rs          # MS1 isotope envelope extraction
```

Run the test suite from the `ms1rescore/` directory:

```bash
cd ms1rescore && pytest   # testpaths = ["ms1rescore/tests"] in pyproject.toml
```

---

## Core design principles

### 1. Symmetric target-decoy computation

Every function that computes features must be blind to `is_decoy`. The symmetry guarantee is enforced at the API level: none of the feature computation functions have an `is_decoy` parameter. This means targets and decoys receive features via identical code paths.

### 2. Decoy generation via K/R-preserving protein-level shuffle

`_shuffle_protein(seq, random_state=42)` in [candidates.py](ms1rescore/candidates.py) keeps K and R residues at their original positions and randomly shuffles all other residues (using a seeded RNG for reproducibility), then digests the shuffled sequence with the same trypsin rules as the target.

**Why K/R-preserving shuffle instead of K/R-preserving reversal:** Reversal with K/R fixed produces decoy peptides that are often isobaric with targets — the elemental composition of the non-K/R residues is unchanged (same multiset, just reversed). This makes isotope envelope features (`theo_isotope_cosine`, `theo_isotope_chi2`, `theo_isotope_kl`, `isotope_envelope_*`) non-discriminative. Shuffling (rather than reversing) the non-K/R residues changes which residues appear in each tryptic fragment, breaking elemental composition conservation at the peptide level.

**Why keep K/R in place:** This preserves tryptic cleavage sites, so the decoy protein is digested at exactly the same positions as the target. Decoy peptides therefore have the same length distribution and the same C-terminal residue as their target counterparts, keeping the TDC null model valid.

### 3. Neutral mass matching for LC-MS/MS

MALDI features are detected as [M+H]+ (charge 1). LC-MS/MS MS2 scans are acquired at charge 2, 3, etc. Matching is done by comparing **neutral masses** (`maldi_mz - PROTON` vs `ms2_precursor_mz * charge - charge * PROTON`), not m/z values directly. Matching on m/z alone gives ~88/1398 features with MS2 scans; neutral mass matching gives ~1067/1398.

### 4. Multi-charge XIC extraction

XICs in LC-MS/MS are extracted at charges 1, 2, 3, 4 for each MALDI feature's neutral mass. The charge producing the highest XIC peak intensity is selected as the "best charge" and stored as a feature (`lcms_xic_best_charge`). Without multi-charge search, only ~595/1398 features have detectable XICs; with it, ~1085/1398 do.

---

## Modules

### `maldi_extraction.py`

Converts raw Bruker `.d`/TSF data into the NPZ format consumed by the rest of the pipeline via `imzy`. CLI flag: `--maldi-raw`.

Install: `pip install ms1rescore[maldi]` (installs `imzy`).

#### `extract_maldi_data(d_path, ppm_bin, extraction_ppm, matching_ppm, min_fraction, feature_mzs, images_path, image_batch_size, output_npz, output_spatial_tsv, output_dir, verbose)`

Three-step extraction:

1. **Feature detection** — `detect_features(reader, ppm_bin, min_fraction)`. Streams all pixels, builds a log-ppm histogram, and greedily merges adjacent bins to produce consensus feature m/z values. Works directly on centroid data (instrument-picked peaks). Profile data is not handled here — use `maldi_imzml.py` for profile imzML.

2. **`extract_ion_images(reader, feature_mzs, ppm=25.0)`** — tries fast extraction paths first (`_extract_centroid_fast` for Bruker TSF, `_extract_profile_fast` cumsum trick for profile), then falls back to `reader.get_ion_images()`. Returns `(n_features, H, W)` float32.

3. **`compute_spatial_features(ion_images, feature_mzs, n_pixels_total)`** — computes per-feature: `fraction_detected`, `n_pixels_detected`, `mean_intensity`, `intensity_p90`, `intensity_sum`, `intensity_cv`, Moran's I (`spatial_autocorrelation`).

**imzy reader API used:**
- `reader.n_pixels`, `reader.is_centroid`, `reader.mz_min`, `reader.mz_max`
- `reader.spectra_iter(silent=False)` — yields `(mzs, ints)` per pixel
- `reader.get_ion_images(mzs, ppm=..., fill_value=0.0)` → `(n_features, H, W)` float32
- Used as context manager: `with imzy.get_reader(d_path) as reader:`

**Two ppm parameters:**

- `extraction_ppm` (default 25.0, CLI `--extraction-ppm`): ion image assembly window. Slightly wider than instrument centroid accuracy to avoid clipping peak tails.
- `matching_ppm` (default 20.0, CLI `--matching-ppm`): downstream candidate-to-feature linking in `match_to_maldi_features`. Not used internally by the extraction functions.

**Deprecated: LC-MS/MS guided extraction** (`_features_from_lcms_file_diagnostic`):

Kept for diagnostic/legacy use only. Do not use for rescoring — it pre-selects features guaranteed to match LC-MS/MS candidates, making scoring trivial.

---

### `maldi_imzml.py`

SCiLS Lab-style interval-based m/z feature extraction for imzML data. CLI flag: `--maldi-imzml`.

Install: `pip install ms1rescore[maldi]` (installs `pyimzml`).

#### Algorithm

1. **Build mean spectrum** across all pixels on a common m/z grid (`mz_grid_resolution` Da resolution). Fast path for aligned profile data (all spectra share the same m/z axis): direct array sum. General path: `np.add.at` with grid index = `round((mz - mz_min) / resolution)`, O(n_peaks) per pixel. Each pixel spectrum is optionally RMS- or TIC-normalized before accumulation so every pixel contributes equally.

2. **Baseline correction** (optional, `baseline_correction=True`): rolling-minimum filter (`scipy.ndimage.minimum_filter1d`) + Gaussian smoothing of the baseline estimate, subtracted from the mean spectrum before peak detection. Window width in points computed at median m/z from `baseline_window_ppm`.

3. **Detect intervals** on the (optionally baseline-corrected) mean spectrum: Savitzky-Golay smooth → `find_peaks(height=threshold)` → valley-to-valley boundaries. The height threshold matches the paper's "4% relative intensity threshold" (peaks above 4% of the base peak, as an absolute height filter). **Do not use scipy's topographic `prominence=` here** — prominence is the peak height above its lowest connecting saddle to a higher peak, which can be far below the absolute height for peaks surrounded by moderately intense neighbours, causing genuine peaks to be missed. When `local_prominence_window_da > 0`, the threshold for each peak is `peak_prominence × local_max(±window)` (sliding-window local max) instead of `peak_prominence × global_max`. This reduces the effective threshold in low-signal m/z regions (e.g. >1600 Da) where the global threshold would suppress genuine peptide peaks. For each detected peak, left boundary = last valley before the peak, right boundary = first valley after the peak. Fallback to `mz_apex ± ppm_tolerance × mz_apex × 1e-6` when no flanking valley exists. Intervals narrower than `min_interval_width_ppm` are symmetrically expanded.

4. **Recalibration** (optional, `calibrant_mzs` non-empty): for each calibrant m/z, find the nearest detected apex within `calibrant_tol_ppm`; fit a linear ppm offset vs m/z model; apply correction to all interval apices and boundaries.

5. **Deisotoping** (optional, `deisotope=True`): uses the `ms_deisotope` package (`deconvolute_peaks()`) to remove isotope satellite peaks. A pre-merge step (`_merge_duplicate_intervals`) collapses near-identical apices within `tol_da=0.001` Da (taking max intensity) before deconvolution.

   `ms_deisotope` fits averagine-based isotope envelopes using MSDeconV scoring. Peaks that appear as k>0 positions in a fitted envelope (score ≥ `deisotope_min_score`) are removed as satellites. Peaks not assigned to any envelope are kept.

   Key behavioral properties:
   - Envelopes where M+k is far more intense than M0 score below threshold and are not removed (ms_deisotope returns empty `peak_set` for physically inconsistent intensity ratios).
   - Error tolerance is in ppm (`deisotope_error_ppm`), not Da. At 1000 Da, 15 ppm ≈ 0.015 Da (stricter than the old 0.15 Da default).
   - Charge range is configurable via `deisotope_charge_range` (default `(1, 1)` for MALDI [M+H]+).
   - Averagine model is configurable: `"peptide"` (default), `"glycopeptide"`, `"glycan"`, `"heparin"`.

   **Note**: in single-cell MALDI-MSI data, M+1 peaks often survive deisotoping because heterogeneous cell expression patterns cause the mean spectrum M0/M+1 ratio to be lower than the averagine prediction. This is a fundamental limitation of mean-spectrum deisotoping; per-pixel deisotoping would be needed to eliminate all isotope peaks reliably.

6. **Mass defect filter** (optional, `filter_mass_defect=True`): Senko-plot peptide corridor. For each [M+H]+ apex: `neutral = apex − 1.007276`, `nominal = floor(neutral)`, `defect = neutral − nominal` (always in [0,1)), `expected = 0.000509 × nominal` (averagine slope, positive). Keep if `|defect − expected| ≤ mass_defect_halfwidth`. Default halfwidth 0.5 passes all peaks; use 0.25 to filter lipids/matrix while retaining all tryptic peptides in 800–2000 Da. **Do not use `round()` for nominal mass** — for neutrals with fractional part >0.5 it flips the defect sign and breaks the corridor test.

7. **Integrate pixels** over intervals. RMS normalization (if `normalize_rms=True`, takes priority) or TIC normalization (if `normalize_tic=True`) is applied to each pixel spectrum before interval integration. Per interval: sum all intensities within `[mz_start, mz_end]` (`use_apex=False`) or take the apex intensity (`use_apex=True`).

8. **Filter intervals**: keep intervals where mean intensity ≥ `min_intensity` AND pixel fraction with non-zero signal ≥ `min_pixel_fraction`.

#### `SCiLSConfig` dataclass

```python
SCiLSConfig(
    ppm_tolerance=10.0,           # fallback interval half-width (ppm)
    smoothing_window=7,
    smoothing_polyorder=2,
    peak_prominence=0.01,         # min peak prominence as fraction of mean-spectrum max
    min_intensity=0.0,
    min_pixel_fraction=0.01,
    min_interval_width_ppm=2.0,
    normalize_tic=True,           # TIC-normalize per pixel (default)
    normalize_rms=False,          # RMS-normalize per pixel (takes priority over TIC; matches SCiLS default)
    use_apex=False,               # False = sum; True = apex intensity
    mz_grid_resolution=0.001,    # Da, for mean spectrum grid
    # Baseline correction
    baseline_correction=False,
    baseline_window_ppm=500.0,
    # Recalibration
    calibrant_mzs=[],             # theoretical m/z of internal standards
    calibrant_tol_ppm=200.0,
    # Deisotoping (ms_deisotope-based)
    deisotope=False,
    deisotope_averagine="peptide",   # "peptide", "glycopeptide", "glycan", "heparin"
    deisotope_scorer="MSDeconVFitter",  # or "PenalizedMSDeconVFitter"
    deisotope_min_score=10.0,        # MSDeconV score threshold; lower = more aggressive
    deisotope_charge_range=(1, 1),   # MALDI is [M+H]+; widen for multi-charge data
    deisotope_error_ppm=15.0,        # m/z matching tolerance in ppm (15 ppm ≈ 0.015 Da at 1000 Da)
    # Senko mass defect filter
    filter_mass_defect=False,
    mass_defect_halfwidth=0.5,    # 0.5 = all pass; 0.15–0.20 = meaningful peptide filter
    # Local adaptive prominence (0 = global max reference)
    local_prominence_window_da=0.0,  # >0 enables sliding-window local max; suggested 200 Da
)
```

#### Published pipeline replication (PXD056528 CHCA dataset)

The paper describes: SCiLS RMS normalization → mMass baseline correction → peak picking at 4% relative threshold → deisotoping → recalibration with trypsin autolysis peaks (842.51, 870.54, 1045.56 Da) → Senko mass defect filter → spatial filter (fraction of pixels).

```bash
ms1rescore \
  -f data/PXD056528/uniprot_human_reviewed.fasta \
  -l data/PXD056528/231212_AG_11.mzML \
  -l data/PXD056528/231212_AG_12.mzML \
  -l data/PXD056528/231212_AG_21.mzML \
  --maldi-raw data/PXD056528/MALDI_MSI/20221013_SingleCells_CHCA.d \
  --msf data/PXD056528/240125_AG_DDA.msf \
  --model svm \
  --normalize-rms \
  --baseline-correction \
  --peak-prominence 0.04 \
  --calibrant-mzs 842.51 870.54 1045.56 \
  --deisotope \
  --filter-mass-defect --mass-defect-halfwidth 0.25 \
  --min-fraction 0.01 \
  --smoothing-window 11 --smoothing-polyorder 2 \
  --output-dir results/replicated_pipeline/ \
  --extra-fasta data/contaminants.fasta \
  -v
```

**Parameter notes:**
- `--normalize-rms`: matches SCiLS Lab default (RMS, not TIC). Takes priority over the default TIC normalization.
- `--peak-prominence 0.04`: matches the paper's "4% relative intensity threshold".
- `--calibrant-mzs`: trypsin autolysis peaks used as internal mass standards. `--calibrant-tol-ppm` defaults to 200 ppm (wide enough to find them before recalibration).
- `--mass-defect-halfwidth 0.25`: covers all 22 GT tryptic peptides in 800–2000 Da while filtering lipids/matrix. Use 0.5 (default) to disable the filter.
- `--picking-height 0.75` (default): matches the mMass "picking height 75%" setting. Computes apex m/z as the midpoint of the two interpolated crossings at 75% of the peak maximum, giving a more accurate centroid for asymmetric peaks. Use `--picking-height 0.0` to revert to the raw smoothed-spectrum apex.
- `--smoothing-window 11 --smoothing-polyorder 1`: best config from parameter sweep on this dataset (v11: 17/22 GT features matched).
- `--deisotope-min-score 10.0` (default): MSDeconV score threshold for accepting an isotope envelope. Lower values remove more peaks; raise to 20+ to be conservative. At 10.0, a clear 3-peak M0/M+1/M+2 pattern matching averagine scores ~20.
- `--deisotope-error-ppm 15.0` (default): m/z tolerance for envelope fitting. At 1000 Da, 15 ppm ≈ 0.015 Da. The old `--deisotope-tol-da 0.15` was 150 ppm at 1000 Da (10× looser).
- `--deisotope-averagine peptide` (default): averagine model. Use `glycopeptide`, `glycan`, or `heparin` for non-peptide analytes.
- `--deisotope-scorer MSDeconVFitter` (default): scoring function. `PenalizedMSDeconVFitter` applies an additional penalty for charge-state ambiguity.

#### `extract_scils_features(imzml_path, config, output_dir, visualize)`

Returns `(intervals, intensity_matrix, pixel_coords)`:
- `intervals`: list of `(mz_start, mz_end, mz_apex)` tuples, one per kept interval
- `intensity_matrix`: `np.ndarray` shape `(n_pixels, n_intervals)`, float32
- `pixel_coords`: list of `(x, y)` 0-based tuples, same row order as `intensity_matrix`

When `visualize=True`, saves 4 PNG files to `output_dir`:
1. Mean spectrum with interval shading
2. Interval apex m/z histogram
3. Per-interval pixel fraction and mean intensity distributions
4. Ion image mosaic for top 9 intervals by mean intensity (γ=0.5, `hot` colormap)

**pyimzml API used:**
- `parser.coordinates` — list of (x, y, z) 1-based tuples
- `parser.spectrum_mode` — `'profile'` | `'centroid'` | `None`
- `parser.getspectrum(i)` — seeks ibd file, returns `(mzs, ints)` arrays

#### CLI flags

| Flag | Format | Feature detection | Ion images | Spatial features |
|---|---|---|---|---|
| `--maldi-npz PATH` | NumPy NPZ | No | Yes (if `"images"` key) | No |
| `--maldi-mzs PATH` | Plain text m/z list | No | No | No |
| `--maldi-raw PATH` | Bruker `.d` | Histogram binning (centroid) | Yes | Yes |
| `--maldi-imzml PATH` | imzML + ibd | SCiLS interval detection | No (interval matrix) | No |

---

### `utils.py`

Shared mathematical utilities. No external state.

| Function | Description |
|---|---|
| `theoretical_isotope_distribution(n_C, n_H, n_N, n_O, n_S, n_peaks=4)` | brainpy Mercury algorithm; `@lru_cache` keyed on composition tuple |
| `composition_from_sequence(peptide)` | Element counts from amino acid sequence |
| `averagine_composition(mass)` | Averagine model composition |
| `cosine_similarity(a, b)` | Safe cosine with zero-vector guard |
| `spectral_angle(a, b)` | 1 - arccos(cosine) / π |
| `mz_to_mass(mz, charge)` | Neutral mass from m/z and charge |
| `mass_to_mz(mass, charge)` | m/z from neutral mass and charge |
| `ppm_error(observed, theoretical)` | Signed ppm |

Constants: `NEUTRON = 1.003355`, `PROTON = 1.007276`

**`theoretical_isotope_distribution` — normalization scheme:** brainpy is called with `npeaks = max(n_peaks, 6)` to ensure the normalization denominator captures essentially all isotope signal. The returned intensities are normalized over all returned peaks (full-spectrum norm), then truncated to `n_peaks`. As a result, the sum of the returned array is < 1 for `n_peaks < 6` (remaining signal is in M+3+). This is more physically correct than the prior Poisson approach, which renormalized only over the first 3 peaks. The `lru_cache` makes repeated calls with the same composition free — the vectorized path in `maldi_features.py` builds a dict of unique compositions first to minimize brainpy calls to O(unique compositions).

### `candidates.py`

Two candidate generation strategies are supported, both producing a DataFrame with one row per (peptide, MALDI feature) pair.

**`digest_fasta()` — Strategy A (full FASTA):**
1. Phase 1 (pyteomics): `pyteomics.parser.cleave()` for tryptic digestion. Decoys generated by `_shuffle_protein()` (K/R-preserving shuffle, seeded at 42).
2. Phase 2 (Rust or pyteomics fallback): `compute_peptide_masses()` from `ms1rescore_rs` computes mass, [M+H]+ m/z, and elemental composition (n_C, n_H, n_N, n_O, n_S).

**`digest_identified_proteins()` — Strategy C (LC-MS/MS-guided):**
See the [Candidate generation strategies](#candidate-generation-strategies) section below.

`match_to_maldi_features()` uses `match_mz()` from `ms1rescore_rs` (binary search) or Python fallback. Returns a candidates DataFrame with one row per (peptide, MALDI feature) pair. Protein-level features (`protein_n_features`, `n_candidates`) are computed over all candidates symmetrically.

Key parameters: `maldi_intensities`, `maldi_intensities_p90`, `maldi_intensities_sum` (each a numpy array aligned with `maldi_mzs`) populate `feature_intensity`, `feature_intensity_p90`, and `feature_intensity_sum` respectively, from which `log_maldi_intensity`, `log_maldi_intensity_p90`, and `log_maldi_intensity_sum` are derived. Prefer `intensity_p90` from `compute_spatial_features()` over mean-of-nonzero for the main intensity feature. In `pipeline.py`, these are read from `spatial_features` columns when available.

### `lcms_evidence.py`

The most complex module. Handles all LC-MS/MS evidence extraction.

#### `LCMSData` dataclass

Holds all MS1 and MS2 scan data loaded from mzML or Bruker `.d` files. Lazily computes:
- `_ms2_neutral_mass`: neutral mass from `ms2_precursor_mz * charge - charge * PROTON`
- `_ms2_mass_sort_idx`: argsort for binary search over neutral masses

#### Key functions

| Function | Description |
|---|---|
| `load_lcms_data(mzml_paths, cache_path)` | Load mzML via pyteomics or Bruker `.d` via alphatims; routes based on extension |
| `load_lcms_data_from_d(d_path, cache_path)` | Load timsTOF `.d` folder with alphatims; MS1 per-frame, MS2 vectorised via `index_precursors()` |
| `_find_matching_ms2_scans(neutral_mass, lcms_data, ppm)` | Binary search over MS2 neutral masses |
| `get_ms2pip_predictions(pairs, model, cache_path)` | Batch MS2PIP predictions for `(peptide, charge)` pairs. Import: `from ms2pip.core import predict_batch` |
| `finetune_deeplc(msf_path, cache_path)` | Fine-tune DeepLC on PD TargetPsms (q≤0.01) |
| `get_deeplc_predictions(peptides, model, cache_path)` | Batch DeepLC RT predictions |
| `extract_all_xics(unique_mzs, lcms_data, ppm)` | Multi-charge XIC extraction; uses Rust if available |
| `compute_all_lcms_evidence(candidates_df, ...)` | Main entry point: returns dict mapping candidate index → feature dict |

#### `compute_all_lcms_evidence` structure

Restructured to be **feature-first**, not candidate-first:
1. Pre-compute per MALDI feature (1,398 iterations, not 707K):
   - Matching MS2 scan indices (by neutral mass)
   - XIC at charges 1-4; select best charge; store `best_xic_mz`, `best_ms1_idx`
   - XIC features: `xic_max_intensity`, `xic_n_scans`, `xic_snr`
2. Per-candidate loop (707K iterations): only peptide-specific computations:
   - Spectral angle vs MS2PIP prediction (peptide+charge specific)
   - RT residual (peptide-specific DeepLC prediction)
   - MS1 isotope cosine at best XIC scan

#### DeepLC finetuning SQL

```sql
SELECT DISTINCT Sequence AS peptide, RetentionTime AS rt
FROM TargetPsms
WHERE PercolatorqValue <= 0.01 AND RetentionTime IS NOT NULL
```
`RetentionTime` is taken directly from `TargetPsms` — no join needed.

### `maldi_features.py`

MALDI-side features. All functions take the candidates DataFrame and return it with new columns added.

#### Ion-image colocalization and spatial autocorrelation

All four ion-image feature functions are performance-critical. Their design:

**`_pearson_r_matrix(ion_images, ion_image_mzs)`** — shared helper. Stacks all valid (non-constant) ion images into a `(n_valid, n_pixels)` float32 matrix and computes the full `(n_valid, n_valid)` Pearson correlation matrix in a single BLAS `dgemm` call via `np.corrcoef`. Called once by `feature_generator.compute_all_features` and passed as `_corr_cache` to all three colocalization functions to avoid 3× redundant BLAS work.

**`compute_colocalization_features()`** — within-protein Pearson correlations:
1. `_pearson_r_matrix` → full corr matrix (shared with other functions)
2. Pandas self-join on `protein` to enumerate all within-protein feature pairs (O(Σ k²) rows where k = features per protein, typically small)
3. Vectorized `corr_matrix[idx_a, idx_b]` lookup on the join result
4. `groupby(['feature_mz', 'protein']).agg(mean/max/median/count)` → merge back onto candidates

**`compute_isotopologue_colocalization()`** and **`compute_adduct_colocalization()`** — both use `_find_partner_indices` (vectorized `searchsorted` + nearest-neighbour check) to locate M+1/M+2 or Na/K/CHCA adduct images for all features simultaneously, then slice the shared corr matrix.

**`compute_spatial_autocorrelation_full()`** — Moran's I and Geary's C:
- Replaces per-feature `scipy.signal.convolve2d` with `_neighbor_sum_batch`: batched numpy 8-neighbour sum using zero-padded slicing, no scipy dependency.
- Processes features in chunks of `chunk_size=200` (caps peak RAM at ~150 MB per chunk for a 49 K-pixel image at float32).
- Chunks dispatched to a `ThreadPoolExecutor` — numpy releases the GIL for large array ops, so threads run in parallel.
- Uses float32 throughout; only final reduction sums accumulate in float64.

**Benchmark at 1398 features, 220×225 px (49,500 pixels), realistic candidate set:**

| Step | Time |
|---|---|
| `_pearson_r_matrix` (shared, once) | ~1.4 s |
| `compute_colocalization_features` | ~0.03 s |
| `compute_isotopologue_colocalization` | ~0.01 s |
| `compute_adduct_colocalization` | ~0.01 s |
| `compute_spatial_autocorrelation_full` | ~0.8 s |

#### `compute_theoretical_isotope_features()`

Uses `theoretical_isotope_distribution()` from `utils.py` (brainpy-backed, `lru_cache`). The hot path deduplicates compositions before calling brainpy:

```python
comp_cols = ["n_C", "n_H", "n_N", "n_O", "n_S"]
unique_comps = {tuple(row) for row in df[comp_cols].astype(int).values}
iso_cache = {k: theoretical_isotope_distribution(*k, n_peaks=3) for k in unique_comps}
dist = np.array([iso_cache[tuple(row)] for row in df[comp_cols].astype(int).values])
theo_m0, theo_m1, theo_m2 = dist[:, 0], dist[:, 1], dist[:, 2]
```

The averagine comparison path (for `averagine_deviation` features) uses the same function with `n_S=0` (consistent with the prior behaviour of omitting sulfur from the averagine model). Uses `n_C, n_H, n_N, n_O, n_S, mass` columns from `digest_fasta()`.

### `feature_generator.py`

Orchestrates feature computation and PSMList construction.

`candidates_to_psm_list()` uses `itertuples()` (not `iterrows()`) for ~5x speedup. MALDI PSMs are always charge 1 (`Peptidoform(f"{peptide}/1")`).

Two named feature group lists are exported from this module — see "Feature groups" section below.

### Feature groups

`feature_generator.py` exports three lists that are importable directly:

```python
from ms1rescore.feature_generator import (
    MALDI_INTRINSIC_FEATURES, PROTEIN_LEVEL_FEATURES, LCMS_PRIOR_FEATURES
)
```

**`MALDI_INTRINSIC_FEATURES`** — features passed to the ranker by default; computable from MALDI data alone plus in-silico properties:
- Mass accuracy: `ppm_error_abs`, `ppm_rank`, `ppm_best_ratio`
- Ambiguity: `n_candidates`, `log_n_candidates`
- Peptide: `peptide_length`, `n_missed_cleavages`, `has_modifications`
- MALDI signal: `log_maldi_intensity`
- Theoretical isotope: `theo_isotope_cosine`, `theo_isotope_chi2`, `theo_isotope_kl`, `theo_has_sulfur`, `averagine_deviation`, `averagine_deviation_sulfur`, `theo_m1_ratio_diff`, `theo_m2_ratio_diff`
- Ionization priors: `n_arginine`, `n_basic_residues`, `n_phenylalanine`, `n_aromatic`, `gravy_score`, `charge_proxy`
- Spatial (optional): `spatial_autocorrelation`, `fraction_detected`, `intensity_cv`, `log_mean_intensity`, `spatial_entropy`
- Observed isotope envelope (optional): `isotope_envelope_cosine`, `isotope_envelope_pearson`, `isotope_envelope_mse`, `isotope_m1_ratio_diff`, `isotope_m2_ratio_diff`, `isotope_n_matched`

**`PROTEIN_LEVEL_FEATURES`** — excluded from the ranker by default; opt-in via `--use-protein-level-feats`:
- Protein consistency: `protein_n_features`, `log_protein_n_features`, `protein_coverage`, `protein_rank`, `protein_best_ratio`
- Protein co-localization (optional, requires ion_images): `protein_colocalization`, `protein_colocalization_max`, `protein_colocalization_median`, `protein_colocalization_n_partners`

**Why excluded by default:** these features aggregate counts and correlations over all candidates sharing a protein, including decoys. A decoy peptide whose protein happens to have many target matches inherits artificially high `protein_n_features` / colocalization values. This breaks TDC null-model symmetry. Use `--use-protein-level-feats` only if you understand and accept this trade-off.

**`LCMS_PRIOR_FEATURES`** — two sub-groups, both excluded from the ranker:

*mzML-derived* (`_LCMS_MZML_FEATURES`): `lcms_ms2_spectral_angle`, `lcms_ms2_n_matches`, `lcms_xic_max_intensity`, `lcms_xic_n_scans`, `lcms_xic_snr`, `lcms_xic_best_charge`, `lcms_rt_residual`, `lcms_ms1_isotope_cosine`, `theo_m1_ratio_diff_lcms`, `theo_m2_ratio_diff_lcms`

*ID-derived* (`_LCMS_ID_FEATURES`, Strategy C only): `lcms_q_value`, `lcms_pep`, `lcms_score`, `n_psms`, `lcms_intensity`, `source_lcms_confirmed`

**Design rationale:** LC-MS/MS features are explicitly excluded from the ranker/SVM training set. Passing them to the model would cause it to score LC-MS/MS identification quality rather than MALDI match quality. Instead, LC-MS/MS evidence is applied as a multiplicative Bayesian prior *after* MALDI-intrinsic scoring (see `compute_lcms_prior()` in `pipeline.py`). `source_lcms_confirmed` receives 2× weight in the prior because a direct LC-MS/MS identification is very strong evidence. `lcms_q_value` and `lcms_pep` are inverted (1 − value) before normalization since lower values indicate stronger evidence.

`get_feature_names()` returns `MALDI_INTRINSIC_FEATURES + LCMS_PRIOR_FEATURES` for backwards compatibility (optional groups included only when data was computed).

---

## Candidate generation strategies

### Strategy A — full FASTA (default)

`digest_fasta(fasta_path, ...)` digests all proteins in the FASTA and generates K/R-preserving shuffled decoys. Used when `rescore()` is called without `lcms_peptides_path`.

### Strategy C — LC-MS/MS guided

Activated in `rescore()` by passing `lcms_peptides_path`. Implemented in `lcms_ids.py` + `digest_identified_proteins()`.

**Candidate set** = in-silico digest of identified proteins ∪ directly identified LC-MS/MS peptides.

The `source` column on the candidates DataFrame encodes the origin of each row:

| `source` value | Meaning |
|---|---|
| `"protein_digest"` | From in-silico digest of an identified protein; not directly observed in LC-MS/MS |
| `"lcms_confirmed"` | Directly identified in LC-MS/MS at the specified FDR; also included in digest when protein is identified |
| `"decoy"` | K/R-preserving shuffle of an identified protein, or peptide-level shuffle for novel confirmed sequences |

Novel `"lcms_confirmed"` sequences (peptides passing FDR but whose parent protein is not in the FASTA, or not reachable by tryptic digestion at the chosen parameters) are added as targets. A K/R-preserving shuffle of the peptide sequence itself is used as their decoy.

LC-MS/MS evidence is joined onto target rows from `lcms_ids.peptides`; decoy rows always get `NaN`. The binary `source_lcms_confirmed` feature is computed in `compute_all_features()` and is 1.0 for any `"lcms_confirmed"` candidate.

**Fallback:** if `digest_identified_proteins()` returns 0 rows (e.g. no FASTA proteins found), `rescore()` falls back to Strategy A with a warning.

### `lcms_ids.py`

Parses LC-MS/MS identification results into an `LCMSIds(proteins, peptides)` namedtuple.

| Component | Type | Content |
|---|---|---|
| `proteins` | `set[str]` | Normalised accessions passing `protein_fdr` |
| `peptides` | `pd.DataFrame` | Unique sequences passing `peptide_fdr`, with evidence columns |

**Peptide DataFrame columns:** `sequence`, `peptidoform`, `protein`, `q_value`, `pep`, `score`, `n_psms`, `charge`, `rt_mean`, `lcms_intensity`

**Supported formats** (pass as `lcms_id_format` to `rescore()`):

| Format | Files needed | Notes |
|---|---|---|
| `"percolator"` (default) | `peptides_path` (required), `proteins_path` (optional), `psms_path` (optional) | Column names auto-discovered by partial lowercase match; psms file used for RT/intensity aggregation |
| `"mzidentml"` | single mzIdentML file as `peptides_path` | q-value from CV `MS:1002354`, PEP from `MS:1002356` |
| `"psm_utils"` | any psm_utils-supported file as `peptides_path` | Aggregated to peptide level |
| `"msf"` | ProteomeDiscoverer `.msf` SQLite file as `peptides_path` | Queries `TargetPsms` joined with `TargetProteins`; filters by `PercolatorqValue <= peptide_fdr`; no separate PEP stored (`pep` column is NaN). In the CLI, passing `--msf` without `--lcms-peptides` automatically activates Strategy C using the MSF as the ID source. |

**Accession normalisation** (`_normalize_accession`): strips UniProt/RefSeq prefixes before comparing against the FASTA:

```
sp|P12345|GENE_HUMAN  →  P12345
tr|A0A000|GENE_HUMAN  →  A0A000
P12345 some description  →  P12345
P12345  →  P12345
```

`filter_fasta_to_proteins(fasta_path, protein_accessions)` warns if fewer than 50% of requested accessions are found — this typically indicates an accession format mismatch between the LC-MS/MS search database and the supplied FASTA.

### `pipeline.py`

`rescore()` is the end-to-end entry point. Steps 1-8 are identical for both backends; step 9 diverges:

1. Generate candidates (Strategy A or C) + match to MALDI features
2. Load LC-MS/MS data
3. Find MS2 matches by neutral mass; run MS2PIP only for `(peptide, charge)` pairs at features with observed MS2 scans
4. DeepLC: optionally fine-tune on PD MSF, then predict RT for all unique peptides
5. Compute LC-MS/MS evidence features
6. Extract LC-MS/MS envelopes from XIC best scans (if MALDI envelopes provided)
7. Compute all features
8. Build PSMList + populate rescoring features
9. Rescore using selected backend (see "Rescoring backends" below)

### Two-pass scoring logic

All backends follow the same two-pass structure:

1. **Round 1** — score all candidates globally. The model does not use per-feature grouping; every candidate is treated on equal footing.
2. **Per-feature winner selection** (`_select_feature_winners`) — for each MALDI m/z feature, retain only the highest round-1 score candidate. Produces `winners_df` (~N rows for N features).
3. **Round 2** — retrain/rescore on the winner subset only. Because each feature contributes exactly one candidate, this is a cleaner training set than the full candidate pool.
4. **FDR** — standard TDC (`_tdc_qvalues`) over all winners sorted by round-2 score. Q-values propagated to non-winners as NaN.

`result_df` contains all candidates. Round-2 score, q-value, and reweighted columns are NaN for non-winners. `is_tdc_winner` marks the round-1 winner per feature.

### Rescoring backends

`rescore()` accepts a `model` parameter:

**`model="svm"` (default):** mokapot `PercolatorModel` trained on `MALDI_INTRINSIC_FEATURES`.
- Round 1: train on all candidates, get `svm_score_r1`.
- Round 2: rebuild PSMList from `winners_df`, retrain mokapot, get `svm_score_r2`.
- FDR: `_tdc_qvalues(svm_score_r2, is_decoy_winners)` → `q_value`.
- Returns `(psm_list, result_df, feature_names)` where `result_df` has columns: `peptide`, `protein`, `feature_mz`, `feature_idx`, `is_decoy`, `svm_score_r1`, `svm_score_r2`, `q_value`, `is_tdc_winner`, `reweighted_score`, `reweighted_q_value`.

**`model="catboost"`:** Semi-supervised `CatBoostRanker` (iterations=500, YetiRank loss). Training uses only `MALDI_INTRINSIC_FEATURES`. Pseudo-label iteration (applied independently in each round):
1. Seed positives: `ppm_error_abs < init_ppm_threshold` AND `theo_isotope_cosine > init_isotope_threshold` (configurable, defaults 2.0 ppm / 0.7 cosine)
2. Train on positives + all decoys; predict scores on all candidates
3. Compute TDC q-values; expand positives to targets with q ≤ 0.05
4. Repeat until <1% change in positive set size or 5 iterations maximum
- Round 1 trains on all candidates → `catboost_score_r1`; round 2 retrains on `winners_df` → `catboost_score_r2`.
- Returns `(psm_list, result_df, feature_names)` where `result_df` has columns: `peptide`, `protein`, `feature_mz`, `feature_idx`, `is_decoy`, `catboost_score_r1`, `catboost_score_r2`, `q_value`, `is_tdc_winner`, `reweighted_score`, `reweighted_q_value`.

**`model="generative"`:** Probabilistic generative scorer. No training. Implemented in `probabilistic_scorer.py`.
- Estimates noise parameters label-free from the best-ppm non-decoy candidate per MALDI feature (proxy for true positives).
- Computes a log-sum generative score from independent half-normal / normal likelihoods for: ppm error, isotope cosine deviation from 1.0, CCS deviation (if im2deep present), spatial autocorrelation (if spatial features present).
- Adds per-feature ranking features in step 7b: `generative_score`, `generative_score_rank`, `generative_score_gap`, `generative_score_z`, plus diagnostic columns `Tm`, `Dm`, `Delta_m`, `generative_q_value`, `generative_pep`, `is_tdc_winner` (from the internal `estimate_fdr` call — **not used for final FDR**).
- Round-1 scores = `generative_score` from step 7b. Round-2 re-estimates noise parameters on `winners_df` (cleaner proxy set) and recomputes log-likelihoods via `estimate_noise_params` + `compute_generative_scores`.
- FDR: `_tdc_qvalues(generative_score_r2, is_decoy_winners)` — standard TDC, not margin-based.
- Returns `(psm_list, result_df, feature_names)` with columns: `peptide`, `protein`, `feature_idx`, `is_decoy`, `generative_score_r1`, `generative_score_r2`, `Delta_m`, `q_value`, `is_tdc_winner`, `reweighted_score`, `reweighted_q_value`.

**Generative pre-scoring for SVM/CatBoost** (`compute_generative=True`, default): when model is `"svm"` or `"catboost"`, the generative scorer runs first (step 7b) and its four ranking features (`generative_score`, `generative_score_rank`, `generative_score_gap`, `generative_score_z`) are added to `MALDI_INTRINSIC_FEATURES` before training. These features are carried into `winners_df` unchanged for the round-2 model — no recomputation needed.

**LC-MS/MS prior reweight** (applied after all backends to winners only): `compute_lcms_prior()` min-max normalizes each `LCMS_PRIOR_FEATURES` column, averages the normalized values (excluding all-zero features), and multiplies the resulting weight into the round-2 score. `reweighted_score = round2_score × lcms_prior`. Q-values are recomputed on the reweighted scores with a second `_tdc_qvalues` call. Only winner rows receive non-NaN values.

---

## Rust extension (`ms1rescore-rs`)

Built with PyO3 + rayon. Exposed functions:

| Function | Module | Description |
|---|---|---|
| `compute_peptide_masses(sequences)` | digest.rs | Returns (masses, mh_mzs, n_C, n_H, n_N, n_O, n_S) for a list of peptide sequences |
| `match_mz(feature_mzs, peptide_mzs, ppm)` | digest.rs | Binary search m/z matching; returns (feat_idx, pep_idx, ppm_errors) |
| `extract_xics_batch(ms1_rts, ms1_mz_arrays, ms1_int_arrays, target_mzs, ppm)` | xic.rs | Parallel XIC extraction for multiple target m/z values across all MS1 scans |
| `extract_ms1_envelopes_batch(ms1_mz_arrays, ms1_int_arrays, scan_indices, target_mzs, charge, n_peaks, ppm)` | isotope.rs | MS1 isotope envelope extraction at specified scans |
| `spectral_angles_batch(pred_mzs, pred_ints, obs_mzs, obs_ints, fragment_tol_da)` | spectral.rs | Batch spectral angle; requires ≥3 matched fragments, else returns 0.0 |
| `compute_ionization_features(sequences)` | features.rs | Returns (n_R, n_K, n_H, n_F, n_W, n_Y, gravy, pi) — 8 arrays; parallel rayon |
| `compute_property_features(sequences)` | features.rs | Returns (n_D, n_E, n_C, n_P, n_M, n_W, n_Y, seq_len, nterm_code, pi) — 10 arrays; parallel rayon |
| `count_missed_cleavages_batch(sequences)` | features.rs | K/R not followed by P, excluding last residue; parallel rayon |

All functions gracefully fall back to Python equivalents if the Rust extension is not importable. The Python code checks `from ms1rescore_rs import <function>` inside try/except blocks.

**Peptide sequence feature performance** (707K rows, 395K unique sequences):

| Function | Before (Python) | After (Rust rayon) | Speedup |
|---|---|---|---|
| `compute_peptide_properties` | 0.62 s | 0.50 s | 1.2× |
| `compute_maldi_ionization_features` | 0.80 s | 0.34 s | 2.3× |
| `compute_peptide_property_features` | 1.34 s | 0.43 s | 3.1× |

pI values from Rust are bit-for-bit identical to the Python bisection implementation.

### Building the Rust extension

```bash
cd ms1rescore/ms1rescore-rs
VIRTUAL_ENV=/home/robbe/.pyenv/versions/MSIscore maturin develop --release
```

**Important:** `maturin develop` must target the same Python environment as the Jupyter kernel. The notebook uses `/home/robbe/.pyenv/versions/MSIscore`. Using `maturin develop` without `VIRTUAL_ENV` will install into the wrong environment.

The Rust `target/` directory can be 1-2 GB. Delete it with `rm -rf ms1rescore/ms1rescore-rs/target/` if disk space is low (it is rebuilt on the next `maturin develop`).

---

## Environment

- Python: `/home/robbe/.pyenv/versions/MSIscore`
- Notebook kernel: MSIscore
- Key packages: `ms2pip>=4.0.0a1`, `deeplc>=4.0.0a1`, `mokapot>=0.10`, `pyteomics>=4.7`, `psm_utils>=1.1`, `brain-isotopic-distribution>=1.5` (PyPI name for `brainpy`), `catboost>=1.2` (optional, for `model="catboost"`), `alphatims>=1.0` (optional, for reading Bruker `.d` files directly)
- ms2pip import: `from ms2pip.core import predict_batch` (not `from ms2pip import predict_batch`)
- `brain-isotopic-distribution` is a transitive dependency of `ms-deisotope` but is listed explicitly in `pyproject.toml` as a core dependency because `theoretical_isotope_distribution()` in `utils.py` imports `from brainpy import isotopic_variants` directly.

Install the package in editable mode:
```bash
pip install -e ms1rescore/
# With CatBoost support:
pip install -e "ms1rescore/[catboost]"
# With Bruker timsTOF .d support:
pip install -e "ms1rescore/[timstof]"
```

---

## Notebooks and scripts

- **`notebooks/04_maldi_rescoring.ipynb`**: End-to-end notebook. Cells follow pipeline steps 1-9.
- **`scripts/visualize_ms1rescore_features.py`**: Feature visualization script. For each selected MALDI feature: summary heatmaps, feature distribution histograms, bar charts. For each of 3 selected candidates per feature (best-ppm target, best-SA target, best-ppm decoy): XIC chromatogram, MS2 mirror plot, isotope envelope comparison, ion image, mass accuracy bar, feature card.

---

## Known issues and limitations

### Low spectral angles (~0.011 mean)

The mean spectral angle is low because most of the 707K candidates are random mass coincidences. The max is ~0.402 for correct peptides. This is expected: centroided Thermo MS2 spectra have 12-34 peaks, and most MALDI features do not have a matching LC-MS/MS MS2 scan at all. The feature is still informative as a discriminator.

### Cache invalidation on numpy version change

`ms2pip_predictions.pkl` and `lcms_data.pkl` caches may fail to unpickle if numpy version changes between the session that wrote them and the session that reads them. Delete and rebuild:
```bash
rm notebooks/cache/ms2pip_predictions.pkl
rm notebooks/cache/lcms_data.pkl
```

### `maldi_intensities_p90` must be passed explicitly

`match_to_maldi_features()` accepts `maldi_intensities`, `maldi_intensities_p90`, and `maldi_intensities_sum`. If none are passed, all log-intensity features are 0. In `pipeline.py`, these are read from `spatial_features` columns (`intensity_p90`, `intensity_sum`, `mean_intensity`) when available. In standalone use (e.g. the viz script), pass at least `maldi_intensities` from ion images or spatial features.

### Scale

With the human FASTA (~20K proteins) and 1,398 MALDI features at 20 ppm:
- ~707K candidates (target + decoy)
- ~542K unique peptides
- ~1,067/1,398 features with MS2 matches
- ~1,085/1,398 features with detectable XIC (multi-charge)
- MS2PIP is run for ~683K unique (peptide, charge) pairs (only at features with observed MS2 scans)
