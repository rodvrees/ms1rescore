# MSI-PICASSO — CLAUDE.md

## Purpose

`MSI-PICASSO` is a symmetric target-decoy rescoring package for MALDI-MSI MS1 data. It takes a protein FASTA, MALDI data (a raw Bruker `.d`, an imzML, or a feature m/z list), and optional LC-MS/MS mzML files, then produces FDR-controlled peptide identifications via LDA-based semi-supervised rescoring (default) or a QDA alternative.

**Why it was built this way:** The prior approach (ms2rescore "Approach B") used ProteomeDiscoverer (PD) output for candidates and features. This introduced label leakage — `lcms_xcorr` (a PD search engine score) had AUC 0.993 and was a near-perfect surrogate for the PD target/decoy label, making rescoring trivial but invalid. This package replaces that with:
- Candidates from in-silico tryptic digest of forward + shuffled FASTA (no PD)
- All LC-MS/MS features computed from raw mzML (no PD-derived scores)
- Strict symmetric design: **no feature computation function takes `is_decoy` as a parameter**

**Two MALDI input modes — and the intended direction of travel.** The MALDI signal for each candidate can be obtained two ways:
1. **Feature-list mode (current default, baseline).** Features are detected/supplied first (peak picking on the `.d`, an imzML interval list, or a plain m/z text file) and candidates are matched against that fixed grid. Simple and fast, but the candidate set is capped by whatever the detector found, and decoy anchors must be reconciled with a pre-detected grid (the source of the `mz_shift` snapping and target-multiplicity problems documented below).
2. **Raw-query mode (`--maldi-query-raw`, the optimization target).** Candidates drive extraction: the ion image (and observed centroid m/z and CCS) for every candidate — target *and* decoy — is queried on demand directly from the raw `.d` at the candidate's own m/z, with no pre-detection step. This removes the detection bottleneck and makes the target/decoy null cleaner (every decoy gets a genuine on-demand image at its own anchor rather than being snapped onto a foreign detected peak). **The goal is to make raw-query the default once it is fast and robust enough**; the work needed to get there (extraction speed on profile-mode data, ppm recomputation from observed centroids, ion-mobility/CCS extraction) is the main optimization frontier of this codebase. Treat raw-query as the path being hardened toward default, and feature-list mode as the legacy baseline it is meant to replace.

---

## Repository layout

```
MSI-PICASSO/
├── pyproject.toml              # Package metadata and dependencies
├── CLAUDE.md                   # This file
├── msi_picasso/                # Python package (import name: msi_picasso)
│   ├── __init__.py             # __version__ = "0.1.0"
│   ├── utils.py                # Shared math utilities
│   ├── candidates.py           # FASTA digest + MALDI m/z matching + decoy generation (shuffle, mz_shift, mz_shuffle, entrapment)
│   ├── lcms_ids.py             # Parse LC-MS/MS IDs for Strategy C candidate generation
│   ├── lcms_evidence.py        # LC-MS/MS feature extraction
│   ├── maldi_extraction.py     # Raw MALDI extraction: feature detection, ion images, spatial features
│   ├── maldi_query.py          # Raw-query mode: ion images (query_raw_maldi) + observed peak centroids & CCS (extract_observed_feature_stats_raw) from candidate m/z
│   ├── maldi_imzml.py          # SCiLS Lab-style interval extraction for imzML data
│   ├── maldi_features.py       # MALDI-side rescoring features (incl. TIC-masked / NMF colocalization)
│   ├── feature_generator.py    # Orchestration + PSMList construction; feature-group definitions
│   ├── pipeline.py             # End-to-end pipeline function; rescoring backends (lda/qda); priors
│   ├── probabilistic_scorer.py # Generative pre-scorer (legacy; was the SVM/CatBoost step-7b feature source — those backends are removed)
│   ├── config_parser.py        # Cascade config merge + jsonschema validation (package_data/config_*.json)
│   ├── cli.py                  # argparse CLI entry point (`picasso` command)
│   ├── debug_viz.py            # Debug figure generation (saved when --verbose)
│   ├── package_data/           # config_default.json + config_schema.json
│   └── tests/                  # Unit tests (pytest; testpaths configured in pyproject.toml)
│       ├── fixtures/           # Static test fixtures (e.g. test_maldi.mgf)
│       ├── test_balanced_shuffle.py
│       ├── test_best_feature_init.py
│       ├── test_calibration_selection.py
│       ├── test_candidates.py
│       ├── test_colocalization_mask.py       # TIC on-tissue masking
│       ├── test_config_parser.py
│       ├── test_debug_viz.py
│       ├── test_deisotoping.py
│       ├── test_entrapment_decoys.py
│       ├── test_evidence_score.py            # pre-existing broken (stale import)
│       ├── test_isotope_distribution.py
│       ├── test_lcms_apex_features.py
│       ├── test_lcms_ids.py
│       ├── test_lda_backend.py
│       ├── test_lda_cv.py                     # cross-validated LDA scoring
│       ├── test_maldi_extraction.py          # pre-existing broken (stale import)
│       ├── test_maldi_query.py                # raw-query extraction
│       ├── test_mz_shift.py
│       ├── test_mz_shuffle.py
│       ├── test_nmf_colocalization.py        # NMF substructure colocalization
│       ├── test_pep.py
│       ├── test_protein_coverage.py          # protein_coverage label-leak fix
│       ├── test_qda_backend.py
│       └── test_spatial_ranker_features.py
└── msi-picasso-rs/             # Rust extension (PyO3 + rayon)
    ├── Cargo.toml
    └── src/
        ├── lib.rs              # PyO3 module definition
        ├── digest.rs           # Peptide mass + composition + m/z matching
        ├── xic.rs              # Parallel XIC extraction
        ├── spectral.rs         # Spectral angle computation
        ├── isotope.rs          # MS1 isotope envelope extraction (LC-MS/MS)
        ├── features.rs         # Ionization and property features (rayon parallel)
        ├── ion_image.rs        # Profile-mode pixel window integration (rayon)
        └── maldi_isotope.rs    # MALDI M+1/M+2 mean intensity via CSR streaming pass
```

Run the test suite from the `MSI-PICASSO/` directory:

```bash
cd MSI-PICASSO && pytest   # testpaths = ["MSI-PICASSO/tests"] in pyproject.toml
```

---

## Core design principles

### 1. Symmetric target-decoy computation

Every function that computes features must be blind to `is_decoy`. The symmetry guarantee is enforced at the API level: none of the feature computation functions have an `is_decoy` parameter. This means targets and decoys receive features via identical code paths.

### 2. Decoy generation via K/R-preserving protein-level shuffle

`_shuffle_protein(seq, random_state=42)` in [candidates.py](msi_picasso/candidates.py) keeps K and R residues at their original positions and randomly shuffles all other residues (using a seeded RNG for reproducibility), then digests the shuffled sequence with the same trypsin rules as the target.

**Why K/R-preserving shuffle instead of K/R-preserving reversal:** Reversal with K/R fixed produces decoy peptides that are often isobaric with targets — the elemental composition of the non-K/R residues is unchanged (same multiset, just reversed). This makes isotope envelope features (`theo_isotope_cosine`, `theo_isotope_chi2`, `theo_isotope_kl`, `isotope_envelope_*`) non-discriminative. Shuffling (rather than reversing) the non-K/R residues changes which residues appear in each tryptic fragment, breaking elemental composition conservation at the peptide level.

**Why keep K/R in place:** This preserves tryptic cleavage sites, so the decoy protein is digested at exactly the same positions as the target. Decoy peptides therefore have the same length distribution and the same C-terminal residue as their target counterparts, keeping the TDC null model valid.

### 3. Neutral mass matching for LC-MS/MS

MALDI features are detected as [M+H]+ (charge 1). LC-MS/MS MS2 scans are acquired at charge 2, 3, etc. Matching is done by comparing **neutral masses** (`maldi_mz - PROTON` vs `ms2_precursor_mz * charge - charge * PROTON`), not m/z values directly. Matching on m/z alone gives ~88/1398 features with MS2 scans; neutral mass matching gives ~1067/1398.

### 4. DeepLC-anchored MS1 features (fully symmetric)

LC-MS/MS MS1 features are computed using the DeepLC predicted retention time as the anchor for each candidate. For each candidate (target or decoy), the DeepLC RT prediction is used to locate the nearest MS1 scan, and signal, SNR, and isotope envelope features are extracted at that scan. This is fully symmetric: targets and decoys receive identical treatment because DeepLC predictions do not depend on `is_decoy`. No XIC extraction is performed.

**Why not XICs:** XIC extraction is inappropriate for DDA LC-MS/MS data (a peptide may appear in only one or a few MS2 events, and XIC apex selection is unreliable). Using the search engine's identified RT (MS2 scan RT) as the anchor would break TDC symmetry (decoys would never have an identified RT). DeepLC predicted RT is the only symmetric, model-based RT anchor available for all candidates.

### 5. Raw-query mode inverts the pipeline ordering (the intended future default)

In the legacy feature-list path the MALDI feature list is detected/supplied first and candidates are matched against it (`_load_maldi() → generate_candidates() → match_to_features()`). With `maldi_query_raw=True` the ordering inverts: candidates are generated first against the digest m/z grid, then `query_raw_maldi` extracts ion images from the raw `.d` at `candidates_df["feature_mz"]` (`generate_candidates() → query_raw_maldi(candidate mzs)`). Per-feature intensities are mapped back onto the candidate rows after extraction. **The guarantee that `feature_mz` on `mz_shift` decoy rows is the shifted (off-target) m/z — not the original peptide m/z — is load-bearing here:** the raw query extracts the decoy's ion image at the shifted anchor, which is the correct (foreign) signal for that decoy.

**This is the mode being optimized toward becoming the default** (see Purpose). When extending the pipeline, prefer making a feature work correctly and efficiently in raw-query first; feature-list mode is the fallback baseline. The standing work items on this frontier:
- **Extraction speed.** Profile-mode `.d` extraction is the dominant cost (~5 min/pass for ~3 K m/z over ~48 K pixels). Faster windowed/streaming extraction (the Rust `ion_image.rs` path, batching, on-tissue-only pixel reads) is the main lever for making raw-query cheap enough to default. **Reuse across runs:** `rescore(raw_query_cache=...)` is a bidirectional cache — pass `None` to always extract; pass `{}` to extract once and populate it; pass the populated dict to reuse the full-grid ion images / observed centroids / CCS without touching the `.d`. The candidate m/z grid is fixed by the digest + decoy method, so a parameter sweep that varies only scoring params (e.g. `scripts/grid_search.py` in raw-query mode) extracts once on the first run and reuses it for all the rest. The cached arrays cover the whole grid, a superset of any run's `query_mzs`, so they are reused as-is (extra ion images are ignored by the `feature_mz → image` lookups).
- **Symmetric observed statistics.** `ppm_error` is recomputed from the observed peak centroid in each candidate's own window (`_recompute_ppm_from_centroids`), worst-case-filled when no peak is found, so targets and decoys are measured identically; observed CCS is keyed per feature_idx (`_observed_ccs_by_feature_idx`). These must stay symmetric as raw-query grows.
- **Ion mobility / CCS.** Observed CCS is extracted from the `.d` via `alphatims` (Mason-Schamp 1/K0→CCS); raw-query is where CCS becomes a first-class observed quantity rather than a prediction-only feature.

---

## Modules

### `maldi_extraction.py`

Converts raw Bruker `.d`/TSF data into the NPZ format consumed by the rest of the pipeline via `imzy`. CLI flag: `--maldi-raw`.

Install: `pip install MSI-PICASSO[maldi]` (installs `imzy`).

#### `extract_maldi_data(d_path, ppm_bin, extraction_ppm, matching_ppm, min_fraction, feature_mzs, images_path, image_batch_size, output_npz, output_spatial_tsv, output_dir, verbose)`

Returns a 5-tuple: `(feature_mzs, ion_images, extra_ion_images, spatial_df, maldi_envelopes)`.

`extra_ion_images` is a `dict | None` with keys `"m1"`, `"m2"`, `"na"`, `"k"`, `"chca"`, each an `(N, H, W)` float32 array of ion images extracted at M+1/M+2 and adduct (Na, K, CHCA) shifted m/z positions. Used by `compute_isotopologue_colocalization` and `compute_adduct_colocalization` in `maldi_features.py` to compute direct per-feature Pearson r without requiring isotopologue/adduct peaks to be present in the feature list. Set to `None` when images are loaded from a pre-computed NPZ `images_path` or when data is sourced from imzML (those paths do not have a live reader for extra extraction).

Four-step extraction:

1. **Feature detection** — `detect_features(reader, ppm_bin, min_fraction)`. Streams all pixels, builds a log-ppm histogram, and greedily merges adjacent bins to produce consensus feature m/z values. Works directly on centroid data (instrument-picked peaks). Profile data is not handled here — use `maldi_imzml.py` for profile imzML.

2. **`extract_ion_images(reader, feature_mzs, ppm=25.0)`** — tries fast extraction paths first (`_extract_centroid_fast` for Bruker TSF, `_extract_profile_fast` cumsum trick for profile), then falls back to `reader.get_ion_images()`. Returns `(n_features, H, W)` float32.

3. **Extra ion image extraction** (RAM path only, when `images_path is None`) — calls `extract_ion_images` five more times at `feature_mzs + delta` for M+1 (+1.003355 Da), M+2 (+2.006710 Da), Na adduct (+21.9819 Da), K adduct (+37.9559 Da), and CHCA matrix adduct (+171.0320 Da). Saved to NPZ as `extra_m1`, `extra_m2`, `extra_na`, `extra_k`, `extra_chca` keys alongside `mzs`, `images`, `x_coords`, `y_coords`.

4. **`compute_spatial_features(ion_images, feature_mzs, n_pixels_total)`** — computes per-feature: `fraction_detected`, `n_pixels_detected`, `mean_intensity`, `intensity_p90`, `intensity_sum`, `intensity_cv`, Moran's I (`spatial_autocorrelation`).

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

Install: `pip install MSI-PICASSO[maldi]` (installs `pyimzml`).

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
picasso \
  -f data/PXD056528/uniprot_human_reviewed.fasta \
  -l data/PXD056528/231212_AG_11.mzML \
  -l data/PXD056528/231212_AG_12.mzML \
  -l data/PXD056528/231212_AG_21.mzML \
  --maldi-raw data/PXD056528/MALDI_MSI/20221013_SingleCells_CHCA.d \
  --msf data/PXD056528/240125_AG_DDA.msf \
  --model lda \
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

| Flag | Format | Feature detection | Ion images | Spatial features | Extra images (adducts) |
|---|---|---|---|---|---|
| `--maldi-npz PATH` | NumPy NPZ | No | Yes (if `"images"` key) | No | From NPZ `extra_*` keys |
| `--maldi-mzs PATH` | Plain text m/z list | No | No | No | No |
| `--maldi-raw PATH` | Bruker `.d` | Histogram binning (centroid) | Yes | Yes | Yes |
| `--maldi-d PATH` | Bruker `.d` (alias for `--maldi-raw`) | Histogram binning (centroid) | Yes | Yes | Yes |
| `--maldi-imzml PATH` | imzML + ibd | SCiLS interval detection | Yes (reconstructed from interval matrix) | Yes | No |
| `--maldi-query-raw` (modifier, with `--maldi-raw`/`--maldi-d`) | Bruker `.d` | None — candidate m/z drive extraction | Yes (at candidate m/z) | Yes | Yes |

**Raw-query mode (`--maldi-query-raw`):** A modifier on `--maldi-raw`/`--maldi-d` (not a standalone source). Instead of detecting a feature list first, candidates are generated first (against the digest m/z grid) and `maldi_query.query_raw_maldi` extracts ion images directly from the `.d` at `candidates_df["feature_mz"]` (zero-signal features are retained, so decoys in empty m/z space yield genuine zero-signal images). The extraction reuses `extract_maldi_data(feature_mzs=query_mzs, drop_zero_signal=False)`, so the 5-tuple return and all downstream code are unchanged. With `decoy_method="mz_shift"` and `mz_shift_delta_min < 10`, a `UserWarning` flags that small shifts may land in empty m/z space.

**Observed peak centroids, ppm, and CCS in raw-query mode:** `imzy` (used for the ion images) exposes neither the per-peak m/z centroid nor mobility, so `maldi_query.extract_observed_feature_stats_raw` opens the `.d` a *second* time with `alphatims` and, in one pass, computes per candidate window (pure vectorised helper `_weighted_mean_in_windows`): (1) the intensity-weighted **observed peak centroid m/z**, and (2) the intensity-weighted mean 1/K0 → **observed CCS** via `one_over_k0_to_ccs`.

- **ppm recompute (symmetric).** In raw-query mode candidates are matched against the *theoretical* digest grid (`maldi_mzs = unique(peptide_db["mh_mz"])`), so the usual `(feature_mz − mh_mz)` ppm is 0 for every self-match and decoys inherit 0. The pipeline replaces it via `_recompute_ppm_from_centroids`: for **every** candidate row (target and decoy identically), `ppm_error = (observed_centroid − feature_mz) / feature_mz × 1e6`, where `feature_mz` is the candidate's own queried anchor (peptide [M+H]+ for a target, the shifted m/z for an `mz_shift` decoy). This is a real mass-accuracy measurement bounded by ±`extraction_ppm`, computed from each candidate's own window with **no inheritance from the paired target** and **no label leak** (a decoy's ppm never references the peptide mass). Applies to all decoy methods (shuffle/entrapment matched to the grid are 0 by construction too). A window with **no observed peak** (e.g. an `mz_shift` decoy shifted into empty m/z) has unmeasurable mass accuracy and is assigned the **worst-case ppm** (`worst_case_ppm=extraction_ppm`, the window edge — the worst in-distribution value), so empty-signal candidates are penalised on ppm rather than median-imputed to an average value by the LDA. If extraction produced no centroids at all (alphatims missing / no signal anywhere), `ppm_error` is left as the matched-grid value instead. Needs only m/z + intensity, so it works even without a TIMS dimension.
- **observed CCS** builds `observed_ccs_per_feature` via `_observed_ccs_by_feature_idx` — keyed by the candidates' own `feature_idx` (bridged through `feature_mz`, since in raw-query `feature_idx` indexes the digest grid, not the query m/z), matching how `compute_im2deep_features` consumes it. This unlocks the IM2Deep CCS features, the `match_ccs` filter, and `mob_coloc`. `NaN` (→ `observed_ccs_per_feature=None`) when `alphatims` is missing or the data has no TIMS dimension (TSF).

Note: `query_raw_maldi` does not return `pixel_coords`, so `ppm_error_calibrated_z` is unavailable in raw-query mode.

**Mobility colocalization wiring:** `mob_coloc` (opt-in, `--mob-coloc`) requires `im2deep_predicted_ccs`, which now exists in raw-query mode once observed CCS is extracted. `compute_mobility_colocalization_features` reads the `.d` itself via `alphatims` + `MaldiFrameInfo` X/Y, so it is independent of the imzy ion images and the raw-query grid swap. (Earlier the pipeline call omitted the required `tdf_path` argument and the `TypeError` was swallowed by a `try/except`, so `mob_coloc` never ran in any mode; the call now passes `tdf_path` and `mob_window_multiplier`.)

**`--maldi-d` vs `--maldi-imzml`:** When raw Bruker `.d` data is available, prefer `--maldi-d`. It extracts ion images directly from raw spectra, includes adduct/isotopologue extra images (`extra_ion_images`), and is not affected by SCiLS recalibration offsets. `--maldi-imzml` reconstructs ion images from SCiLS-integrated interval intensities — all spatial features are computed, but adduct images (`na`, `k`, `chca`) are unavailable because the interval list covers only detected monoisotopic peaks. Note that SCiLS-exported imzML m/z values may differ from raw Bruker calibration by 10–80+ ppm; verify alignment with `Amy_TMA_MS1.d` before using `--maldi-d` with a SCiLS feature CSV.

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

**`digest_fasta(fasta_path, ..., generate_decoys=True)` — Strategy A (full FASTA):**
1. Phase 1 (pyteomics): `pyteomics.parser.cleave()` for tryptic digestion. Decoys generated by `_shuffle_protein()` (K/R-preserving shuffle, seeded at 42) when `generate_decoys=True`.
2. Phase 2 (Rust or pyteomics fallback): `compute_peptide_masses()` from `ms1rescore_rs` computes mass, [M+H]+ m/z, and elemental composition (n_C, n_H, n_N, n_O, n_S).

**`digest_identified_proteins(..., generate_decoys=True)` — Strategy C (LC-MS/MS-guided):**
See the [Candidate generation strategies](#candidate-generation-strategies) section below. Pass `generate_decoys=False` to suppress decoy generation.

**`generate_mz_shift_candidates(target_df, feature_mzs, ..., snap_to_features=True)` — observation-space m/z-shift decoys:** For each unique target peptide, samples a random delta in `[delta_min, delta_max]` Da (alternating sign). Two placement modes:
- **`snap_to_features=True` (default, feature-list mode):** snaps the shifted query to the nearest MALDI feature within `snap_tolerance_ppm`, rejecting snaps within `matching_ppm` of any target peptide m/z (collision filter). `feature_idx` is the snapped grid index.
- **`snap_to_features=False` (raw-query mode):** the decoy feature *is* the exact shifted m/z `mh_mz ± delta` (no snap — raw-query images any m/z on demand), accepted only if it does not collide (within `matching_ppm`) with a target peptide m/z **or** with an already-assigned decoy m/z. This guarantees one **distinct** feature per decoy (no clustering onto shared grid points, which otherwise collapses many decoys onto few features and skews the per-feature-winner T:D ratio). Each decoy gets a unique `feature_idx` past the grid range `[0, len(feature_mzs))`.

Targets are matched normally; decoy rows reuse the target sequence with `is_decoy=True`, `source="decoy_mz_shift"`, `feature_mz=` the shifted m/z, and a diagnostic `decoy_delta_da` (NaN for targets). `ppm_error` in feature-list mode is copied from the peptide's best target match (non-discriminative); in raw-query it is recomputed from observed peak centroids in `pipeline.py` (`_recompute_ppm_from_centroids`). **`feature_mz` on decoy rows being the shifted m/z is load-bearing for raw-query mode** (`maldi_query.query_raw_maldi`). The pipeline passes `snap_to_features=not maldi_query_raw`.

**`generate_mz_shuffle_candidates(target_df, feature_mzs, ...)` — m/z-assignment-shuffle decoys:** Matches targets normally, takes each unique target peptide's representative feature (lowest `ppm_error_abs`), and forms decoys by a **derangement** of the peptide→feature assignment: peptide `i` is relocated onto the feature of peptide `σ(i)`, where `σ` is a mass-sorted cyclic rotation by `k ∈ [n/4, 3n/4)` ranks (guarantees no fixed point and a large mass gap → never self- or near-isobaric). Decoy rows: `is_decoy=True`, `source="decoy_mz_shuffle"`, `feature_mz`/`feature_idx` = the assigned (other peptide's) feature — so each decoy is **co-located on the identical ion image as that feature's target**. `ppm_error` is inherited from the peptide's own best target match (non-discriminative; **not** computed against the decoy's feature, which would be a fake discriminator absent from real false positives). `decoy_delta_da = assigned_feature_mz − peptide_mh_mz`. The key property: feature-quality features are identical between a feature's target and decoy, so discrimination is forced onto the peptide↔observation match (CCS, isotope). **Targets are deduplicated to one representative row per unique peptide** (the lowest-|ppm| match, the same `best` set the derangement is built from) before being concatenated with the decoys. This is required for a 1:1 null: a target peptide whose m/z falls within `matching_ppm` of several MALDI peaks would otherwise contribute multiple target rows against its single decoy, producing a ~(mean features/peptide):1 target:decoy imbalance (e.g. 5901:2895) with most target rows left without a co-located decoy. Deduplicating loses no unique peptide identifications (only redundant near-isobaric secondary matches). Decoy rows are given a **separate protein namespace** (`protein = "DECOY_" + source_protein`) so protein-level features (`protein_colocalization*`, `protein_n_features`, `protein_coverage`, …) are computed within class — never pooling a decoy with its source target's protein. `protein_tryptic_count` is inherited from the source protein (the `DECOY_` prefix is stripped for the lookup). Returns the combined target+decoy frame.

**`load_entrapment_candidates(entrapment_fasta, target_df, feature_mzs, ...)` — entrapment decoys:** Digests a foreign-organism FASTA with the same trypsin rules, computes [M+H]+ m/z, then applies a **contamination filter** — `match_mz(target_mzs, entrapment_mzs, matching_ppm)` removes any entrapment peptide isobaric with a target. *Rationale:* an isobaric entrapment peptide would inherit the real biological signal at its m/z, making the null artificially good; this is a contamination filter, not a decoy-selection step. The collision rate is logged and a >10% rate warns of m/z-space overlap between the entrapment organism and the sample proteome. Surviving peptides are matched to features via `match_to_maldi_features` and flagged `is_decoy=True`, `protein="ENTRAPMENT_{accession}"`, `source="entrapment"`. Returns matched decoy rows only (the pipeline concatenates them with the matched target candidates).

`match_to_maldi_features()` uses `match_mz()` from `ms1rescore_rs` (binary search) or Python fallback. Returns a candidates DataFrame with one row per (peptide, MALDI feature) pair. Protein-level features (`protein_n_features`, `n_candidates`) are computed over all candidates symmetrically.

**`protein_coverage` — label-leak fix (symmetric numerator + true-digest denominator).** `protein_coverage` is the fraction of a protein's tryptic peptides that are observed, computed in `compute_protein_consistency_features`. It previously used `protein_n_features / protein_tryptic_count`, which leaked the target/decoy label: every decoy peptide is placed on exactly one feature by construction (`mz_shift`/`mz_shuffle`), so a decoy protein's `n_features` equals its observed-peptide count and — because `protein_tryptic_count` was the candidate-pool count, not the full digest — coverage was pinned to exactly **1.0 for every decoy**, while target peptides matching several near-isobaric features pushed target coverage above 1. Two changes fix it: (1) the **numerator** counts distinct observed *peptides* (`protein_n_peptides = groupby(protein).peptide.nunique()`), which is symmetric because a protein and its `DECOY_`/`ENTRAPMENT_` namespace share the same peptide set; (2) the **denominator** is overridden in `pipeline.py` (just before Step 6) with the *true full tryptic digest count* per protein, computed from `peptide_db` (the complete length-filtered digest, before m/z matching) and keyed by base accession so decoys inherit their source protein's count. Result: coverage ∈ (0,1], symmetric (target ≈ decoy per protein), and non-degenerate (real variation across proteins). `protein_tryptic_count` is consumed only by `protein_coverage`. (Note: the decoy protein namespacing and `protein_tryptic_count` *inheritance* were already correct — the leak was in the metric definition, not the decoy setup.)

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
| `load_lcms_data(mzml_paths)` | Load mzML via pyteomics or Bruker `.d` via alphatims; routes based on extension |
| `load_lcms_data_from_d(d_path)` | Load timsTOF `.d` folder with alphatims; MS1 per-frame, MS2 vectorised via `index_precursors()` |
| `_find_matching_ms2_scans(neutral_mass, lcms_data, ppm)` | Binary search over MS2 neutral masses |
| `get_ms2pip_predictions(pairs, model)` | Batch MS2PIP predictions for `(peptide, charge)` pairs. Import: `from ms2pip.core import predict_batch` |
| `finetune_deeplc(msf_path)` | Fine-tune DeepLC on PD TargetPsms (q≤0.01) |
| `finetune_deeplc_from_df(rt_df)` | Fine-tune DeepLC from a DataFrame with `sequence`/`rt_mean` columns (minutes); used for FragPipe input |
| `get_deeplc_predictions(peptides, model)` | Batch DeepLC RT predictions |
| `extract_all_xics(unique_mzs, lcms_data, ppm)` | XIC extraction utility (available but not used in the main pipeline) |
| `compute_all_lcms_evidence(candidates_df, ...)` | Main entry point: returns dict mapping candidate index → feature dict |

#### `compute_all_lcms_evidence` structure

**DeepLC-anchored, fully symmetric.** No XIC extraction. All MS1 features are computed at the nearest MS1 scan to the DeepLC predicted RT.

1. Pre-compute per MALDI feature (1,398 iterations):
   - Matching MS2 scan indices (by neutral mass)
2. Per-candidate loop (707K iterations): all peptide-specific computations:
   - **MS2 RT filter**: when `rt_window_min > 0`, `predicted_rt` is fetched before the MS2 loop and the neutral-mass-matched scan list is further restricted to scans whose `ms2_precursor_rt` is within ±`rt_window_min` of `predicted_rt`. `lcms_ms2_n_matches` reflects this filtered count. When `rt_window_min == 0`, only neutral-mass matching applies (original behaviour).
   - Spectral angle vs MS2PIP prediction (peptide+charge specific). `lcms_ms2_spectral_angle` = NaN when no MS2PIP prediction is available; 0.0 when a prediction exists but fewer than 3 fragments match.
   - `lcms_ms2_rt_delta` (F4): |RT of the highest-SA MS2 scan − predicted_rt|. NaN when no MS2 match passes both filters.
   - DeepLC predicted RT → MS1 scan window (cached per unique peptide sequence). When `rt_window_min > 0`: all MS1 scans within ±`rt_window_min`; when `rt_window_min == 0`: single nearest scan.
   - `lcms_ms1_intensity`: log1p of summed signal in ±ppm window at precursor m/z across selected scans
   - `lcms_ms1_snr`: log10(signal / median_background) when both signal > 0 and background > 0; log10(signal) when signal > 0 but no non-zero background found; 0.0 when signal = 0
   - Isotope envelope [M0, M+1, M+2] from `_extract_ms1_envelope` at the **LC-MS/MS charge** (charge of the highest-SA MS2 scan, fallback charge 1). The envelope m/z is `(neutral_mass + z * PROTON) / z` with peak spacing `NEUTRON / z`.
   - `lcms_ms1_isotope_cosine`: cosine similarity of observed vs theoretical envelope
   - `theo_m1_ratio_diff_lcms`, `theo_m2_ratio_diff_lcms`: |obs_ratio − theo_ratio| for M+1/M0 and M+2/M0
   - **Apex window features** (F1–F3, only when `rt_window_min > 0`): computed at `lc_mz` (LC-MS/MS charge m/z) over the same scan window:
     - `lcms_ms1_apex_rt_delta`: |RT of the max-signal scan − predicted_rt|. NaN when no signal found.
     - `lcms_ms1_frac_apex_signal`: signal at the DeepLC-nearest scan / signal at the apex scan. 0.0 when no signal. Equals 1.0 when predicted_rt falls exactly at the elution apex.
     - `lcms_ms1_n_scans_with_signal`: count of scans in the window with non-zero signal at `lc_mz ± ppm_tolerance`.
   - If `maldi_envelopes` provided: MALDI vs LC-MS/MS envelope comparison → `isotope_envelope_cosine`, `isotope_envelope_pearson`, `isotope_envelope_mse`, `log_isotope_m1_ratio_diff`, `log_isotope_m2_ratio_diff`, `isotope_n_matched`
     - **`log_isotope_m1_ratio_diff`**: `log1p(|MALDI_M+1/MALDI_M0 − LCMS_M+1/LCMS_M0|)`. Compares the M+1/M0 isotope intensity ratio observed in the MALDI ion images (mean pixel intensities across the tissue section) to the same ratio measured in the LC-MS/MS MS1 spectrum (summed over scans within the DeepLC RT window, normalized to sum to 1). A value near 0 means the MALDI isotope pattern is concordant with the LC-MS/MS observation for the same candidate peptide; higher values indicate discordance, which can arise from chimeric MALDI features or mass coincidences. The log1p transform compresses the range and reduces sensitivity to outliers. NaN when `maldi_envelopes` is not provided or when either M0 signal is zero.
     - **`log_isotope_m2_ratio_diff`**: `log1p(|MALDI_M+2/MALDI_M0 − LCMS_M+2/LCMS_M0|)`. Same as `log_isotope_m1_ratio_diff` but for the M+2 peak, which is particularly informative for sulfur-containing peptides (elevated M+2 relative to averagine) and heavier peptides (> ~1500 Da, where M+2 intensity approaches or exceeds M+1). NaN when fewer than 3 isotope peaks are available (k < 3).

**`rt_window_min` is set to `2.0 × p95_mae`** of DeepLC calibration residuals in `pipeline.py`, where `p95_mae` is the 95th-percentile absolute error over the fine-tuning calibration set. When no calibration is performed, `rt_window_min = 0.0` and all window-based features (F1–F4 plus the MS2 RT filter) fall back to their sentinel values automatically.

**Signature:**
```python
compute_all_lcms_evidence(
    candidates_df, lcms_data, ms2pip_cache,
    deeplc_cache=None,     # peptide → predicted RT (minutes)
    maldi_envelopes=None,  # feature_mz → normalized envelope array
    ppm_tolerance=20.0,
    fragment_tol_da=0.02,
    rt_window_min=0.0,     # ±window for MS2 RT filter and apex features
) -> dict[int, dict[str, float]]
```

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

**On-tissue pixel masking (TIC) — required for valid colocalization.** Every MALDI ion image is ~0 in the unmeasured padding around the acquired pixel grid and broadly follows the tissue footprint within it (every peak is near-0 off-tissue, positive on-tissue). A raw Pearson r between two ion images is therefore dominated by this shared on/off-tissue structure and is inflated toward the tissue outline for *any* pair of images, real or decoy. Empirically (see `notebooks/gt_protein_ion_images.ipynb`) the within-protein mean pairwise r is ~0.76–0.84 and **decoy proteins colocalize as well as or better than targets** — the feature measures "is this on tissue," not protein-specific co-distribution. `compute_tissue_mask(ion_images, tic_quantile=0.0)` (in `maldi_features.py`) builds a flattened `(H*W,)` boolean on-tissue mask from a TIC proxy (per-pixel sum over all ion images): TIC == 0 padding is always dropped; `tic_quantile > 0` raises the threshold to that quantile of the measured-pixel TIC, additionally trimming low-signal edges. `compute_all_features` builds the mask once and threads it (as `pixel_mask`) into `_pearson_r_matrix` and the iso/adduct functions so the correlation is computed over on-tissue pixels only. Exposed via `--coloc-tic-quantile` (config `coloc_tic_quantile`, default `0.0` = drop only padding). Image validity (non-constant) is assessed on the masked pixels too.

**`_pearson_r_matrix(ion_images, ion_image_mzs, pixel_mask=None)`** — shared helper. Stacks all valid (non-constant) ion images into a `(n_valid, n_pixels)` float32 matrix and computes the full `(n_valid, n_valid)` Pearson correlation matrix in a single BLAS `dgemm` call via `np.corrcoef`. When `pixel_mask` is given, the columns are restricted to the selected on-tissue pixels before centring/normalising. Called once by `feature_generator.compute_all_features` and passed as `_corr_cache` to all three colocalization functions to avoid 3× redundant BLAS work.

**`compute_colocalization_features()`** — within-protein Pearson correlations:
1. `_pearson_r_matrix` → full corr matrix (shared with other functions)
2. Pandas self-join on `protein` to enumerate all within-protein feature pairs (O(Σ k²) rows where k = features per protein, typically small)
3. Vectorized `corr_matrix[idx_a, idx_b]` lookup on the join result
4. `groupby(['feature_mz', 'protein']).agg(...)` → merge back onto candidates

In addition to `protein_colocalization` (mean), `_max`, `_median`, `_n_partners`, the same single pass computes **intensity-weighted** and **rank-weighted** aggregations from a per-pair weight `w = sqrt(I_a · I_b)` where `I` is the **linear** `feature_intensity_p90` (fallback `feature_intensity`, else uniform `w=1` so the weighted features equal the plain mean). All blind to `is_decoy`; full column list in `maldi_features._COLOC_FEATURE_COLS`; all added to `PROTEIN_LEVEL_FEATURES` (enter the ranker via `--use-protein-level-feats`):
- `protein_colocalization_weighted` = Σ(w·r)/Σw  (intensity-weighted mean r — down-weights faint partner pairs)
- `protein_colocalization_weighted_max` = max(w·r)
- `protein_colocalization_top{2,3,5}` = mean r over the k highest-weight partner pairs (`sort_values("w").groupby(...).head(k)`; uses all pairs when fewer than k exist, no NaN padding)

**`compute_patch_colocalization_features()`** — patch-level (local) colocalization (opt-in, `--patch-coloc`; also needs `--use-protein-level-feats`). Tiles the `(H,W)` grid into `patch_size`×`patch_size` blocks (default 10), keeps only on-tissue pixels (`pixel_mask`) and skips patches with < ~5 measured pixels, then for each within-protein pair computes the Pearson r over **each patch's pixels** (mean-center + normalize + dot, features constant within a patch skipped). Per pair → mean/max/`>threshold` (default 0.5) across patches; per `(feature_mz, protein)` → `protein_patch_colocalization_mean` (mean over partners of per-pair mean), `_max` (max over partners of per-pair max), `_frac_above` (mean over partners of per-pair fraction). Asks "in how many local neighbourhoods do same-protein peptides co-distribute," sidestepping the global tissue-outline correlation. Purely spatial → symmetric. Logs `kept/total patches` and mean on-tissue pixels/patch (on a TMA most patches are off-tissue and skipped, so it runs well below worst case; the log also flags a mis-sized `patch_size`). Columns in `maldi_features._PATCH_COLOC_COLS`, added to `PROTEIN_LEVEL_FEATURES` (computed only when `patch_coloc=True`; absent columns are dropped by the pool's presence filter otherwise). Config: `patch_size` (`--patch-size`, default 10), `patch_coloc_threshold` (`--patch-coloc-threshold`, default 0.5).

**`_pearson_r_pairwise(images_a, images_b, pixel_mask=None)`** — helper used by isotopologue and adduct colocalization. Takes two `(N, H, W)` float32 arrays and returns a length-N array of per-feature Pearson r values. Uses manual mean-centering and dot product (avoids `np.corrcoef` memory overhead). When `pixel_mask` is given, r is computed over on-tissue pixels only (consistent with `_pearson_r_matrix`). Returns `np.nan` for constant images.

**`compute_isotopologue_colocalization()`** and **`compute_adduct_colocalization()`** — both accept an `extra_ion_images: dict | None` parameter. When provided (keys: `"m1"`, `"m2"`, `"na"`, `"k"`, `"chca"`), they use `_pearson_r_pairwise` to compute direct per-feature Pearson r between M0 images and pre-extracted partner images. This is necessary because MALDI feature lists contain only predefined monoisotopic M0 peaks — M+1/M+2 and adduct peaks are absent from the feature list and cannot be found by index lookup. When `extra_ion_images=None`, the old fallback path uses `_find_partner_indices` (vectorized `searchsorted` + nearest-neighbour check) to locate partner images within the feature list and slices the shared corr matrix — preserved for backwards compatibility (e.g. plain m/z text file or imzML input).

**`compute_nmf_colocalization_features()`** — NMF substructure-sharing colocalization (opt-in, `--nmf-coloc`). Factorises the TIC-normalised on-tissue ion-image matrix (`pixel_mask`) into `nmf_n_components` (default 12) non-negative spatial components via `sklearn.decomposition.NMF` (`_nmf_loading_cosine_matrix`), giving each feature a loading vector over those components. The within-protein pairwise **cosine similarity** of loadings (scale-invariant, so abundance does not matter) is aggregated to `protein_nmf_colocalization` (mean), `_max`, `_median`. This asks whether same-protein peptides occupy the *same tissue substructure* — a sharper question than global ion-image Pearson r, which is dominated by overall tissue morphology. Reuses the same protein self-join machinery as `compute_colocalization_features`; protein-level, so valid only because every decoy method gives decoys a separate `DECOY_`/`ENTRAPMENT_` namespace. NMF fit cost is ~6 s at full scale (~2869 images × 48 K on-tissue pixels, K=12); the dominant cost (ion-image extraction) is already paid by the pipeline. Declared in `feature_generator.NMF_COLOCALIZATION_FEATURES`, computed in `compute_all_features` only when `nmf_coloc=True`, and appended to the ranker pool at runtime in `pipeline.py` when the flag is set (independent of `--use-protein-level-feats`; the order-preserving dedup guard covers any overlap). **Note (this dataset):** on the amyloidosis data it does not discriminate — relocated decoys share substructures as much as same-protein targets (pooled NMF cosine: targets 0.715, decoys 0.790), consistent with the masked-Pearson result; the feature is provided for datasets that do have protein-specific spatial structure. See `notebooks/gt_protein_ion_images.ipynb` §8.

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

Three named feature group lists are exported from this module — see "Feature groups" section below. `LDA_FEATURES` is an alias for `MALDI_INTRINSIC_FEATURES` (currently identical) used by the LDA backend.

### Feature groups

`feature_generator.py` exports three lists that are importable directly:

```python
from MSI-PICASSO.feature_generator import (
    MALDI_INTRINSIC_FEATURES, PROTEIN_LEVEL_FEATURES, LCMS_PRIOR_FEATURES
)
```

**`MALDI_INTRINSIC_FEATURES`** — features passed to the ranker by default; computable from MALDI data alone plus in-silico properties:
- Mass accuracy: `ppm_error_abs`, `ppm_rank`, `ppm_best_ratio`, `ppm_error_calibrated_z` (optional, requires pixel coords)
- Ambiguity: `n_candidates`, `log_n_candidates`
- Peptide (basic): `peptide_length`, `n_missed_cleavages`
- Peptide (extended): `has_oxidized_met`, `has_cys`, `n_proline`, `acidic_residue_density`
- MALDI signal: `log_maldi_intensity_p90`, `log_maldi_intensity_sum`
- Mass defect: `kendrick_mass_defect`, `mass_defect_residual`
- CHCA matrix: `chca_cluster_distance_ppm`
- Theoretical isotope: `theo_isotope_cosine`, `theo_isotope_chi2`, `theo_isotope_kl`, `theo_has_sulfur`, `averagine_deviation`, `averagine_deviation_sulfur`, `theo_m1_ratio_diff`, `theo_m2_ratio_diff`, `monoisotopic_confidence`
- Ionization priors: `n_arginine`, `n_basic_residues`, `n_aromatic`, `gravy_score`, `charge_proxy`
- Ion mobility (optional, requires im2deep + observed CCS): `im2deep_delta_ccs`, `im2deep_abs_delta_ccs_pct`, `im2deep_ccs_zscore`, `im2deep_ccs_rank`, plus the m/z-detrended variants `im2deep_delta_ccs_resid`, `im2deep_abs_delta_ccs_pct_resid`, `im2deep_ccs_zscore_resid`, `im2deep_ccs_rank_resid` (see "m/z-detrended CCS" below)
- Isotopologue co-localization (optional, requires ion_images): `isotope_image_colocalization_m1`, `isotope_image_colocalization_m2`, `isotope_image_colocalization_mean`
- Adduct co-localization (optional, requires ion_images): `adduct_colocalization_na`, `adduct_colocalization_k`, `adduct_colocalization_chca`

**`SPATIAL_PRIOR_FEATURES`** — excluded from the ranker; applied as additive log-prior alongside `LCMS_PRIOR_FEATURES` (via `compute_spatial_prior()` in `pipeline.py`):
- `spatial_autocorrelation`, `fraction_detected`, `intensity_cv`, `log_mean_intensity`, `spatial_entropy`, `spatial_morans_i`, `spatial_gearys_c`

**Why excluded from the ranker:** these are feature-level signals (identical for every candidate at the same MALDI m/z). They cannot discriminate between candidate sequences within a feature, so including them in the ranker adds no signal and may destabilise training. `spatial_gearys_c` is negated before normalization (lower Geary's C = positive autocorrelation = better quality).

**`PROTEIN_LEVEL_FEATURES`** — excluded from the ranker by default; opt-in via `--use-protein-level-feats`:
- Protein consistency: `protein_n_features`, `log_protein_n_features`, `protein_coverage`, `protein_rank`, `protein_best_ratio`
- Protein co-localization (optional, requires ion_images): `protein_colocalization`, `protein_colocalization_max`, `protein_colocalization_median`, `protein_colocalization_n_partners`, `protein_colocalization_weighted`, `protein_colocalization_weighted_max`, `protein_colocalization_top2/top3/top5`; patch-level (opt-in `--patch-coloc`): `protein_patch_colocalization_mean/_max/_frac_above`

**Why excluded by default:** these features aggregate counts and correlations over all candidates sharing a protein. They are only valid when decoys occupy a **separate protein namespace** from targets — every decoy method gives decoys a distinct protein label (`DECOY_…` for shuffle / balanced_shuffle / paired_shuffle / mz_shift / mz_shuffle, `ENTRAPMENT_…` for entrapment), so a decoy is never pooled with its source target's protein. (Before this was fixed, `mz_shift`/`mz_shuffle` decoys kept the real target protein name, which pooled targets and decoys under one protein and made decoy proteins colocalize as well as — or better than — targets: an invalid null.) Even with the namespace correct, these features can interact subtly with the decoy model, so they stay opt-in via `--use-protein-level-feats`.

**`SPATIAL_RANKER_FEATURES`** — excluded from the ranker by default; opt-in via `--use-spatial-ranker-features`. Adds feature-level spatial quality (`spatial_autocorrelation`, `spatial_morans_i`, `spatial_gearys_c`, `fraction_detected`, `intensity_cv`) and protein colocalization (`protein_colocalization`, `protein_colocalization_max`, `protein_colocalization_median`, `protein_colocalization_n_partners`) to the ranker feature pool. The list is appended at runtime in `pipeline.py` only when the flag is active; `MALDI_INTRINSIC_FEATURES` is not modified.

**Permitted decoy methods:** only `entrapment`, `mz_shift`, and `mz_shuffle`. Their decoys land on real MALDI features, so spatial features form a symmetric null (the ranker learns that real peptides at high-quality, spatially structured features score better than decoys at random/foreign anchors). `mz_shuffle` is the ideal case — its decoys are co-located with targets on the identical features, so the spatial features are *exactly* symmetric and cannot bias the null at all. With `shuffle`/`balanced_shuffle`/`paired_shuffle` the flag is **force-disabled with a `UserWarning`** (`_resolve_spatial_ranker_features` in `pipeline.py`): those decoys have no consistent spatial anchor, so spatial features would be asymmetric. The `protein_colocalization_*` members overlap `PROTEIN_LEVEL_FEATURES`; an order-preserving dedup guard in the `pipeline.py` feature-pool assembly prevents double-inclusion when both `--use-protein-level-feats` and `--use-spatial-ranker-features` are active.

**`LCMS_PRIOR_FEATURES`** — excluded from the ranker, applied as an additive log-prior after scoring. All features are derived symmetrically from raw mzML (no search engine scores):

*mzML-derived* (`_LCMS_MZML_FEATURES`):
- MS2: `lcms_ms2_spectral_angle`, `lcms_ms2_n_matches` (both filtered by DeepLC RT window when `rt_window_min > 0`)
- DeepLC-anchored MS1 signal: `lcms_ms1_intensity`, `lcms_ms1_snr`
- DeepLC-anchored MS1 isotope: `lcms_ms1_isotope_cosine`, `theo_m1_ratio_diff_lcms`, `theo_m2_ratio_diff_lcms`, `log_theo_m1_ratio_diff_lcms`, `log_theo_m2_ratio_diff_lcms`
- MALDI vs LC-MS/MS envelope similarity (requires `maldi_envelopes`): `isotope_envelope_cosine`, `isotope_envelope_pearson`, `isotope_envelope_mse`, `isotope_n_matched`, `isotope_absolute_diff`, `log_isotope_m1_ratio_diff`, `log_isotope_m2_ratio_diff`
- DeepLC-anchored RT-consistency (requires `rt_window_min > 0`): `lcms_ms1_apex_rt_delta`, `lcms_ms1_frac_apex_signal`, `lcms_ms1_n_scans_with_signal`, `lcms_ms2_rt_delta`

Note: all six MALDI-vs-LC-MS/MS envelope similarity features (`isotope_envelope_cosine`, `isotope_envelope_pearson`, `isotope_envelope_mse`, `isotope_n_matched`, `isotope_absolute_diff`, `log_isotope_m1_ratio_diff`, `log_isotope_m2_ratio_diff`) are in `_LCMS_MZML_FEATURES` and applied as a prior. They require `maldi_envelopes` to be non-None; when absent, all values are 0.0 and the columns are skipped by `compute_lcms_prior` (min == max → all-NaN after normalisation).

*CCS-derived* (`_LCMS_CCS_FEATURES`, optional): `lcms_ccs_delta`, `lcms_ccs_abs_pct`

Note: `_LCMS_ID_FEATURES` (`lcms_q_value`, `lcms_pep`, `lcms_score`, `n_psms`, `lcms_intensity`) are populated in the candidates DataFrame by Strategy C but are **not** included in `LCMS_PRIOR_FEATURES`. Using ID-derived features in the prior would give LC-MS/MS confirmed targets different treatment than decoys, breaking TDC symmetry.

**Design rationale:** LC-MS/MS features are explicitly excluded from the ranker training set. Instead, LC-MS/MS evidence is applied as an additive log-prior *after* MALDI-intrinsic scoring (see `compute_lcms_prior()` and `compute_spatial_prior()` in `pipeline.py`). `compute_lcms_prior` min-max normalizes each mzML feature (using `np.nanmin`/`np.nanmax` to ignore NaN), fills NaN sentinel values (e.g. `lcms_ms2_spectral_angle` when no MS2PIP prediction) with the column minimum after normalization so they do not penalize candidates, and returns the `np.nanmean` across features as a per-candidate weight in (0, 1]. Features that are all-NaN after normalization are skipped. The log of this weight (and the spatial prior) is added to the round-2 score before FDR computation.

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
| `"decoy"` | K/R-preserving shuffle of an identified protein (digest mode), or pseudo-protein concat decoy (LC-only mode) |

**Decoy generation — two sub-cases:**

*With `--digest` (fasta_path not None):* Identified proteins are shuffled at the protein level (`_shuffle_protein`) and re-digested. Novel confirmed sequences not reachable from the digest get a per-peptide K/R-preserving shuffle as their decoy. `protein = "DECOY_{accession}"`.

*Without `--digest` (fasta_path=None, LC-only mode):* All confirmed peptides are novel. Per-peptide shuffle would produce decoys with identical elemental composition (same residue multiset, just reordered) — making isotope envelope features (`theo_isotope_cosine`, `theo_isotope_chi2`) non-discriminative. Instead, the **concatenated pseudo-protein** strategy is used:
1. All confirmed sequences are sorted and concatenated into a single pseudo-protein string.
2. `_shuffle_protein(pseudo_protein, random_state=42)` redistributes non-K/R residues across the entire sequence.
3. The shuffled pseudo-protein is digested with the same trypsin rules (same `missed_cleavages`, `min_length`, `max_length`).
4. Exact matches with any target sequence are removed.
5. If more decoys than targets: subsample to match (seeded). If fewer: warn about sub-1:1 TDC ratio.
6. All decoys get `protein = "DECOY_concat"` and `source = "decoy"`.

This ensures decoy peptides draw non-K/R residues from across the shuffled pool of all target peptides, breaking the isobaric property of per-peptide shuffle.

LC-MS/MS evidence is joined onto target rows from `lcms_ids.peptides`; decoy rows always get `NaN`.

**Fallback:** if `digest_identified_proteins()` returns 0 rows (e.g. no FASTA proteins found), `rescore()` falls back to Strategy A with a warning.

**`is_decoy` dtype:** Both `digest_fasta()` and `digest_identified_proteins()` enforce `df["is_decoy"] = df["is_decoy"].astype(bool)` before returning. This prevents pandas `object`-dtype booleans (which arise from `pd.concat` with empty DataFrames) from breaking `~df["is_decoy"]` boolean indexing downstream.

### `lcms_ids.py`

Parses LC-MS/MS identification results into an `LCMSIds(proteins, peptides)` namedtuple.

| Component | Type | Content |
|---|---|---|
| `proteins` | `set[str]` | Normalised accessions passing `protein_fdr` |
| `peptides` | `pd.DataFrame` | Unique sequences passing `peptide_fdr`, with evidence columns |

**Peptide DataFrame columns:** `sequence`, `peptidoform`, `protein`, `q_value`, `pep`, `score`, `n_psms`, `charge`, `rt_mean`, `lcms_intensity`

**RT unit normalisation** (`_parse_psm_utils`): after aggregating PSMs to peptide level, if the median `rt_mean` exceeds 200 the values are divided by 60 (seconds → minutes). This matches the same pattern used in `_join_psm_rt_intensity`. FragPipe PSM TSV files report `Retention` in seconds (typical range 131–2564 s); all other supported formats report in minutes.

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

`rescore()` is the end-to-end entry point. Accepts `extra_ion_images: dict | None` (keys: `"m1"`, `"m2"`, `"na"`, `"k"`, `"chca"`) and passes it through to `compute_all_features` → `compute_isotopologue_colocalization` / `compute_adduct_colocalization`. Populated by `_load_maldi` in `cli.py` from the Bruker RAM extraction path or from NPZ `extra_*` keys. Steps 1-8 are identical for both backends; step 9 diverges:

**Key `rescore()` parameters exposed since v0.1:**

| Parameter | Default | Description |
|---|---|---|
| `matching_ppm` | 20.0 | ppm window for candidate-to-feature linking in `match_to_maldi_features` — separate from `ppm_tolerance` (extraction window) |
| `fragment_tol_da` | 0.02 | MS2 fragment matching tolerance (Da) for spectral angle computation |
| `winner_percentile` | 0.02 | `_select_feature_winners` quality filter: features whose R1 winner score falls below this quantile of all winner scores are dropped before R2 training |
| `rt_window_multiplier` | 2.0 | RT window = multiplier × p95 DeepLC MAE; controls ±window for MS1/MS2 RT filtering |
| `lcms_prior_weight` | 1.0 | Multiplicative weight on the LC-MS/MS log-prior in reweighted scoring |
| `spatial_prior_weight` | 1.0 | Multiplicative weight on the spatial quality log-prior in reweighted scoring |
| `match_ccs` | False | Enable CCS-based candidate filtering after IM2Deep finetuning (see below) |
| `ccs_window_multiplier` | 2.0 | CCS filter threshold = multiplier × p95 \|delta_CCS%\| on single-candidate calibration set |
| `entrapment_fasta` | None | Foreign-organism FASTA used as the null when `decoy_method="entrapment"` (required for that method) |
| `maldi_query_raw` | False | Raw-query mode: extract ion images directly at candidate-derived m/z (requires `maldi_d_path`); inverts pipeline ordering |
| `maldi_d_path` | None | Raw Bruker `.d` directory; required when `maldi_query_raw=True` |
| `extraction_ppm` | 25.0 | Ion image extraction half-window (ppm) for raw-query mode |
| `use_spatial_ranker_features` | False | Include `SPATIAL_RANKER_FEATURES` in the ranker; only valid with `decoy_method` ∈ {entrapment, mz_shift} |

**Decoy mode parameter:** `decoy_method` (str, default `"shuffle"`) controls Step 1c:
- `"shuffle"` — standard K/R-preserving protein shuffle (via `digest_fasta` / `digest_identified_proteins`). Decoys are sequence-space decoys with distinct elemental compositions (different `theo_isotope_cosine`).
- `"mz_shift"` — observation-space m/z-shift decoys via `generate_mz_shift_candidates()`. Each unique target peptide is shifted by a random delta in `[mz_shift_delta_min, mz_shift_delta_max]` Da (sign alternates). In feature-list mode the shift is snapped to the nearest MALDI feature within `mz_shift_snap_tolerance_ppm`; **in raw-query mode (`maldi_query_raw`) snapping is disabled** (`snap_to_features=False`) so each decoy sits at its exact shifted m/z on a distinct feature — avoiding the decoy-clustering that otherwise collapses decoys onto few grid points and skews the winner T:D ratio (see candidates.py). A collision filter rejects shifts within `matching_ppm` of a target m/z (and, in raw-query, of an already-used decoy m/z). **`feature_mz` on mz_shift decoy rows is the shifted (off-target) m/z, not the original peptide m/z** — load-bearing for raw-query. `ppm_error` is copied from the peptide's best target match in feature-list mode (non-discriminative) and recomputed from observed peak centroids in raw-query mode. `source = "decoy_mz_shift"`. Decoys get a **separate protein namespace** (`DECOY_<protein>`) so protein-level features stay within-class. **Compatible with `use_spatial_ranker_features`** (decoys land on real MALDI features → genuine spatial signal at the anchor; an acceptable null).
- `"mz_shuffle"` — m/z-assignment-shuffle decoys via `generate_mz_shuffle_candidates()`. A derangement of the peptide→feature assignment: each real target peptide is relocated onto **another peptide's real feature** (mass-sorted rotation σ with no fixed point and a large mass gap, so never self- or near-isobaric). Result: decoy features = the target feature set, **co-located 1 target + 1 decoy per feature on the identical ion image**, so feature-quality features (`fraction_detected`, intensity, spatial, colocalization) are *identical* between a feature's target and decoy and contribute **zero** to the target/decoy separation — the ranker is forced onto the peptide-specific predicted-vs-observed match (`im2deep_*` CCS, isotope). Restores genuine per-feature competition (contested features, which `mz_shift` lacks). `ppm_error` is inherited from the peptide's best target match (non-discriminative; recomputed anchor-relative in raw-query) — it must **not** be computed against the decoy peptide's own mass, since that mismatch does not exist for real false positives (which match within tolerance) and would make the null anti-conservative. `source = "decoy_mz_shuffle"`. **Recommended with `use_spatial_ranker_features`** — this is the case where the quality features are exactly symmetric. Mild conservative bias (the decoy's "wrong peptide" is another real peptide-like peak; real false positives may be easier-to-reject non-peptides). **CCS / mobility handling:** any feature that uses the candidate's *predicted* CCS/mobility to gate or compare against the observed feature leaks the m/z baseline for `mz_shuffle` (decoys relocated far in mass; CCS / 1-K0 ∝ m/z). The pipeline therefore **excludes `_MZ_SHUFFLE_CCS_LEAK_FEATURES` from the ranker for `mz_shuffle`** — the raw `im2deep_*` CCS scalars **and** the mobility-gated colocalizations (`isotope_colocalization_*_mob`, `adduct_colocalization_*_mob`, which filter the shared ion image with each candidate's own predicted-1/K0 window, so a decoy's window misses the heavy feature's peak) — keeping only the m/z-detrended `*_resid` CCS features. The **non-gated** colocalizations (`isotope_image_colocalization_*`, `adduct_colocalization_*`) stay: they read the shared co-located ion image, so they are exactly symmetric (AUC ≈ 0.5). Other decoy methods keep all of these. **Do not combine `mz_shuffle` with `--match-ccs`** — the CCS prefilter would remove ~all decoys (they fail CCS by design), collapsing the null; let CCS discriminate in the ranker instead.

**m/z-detrended CCS (`*_resid`):** `compute_im2deep_features` fits a power-law CCS↔m/z trend `g(mz)=A·mz^B` on the calibration peptides (`_ccs_mz_baseline`) and subtracts the expected m/z-gap CCS difference `g(feature_mz) − g(mh_mz)` from the raw delta. For targets (`feature_mz == mh_mz`) the baseline is 0 so `*_resid == raw`; for decoys whose peptide m/z differs from the feature m/z (`mz_shift`/`mz_shuffle`/`entrapment`) it removes the trivial m/z-gap separation, leaving the conformational mismatch (`conf(observed) − conf(predicted)`), which is exchangeable with an isobaric false positive. A leak-check log line reports `|corr(raw Δ, decoy_delta_da)|` vs `|corr(residual Δ, decoy_delta_da)|` — the residual should be near 0.
- `"entrapment"` — decoys are tryptic peptides from a foreign-organism FASTA (`entrapment_fasta`) via `load_entrapment_candidates()`. A contamination filter removes any entrapment peptide isobaric (within `matching_ppm`) with a target; surviving peptides are matched to MALDI features exactly as targets. `protein="ENTRAPMENT_{accession}"`, `source="entrapment"`. **Compatible with `use_spatial_ranker_features`.**
- `"balanced_shuffle"` — iterative K/R-preserving protein shuffle with MALDI-match filtering via `generate_balanced_shuffle_candidates()`. Runs up to `max_shuffle_rounds` (default 50) rounds of protein-level shuffle, keeping only decoy peptides that match a MALDI feature within `matching_ppm`. Subsamples the collected pool to `target_ratio * N_target` (default 1.0). Unlike standard shuffle (one decoy per target regardless of MALDI match), this ensures decoys compete in the same observation space as targets and achieves ~1:1 T:D even when the MALDI feature list is sparse. LC-MS/MS evidence columns are NaN for all decoy rows (shuffle decoys have different sequences; inheriting evidence would break TDC symmetry). `source = "decoy_balanced_shuffle"`. **Not compatible with `use_spatial_ranker_features`** (no consistent spatial anchor).
- `"paired_shuffle"` — same shuffle pool as `balanced_shuffle` but decoys are feature-occupancy-matched (`selection_mode="feature"`), drawn at the same m/z features that targets occupy to maximise per-feature competition. **Not compatible with `use_spatial_ranker_features`.**

CLI flags: `--decoy-method {shuffle,mz_shift,mz_shuffle,entrapment,balanced_shuffle,paired_shuffle}`, `--entrapment-fasta PATH`, `--mz-shift-delta-min FLOAT`, `--mz-shift-delta-max FLOAT`, `--mz-shift-snap-tolerance-ppm FLOAT`, `--max-shuffle-rounds INT`, `--decoy-target-ratio FLOAT`, `--maldi-query-raw`, `--use-spatial-ranker-features`.

1. Generate candidates (Strategy A or C) + match to MALDI features
2. Load LC-MS/MS data
3. Find MS2 matches by neutral mass; run MS2PIP only for `(peptide, charge)` pairs at features with observed MS2 scans
4. DeepLC: optionally fine-tune on PD MSF or FragPipe RT table, then predict RT for all unique peptides
5. Compute LC-MS/MS evidence features (DeepLC-anchored MS1 features; fully symmetric)
6. Compute all features (includes IM2Deep finetuning on `n_candidates==1` matches when CCS data is available)
6b. *(optional)* **CCS filter** — when `match_ccs=True`: compute p95 of `im2deep_abs_delta_ccs_pct` on single-candidate matches; remove all candidates where `im2deep_abs_delta_ccs_pct > ccs_window_multiplier × p95`; recompute `n_candidates` and `log_n_candidates`. LDA positive seeding at step 9 uses post-filter `n_candidates`, so newly unambiguous features (filtered down to one candidate) contribute as seed positives.
7. Build PSMList + populate rescoring features
8. Rescore using selected backend (see "Rescoring backends" below)

### Two-pass scoring logic

All backends follow the same two-pass structure:

1. **Round 1** — score all candidates globally. The model does not use per-feature grouping; every candidate is treated on equal footing.
2. **Per-feature winner selection** (`_select_feature_winners`) — for each MALDI m/z feature, retain only the highest round-1 score candidate. Produces `winners_df` (~N rows for N features). A quality filter is then applied: features whose winner R1 score falls below `np.quantile(winner_scores, winner_percentile)` are dropped (`is_tdc_winner=False`, `q_value=NaN`). `winner_percentile` defaults to 0.02 (2nd percentile); raise it to filter more aggressively before R2 training.
3. **Round 2** — retrain/rescore on the winner subset only. Because each feature contributes exactly one candidate, this is a cleaner training set than the full candidate pool.
4. **FDR** — standard TDC (`_tdc_qvalues`) over all winners sorted by round-2 score. Q-values propagated to non-winners as NaN.

**Single-round toggle (`single_round=True`, `--single-round`).** Skips step 3: the FDR is computed directly on the R1 winner scores (`scores2 = scores1[winner_pos]`), and the R2 importance/struct outputs reuse the R1 model. Steps 1, 2, 4 are unchanged — crucially, the per-feature winner selection (the **target-vs-decoy competition** that defines the TDC population) still runs, so the FDR semantics are identical; only the final discriminant refit is dropped. Motivated by raw-query (`--maldi-query-raw`): with `--matching-ppm 0` there is ~1 target per feature and (with `mz_shuffle`) one co-located decoy, so R1 already trains on a clean ~1:1 target:decoy set and R2 typically adds little. The winner selection collapses *target-vs-target* ambiguity — moot in raw-query — but the *target-vs-decoy* duel it performs is not, which is why winner selection is kept and only R2 is optional. Implemented for `lda`/`svm` (shared branch) and `qda` (which reuses R1 posterior probabilities for PEP). Use it to A/B whether R2 changes the IDs-at-1%-FDR on a given run. **Debug figures** are kept honest when `single_round`: `save_debug_figures(single_round=True)` suppresses the round-2 feature-importance panel and relabels the round-2/final panels in the score-PP, score-distribution, PEP-mixture, and feature-distribution figures as "final (R1 winners)" instead of "R2" (the `*_score_r2` column is retained as the final-score slot; only labels change). The R2 importance TSVs / score pickles are not written.

`result_df` contains all candidates. Round-2 score, q-value, and reweighted columns are NaN for non-winners. `is_tdc_winner` marks the round-1 winner per feature.

### Rescoring backends

`rescore()` accepts a `model` parameter:

**`model="lda"` (default):** Semi-supervised `LinearDiscriminantAnalysis` (sklearn) on `MALDI_INTRINSIC_FEATURES`. No extra dependencies beyond sklearn (always installed). Converges quickly and produces clean feature importances.

Preprocessing: ±inf replaced with NaN, then `SimpleImputer(strategy="median")` + `StandardScaler` inside a sklearn `Pipeline`. LDA is configured with `solver="lsqr"` and `shrinkage="auto"` (Ledoit-Wolf regularisation).

**Shared linear routine.** LDA and SVM (below) are both linear, `decision_function`-based classifiers and share a single implementation, `_rescore_linear(..., make_clf, clf_name)` (`pipeline.py`). `make_clf` is a zero-arg factory for the final pipeline estimator and `clf_name` its step key / log tag; `_rescore_lda` and `_rescore_svm` are thin wrappers. The entire dispatch branch (`if model in ("lda","svm")`) is shared — winner selection, TDC, PEP-from-scores, reweighting — with score columns tagged `f"{model}_score_r1/r2"` and importance TSVs `17_debug_{model}_importances_r{1,2}.tsv`. Importances are `coef_[0]` for both.

**Cross-validated scoring (LDA, SVM, and QDA).** The pseudo-label iteration scores candidates **out-of-fold** (`_cv_semisup_scores`, default `cv_folds=3`, stratified by `is_decoy` via `_make_fold_ids`): at each iteration every candidate is scored by a model trained on the *other* folds' positives/decoys, so no row is scored by a model that trained on it. This prevents the semi-supervised discriminant from **overfitting** — i.e. manufacturing target/decoy separation by fitting noise in high-dimensional feature space, which would make the TDC FDR anti-conservative. (Diagnostic: a plain in-sample LDA on a fair `mz_shuffle` null gave in-sample AUC 0.71 but 5-fold CV AUC 0.45 — at chance; the in-sample separation was entirely overfit.) Folds are fixed across iterations; the function falls back to in-sample scoring only when there are fewer than `2·cv_folds` targets or decoys (logged). The returned **feature importances / structure coefficients come from a model fit on all positives+decoys** (reporting only — never the FDR scores, which are strictly out-of-fold). `_tdc_qvalues`, winner selection, and reweighting all operate on the out-of-fold scores.

**Round-1 seed — `_find_best_feature_labels` (Mokapot-style):**

For each feature column and each ranking direction (ascending / descending), TDC q-values are computed and the number of targets at q ≤ `train_fdr` is counted. Sub-ULP random noise (`np.random.default_rng(0).uniform(-1e-9, 1e-9)`) is added before argsort to break ties and prevent row-order bias (targets listed before decoys in the DataFrame would otherwise receive artificially low q-values under stable sort). The (feature, direction) pair yielding the most targets is selected.

If the best single-feature result is below `min_pair_threshold` (default 10) targets, all pairwise sums and differences of eligible features are tried on standardised columns. The composite score beating the single-feature result is used if one exists.

Columns in `_BEST_FEAT_SKIP` are excluded from both the single-feature and pairwise sweeps because they measure amino acid composition rather than spectral quality and can produce spurious pseudo-positives when shuffled decoys have a different residue composition than targets:

```python
_BEST_FEAT_SKIP = {
    "peptide_length", "n_missed_cleavages",         # basic sequence
    "has_oxidized_met", "has_cys", "n_proline", "acidic_residue_density",  # composition
    "n_arginine", "n_basic_residues", "n_aromatic", "gravy_score", "charge_proxy",  # ionisation
}
```

**Fallback chain when best-feature init yields 0 targets:**
1. `ppm_error_abs < init_ppm_threshold` OR `n_candidates == 1`
2. If that also yields nothing: top `r1_seed_percentile` (default 10%) of targets by `ppm_error_abs`

**Pseudo-label iteration** (up to `max_iter` rounds, default 5): train on seed positives (+1) + all decoys (−1), excluding unlabelled targets (0) from the training set; score all candidates with `decision_function`; recompute TDC q-values; promote all targets at q ≤ `train_fdr` to +1. Stop when the positive count changes by < 1% or no positives remain.

**Round-2 seed:** top `r2_seed_percentile` (default 20%) of target TDC winners by R1 score — i.e. targets with `R1_score ≥ np.percentile(target_winner_scores, 100*(1−r2_seed_percentile))`. Ppm-based seeding is not used for R2 because after winner selection most targets already satisfy `ppm_error_abs < init_ppm_threshold` and the criterion becomes uninformative.

**Key parameters** (all configurable via CLI / TOML):

| Parameter | Default | Effect |
|---|---|---|
| `init_ppm_threshold` | 5.0 | ppm cutoff for the ppm-based fallback seed |
| `train_fdr` | 0.01 | q-value threshold for pseudo-label promotion |
| `max_iter` | 5 | Maximum pseudo-label iterations |
| `min_pair_threshold` | 10 | Min targets required from single feature before trying pairs |
| `r1_seed_percentile` | 0.10 | Top fraction of targets by ppm used as last-resort R1 seed |
| `r2_seed_percentile` | 0.20 | Top fraction of target winners by R1 score used as R2 seed |

Feature importances: `|coef_[0]|` from the final Pipeline LDA. Saved to `17_debug_lda_importances_r1/r2.tsv` when `--verbose`.

Returns `(psm_list, result_df, feature_names)` where `result_df` has columns: `peptide`, `protein`, `feature_mz`, `feature_idx`, `is_decoy`, `lda_score_r1`, `lda_score_r2`, `q_value`, `is_tdc_winner`, `reweighted_score`, `reweighted_q_value`.

---

**`model="svm"`:** Semi-supervised `sklearn.svm.LinearSVC` (`penalty="l2"`, `loss="squared_hinge"`, `C=svm_c` (default 1.0, `--svm-c`), `dual="auto"`, `max_iter=2000`) on `MALDI_INTRINSIC_FEATURES`. Shares `_rescore_linear` with LDA, so the median imputer (LinearSVC rejects NaN), StandardScaler, out-of-fold CV, pseudo-label iteration, and `coef_[0]` importances are identical; only the final estimator differs. `LinearSVC.decision_function()` provides the per-candidate score the CV machinery needs. Result columns: `svm_score_r1`, `svm_score_r2`. A fast linear alternative to LDA for benchmarking (the two often track closely; SVM's hinge loss is less sensitive to non-Gaussian feature tails).

> **Removed backends (distinct from the above):** the *old* `model="svm"` (mokapot `PercolatorModel`) and `model="catboost"` (`CatBoostRanker`) were removed. The current `svm` is sklearn `LinearSVC` and adds no dependency. `--model` accepts `{lda, qda, svm}`. The `mokapot`/`catboost` packages and `probabilistic_scorer.py` (the former SVM/CatBoost step-7b feature source) are no longer used by the scoring path.

**`model="qda"`:** Semi-supervised `QuadraticDiscriminantAnalysis(reg_param=0.1)` on `MALDI_INTRINSIC_FEATURES`. Same pseudo-label iteration and seed logic as LDA. `reg_param=0.1` regularizes the per-class covariance toward a scaled identity matrix. Returns `(psm_list, result_df, feature_names)` where `result_df` has columns analogous to LDA but with `qda_score_r1`, `qda_score_r2`.


**Post-scoring reweighting** (applied after all backends to winners only):

`compute_lcms_prior()` min-max normalizes each `LCMS_PRIOR_FEATURES` column using `nanmin`/`nanmax`. NaN values (e.g. `lcms_ms2_spectral_angle` when no prediction exists) are treated as the column minimum after normalization so they contribute 0 without inflating the mean. The `nanmean` across available features returns a per-candidate weight in (0, 1]; candidates where all features are NaN receive weight 1.0 (no penalty).

`compute_spatial_prior()` min-max normalizes spatial quality features (`spatial_autocorrelation`, `spatial_gearys_c` negated, etc.) for the winner subset and returns a per-candidate weight in (0, 1]. Returns 1.0 if no informative spatial features are present.

Both priors are combined as a **weighted additive log-prior** (not a multiplicative prior):

```
reweighted_score = round2_score
                 + lcms_prior_weight  * log(lcms_prior)
                 + spatial_prior_weight * log(spatial_prior)
```

`lcms_prior_weight` and `spatial_prior_weight` (both default 1.0) scale the contribution of each prior independently. Values > 1 amplify the prior's influence; 0 disables it entirely. Multiplicative combination would invert the ranking for negative scores (a bad candidate with low prior would become less negative, i.e. higher ranked). The additive log form penalises low-prior candidates uniformly regardless of score sign. Q-values are recomputed on the reweighted scores with a second `_tdc_qvalues` call. Only winner rows receive non-NaN values.

---

## Rust extension (crate `MSI-PICASSO-rs`, dir `msi-picasso-rs/`, import `ms1rescore_rs`)

Built with PyO3 + rayon; imported in Python as `from ms1rescore_rs import ...`. Exposed functions:

| Function | Module | Description |
|---|---|---|
| `compute_peptide_masses(sequences)` | digest.rs | Returns (masses, mh_mzs, n_C, n_H, n_N, n_O, n_S) for a list of peptide sequences |
| `match_mz(feature_mzs, peptide_mzs, ppm)` | digest.rs | Binary search m/z matching; returns (feat_idx, pep_idx, ppm_errors) |
| `extract_xics_batch(ms1_rts, ms1_mz_arrays, ms1_int_arrays, target_mzs, ppm)` | xic.rs | Parallel XIC extraction for multiple target m/z values across all MS1 scans |
| `extract_ms1_envelopes_batch(ms1_mz_arrays, ms1_int_arrays, scan_indices, target_mzs, charge, n_peaks, ppm)` | isotope.rs | MS1 isotope envelope extraction at specified scans |
| `spectral_angles_batch(pred_mzs, pred_ints, obs_mzs, obs_ints, fragment_tol_da)` | spectral.rs | Batch spectral angle; returns 0.0 when fewer than 3 fragments match (genuine poor match). NaN (no-prediction sentinel) is set at the Python level in `compute_all_lcms_evidence` when no MS2PIP prediction exists for a candidate. |
| `compute_ionization_features(sequences)` | features.rs | Returns (n_R, n_K, n_H, n_F, n_W, n_Y, gravy, pi) — 8 arrays; parallel rayon |
| `compute_property_features(sequences)` | features.rs | Returns (n_D, n_E, n_C, n_P, n_M, n_W, n_Y, seq_len, nterm_code, pi) — 10 arrays; parallel rayon |
| `count_missed_cleavages_batch(sequences)` | features.rs | K/R not followed by P, excluding last residue; parallel rayon |
| `compute_maldi_isotope_means(flat_mzs, flat_ints, pixel_offsets, target_mzs, ppm_tolerance)` | maldi_isotope.rs | Mean intensity per target m/z across all MALDI pixels in a single CSR streaming pass (rayon). Replaces two `get_ion_images()` calls for M+1/M+2 extraction. Returns list of float64, one per target m/z. |
| `accumulate_profile_chunk(pixel_matrix, lo_indices, hi_indices)` | ion_image.rs | Profile-mode pixel window integration. Takes `(n_pixels, n_mz)` float32 C-contiguous array and per-feature `[lo, hi)` index pairs; returns flat float32 array of shape `(n_pixels * n_features,)` to be reshaped in Python. Rayon parallel over pixels. |

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
cd MSI-PICASSO/msi-picasso-rs
VIRTUAL_ENV=/home/robbe/.pyenv/versions/3.11.11/envs/MSIscore \
  /home/robbe/.pyenv/versions/3.11.11/envs/MSIscore/bin/maturin develop --release
```

**Important:** `maturin develop` must target the same Python environment as the Jupyter kernel. The MSIscore venv lives at `/home/robbe/.pyenv/versions/3.11.11/envs/MSIscore` (the symlink `/home/robbe/.pyenv/versions/MSIscore` also works for `VIRTUAL_ENV`, but call the venv's own `maturin` binary explicitly to avoid picking up the wrong interpreter). Without `VIRTUAL_ENV` and the correct binary, maturin installs into the base pyenv Python, not the venv.

The Rust `target/` directory can be 1-2 GB. Delete it with `rm -rf MSI-PICASSO/msi-picasso-rs/target/` if disk space is low (it is rebuilt on the next `maturin develop`).

---

## Environment

- Python: `/home/robbe/.pyenv/versions/MSIscore`
- Notebook kernel: MSIscore
- Key packages: `ms2pip>=4.0.0a1`, `deeplc>=4.0.0a1`, `pyteomics>=4.7`, `psm_utils>=1.1`, `brain-isotopic-distribution>=1.5` (PyPI name for `brainpy`), `alphatims>=1.0` (optional, for reading Bruker `.d` files directly). The `mokapot` and `catboost` packages are no longer required — the SVM/CatBoost backends were removed (LDA/QDA only).
- ms2pip import: `from ms2pip.core import predict_batch` (not `from ms2pip import predict_batch`)
- `brain-isotopic-distribution` is a transitive dependency of `ms-deisotope` but is listed explicitly in `pyproject.toml` as a core dependency because `theoretical_isotope_distribution()` in `utils.py` imports `from brainpy import isotopic_variants` directly.

Install the package in editable mode:
```bash
pip install -e MSI-PICASSO/
# With Bruker timsTOF .d support:
pip install -e "MSI-PICASSO/[timstof]"
```

---

## Notebooks and scripts

- **`notebooks/04_maldi_rescoring.ipynb`**: End-to-end notebook. Cells follow pipeline steps 1-9.
- **`notebooks/gt_protein_ion_images.ipynb`**: Protein-colocalization diagnostic. For a set of ground-truth amyloidosis proteins, digests the tryptic peptides (LC-identified vs not), builds `mz_shuffle`-style decoy counterparts (1 feature per peptide, matching the post-dedup pipeline), extracts ion images from the raw `.d`, and reports within-protein colocalization three ways: raw vs TIC-masked Pearson r (`COLOC_TIC_QUANTILE` knob, §5/7) and NMF substructure-loading cosine (`N_NMF_COMPONENTS`, §8). Built by `/tmp/build_gt_nb.py` (not in-repo). Conclusion baked into the narrative: colocalization is non-discriminative here (decoy ≈ target); any earlier apparent signal was the target feature-multiplicity artifact removed by the dedup fix.
- **`scripts/visualize_ms1rescore_features.py`**: Feature visualization script. For each selected MALDI feature: summary heatmaps, feature distribution histograms, bar charts. For each of 3 selected candidates per feature (best-ppm target, best-SA target, best-ppm decoy): XIC chromatogram, MS2 mirror plot, isotope envelope comparison, ion image, mass accuracy bar, feature card.

---

## Known issues and limitations

### Low spectral angles (~0.011 mean)

The mean spectral angle is low because most of the 707K candidates are random mass coincidences. The max is ~0.402 for correct peptides. This is expected: centroided Thermo MS2 spectra have 12-34 peaks, and most MALDI features do not have a matching LC-MS/MS MS2 scan at all. The feature is still informative as a discriminator.

### `maldi_intensities_p90` must be passed explicitly

`match_to_maldi_features()` accepts `maldi_intensities`, `maldi_intensities_p90`, and `maldi_intensities_sum`. If none are passed, all log-intensity features are 0. In `pipeline.py`, these are read from `spatial_features` columns (`intensity_p90`, `intensity_sum`, `mean_intensity`) when available. In standalone use (e.g. the viz script), pass at least `maldi_intensities` from ion images or spatial features.

### Scale

With the human FASTA (~20K proteins) and 1,398 MALDI features at 20 ppm:
- ~707K candidates (target + decoy)
- ~542K unique peptides
- ~1,067/1,398 features with MS2 matches
- MS2PIP is run for ~683K unique (peptide, charge) pairs (only at features with observed MS2 scans)
- DeepLC predictions run for all ~542K unique peptide sequences; each candidate is anchored to its peptide's nearest MS1 scan

---

## Known result biases

Each entry names the bias, which decoy method(s) and pipeline mode it affects, why it arises, and what the current mitigation is.

---

### 1. `peptide_length` leaks the mass-sorted derangement in the zero-signal subpopulation (`mz_shuffle` + `--maldi-query-raw`)

**Affects:** `decoy_method="mz_shuffle"` in raw-query mode when `drop_zero_signal=False`.

**Why it arises.** The mz_shuffle derangement is a mass-sorted cyclic rotation: each target peptide is relocated onto the feature of a peptide ~n/4–3n/4 positions away in mass-sorted order. Consequently, the decoy peptide co-located on a given feature has a systematically *different* mass (and therefore a different `peptide_length`) from the target peptide. For features that have genuine MALDI signal (`feature_intensity_sum > 0`), MALDI-dependent features (`log_maldi_intensity*`, `ppm_error`, isotope, CCS, etc.) provide the discrimination signal and overwhelm `peptide_length`. For **zero-signal features** — m/z bins with no detected peak across all pixels — all MALDI-dependent features collapse to 0 or their imputed median, leaving only sequence-derived features (`peptide_length`, amino acid composition) to separate targets from decoys. Because `peptide_length` ∝ mass and the derangement enforces a large mass gap, `peptide_length` achieves AUC ≈ 0.91 in the zero-signal subpopulation. This skews the FDR: the semi-supervised ranker trains on these easy zero-signal decoys and assigns artificially high scores to short (or long) target peptides regardless of MALDI evidence.

**Mitigation.** `drop_zero_signal=True` (CLI `--drop-zero-signal`, default in raw-query mode) removes all candidates where `feature_intensity_sum == 0` before scoring. The mask is `is_decoy`-blind: under mz_shuffle co-located target and decoy share the identical ion image, so both are dropped together, preserving the 1:1 T:D ratio. This is the right fix; retaining zero-signal rows only adds noise because no MALDI evidence supports them.

**Symptom.** If zero-signal rows are retained, the ranked-list shows an inflated tail of short peptides (low mass → negative mass-sorted offset → `peptide_length` of decoy > target) passing FDR, with no supporting MALDI ion image.

**Note.** `peptide_length` (and all `_BEST_FEAT_SKIP` features) are already excluded from the round-1 seed initialization sweep (`_find_best_feature_labels`) for the same underlying reason: composition features can produce spurious pseudo-positives when targets and decoys have different sequence distributions.

---

### 2. Raw Pearson colocalization inflated by tissue morphology (all decoy methods)

**Affects:** all modes when `compute_colocalization_features` or `compute_nmf_colocalization_features` is used without TIC masking, and even with TIC masking on this dataset.

**Why it arises.** Any MALDI ion image is ~0 in the unmeasured off-tissue padding and broadly tracks the tissue footprint on-tissue. The shared on/off-tissue structure dominates the raw Pearson r between any two ion images: mean within-protein r ~0.76–0.84, with mz_shuffle decoys at 0.778 — *higher* than all targets (0.736). TIC masking (`--coloc-tic-quantile`) restricts computation to on-tissue pixels and halves the absolute values (masked targets 0.279, decoys 0.301), but does not reverse the decoy ≥ target ordering. The underlying reason is that the mz_shuffle null is symmetric by design: relocated decoys land on real confirmed ion images, and every real image shares the same few tissue substructures, so any protein-level aggregation is equally coherent for targets and decoys. NMF substructure cosine shows the same pattern (targets 0.715, decoys 0.790).

**Mitigation.** Keep protein colocalization out of the default ranker feature set. TIC-masking code (`compute_tissue_mask` in `maldi_features.py`) is retained because it is strictly more correct than raw Pearson r and would matter on a dataset with genuine protein-specific spatial structure. The non-discriminativeness is specific to this amyloidosis data; do not assume it generalizes to all datasets.

---

### 3. `ppm_error` non-discriminative for `mz_shift` decoys in feature-list mode (by design)

**Affects:** `decoy_method="mz_shift"` in feature-list mode (no `--maldi-query-raw`).

**Why it arises.** In feature-list mode, the `ppm_error` for an mz_shift decoy is copied from the target peptide's best match ppm — the delta between the peptide's theoretical [M+H]+ and the feature it was matched to — rather than computed against the decoy's shifted (off-target) feature m/z. This makes `ppm_error` identical for a target and its mz_shift decoy, contributing zero discrimination signal.

**Why this is intentional.** Computing `ppm_error` against the decoy's snapped-to feature m/z (which could be many Da away) would artificially penalise decoys for a large ppm offset that does not correspond to any real ambiguity a false positive would have. Real false positives match within the ppm window by definition; the decoy's shifted anchor is not a mass-match, so penalising it on that delta would make the null anti-conservative.

**In raw-query mode** this limitation is removed: `_recompute_ppm_from_centroids` computes `ppm_error` from the observed peak centroid in each candidate's own extraction window, identically for targets and decoys. An mz_shift decoy shifted into empty m/z space receives the worst-case ppm (`extraction_ppm`) rather than a median-imputed value, so zero-signal decoys are penalised rather than treated as average-quality (see `pipeline.py:_recompute_ppm_from_centroids`).

---

### 4. CCS / `im2deep_*` features leak the m/z baseline for `mz_shuffle` decoys

**Affects:** `decoy_method="mz_shuffle"` when IM2Deep CCS features are active.

**Why it arises.** CCS is approximately proportional to m/z (Mason-Schamp relation; larger peptides drift slower). Under mz_shuffle, a decoy peptide is placed on a feature that belongs to a peptide of systematically different mass (the derangement enforces a large mass gap). The observed CCS at that feature reflects the feature peptide's mass, while the predicted CCS from IM2Deep reflects the decoy peptide's mass. The raw delta (`im2deep_delta_ccs`, `im2deep_abs_delta_ccs_pct`, `im2deep_ccs_zscore`, `im2deep_ccs_rank`) therefore encodes the mass gap, not a conformational mismatch — creating a trivial discriminator that does not exist for real isobaric false positives. Mobility-gated colocalization features (`isotope_colocalization_*_mob`, `adduct_colocalization_*_mob`) are also affected because they filter the shared co-located ion image through each candidate's own predicted 1/K0 window; the decoy's window misses the real feature's peak.

**Mitigation.** For `mz_shuffle`, the pipeline automatically excludes `_MZ_SHUFFLE_CCS_LEAK_FEATURES` (raw CCS scalars and mobility-gated colocalization) from the ranker and uses only the m/z-detrended residual features (`*_resid`). The residuals subtract the expected CCS difference due to the m/z gap (fitted as a power-law `CCS = A · mz^B` on calibration peptides), leaving only the conformational mismatch — which is exchangeable with a real isobaric false positive. A diagnostic log line reports `|corr(raw Δ, decoy_delta_da)|` vs `|corr(residual Δ, decoy_delta_da)|`; the residual should be near 0.

---

### 5. `protein_coverage` pinned at 1.0 for decoys (historical, now fixed)

**Affected versions:** before the symmetric numerator + true-digest denominator fix.

**Why it arose.** The original `protein_coverage = protein_n_features / protein_tryptic_count` leaked the target/decoy label: each decoy peptide is placed on exactly one feature by construction (`mz_shift`/`mz_shuffle`), so `protein_n_features` equalled the candidate count for every decoy protein, and since `protein_tryptic_count` was the candidate-pool count (not the full digest), coverage was pinned to exactly 1.0. Targets matching several near-isobaric features could exceed 1.0. The LDA therefore trivially separated targets from decoys on this feature.

**Current state.** Fixed: the numerator counts distinct observed *peptides* (symmetric, because target and DECOY_ protein share the same peptide set) and the denominator is the true full-tryptic-digest count per protein (from `peptide_db`, applied in `pipeline.py` just before Step 6). Coverage ∈ (0, 1], symmetric, and non-degenerate across proteins.

---

### 6. mz_shuffle target multiplicity creating a ~2:1 T:D imbalance (historical, now fixed)

**Affected versions:** before the target-dedup fix in `generate_mz_shuffle_candidates`.

**Why it arose.** A target peptide whose [M+H]+ m/z falls within `matching_ppm` of several MALDI features contributed one target row per feature, but the derangement assigned it only one decoy (one relocated feature). A 1,398-feature run produced a 5901:2895 imbalance (~2:1). The excess target rows had no co-located decoy, so the per-feature winner competition was effectively uncontested for those targets, inflating identifications without a valid null.

**Current state.** Fixed: targets are deduplicated to one representative row per unique peptide (lowest `|ppm_error|`) before the derangement is built and the combined frame is returned. No unique peptide identification is lost; only redundant near-isobaric secondary matches are dropped.

---

## Configuration

MSI-PICASSO uses `cascade_config` for hierarchical configuration. Priority (lowest to highest):

1. `MSI-PICASSO/package_data/config_default.json` — all defaults
2. User `--config-file` (JSON or TOML)
3. CLI arguments (explicit only; `None` values never override lower-priority sources)

The merged config is written to `<output_dir>/.full_config.json` at the start of every run for reproducibility.

**Explicit falsy values are honored.** `cascade_config`'s merge rule (`elif v or k not in original`) silently drops a falsy-but-explicit scalar (`0`, `0.0`, `""`) when the key already has a value, keeping the lower-priority default. `parse_configurations` corrects this: after the cascade merge it re-applies user-provided non-None values (`_apply_explicit_overrides`) and then re-validates against the schema with `jsonschema`. Consequence: an explicit `0` is respected instead of being silently masked by the default. In particular `--matching-ppm 0` is honored — it means exact matching / no collision tolerance (useful in raw-query, where the matching grid is the exact peptide masses and a tolerance only inflates the target-feature count via near-isobaric cross-matches). `matching_ppm`'s schema constraint is therefore `minimum: 0` (not `exclusiveMinimum`); negative values are still rejected, and `None` still means "unset" and never overrides. `extraction_ppm`/`ppm_bin` keep `exclusiveMinimum: 0` (a zero-width extraction window is degenerate).

### Adding a new configurable parameter

1. Add it to `package_data/config_default.json` with a default value.
2. Add a schema entry in `package_data/config_schema.json` (type, constraints; use `["type", "null"]` to allow CLI `None` passthrough).
3. In `cli.py`: add `--param-name` to the appropriate argument group with `default=None`; add the snake_case name to `_TOP_LEVEL_ATTRS` (and to `_STORE_TRUE_ATTRS` for boolean flags); add it to the `rescore()` call at the bottom of `main()`.
4. Add it to the `rescore()` signature in `pipeline.py` with the same default as the JSON.
5. Add a test in `tests/test_config_parser.py`.

### Feature composition via config

Instead of editing `feature_generator.py`, specify in a TOML config file:

```toml
[MSI-PICASSO]
features_preset = "all"          # "all" | "main"
features_exclude = ["peptide_length", "adduct_colocalization_chca"]
```

`MALDI_INTRINSIC_FEATURES` and `MAIN_FEATURES` in `feature_generator.py` remain the canonical full definitions. Exclusions are applied at runtime in `pipeline.py`.

### MALDI extraction parameters

Raw extraction parameters (`ppm_bin`, smoothing, deisotoping, etc.) live under `[MSI-PICASSO.maldi_extraction]` in TOML or `"maldi_extraction": {...}` in JSON. The default assumption is pre-extracted features (NPZ or m/z list); these parameters are only active when `--maldi-raw` or `--maldi-imzml` is used.

Example config file:

```toml
[MSI-PICASSO]
model = "lda"
decoy_method = "balanced_shuffle"
features_exclude = ["peptide_length"]

[MSI-PICASSO.maldi_extraction]
matching_ppm = 15.0
deisotope = true
deisotope_min_score = 12.0
```

```bash
picasso --config-file my_config.toml --maldi-d data/MALDI.d --fasta human.fasta
```

(The console-script is `picasso`; `msi-picasso` is an alias for the same entry point.)

---

## Reference pipeline commands (amyloidosis dataset)

**Raw-query (the direction being optimized toward default).** Candidates drive on-demand extraction from the raw `.d`; no pre-detected feature list. This is the configuration current development targets:

```bash
picasso \
  -f /home/robbe/MALDI_MSI_score/data/uniprot_human_reviewed.fasta \
  --maldi-d /home/robbe/MALDI_MSI_score/data/amyloidosis/Amy_TMA_MS1.d --maldi-query-raw \
  --lcms-peptides /home/robbe/MALDI_MSI_score/data/amyloidosis/fragpipe_output_amyloidosis/Amyl_tissue_psm.tsv \
  --lcms-id-format psm_utils \
  --decoy-method mz_shuffle \
  --model lda \
  --output-dir /home/robbe/MALDI_MSI_score/results/mz_shuffle/ \
  -v --im2deep-calibration linear \
  --debug-gt /home/robbe/MALDI_MSI_score/data/amyloidosis/GT_peptides.txt \
  --matching-ppm 0 --n-interaction-features 0
```

Notes for raw-query:
- `--decoy-method mz_shuffle`: co-located 1 target + 1 decoy per peptide (after the target-dedup fix), the cleanest symmetric null for raw-query. Do **not** combine with `--match-ccs` (it would remove ~all decoys by design). `mz_shift` and `entrapment` are the other raw-query-appropriate methods.
- `--matching-ppm 0`: in raw-query the grid *is* the exact peptide masses, so 0-ppm = one feature per peptide (explicit 0 is honored, see config notes).
- Optional: `--use-spatial-ranker-features`, `--use-protein-level-feats`, `--coloc-tic-quantile`, `--nmf-coloc`, `--mob-coloc` (CCS available in raw-query).

**Feature-list baseline (legacy).** Uses a pre-detected feature list; kept for comparison until raw-query is fast/robust enough to default:

```bash
picasso \
  -f .../uniprot_human_reviewed.fasta \
  --maldi-raw .../Amy_TMA_MS1.d \
  --feature-mzs .../ff_with_new_algo._amyloidosies_Lme48.csv \
  --lcms-peptides .../Amyl_tissue_psm.tsv --lcms-id-format psm_utils \
  --model lda --decoy-method balanced_shuffle \
  -v --im2deep-calibration linear --n-interaction-features 0 \
  --output-dir .../results/new_algo_lda/
```

Key parameter choices (both modes):
- `--model lda`: LDA default; no extra dependencies, fast cross-validated pseudo-label iteration.
- `--decoy-method`: `mz_shuffle` for raw-query; `balanced_shuffle` for the feature-list baseline (retains only shuffled peptides matching a feature, preventing decoy starvation on sparse lists).
- `--n-interaction-features 0`: polynomial interaction expansion disabled (default 5 was found to hurt performance).
- `--im2deep-calibration linear`: IM2Deep CCS calibration mode.
- `--debug-gt`: ground truth peptide list for diagnostic FDR plots (not used in scoring).
