# MSI-PICASSO — CLAUDE.md

**Scope of this document:** what the code *is* and what it must *never* do.
It contains no experimental results. Everything about what has been tried, what it
produced, and what to try next lives in **`/home/robbe/MALDI_MSI_score/PROGRESS.md`** —
read that first for project state. Keeping results out of here is deliberate: the previous
version of this file mixed findings from superseded decoy methods into the reference
material, and those findings were then applied to configurations where they did not hold.

Superseded material is archived in `/home/robbe/MALDI_MSI_score/docs/archive/`
(`CLAUDE_pre_3008.md`, `ARCHITECTURE.md`, `AUDIT.md`, `FIX_KIDNEY_DISCREPANCY.md`).

---

## Purpose

`MSI-PICASSO` is a symmetric target-decoy rescoring package for MALDI-MSI MS1 data. It
takes MALDI data (a raw Bruker `.d`) plus an LC-MS/MS PSM table and produces FDR-controlled
MALDI peptide identifications by semi-supervised rescoring.

**Goal:** raise the number of confident IDs on the ground-truth datasets — amyloidosis,
her2, kidney (all TIMS, in-situ iprm-PASEF MS/MS ground truth), and `SC_Heeren` (no ion
mobility). Nothing is fixed: decoy method, scoring backend, and feature set are all in play.

**Why it was built this way:** the prior approach (ms2rescore "Approach B") used
ProteomeDiscoverer output for candidates and features, which leaked the label — `lcms_xcorr`,
a search-engine score, had AUC 0.993 against the PD target/decoy label. This package removes
all PD-derived scores; every feature is computed from the raw `.d` (and optionally raw
LC-MS/MS) under the symmetry invariant below.

---

## Critical invariants

Violating any of these silently corrupts the FDR. They are listed first because they are
the only part of this file that is genuinely load-bearing.

1. **No feature computation function takes `is_decoy`.** Targets and decoys receive features
   through identical code paths. This is enforced at the API level — grep for `is_decoy` in
   `maldi_features.py`, `feature_generator.py`, `lcms_evidence.py`, `utils.py` and
   `maldi_query.py` returns nothing.
2. **A decoy row's `feature_mz` must be the decoy's own m/z** — for `substitution` the
   substituted peptide's [M+H]+, for `mz_shift` the shifted m/z. Never the source target's.
   Load-bearing for raw-query mode, which extracts each candidate's ion image at its own
   anchor.
3. **Composition features stay out of `_BEST_FEAT_SKIP` and out of the seed.** Several decoy
   methods alter elemental composition, so composition features separate target from decoy
   as an artifact of decoy construction. Seeding on them makes the FDR anti-conservative.
4. **`is_decoy` must be cast to `bool` dtype** before returning a candidates frame.
   `pd.concat` with an empty frame yields `object`-dtype booleans, which break
   `~df["is_decoy"]` indexing downstream.
5. **Explicit falsy config values are honored** via `_apply_explicit_overrides` in
   `config_parser.py`. `--matching-ppm 0` means exact matching and is not masked by the
   default of 20.0. `cascade_config`'s own merge rule would drop it.
6. **`maturin develop` must target the venv explicitly** (see Build below), or the extension
   installs into the base pyenv Python and the package silently falls back to slow Python
   paths.
7. **`has_oxidized_met` is really `has_methionine`** (`maldi_features.py:1005`). Plain-sequence
   candidates never carry `M[Oxidation]` annotations, so the literal definition is
   unreachable. Rename it or wire up modification detection; do not trust the name.

---

## Repository layout

The git repo is `MSI-PICASSO/` only. The parent `/home/robbe/MALDI_MSI_score/` is **not**
under version control, so `configs/`, `results/`, `scripts/`, `data/` and `docs/` are
untracked by design. Run provenance is instead recovered from the `.full_config.json` that
`cli.py` writes into every output directory.

```
MSI-PICASSO/
├── pyproject.toml              # testpaths = ["msi_picasso/tests"]
├── CLAUDE.md                   # this file
├── msi_picasso/
│   ├── cli.py             2030 # argparse CLI (`picasso` / `msi-picasso`), MALDI input dispatch
│   ├── config_parser.py    144 # cascade_config merge + jsonschema validation
│   ├── candidates.py      1987 # FASTA digest, all decoy generators, match_to_maldi_features
│   ├── lcms_ids.py         722 # parse LC-MS/MS IDs -> identified proteins + peptides
│   ├── lcms_evidence.py    965 # raw LC-MS/MS evidence features (MS2PIP, DeepLC-anchored MS1)
│   ├── maldi_extraction.py 1237# raw Bruker .d extraction via imzy: ion images, spatial feats
│   ├── maldi_query.py      391 # raw-query mode + observed centroids/CCS (alphatims) + disk cache
│   ├── maldi_imzml.py     1174 # SCiLS-style interval extraction from imzML via pyimzml (legacy)
│   ├── maldi_features.py  2657 # all MALDI-side features: mass accuracy, colocalization, isotope
│   ├── feature_generator.py 591# feature-group constants + compute_all_features
│   ├── pipeline.py        3324 # rescore() orchestrator, scoring backends, TDC q-values, PEP
│   ├── debug_viz.py       4049 # debug figures (--verbose)
│   ├── utils.py             99 # shared math (brainpy isotopes, spectral angle, mass constants)
│   ├── package_data/           # config_default.json + config_schema.json
│   └── tests/                  # 50 test modules
└── msi-picasso-rs/             # PyO3 + rayon extension, crate "MSI-PICASSO-rs"
    └── src/{lib,digest,features,ion_image,isotope,maldi_isotope,mob_coloc,spectral,xic}.rs
```

**The Rust crate imports as `ms1rescore_rs`** — the module name was never renamed with the
package. Every call site wraps it in `try/except ImportError` with a Python fallback.

---

## Pipeline flow

1. **Entry** — `cli.main()` (`cli.py:1458`) → `parse_configurations(...)["MSI-PICASSO"]`.
2. **Config cascade** (`config_parser.py:82`) — `package_data/config_default.json` → user
   JSON/TOML → explicit CLI args. `store_true` flags are converted `False → None`
   (`cli.py:1479`) so argparse defaults do not clobber config-file values.
3. **MALDI input dispatch** (`cli.py:1613`), mutually exclusive:
   `maldi_npz | maldi_mzs | maldi_raw | maldi_imzml | maldi_d`.
   With `maldi_d` + `maldi_query_raw=true` extraction is **deferred** into `rescore()`.
   This is what every current config uses.
4. **`rescore()`** (`pipeline.py:1555`):
   - Step 1 candidate generation; 1b extra FASTA; 1c decoy generation (`pipeline.py:2093`)
   - Raw-query extraction (`pipeline.py:2277`) — `query_raw_maldi()` +
     `extract_observed_feature_stats_raw()`, then symmetric `ppm_error` recomputation from
     the observed centroids
   - Calibration-peptide selection for DeepLC / IM2Deep finetuning (`pipeline.py:2412`)
   - Steps 2–5 LC-MS/MS branch, skipped when no mzML or `lcms_prior_weight == 0`
   - Step 6 `compute_all_features()`; then optional `drop_zero_signal`, CCS filter,
     mobility-filtered colocalization, spatial-ranker features
   - Step 8 build `PSMList`; Step 9 rescoring, winner selection, TDC q-values, PEP
5. **Output** — `_write_results()` (`cli.py:220`), debug figures via `save_debug_figures()`.

### Two-pass scoring

Round 1 scores all candidates → per-feature winner selection (`_select_feature_winners`)
keeps the top-scoring candidate per MALDI m/z → Round 2 retrains on winners only → TDC
q-values over winners.

`--single-round` skips Round 2 only. **Winner selection still runs**, so the
target-vs-decoy competition that defines the TDC population is unchanged and the FDR
semantics are identical; only the final discriminant refit is dropped.

---

## Reading MALDI data

Two readers exist. Only the first is on the active path.

**imzy (active).** `maldi_extraction.py:200` — `imzy.get_reader(d_path)` dispatches Bruker
`.d` (TDF/TSF, bundled `libtimsdata.so`) and `.imzML`. `extract_maldi_data()`
(`maldi_extraction.py:865`) is the single public entry, returning
`(feature_mzs, ion_images, extra_ion_images, spatial_df, maldi_envelopes)`.

**pyimzml (legacy).** `maldi_imzml.extract_scils_features()`, reached only via
`--maldi-imzml`. SCiLS-style interval extraction. No current config uses it.

Performance-relevant details:

- **`_extract_centroid_fast`** (`maldi_extraction.py:193`) — TSF only. Pre-converts feature
  m/z windows to raw spectral index windows once from the reference frame's calibration,
  then makes one DLL call per pixel. Accepts <5 ppm systematic calibration error. Without
  it, imzy's two-calls-per-pixel pattern is ~3 hours for 49 K pixels on a network filesystem.
- **`_extract_profile_fast_multi`** (`maldi_extraction.py:351`) — the default in-RAM path.
  Extracts all six feature sets (main, M+1, M+2, Na, K, CHCA) in a **single**
  `spectra_iter()` pass, buffering 512 pixels at a time into the Rust
  `accumulate_profile_chunk` (rayon).
- **imzy writes its own caches**: an `.icache` npz beside each imzML holding per-spectrum
  `.ibd` byte offsets and coordinates (so opening a 133 MB imzML is an npz load, not an XML
  parse), and a `.icache/frame_index_cache.npz` inside each Bruker `.d`.
- **RAM vs memmap** — by default the full `(n_features, H, W)` float32 array lives in RAM.
  Passing `images_path` switches to a `np.memmap` written in `image_batch_size` batches.

### The `.d` is opened more than once

`imzy` exposes neither per-peak centroid m/z nor mobility, so raw-query mode opens the `.d`
a second time with `alphatims` (`maldi_query.py:197`) for observed peak centroids, observed
CCS, and the mobility peak-quality descriptors. Mobility colocalization
(`maldi_features.py`) streams the TDF a third time.

**This second pass dominates runtime** (roughly 40 minutes of a ~69 minute amyloidosis run,
streaming ~4e9 raw peaks) while producing only a handful of arrays of `len(query_mzs)`.
Two caches exist for it:

- **`raw_query_cache`** (`pipeline.py:1566`, logic at `2292`) — an in-process dict covering
  the *whole* extraction (`maldi_mzs`, `ion_images`, `extra_ion_images`, `spatial_features`,
  `maldi_envelopes`, `ccs_arr`, `centroid_arr`, `peak_quality`). Pass `None` to always
  extract, `{}` to extract once and populate, or a populated dict to reuse without touching
  the `.d`. Used by `scripts/grid_search.py`. Dies with the process.
- **`--raw-query-cache-dir`** (`raw_query_cache_dir`) — persists just the alphatims stats to
  an `.npz` keyed by a SHA-256 of the `.d` path, the **full** query m/z grid, and the window
  parameters. Survives across processes. Hashing the whole grid means a changed candidate
  set (different decoy method, digest, or FASTA) misses rather than silently reusing stale
  statistics. This is the cache to use for the experiment cycle; the ion images are only
  ~5 minutes of work but several GB of array, so they are deliberately *not* persisted.

Both are valid for exactly one reason: the candidate m/z grid is fixed by the digest plus
the decoy method, so it is constant across runs that vary only scoring parameters.

---

## Decoy generation

`decoy_method` selects the Step-1c generator. All five are supported and all remain
candidates for improving results. **This section describes capability only** — for how each
one has actually performed, see PROGRESS.md §4, where every finding carries the
configuration it was established under.

| method | what it does | preserves | notes |
|---|---|---|---|
| `substitution` | substitutes `substitution_n_residues` interior non-K/R residues, one decoy per unique target | length, cleavage sites | **changes elemental composition** — see invariant 3. Mass shift ~1–50 Da, so CCS features stay usable and it is compatible with `--match-ccs`. |
| `mz_shift` | shifts the query m/z by a random delta in `[delta_min, delta_max]` Da | sequence exactly | in raw-query, snapping is disabled so each decoy sits at its exact shifted m/z on a distinct feature |
| `mz_shuffle` | derangement of the peptide→feature assignment (mass-sorted rotation) | sequence exactly | decoys are **co-located** with targets on identical ion images, so feature-quality features are exactly symmetric. **Do not combine with `--match-ccs`** — it would remove ~all decoys by design. Raw CCS scalars and mobility-gated colocalizations are auto-excluded (`_MZ_SHUFFLE_CCS_LEAK_FEATURES`); only `*_resid` variants are kept. |
| `entrapment` | tryptic peptides from a foreign-organism FASTA (`entrapment_fasta`), isobaric-with-target ones filtered out | — | `protein="ENTRAPMENT_{acc}"` |
| `balanced_shuffle` / `paired_shuffle` | iterative K/R-preserving protein shuffle, keeping only decoys that match a MALDI feature | cleavage sites | achieves ~1:1 T:D on sparse feature lists. **Not compatible with `use_spatial_ranker_features`** (no consistent spatial anchor). |

Every method places decoys in a **separate protein namespace** (`DECOY_…` / `ENTRAPMENT_…`)
so protein-level features are computed within class and a decoy is never pooled with its
source target's protein.

`_SPATIAL_RANKER_OK_DECOYS` (`pipeline.py:53`) = `{entrapment, mz_shift, mz_shuffle,
substitution}`. With any other method `use_spatial_ranker_features` is force-disabled with a
`UserWarning`.

---

## Scoring backends

`--model` accepts **`{lda, qda, svm, gbt, rbf_svm}`** (`cli.py:819`).

| model | estimator | notes |
|---|---|---|
| `lda` | `LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")` | package default; importances are `coef_[0]` |
| `svm` | `sklearn.svm.LinearSVC` | shares `_rescore_linear` with `lda`; adds no dependency |
| `rbf_svm` | `sklearn.svm.SVC(kernel="rbf")` | nonlinear; no `coef_`, so importances are reported as \|structure coefficient\|. `rbf_svm_gamma` accepts `"scale"`/`"auto"` or a float. Training is O(N²). |
| `qda` | `QuadraticDiscriminantAnalysis(reg_param=0.1)` | reuses R1 posteriors for PEP under `--single-round` |
| `gbt` | gradient-boosted trees (`_rescore_gbt`, `pipeline.py:991`) | `gbt_n_estimators`, `gbt_max_depth`, `gbt_learning_rate`; never benchmarked on the three datasets |

All backends share the same semi-supervised loop: seed → pseudo-label iteration → winner
selection → TDC.

**Out-of-fold scoring.** `_cv_semisup_scores` (default `cv_folds=3`, stratified by
`is_decoy`) scores every candidate with a model trained on the other folds. Feature
importances come from a model fit on everything, but the FDR scores are strictly
out-of-fold.

**Round-1 seed** — `_find_best_feature_labels` (`pipeline.py:456`) sweeps each feature and
both ranking directions, counting targets at q ≤ `train_fdr`. Sub-ULP random noise breaks
ties so row order cannot bias the result. If the best single feature yields fewer than
`min_seed_positives` targets it escalates to pairwise sums/differences on standardised
columns, then to a depth-3 `DecisionTreeClassifier`. Columns in `_BEST_FEAT_SKIP`
(composition and ionisation features) are excluded throughout — invariant 3.

Fallback chain when that yields nothing: `ppm_error_abs < init_ppm_threshold` OR
`n_candidates == 1`; then the top `r1_seed_percentile` of targets by `ppm_error_abs`.

**Post-scoring reweighting** (winners only) is an **additive log-prior**, not multiplicative:

```
reweighted_score = round2_score
                 + lcms_prior_weight   * log(lcms_prior)
                 + spatial_prior_weight * log(spatial_prior)
```

Multiplicative combination would invert the ranking for negative scores.

---

## Feature groups

Defined in `feature_generator.py`; import as `from msi_picasso.feature_generator import ...`.

| constant | line | in the ranker? |
|---|---|---|
| `MALDI_INTRINSIC_FEATURES` | 53 | yes, by default |
| `PROTEIN_LEVEL_FEATURES` | 103 | opt-in `--use-protein-level-feats` |
| `REGION_COLOCALIZATION_FEATURES` | 133 | opt-in `--region-coloc` |
| `WITHIN_REGION_COLOCALIZATION_FEATURES` | 144 | opt-in `--within-region-coloc` (experimental) |
| `LCMS_PRIOR_FEATURES` | 188 | no — applied as an additive log-prior |
| `SPATIAL_RANKER_FEATURES` | 210 | opt-in `--use-spatial-ranker-features` |
| `MOB_QUALITY_FEATURES` | 230 | gated by `_MOB_QUALITY_DEFAULT_DECOYS` (`pipeline.py:62`) |
| `MAIN_FEATURES` | 243 | subset selector |
| `FEATURE_NAN_FILL` | 286 | per-feature NaN sentinels |

`MOB_QUALITY_FEATURES` = `mob_2d_concentration`, `mob_k0_spread`, `mob_mz_spread_ppm`,
`mob_peak_snr` — intrinsic joint (m/z, intensity, 1/K0) peak-quality descriptors, available
only when the dataset has a TIMS dimension.

**Why `LCMS_PRIOR_FEATURES` are excluded from the ranker:** LC-MS/MS ID-derived features
(`lcms_q_value`, `lcms_pep`, `lcms_score`, `n_psms`, `lcms_intensity`) would give confirmed
targets different treatment from decoys, breaking TDC symmetry. They are populated on the
candidates frame by Strategy C but never enter the prior either.

**Why `PROTEIN_LEVEL_FEATURES` are opt-in:** they aggregate over all candidates sharing a
protein, and are only valid because decoys occupy a separate protein namespace. Even so they
can interact subtly with the decoy model.

Compose the active set via config rather than by editing the module:

```toml
[MSI-PICASSO]
features_preset  = "all"          # "all" | "main"
features_exclude = ["peptide_length", "adduct_colocalization_chca"]
```

---

## Candidate generation

**Strategy C (current)** — activated by passing `lcms_peptides_path`. Candidates are the
in-silico digest of identified proteins ∪ the directly identified LC-MS/MS peptides. The
`source` column records the origin: `protein_digest`, `lcms_confirmed`, or `decoy`.
Without `--digest` (no FASTA) all confirmed peptides are novel, so decoys are built from a
**concatenated pseudo-protein** — all target sequences concatenated, shuffled once, then
re-digested — because a per-peptide shuffle would give decoys identical elemental
composition and make isotope features non-discriminative.

**Strategy A (legacy)** — `digest_fasta()` over a whole FASTA with K/R-preserving shuffled
decoys. Used when `rescore()` gets no `lcms_peptides_path`. Add `--digest` to combine it
with Strategy C.

`protein_coverage` counts distinct observed *peptides* over the **true full tryptic digest
count** (overridden in `pipeline.py` from `peptide_db` before Step 6). Both halves matter:
the earlier `protein_n_features / candidate_pool_count` form pinned every decoy protein to
exactly 1.0 and leaked the label.

### LC-MS/MS ID formats

`lcms_id_format` ∈ `percolator` (default), `mzidentml`, `psm_utils`, `msf`, `ms2rescore`.
With `psm_utils`, `psm_utils_reader` picks the concrete reader (`fragpipe`,
`proteome_discoverer`, `tsv`, …). Accessions are normalised (`sp|P12345|GENE_HUMAN` →
`P12345`) before comparison with the FASTA; a <50% match rate warns of a database mismatch.
FragPipe reports `Retention` in **seconds** — `_parse_psm_utils` divides by 60 when the
median exceeds 200.

---

## Configuration

Priority, lowest to highest: `package_data/config_default.json` → `--config-file`
(JSON/TOML) → explicit CLI arguments (`None` never overrides). The merged config is written
to `<output_dir>/.full_config.json` at the start of every run — **this is the run provenance
record**, and `scripts/scoreboard.py --diff` reads it.

The TOML table name is `[MSI-PICASSO]`, with `[MSI-PICASSO.maldi_extraction]` and
`[MSI-PICASSO.im2deep]` subtables.

Package defaults worth knowing, because the checked-in configs override all of them:

| key | package default | baseline configs use |
|---|---|---|
| `model` | `lda` | `rbf_svm` |
| `decoy_method` | `balanced_shuffle` (`rescore()` signature says `shuffle`) | `substitution` |
| `substitution_n_residues` | `1` | `2` |
| `train_fdr` | `0.05` | `0.1` (amyloidosis) / `0.3` (her2, kidney) |
| `init_ppm_threshold` | `2.0` (`rescore()` signature says `5.0`) | `10.0` / `5.0` |
| `min_seed_positives` | `50` | `125` / `20` |
| `matching_ppm` | `20.0` | `0` (exact; see invariant 5) |

### Adding a configurable parameter

1. `package_data/config_default.json` — add the key with its default.
2. `package_data/config_schema.json` — add the type (use `["type", "null"]` to allow a CLI
   `None` passthrough). The schema rejects unknown keys, so this step is mandatory.
3. `cli.py` — add `--param-name` with `default=None`; add the snake_case name to
   `_TOP_LEVEL_ATTRS` (and `_STORE_TRUE_ATTRS` for boolean flags); pass it in the `rescore()`
   call at the bottom of `main()`.
4. `pipeline.py` — add it to the `rescore()` signature with the same default.
5. Add a test in `tests/test_config_parser.py`.

---

## Environment, build, tests

```bash
# interpreter — the bare pyenv python does NOT have the dependencies
/home/robbe/.pyenv/versions/MSIscore/bin/python

pip install -e MSI-PICASSO/            # or "MSI-PICASSO/[timstof]" for Bruker .d support
```

Sibling checkouts `ms2rescore/`, `ms2pip/`, `psm_utils/`, `IM2Deep/`, `ms2rescore-rs/` are
upstream tools on custom branches, all installed editable into the same env.

**Rust extension** (invariant 6):

```bash
cd MSI-PICASSO/msi-picasso-rs
VIRTUAL_ENV=/home/robbe/.pyenv/versions/3.11.11/envs/MSIscore \
  /home/robbe/.pyenv/versions/3.11.11/envs/MSIscore/bin/maturin develop --release
```

`target/` reaches 1–2 GB; `rm -rf` it if disk is tight, it rebuilds.

**Tests** — `pytest` from `MSI-PICASSO/`, `testpaths = ["msi_picasso/tests"]`.
The suite currently has known failures; see PROGRESS.md §7 before treating a red run as a
regression.

---

## Scripts

In `/home/robbe/MALDI_MSI_score/scripts/`:

| script | purpose |
|---|---|
| `scoreboard.py` | scrape all `results/*/*/run.log` into one comparison table; `--markdown` for a PROGRESS.md row, `--diff A B` for a settings diff between two runs |
| `validate_results.py` | biological validation: marker recovery, GT recovery, LC-MS concordance |
| `diagnose_gt.py` | per-GT-peptide failure diagnosis against a results dir |
| `grid_search.py` / `analyze_grid_search.py` | parameter sweep (reuses `raw_query_cache`) and its sensitivity analysis |
| `ablation_svm.py` / `ablation_lda.py` | feature ablation |
| `audit_coloc_leak.py` | per-colocalization-column target/decoy AUC and abundance-leak check — run before promoting a coloc feature into the ranker |
| `envelope_qc.py` | isotope-envelope QC |
| `visualize_ms1rescore_features.py` | per-feature, per-candidate target/decoy visualisation |

One-off analyses are parked in `scripts/archive/`.

---

## Running

```bash
picasso -c /home/robbe/MALDI_MSI_score/configs/amyloidosis_substitution.toml
picasso -c /home/robbe/MALDI_MSI_score/configs/her2_test.toml
picasso -c /home/robbe/MALDI_MSI_score/configs/kidney_test.toml
```

For an experiment, override the output directory and reuse the extraction cache:

```bash
picasso -c configs/kidney_test.toml \
        --output-dir results/kidney/E007/ \
        --raw-query-cache-dir .rawquery_cache
```

Convention: one experiment = one ID shared across branch (`exp/E007-<slug>`), config
(`configs/kidney_E007.toml`), results dir (`results/kidney/E007/`), and PROGRESS.md entry.
