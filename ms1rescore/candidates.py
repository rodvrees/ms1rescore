"""FASTA digest, decoy generation, and MALDI m/z matching."""

import logging
import random

import numpy as np
import pandas as pd
from pyteomics import fasta, mass, parser

from ms1rescore.utils import PROTON

logger = logging.getLogger(__name__)


def _shuffle_protein(seq: str, random_state: int = 42) -> str:
    """
    Shuffle non-K/R residues of a protein sequence randomly while keeping
    K and R at their original positions.

    Keeping K/R in place ensures the decoy protein is digested at the same
    tryptic cleavage sites as the target, preserving peptide length and charge
    distributions. Shuffling (rather than reversing) the non-K/R residues
    gives decoy peptides different amino acid compositions from targets of the
    same mass/length, making isotope envelope features (theo_isotope_cosine,
    theo_isotope_chi2, theo_isotope_kl) genuinely discriminative. K/R-fixed
    reversal creates isobaric peptides with near-identical isotope patterns.
    """
    kr_positions = {i for i in range(len(seq)) if seq[i] in "KR"}
    non_kr = [seq[i] for i in range(len(seq)) if seq[i] not in "KR"]
    rng = random.Random(random_state)
    rng.shuffle(non_kr)
    result = list(seq)
    j = 0
    for i in range(len(result)):
        if i not in kr_positions:
            result[i] = non_kr[j]
            j += 1
    return "".join(result)


def digest_fasta(
    fasta_path: str,
    enzyme: str = "trypsin",
    missed_cleavages: int = 1,
    min_length: int = 7,
    max_length: int = 30,
    generate_decoys: bool = True,
) -> pd.DataFrame:
    """
    In-silico tryptic digest of a FASTA file with decoy generation.

    Decoy strategy: reverse each protein keeping K/R at original positions,
    then digest. Produces 1:1 paired decoys with identical mass distribution.

    Returns DataFrame with columns:
        peptide, protein, mass, mh_mz, is_decoy, n_C, n_H, n_N, n_O, n_S
    """

    # Phase 1: Cleave all proteins (pyteomics), collect sequences
    rows = []  # (peptide, protein, is_decoy)
    for desc, seq in fasta.read(fasta_path):
        protein_id = desc.split("|")[1] if "|" in desc else desc.split()[0]
        cleaved = parser.cleave(
            seq,
            parser.expasy_rules.get(enzyme, enzyme),
            missed_cleavages=missed_cleavages,
        )
        for pep in cleaved:
            if min_length <= len(pep) <= max_length:
                rows.append((pep, protein_id, False))

        if generate_decoys:
            decoy_seq = _shuffle_protein(seq)
            cleaved_d = parser.cleave(
                decoy_seq,
                parser.expasy_rules.get(enzyme, enzyme),
                missed_cleavages=missed_cleavages,
            )
            for pep in cleaved_d:
                if min_length <= len(pep) <= max_length:
                    rows.append((pep, f"DECOY_{protein_id}", True))

    df = pd.DataFrame(rows, columns=["peptide", "protein", "is_decoy"])
    df = df.drop_duplicates(subset=["peptide", "is_decoy"])

    # Remove decoys whose sequence is identical to a target (arises when all
    # non-K/R residues in a peptide are identical or there is only one — the
    # K/R-preserving shuffle is then a no-op for that peptide).
    target_seqs = set(df.loc[~df["is_decoy"], "peptide"])
    n_before = df["is_decoy"].sum()
    df = df[~(df["is_decoy"] & df["peptide"].isin(target_seqs))].reset_index(drop=True)
    n_removed = n_before - df["is_decoy"].sum()
    if n_removed > 0:
        logger.debug("  Removed %d decoy sequences identical to a target peptide", n_removed)

    # Phase 2: Compute masses + elemental composition (Rust if available, else pyteomics)
    sequences = df["peptide"].tolist()
    try:
        from ms1rescore_rs import compute_peptide_masses

        masses, mh_mzs, n_cs, n_hs, n_ns, n_os, n_ss = compute_peptide_masses(sequences)
        df["mass"] = masses
        df["mh_mz"] = mh_mzs
        df["n_C"] = n_cs
        df["n_H"] = n_hs
        df["n_N"] = n_ns
        df["n_O"] = n_os
        df["n_S"] = n_ss
        logger.info("  (used Rust backend for mass computation)")
    except ImportError:
        logger.info("  (using pyteomics for mass computation)")
        masses_list = []
        for seq in sequences:
            try:
                comp = mass.Composition(sequence=seq)
                pep_mass = mass.calculate_mass(composition=comp)
                masses_list.append({
                    "mass": pep_mass, "mh_mz": pep_mass + PROTON,
                    "n_C": comp.get("C", 0), "n_H": comp.get("H", 0),
                    "n_N": comp.get("N", 0), "n_O": comp.get("O", 0),
                    "n_S": comp.get("S", 0),
                })
            except Exception:
                masses_list.append({
                    "mass": 0, "mh_mz": 0, "n_C": 0, "n_H": 0,
                    "n_N": 0, "n_O": 0, "n_S": 0,
                })
        mass_df = pd.DataFrame(masses_list)
        for col in mass_df.columns:
            df[col] = mass_df[col].values

    # Remove peptides with unknown amino acids (mass=0)
    df = df[df["mass"] > 0].reset_index(drop=True)
    df["is_decoy"] = df["is_decoy"].astype(bool)
    logger.info(
        f"Digested {fasta_path}: {(~df['is_decoy']).sum()} target, "
        f"{df['is_decoy'].sum()} decoy peptides"
    )
    return df


def match_to_maldi_features(
    maldi_mzs: np.ndarray,
    peptide_db: pd.DataFrame,
    ppm_tolerance: float = 20.0,
    maldi_intensities: np.ndarray | None = None,
    maldi_intensities_p90: np.ndarray | None = None,
    maldi_intensities_sum: np.ndarray | None = None,
) -> pd.DataFrame:
    """
    Match MALDI m/z features to digest candidates within ppm tolerance.

    Uses Rust (ms1rescore_rs) if available for the m/z matching step.

    Parameters
    ----------
    maldi_intensities
        Per-feature intensity array aligned with ``maldi_mzs``.  Prefer
        passing ``maldi_intensities_p90`` (90th-percentile of nonzero pixels)
        rather than mean-of-nonzero, as p90 decouples intensity magnitude from
        spatial coverage.  If only this argument is supplied it is used for
        ``feature_intensity`` (backwards compatibility).
    maldi_intensities_p90
        90th-percentile intensity of nonzero pixels per feature.  Robust
        estimate of peak intensity that is not confounded by spatial coverage
        (``fraction_detected`` handles that separately).  Preferred over
        mean-of-nonzero.  Computed by ``compute_spatial_features`` as
        ``intensity_p90``.
    maldi_intensities_sum
        Sum of nonzero pixel intensities per feature.  Computed by
        ``compute_spatial_features`` as ``intensity_sum``.

    Returns candidate table with columns from peptide_db plus:
        feature_mz, feature_idx, ppm_error, ppm_error_abs,
        protein_n_features, n_candidates
    """
    peptide_mzs = peptide_db["mh_mz"].values

    def _assign_intensities(df: pd.DataFrame, idx) -> None:
        if maldi_intensities_p90 is not None:
            df["feature_intensity_p90"] = maldi_intensities_p90[idx]
        if maldi_intensities_sum is not None:
            df["feature_intensity_sum"] = maldi_intensities_sum[idx]
        if maldi_intensities is not None:
            df["feature_intensity"] = maldi_intensities[idx]

    try:
        from ms1rescore_rs import match_mz

        feat_idx, pep_idx, ppm_errors = match_mz(
            maldi_mzs.tolist(), peptide_mzs.tolist(), ppm_tolerance
        )
        if len(feat_idx) == 0:
            logger.warning("No candidates matched any MALDI features")
            return pd.DataFrame()

        feat_idx = np.array(feat_idx, dtype=np.int64)
        pep_idx = np.array(pep_idx, dtype=np.int64)
        ppm_errors = np.array(ppm_errors)

        result = peptide_db.iloc[pep_idx].copy()
        result = result.reset_index(drop=True)
        result["feature_mz"] = maldi_mzs[feat_idx]
        result["feature_idx"] = feat_idx
        result["ppm_error"] = ppm_errors
        result["ppm_error_abs"] = np.abs(ppm_errors)
        _assign_intensities(result, feat_idx)
        logger.info("  (used Rust backend for m/z matching)")

    except ImportError:
        matches = []
        db_mz = peptide_mzs
        sort_idx = np.argsort(db_mz)
        db_mz_sorted = db_mz[sort_idx]

        for i, mz in enumerate(maldi_mzs):
            tol = mz * ppm_tolerance / 1e6
            lo = np.searchsorted(db_mz_sorted, mz - tol, side="left")
            hi = np.searchsorted(db_mz_sorted, mz + tol, side="right")
            if lo >= hi:
                continue
            candidate_idx = sort_idx[lo:hi]
            candidates = peptide_db.iloc[candidate_idx].copy()
            candidates["feature_mz"] = mz
            candidates["feature_idx"] = i
            candidates["ppm_error"] = (mz - candidates["mh_mz"]) / candidates["mh_mz"] * 1e6
            candidates["ppm_error_abs"] = candidates["ppm_error"].abs()
            _assign_intensities(candidates, i)
            matches.append(candidates)

        if not matches:
            logger.warning("No candidates matched any MALDI features")
            return pd.DataFrame()
        result = pd.concat(matches, ignore_index=True)

    # Protein-level consistency: count distinct MALDI features per protein
    # Computed over ALL candidates (targets + decoys) — symmetric
    protein_feature_count = result.groupby("protein")["feature_mz"].nunique()
    result["protein_n_features"] = result["protein"].map(protein_feature_count).fillna(0).astype(int)

    # Full tryptic digest count per protein (from peptide_db, before m/z filtering).
    # Used by compute_protein_consistency_features to compute protein_coverage correctly.
    protein_tryptic_counts = peptide_db.groupby("protein")["peptide"].nunique()
    result["protein_tryptic_count"] = result["protein"].map(protein_tryptic_counts).fillna(0).astype(int)

    # Candidates per feature
    result["n_candidates"] = result.groupby("feature_mz")["feature_mz"].transform("count")

    # A10 — Kendrick mass defect (CH₂ reference unit: 14 / 14.01565).
    # KMD = KM − round(KM) ∈ [-0.5, 0.5).
    kendrick_mass = result["feature_mz"] * (14.0 / 14.01565)
    result["kendrick_mass_defect"] = kendrick_mass - np.round(kendrick_mass)

    logger.info(
        f"Matched {result['feature_mz'].nunique()}/{len(maldi_mzs)} features → "
        f"{(~result['is_decoy']).sum()} target + {result['is_decoy'].sum()} decoy candidates"
    )
    return result


def generate_mz_shift_candidates(
    target_df: pd.DataFrame,
    feature_mzs: np.ndarray,
    matching_ppm: float = 20.0,
    delta_min: float = 5.0,
    delta_max: float = 20.0,
    snap_tolerance_ppm: float = 50.0,
    random_state: int = 42,
    maldi_intensities: np.ndarray | None = None,
    maldi_intensities_p90: np.ndarray | None = None,
    maldi_intensities_sum: np.ndarray | None = None,
) -> pd.DataFrame:
    """
    Generate m/z-shifted observation-space decoys and return a combined
    target + decoy candidates DataFrame.

    For each unique target peptide a random delta in [delta_min, delta_max] Da
    is sampled; sign alternates (even index → +, odd index → −).  The shifted
    query is snapped to the nearest MALDI feature.  If that feature is within
    snap_tolerance_ppm of the shifted query and does not collide with any target
    peptide m/z (within matching_ppm), it becomes the decoy feature.  Up to 10
    resamples are attempted before the peptide's decoy is skipped.

    ppm_error on decoy rows is copied from the peptide's best target match
    (minimum ppm_error_abs in target_candidates).  This makes ppm_error
    non-discriminative between a target and its paired decoy, ensuring that
    score separation comes from isotope envelope, spatial, and intensity features.

    decoy_delta_da stores snapped_feature_mz − peptide_mh_mz (the actual mass
    offset to the chosen decoy feature, not the sampled delta).

    Returns a DataFrame with the same schema as match_to_maldi_features()
    plus a decoy_delta_da column (NaN for targets).
    """
    rng = np.random.default_rng(random_state)
    feature_mzs = np.asarray(feature_mzs, dtype=np.float64)
    n_features = len(feature_mzs)
    tol_frac = matching_ppm * 1e-6

    # Unique target peptides indexed 0..N-1
    unique_pep = (
        target_df[~target_df["is_decoy"].astype(bool)]
        .drop_duplicates(subset="peptide")
        .reset_index(drop=True)
    )
    n_unique = len(unique_pep)
    target_mzs_sorted = np.sort(unique_pep["mh_mz"].values.astype(np.float64))

    # Pre-sort feature array once for O(log n) nearest-feature lookup
    feat_sort_idx = np.argsort(feature_mzs)
    feat_sorted = feature_mzs[feat_sort_idx]

    # Per-peptide outputs (-1 = no valid decoy feature found)
    decoy_orig_idx = np.full(n_unique, -1, dtype=np.int64)
    decoy_feat_mz  = np.full(n_unique, np.nan)
    decoy_actual_delta = np.full(n_unique, np.nan)

    for i in range(n_unique):
        orig = float(unique_pep.at[i, "mh_mz"])
        sign = 1.0 if i % 2 == 0 else -1.0
        for _attempt in range(10):
            delta = float(rng.uniform(delta_min, delta_max))
            shifted = orig + sign * delta
            if shifted <= 0:
                sign = 1.0
                continue

            # Find the nearest feature to the shifted query
            pos = int(np.searchsorted(feat_sorted, shifted))
            best_pos, best_dist = -1, np.inf
            for cand in (pos - 1, pos):
                if 0 <= cand < n_features:
                    d = abs(feat_sorted[cand] - shifted)
                    if d < best_dist:
                        best_dist, best_pos = d, cand
            if best_pos < 0:
                continue

            # Snap tolerance check
            if best_dist / shifted * 1e6 > snap_tolerance_ppm:
                sign = -sign
                continue

            nearest_mz = feat_sorted[best_pos]

            # Collision check: snapped feature must not be within matching_ppm
            # of any target peptide m/z (covers self-match implicitly)
            lo = np.searchsorted(target_mzs_sorted, nearest_mz * (1.0 - tol_frac), side="left")
            hi = np.searchsorted(target_mzs_sorted, nearest_mz * (1.0 + tol_frac), side="right")
            if lo < hi:
                sign = -sign
                continue

            decoy_orig_idx[i] = int(feat_sort_idx[best_pos])
            decoy_feat_mz[i] = nearest_mz
            decoy_actual_delta[i] = nearest_mz - orig
            break
        else:
            logger.warning(
                "mz_shift: no valid decoy found for '%s' (mh_mz=%.4f) after 10 attempts",
                unique_pep.at[i, "peptide"], orig,
            )

    valid_mask = decoy_orig_idx >= 0
    n_valid = int(valid_mask.sum())
    logger.info("mz_shift: %d/%d target peptides have valid decoy features", n_valid, n_unique)

    # --- Match targets against MALDI features (normal path) ---
    target_candidates = match_to_maldi_features(
        feature_mzs, target_df, matching_ppm,
        maldi_intensities=maldi_intensities,
        maldi_intensities_p90=maldi_intensities_p90,
        maldi_intensities_sum=maldi_intensities_sum,
    )
    target_candidates["decoy_delta_da"] = np.nan
    if "source" not in target_candidates.columns:
        target_candidates["source"] = "target"

    if n_valid == 0:
        logger.warning("mz_shift: no valid decoy features found — returning target-only candidates")
        return target_candidates

    # Build ppm_error lookup per peptide: use the target match with the smallest
    # ppm_error_abs so the decoy inherits the same non-discriminative ppm value.
    if "ppm_error" in target_candidates.columns and "peptide" in target_candidates.columns:
        _tc = target_candidates[["peptide", "ppm_error", "ppm_error_abs"]].copy()
        _best_idx = _tc.groupby("peptide")["ppm_error_abs"].idxmin()
        pep_ppm_map: pd.Series = (
            _tc.loc[_best_idx, ["peptide", "ppm_error"]]
            .set_index("peptide")["ppm_error"]
        )
    else:
        pep_ppm_map = pd.Series(dtype=float)

    valid_pep_rows = unique_pep[valid_mask].reset_index(drop=True)
    valid_orig_idx = decoy_orig_idx[valid_mask]
    valid_feat_mz  = decoy_feat_mz[valid_mask]
    valid_delta    = decoy_actual_delta[valid_mask]

    # --- Build decoy rows ---
    # LC-MS/MS evidence columns are intentionally preserved from the source target
    # peptide.  The decoy is the same sequence at a different MALDI feature;
    # wiping them would give decoys systematically worse priors, breaking TDC symmetry.
    decoy_df = valid_pep_rows.copy()
    decoy_df["is_decoy"] = True
    decoy_df["source"] = "decoy_mz_shift"
    decoy_df["feature_mz"]  = valid_feat_mz
    decoy_df["feature_idx"] = valid_orig_idx.astype(int)
    # ppm_error copied from the target match — not computed from the decoy feature,
    # because that would be ~δ/mz * 1e6 (thousands of ppm) and leak the label.
    decoy_df["ppm_error"]     = decoy_df["peptide"].map(pep_ppm_map).fillna(0.0)
    decoy_df["ppm_error_abs"] = decoy_df["ppm_error"].abs()
    # actual offset from peptide mass to chosen decoy feature (diagnostic only)
    decoy_df["decoy_delta_da"] = valid_delta

    fi_vals = valid_orig_idx.astype(int)
    if maldi_intensities_p90 is not None:
        decoy_df["feature_intensity_p90"] = maldi_intensities_p90[fi_vals]
    if maldi_intensities_sum is not None:
        decoy_df["feature_intensity_sum"] = maldi_intensities_sum[fi_vals]
    if maldi_intensities is not None:
        decoy_df["feature_intensity"] = maldi_intensities[fi_vals]

    kendrick = decoy_df["feature_mz"].values * (14.0 / 14.01565)
    decoy_df["kendrick_mass_defect"] = kendrick - np.round(kendrick)

    # --- Combine and recompute per-feature statistics ---
    result = pd.concat([target_candidates, decoy_df], ignore_index=True)
    result["is_decoy"] = result["is_decoy"].astype(bool)
    result["n_candidates"] = result.groupby("feature_mz")["feature_mz"].transform("count")
    prot_feat_count = result.groupby("protein")["feature_mz"].nunique()
    result["protein_n_features"] = result["protein"].map(prot_feat_count).fillna(0).astype(int)

    logger.info(
        "mz_shift: %d features → %d target + %d decoy candidates",
        result["feature_mz"].nunique(),
        int((~result["is_decoy"]).sum()),
        int(result["is_decoy"].sum()),
    )
    return result


def generate_balanced_shuffle_candidates(
    fasta_path: str | None,
    lcms_ids,
    feature_mzs: np.ndarray,
    matching_ppm: float = 20.0,
    max_shuffle_rounds: int = 50,
    target_ratio: float = 1.0,
    random_state: int = 42,
    maldi_intensities: np.ndarray | None = None,
    maldi_intensities_p90: np.ndarray | None = None,
    maldi_intensities_sum: np.ndarray | None = None,
    missed_cleavages: int = 2,
    min_length: int = 7,
    max_length: int = 50,
    enzyme: str = "trypsin",
) -> pd.DataFrame:
    """
    Generate balanced shuffle decoys with MALDI-match filtering.

    Runs up to max_shuffle_rounds rounds of K/R-preserving protein shuffle,
    keeping only decoy peptides that match a MALDI feature within matching_ppm.
    Subsample the collected pool to int(target_ratio * N_target) candidates.

    Unlike standard shuffle (one decoy per target regardless of MALDI match),
    this ensures decoys compete in the same observation space as targets.
    LC-MS/MS evidence columns are set to NaN for all decoy rows — shuffle
    decoys have different sequences from their parent targets, so inheriting
    evidence would break TDC symmetry.

    Returns a DataFrame with the same schema as match_to_maldi_features()
    plus decoy_delta_da (NaN for all rows) and source columns.
    """
    feature_mzs = np.asarray(feature_mzs, dtype=np.float64)
    enzyme_rule = parser.expasy_rules.get(enzyme, enzyme)

    # --- Step 1: Generate target peptides (no decoys) ---
    logger.info("balanced_shuffle Step 1: generating target peptides...")
    if lcms_ids is not None:
        target_db = digest_identified_proteins(
            fasta_path=fasta_path,
            lcms_ids=lcms_ids,
            enzyme=enzyme,
            missed_cleavages=missed_cleavages,
            min_length=min_length,
            max_length=max_length,
            generate_decoys=False,
        )
    else:
        if fasta_path is None:
            raise ValueError("fasta_path required when lcms_ids is None")
        target_db = digest_fasta(
            fasta_path,
            enzyme=enzyme,
            missed_cleavages=missed_cleavages,
            min_length=min_length,
            max_length=max_length,
            generate_decoys=False,
        )

    # --- Step 2: Match targets to MALDI features ---
    target_candidates = match_to_maldi_features(
        feature_mzs, target_db, matching_ppm,
        maldi_intensities=maldi_intensities,
        maldi_intensities_p90=maldi_intensities_p90,
        maldi_intensities_sum=maldi_intensities_sum,
    )
    if "source" not in target_candidates.columns:
        target_candidates["source"] = "target"

    n_target = len(target_candidates)
    n_decoys_needed = int(target_ratio * n_target)
    logger.info(
        "balanced_shuffle: %d target candidates, need %d decoys (ratio=%.2f)",
        n_target, n_decoys_needed, target_ratio,
    )

    if n_target == 0 or n_decoys_needed == 0:
        logger.warning("balanced_shuffle: 0 targets — returning targets only")
        target_candidates["decoy_delta_da"] = np.nan
        return target_candidates

    target_seqs = set(target_db["peptide"].values)

    # --- Step 3: Load protein sequences for shuffle ---
    is_lc_only = (fasta_path is None) and (lcms_ids is not None)

    if is_lc_only:
        confirmed_seqs = sorted(set(lcms_ids.peptides["sequence"].values))
        protein_seqs = {"__pseudo__": "".join(confirmed_seqs)}
    elif lcms_ids is not None:
        from ms1rescore.lcms_ids import filter_fasta_to_proteins
        protein_seqs = filter_fasta_to_proteins(fasta_path, lcms_ids.proteins)
        if not protein_seqs:
            logger.warning(
                "balanced_shuffle: no identified proteins in FASTA — using full FASTA"
            )
            protein_seqs = {
                (desc.split("|")[1] if "|" in desc else desc.split()[0]): seq
                for desc, seq in fasta.read(fasta_path)
            }
    else:
        protein_seqs = {
            (desc.split("|")[1] if "|" in desc else desc.split()[0]): seq
            for desc, seq in fasta.read(fasta_path)
        }

    # --- Step 4: Iterative shuffle rounds ---
    # Early stopping is length-aware: continue until every target length bin that
    # can produce MALDI-matching decoys has at least target_ratio * tgt_count entries
    # in the pool, AND the total pool already has enough to subsample.  Lengths for
    # which the pool never produces any decoys are excluded from the coverage check
    # (they cannot be satisfied regardless of round count).
    tgt_len_counts = target_candidates["peptide"].str.len().value_counts()
    pool_len_counts: dict[int, int] = {}
    decoy_pool_parts: list[pd.DataFrame] = []
    n_pool = 0

    for r in range(max_shuffle_rounds):
        # Length-aware early stop: every length bin that has ever produced pool
        # entries must now have at least n_need entries.
        if n_pool >= n_decoys_needed:
            all_covered = all(
                pool_len_counts.get(llen, 0)
                >= int(round(target_ratio * tgt_len_counts.get(llen, 0)))
                for llen in tgt_len_counts.index
                if pool_len_counts.get(llen, 0) > 0
            )
            if all_covered:
                logger.info(
                    "balanced_shuffle: all reachable length bins covered after %d rounds "
                    "(pool %d)",
                    r, n_pool,
                )
                break

        round_rows = []
        for acc, seq in protein_seqs.items():
            shuffled = _shuffle_protein(seq, random_state=random_state + r)
            for pep in parser.cleave(shuffled, enzyme_rule, missed_cleavages=missed_cleavages):
                if min_length <= len(pep) <= max_length and pep not in target_seqs:
                    round_rows.append((pep, f"DECOY_{acc}_r{r}", True))

        if not round_rows:
            continue

        round_df = pd.DataFrame(round_rows, columns=["peptide", "protein", "is_decoy"])
        round_df = round_df.drop_duplicates(subset="peptide")

        seqs_list = round_df["peptide"].tolist()
        try:
            from ms1rescore_rs import compute_peptide_masses
            masses, mh_mzs, n_cs, n_hs, n_ns, n_os, n_ss = compute_peptide_masses(seqs_list)
            round_df["mass"] = masses
            round_df["mh_mz"] = mh_mzs
            round_df["n_C"] = n_cs
            round_df["n_H"] = n_hs
            round_df["n_N"] = n_ns
            round_df["n_O"] = n_os
            round_df["n_S"] = n_ss
        except ImportError:
            masses_list = []
            for seq in seqs_list:
                try:
                    comp = mass.Composition(sequence=seq)
                    pm = mass.calculate_mass(composition=comp)
                    masses_list.append({
                        "mass": pm, "mh_mz": pm + PROTON,
                        "n_C": comp.get("C", 0), "n_H": comp.get("H", 0),
                        "n_N": comp.get("N", 0), "n_O": comp.get("O", 0),
                        "n_S": comp.get("S", 0),
                    })
                except Exception:
                    masses_list.append({
                        "mass": 0, "mh_mz": 0, "n_C": 0,
                        "n_H": 0, "n_N": 0, "n_O": 0, "n_S": 0,
                    })
            mass_df = pd.DataFrame(masses_list)
            for col in mass_df.columns:
                round_df[col] = mass_df[col].values

        round_df = round_df[round_df["mass"] > 0].reset_index(drop=True)
        if len(round_df) == 0:
            continue

        round_matched = match_to_maldi_features(
            feature_mzs, round_df, matching_ppm,
            maldi_intensities=maldi_intensities,
            maldi_intensities_p90=maldi_intensities_p90,
            maldi_intensities_sum=maldi_intensities_sum,
        )
        if len(round_matched) == 0:
            continue

        for llen, cnt in round_matched["peptide"].str.len().value_counts().items():
            pool_len_counts[llen] = pool_len_counts.get(llen, 0) + cnt

        decoy_pool_parts.append(round_matched)
        n_pool += len(round_matched)
        logger.info(
            "balanced_shuffle round %d: %d new decoy candidates (pool %d/%d)",
            r, len(round_matched), n_pool, n_decoys_needed,
        )

    if not decoy_pool_parts:
        logger.warning(
            "balanced_shuffle: no decoys matched MALDI features — returning targets only"
        )
        target_candidates["decoy_delta_da"] = np.nan
        return target_candidates

    decoy_pool = pd.concat(decoy_pool_parts, ignore_index=True)

    # --- Step 5: Length-stratified subsample ---
    # For each target length bin take exactly min(pool_available, n_need) decoys.
    # No fill from other lengths: adding decoys of the wrong length to compensate
    # for truly unreachable lengths would introduce a length bias worse than the
    # slight T:D count deficit.
    dec_lengths = decoy_pool["peptide"].str.len()
    rng = np.random.default_rng(random_state)

    keep_indices: list[int] = []
    unfilled: list[tuple[int, int, int]] = []  # (length, needed, available)
    for length, tgt_count in tgt_len_counts.items():
        dec_at_len = decoy_pool.index[dec_lengths == length].tolist()
        n_need = int(round(target_ratio * tgt_count))
        if n_need == 0:
            continue
        if not dec_at_len:
            unfilled.append((length, n_need, 0))
            continue
        if len(dec_at_len) <= n_need:
            if len(dec_at_len) < n_need:
                unfilled.append((length, n_need, len(dec_at_len)))
            keep_indices.extend(dec_at_len)
        else:
            keep_indices.extend(
                rng.choice(dec_at_len, size=n_need, replace=False).tolist()
            )

    if unfilled:
        logger.warning(
            "balanced_shuffle: insufficient decoys at lengths %s "
            "(format: length:needed/available) — consider increasing --max-shuffle-rounds",
            ", ".join(f"{l}:{n}/{a}" for l, n, a in unfilled),
        )

    decoy_pool = decoy_pool.loc[np.sort(keep_indices)].reset_index(drop=True)
    logger.info(
        "balanced_shuffle: subsampled decoy pool %d → %d (length-stratified, no fill)",
        n_pool, len(decoy_pool),
    )

    # --- Step 6: Mark decoys and wipe LC-MS/MS evidence ---
    decoy_pool["is_decoy"] = True
    decoy_pool["source"] = "decoy_balanced_shuffle"

    _LCMS_EV_PREFIXES = ("lcms_",)
    _LCMS_EV_NAMES = {"n_psms"}
    for col in decoy_pool.columns:
        if any(col.startswith(pfx) for pfx in _LCMS_EV_PREFIXES) or col in _LCMS_EV_NAMES:
            decoy_pool[col] = np.nan

    target_candidates["decoy_delta_da"] = np.nan
    decoy_pool["decoy_delta_da"] = np.nan

    # --- Step 7: Combine and recompute per-feature statistics ---
    result = pd.concat([target_candidates, decoy_pool], ignore_index=True)
    result["is_decoy"] = result["is_decoy"].astype(bool)
    result["n_candidates"] = result.groupby("feature_mz")["feature_mz"].transform("count")
    prot_feat_count = result.groupby("protein")["feature_mz"].nunique()
    result["protein_n_features"] = result["protein"].map(prot_feat_count).fillna(0).astype(int)

    logger.info(
        "balanced_shuffle: %d features → %d target + %d decoy candidates",
        result["feature_mz"].nunique(),
        int((~result["is_decoy"]).sum()),
        int(result["is_decoy"].sum()),
    )
    return result


def digest_identified_proteins(
    fasta_path: str | None,
    lcms_ids,
    enzyme: str = "trypsin",
    missed_cleavages: int = 2,
    min_length: int = 7,
    max_length: int = 30,
    generate_decoys: bool = True,
) -> pd.DataFrame:
    """
    Strategy C hybrid candidate generation.

    Builds a candidate DataFrame from:
    1. In-silico digest of LC-MS/MS-identified proteins (target + K/R-preserving
       shuffled decoy, via ``_shuffle_protein``). Skipped when ``fasta_path`` is
       ``None`` — all confirmed peptides are then treated as novel (LC-only mode).
    2. Directly identified LC-MS/MS peptides (``source="lcms_confirmed"``), even
       if also present in the protein digest.

    A peptide present in both the protein digest and ``lcms_ids.peptides`` is
    labelled ``"lcms_confirmed"``. Novel directly-identified peptides (not
    reachable by digesting the identified proteins) are added as targets with
    K/R-preserving peptide-level decoys.

    LC-MS/MS evidence columns are joined onto target rows and set to NaN for
    all decoy rows: ``lcms_q_value``, ``lcms_pep``, ``lcms_score``,
    ``n_psms``, ``lcms_charge``, ``lcms_rt_mean``, ``lcms_intensity``.

    Parameters
    ----------
    fasta_path
        Path to the protein FASTA file, or ``None`` to skip protein digestion
        and use only the LC-MS/MS identified peptides as candidates.
    lcms_ids
        ``LCMSIds`` namedtuple returned by ``parse_lcms_ids()``.
    enzyme
        Enzyme name recognised by ``pyteomics.parser.expasy_rules``.
    missed_cleavages
        Maximum allowed missed cleavages.
    min_length, max_length
        Peptide length range (inclusive).

    Returns
    -------
    DataFrame with the same columns as ``digest_fasta()`` plus
    ``source``, ``lcms_q_value``, ``lcms_pep``, ``lcms_score``,
    ``n_psms``, ``lcms_charge``, ``lcms_rt_mean``, ``lcms_intensity``.
    """
    _EV_COLS = {
        "q_value": "lcms_q_value",
        "pep": "lcms_pep",
        "score": "lcms_score",
        "n_psms": "n_psms",
        "charge": "lcms_charge",
        "rt_mean": "lcms_rt_mean",
        "lcms_intensity": "lcms_intensity",
        "lcms_ccs": "lcms_ccs",
    }
    _BASE_COLS = ["peptide", "protein", "is_decoy", "mass", "mh_mz",
                  "n_C", "n_H", "n_N", "n_O", "n_S", "source"]

    if fasta_path is not None:
        from ms1rescore.lcms_ids import filter_fasta_to_proteins

        # --- Step 1: Filter FASTA to identified proteins ---
        protein_seqs = filter_fasta_to_proteins(fasta_path, lcms_ids.proteins)

        if not protein_seqs:
            logger.warning(
                "No identified proteins found in FASTA — check accession format. "
                "Continuing with LC-MS/MS confirmed peptides only."
            )
            df = pd.DataFrame(columns=_BASE_COLS)
        else:
            # --- Step 2: Digest identified proteins (target + shuffled decoy) ---
            rows = []  # (peptide, protein, is_decoy)
            for acc, seq in protein_seqs.items():
                cleaved = parser.cleave(
                    seq,
                    parser.expasy_rules.get(enzyme, enzyme),
                    missed_cleavages=missed_cleavages,
                )
                for pep in cleaved:
                    if min_length <= len(pep) <= max_length:
                        rows.append((pep, acc, False))

                if generate_decoys:
                    decoy_seq = _shuffle_protein(seq)
                    cleaved_d = parser.cleave(
                        decoy_seq,
                        parser.expasy_rules.get(enzyme, enzyme),
                        missed_cleavages=missed_cleavages,
                    )
                    for pep in cleaved_d:
                        if min_length <= len(pep) <= max_length:
                            rows.append((pep, f"DECOY_{acc}", True))

            df = pd.DataFrame(rows, columns=["peptide", "protein", "is_decoy"])
            df = df.drop_duplicates(subset=["peptide", "is_decoy"])
    else:
        # LC-only mode: no FASTA digestion — all confirmed peptides will be added
        # as novel targets in Step 5 below.
        logger.info("  No FASTA provided — using LC-MS/MS identified peptides only as candidates.")
        df = pd.DataFrame(columns=_BASE_COLS)

    # --- Step 3: Compute masses (Rust if available, else pyteomics) ---
    sequences = df["peptide"].tolist()
    if sequences:
        try:
            from ms1rescore_rs import compute_peptide_masses

            masses, mh_mzs, n_cs, n_hs, n_ns, n_os, n_ss = compute_peptide_masses(sequences)
            df["mass"] = masses
            df["mh_mz"] = mh_mzs
            df["n_C"] = n_cs
            df["n_H"] = n_hs
            df["n_N"] = n_ns
            df["n_O"] = n_os
            df["n_S"] = n_ss
            logger.info("  (used Rust backend for mass computation)")
        except ImportError:
            logger.info("  (using pyteomics for mass computation)")
            masses_list = []
            for seq in sequences:
                try:
                    comp = mass.Composition(sequence=seq)
                    pep_mass = mass.calculate_mass(composition=comp)
                    masses_list.append({
                        "mass": pep_mass, "mh_mz": pep_mass + PROTON,
                        "n_C": comp.get("C", 0), "n_H": comp.get("H", 0),
                        "n_N": comp.get("N", 0), "n_O": comp.get("O", 0),
                        "n_S": comp.get("S", 0),
                    })
                except Exception:
                    masses_list.append({
                        "mass": 0, "mh_mz": 0, "n_C": 0, "n_H": 0,
                        "n_N": 0, "n_O": 0, "n_S": 0,
                    })
            mass_df = pd.DataFrame(masses_list)
            for col in mass_df.columns:
                df[col] = mass_df[col].values

        df = df[df["mass"] > 0].reset_index(drop=True)

    # --- Step 4: Label source (only rows from protein digest; novel rows set in Step 5) ---
    if len(df) > 0:
        df["source"] = np.where(df["is_decoy"], "decoy", "protein_digest")

    # --- Step 5: Union with directly identified peptides ---
    lcms_pep_df = lcms_ids.peptides
    if len(lcms_pep_df) > 0:
        confirmed_seqs = set(lcms_pep_df["sequence"].values)

        # Mark digest targets that are LC-MS/MS confirmed
        target_mask = ~df["is_decoy"]
        df.loc[target_mask & df["peptide"].isin(confirmed_seqs), "source"] = "lcms_confirmed"

        # Find novel confirmed peptides not reachable from the protein digest
        existing_targets = set(df.loc[target_mask, "peptide"].values)
        novel_seqs = sorted(s for s in confirmed_seqs if s not in existing_targets)

        if novel_seqs:
            logger.info(
                f"  {len(novel_seqs)} LC-MS/MS-confirmed peptides not in protein digest "
                f"— adding as lcms_confirmed targets"
            )
            novel_rows = []

            # Target rows (always the same)
            for seq in novel_seqs:
                prot_row = lcms_pep_df[lcms_pep_df["sequence"] == seq].iloc[0]
                prot = str(prot_row.get("protein", ""))
                novel_rows.append((seq, prot, False))

            if generate_decoys:
                if fasta_path is None:
                    # Concatenated pseudo-protein decoy strategy for LC-only mode.
                    # Per-peptide shuffle produces decoys with identical elemental
                    # composition to their target (same residue multiset, just reordered)
                    # — making isotope envelope features non-discriminative. By
                    # concatenating all target peptides into a pseudo-protein and
                    # shuffling at that level, non-K/R residues are redistributed
                    # across tryptic boundaries, breaking composition conservation.
                    sorted_seqs = sorted(novel_seqs)  # fixed order for reproducibility
                    pseudo_protein = "".join(sorted_seqs)
                    shuffled_pseudo = _shuffle_protein(pseudo_protein, random_state=42)
                    target_set = set(novel_seqs)
                    raw_decoys = list(parser.cleave(
                        shuffled_pseudo,
                        parser.expasy_rules.get(enzyme, enzyme),
                        missed_cleavages=missed_cleavages,
                    ))
                    decoy_peptides = list(dict.fromkeys(
                        p for p in raw_decoys
                        if min_length <= len(p) <= max_length and p not in target_set
                    ))
                    n_targets = len(novel_seqs)
                    if len(decoy_peptides) < n_targets:
                        logger.warning(
                            "Concatenated pseudo-protein decoy digest produced %d decoys "
                            "for %d target peptides — TDC ratio will be < 1:1. Consider "
                            "increasing --missed-cleavages or --max-length.",
                            len(decoy_peptides), n_targets,
                        )
                    elif len(decoy_peptides) > n_targets:
                        decoy_peptides = random.Random(42).sample(decoy_peptides, n_targets)
                    for dec in decoy_peptides:
                        novel_rows.append((dec, "DECOY_concat", True))
                else:
                    # Per-peptide K/R-preserving shuffle for novel sequences that are
                    # not reachable from the protein digest (Strategy C hybrid).
                    for seq in novel_seqs:
                        prot_row = lcms_pep_df[lcms_pep_df["sequence"] == seq].iloc[0]
                        prot = str(prot_row.get("protein", ""))
                        dec = _shuffle_protein(seq, random_state=42)
                        if dec != seq:
                            novel_rows.append((dec, f"DECOY_{prot}", True))

            novel_df = pd.DataFrame(novel_rows, columns=["peptide", "protein", "is_decoy"])
            novel_df = novel_df.drop_duplicates(subset=["peptide", "is_decoy"])

            # Compute masses for novel sequences
            novel_seqs_list = novel_df["peptide"].tolist()
            try:
                from ms1rescore_rs import compute_peptide_masses

                masses, mh_mzs, n_cs, n_hs, n_ns, n_os, n_ss = compute_peptide_masses(novel_seqs_list)
                novel_df["mass"] = masses
                novel_df["mh_mz"] = mh_mzs
                novel_df["n_C"] = n_cs
                novel_df["n_H"] = n_hs
                novel_df["n_N"] = n_ns
                novel_df["n_O"] = n_os
                novel_df["n_S"] = n_ss
            except ImportError:
                novel_masses = []
                for seq in novel_seqs_list:
                    try:
                        comp = mass.Composition(sequence=seq)
                        pm = mass.calculate_mass(composition=comp)
                        novel_masses.append({
                            "mass": pm, "mh_mz": pm + PROTON,
                            "n_C": comp.get("C", 0), "n_H": comp.get("H", 0),
                            "n_N": comp.get("N", 0), "n_O": comp.get("O", 0),
                            "n_S": comp.get("S", 0),
                        })
                    except Exception:
                        novel_masses.append({
                            "mass": 0, "mh_mz": 0, "n_C": 0, "n_H": 0,
                            "n_N": 0, "n_O": 0, "n_S": 0,
                        })
                novel_mass_df = pd.DataFrame(novel_masses)
                for col in novel_mass_df.columns:
                    novel_df[col] = novel_mass_df[col].values

            novel_df = novel_df[novel_df["mass"] > 0].reset_index(drop=True)
            novel_df["is_decoy"] = novel_df["is_decoy"].astype(bool)
            novel_df.loc[~novel_df["is_decoy"], "source"] = "lcms_confirmed"
            novel_df.loc[novel_df["is_decoy"], "source"] = "decoy"
            df = pd.concat([df, novel_df], ignore_index=True)

    # --- Step 6: Join LC-MS/MS evidence columns ---
    for new_col in _EV_COLS.values():
        df[new_col] = np.nan

    if len(lcms_pep_df) > 0:
        ev = lcms_pep_df.drop_duplicates(subset="sequence").set_index("sequence")
        for old_col, new_col in _EV_COLS.items():
            if old_col in ev.columns:
                df[new_col] = df["peptide"].map(ev[old_col])

    # --- Step 7: Wipe evidence for decoys (symmetric TDC requirement) ---
    df["is_decoy"] = df["is_decoy"].astype(bool)
    decoy_mask = df["is_decoy"].values
    for new_col in _EV_COLS.values():
        df.loc[decoy_mask, new_col] = np.nan

    n_confirmed = (df["source"] == "lcms_confirmed").sum()
    n_digest = (df["source"] == "protein_digest").sum()
    n_decoy = (df["source"] == "decoy").sum()
    logger.info(
        f"Strategy C candidates: {n_confirmed} lcms_confirmed + "
        f"{n_digest} protein_digest + {n_decoy} decoy"
    )
    return df
