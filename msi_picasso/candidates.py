"""FASTA digest, decoy generation, and MALDI m/z matching."""

import bisect
import logging
import random

import numpy as np
import pandas as pd
from pyteomics import fasta, mass, parser

from msi_picasso.utils import PROTON

logger = logging.getLogger(__name__)

# paired_shuffle (selection_mode="feature") tuning constants.
# FEATURE_COVERAGE_TARGET: fraction of reachable target-occupied features that
# must have >=1 pool decoy before early stopping the shuffle rounds.
FEATURE_COVERAGE_TARGET = 0.95


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
        cleaved = sorted(parser.cleave(
            seq,
            parser.expasy_rules.get(enzyme, enzyme),
            missed_cleavages=missed_cleavages,
        ))
        for pep in cleaved:
            if min_length <= len(pep) <= max_length:
                rows.append((pep, protein_id, False))

        if generate_decoys:
            decoy_seq = _shuffle_protein(seq)
            cleaved_d = sorted(parser.cleave(
                decoy_seq,
                parser.expasy_rules.get(enzyme, enzyme),
                missed_cleavages=missed_cleavages,
            ))
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
    snap_to_features: bool = True,
    maldi_intensities: np.ndarray | None = None,
    maldi_intensities_p90: np.ndarray | None = None,
    maldi_intensities_sum: np.ndarray | None = None,
) -> pd.DataFrame:
    """
    Generate m/z-shifted observation-space decoys and return a combined
    target + decoy candidates DataFrame.

    For each unique target peptide a random delta in [delta_min, delta_max] Da
    is sampled; sign alternates (even index -> +, odd index -> -).

    Two placement modes:

    - ``snap_to_features=True`` (default, feature-list mode): the shifted query is
      snapped to the nearest MALDI feature.  If that feature is within
      ``snap_tolerance_ppm`` of the shifted query and does not collide with any
      target peptide m/z (within ``matching_ppm``), it becomes the decoy feature.
    - ``snap_to_features=False`` (raw-query mode): the decoy feature *is* the exact
      shifted m/z (``mh_mz ± delta``) — no snapping, because in raw-query any m/z is
      imaged on demand.  The shift is accepted only if it does not collide (within
      ``matching_ppm``) with any target peptide m/z **or** with an already-assigned
      decoy m/z, so every decoy occupies a distinct feature (no clustering onto
      shared grid points).  Each such decoy gets a unique ``feature_idx`` past the
      grid index range ``[0, len(feature_mzs))``.

    Up to 50 resamples are attempted before the peptide's decoy is skipped.

    ppm_error on decoy rows is copied from the peptide's best target match
    (minimum ppm_error_abs in target_candidates).  This makes ppm_error
    non-discriminative between a target and its paired decoy, ensuring that
    score separation comes from isotope envelope, spatial, and intensity features.

    decoy_delta_da stores snapped_feature_mz - peptide_mh_mz (the actual mass
    offset to the chosen decoy feature, not the sampled delta).

    The ``feature_mz`` column on decoy rows is the *shifted* m/z (the snapped
    off-target anchor), NOT the original peptide m/z.  This is load-bearing for
    raw-query mode (see maldi_query.py): the raw query extracts the ion image at
    ``feature_mz``, which for mz_shift decoys is the shifted feature.

    Returns a DataFrame with the same schema as match_to_maldi_features()
    plus a decoy_delta_da column (NaN for targets).
    """
    failed_counter = 0
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
    decoy_feat_mz = np.full(n_unique, np.nan)
    decoy_actual_delta = np.full(n_unique, np.nan)

    # No-snap (raw-query) bookkeeping: a sorted list of assigned decoy m/z so each
    # new decoy lands on a distinct feature, and a running feature_idx disjoint from
    # the grid index range [0, n_features).
    used_decoy_mz: list[float] = []
    next_decoy_idx = n_features

    def _collides_used(mz: float) -> bool:
        """True if `mz` is within matching_ppm of an already-assigned decoy m/z."""
        if not used_decoy_mz:
            return False
        j = bisect.bisect_left(used_decoy_mz, mz * (1.0 - tol_frac))
        return j < len(used_decoy_mz) and used_decoy_mz[j] <= mz * (1.0 + tol_frac)

    for i in range(n_unique):
        orig = float(unique_pep.at[i, "mh_mz"])
        sign = 1.0 if i % 2 == 0 else -1.0
        for _attempt in range(50):
            delta = float(rng.uniform(delta_min, delta_max))
            shifted = orig + sign * delta
            if shifted <= 0:
                sign = 1.0
                continue

            if snap_to_features:
                # Snap the shifted query to the nearest detected MALDI feature.
                pos = int(np.searchsorted(feat_sorted, shifted))
                best_pos, best_dist = -1, np.inf
                for cand in (pos - 1, pos):
                    if 0 <= cand < n_features:
                        d = abs(feat_sorted[cand] - shifted)
                        if d < best_dist:
                            best_dist, best_pos = d, cand
                if best_pos < 0:
                    continue
                if best_dist / shifted * 1e6 > snap_tolerance_ppm:
                    sign = -sign
                    continue
                cand_mz = float(feat_sorted[best_pos])
                cand_idx = int(feat_sort_idx[best_pos])
            else:
                # Raw-query: the decoy feature IS the exact shifted m/z (any m/z is
                # imaged on demand), so there is no nearest-feature snap.
                cand_mz = shifted
                cand_idx = -1  # assigned below, after acceptance

            # Collision check: must not be within matching_ppm of any target peptide
            # m/z (covers self-match implicitly).
            lo = np.searchsorted(target_mzs_sorted, cand_mz * (1.0 - tol_frac), side="left")
            hi = np.searchsorted(target_mzs_sorted, cand_mz * (1.0 + tol_frac), side="right")
            if lo < hi:
                sign = -sign
                continue

            if not snap_to_features:
                # Distinct-feature guarantee: reject a shift that lands on an already
                # assigned decoy m/z, so decoys never cluster onto one feature.
                if _collides_used(cand_mz):
                    sign = -sign
                    continue
                cand_idx = next_decoy_idx
                next_decoy_idx += 1
                bisect.insort(used_decoy_mz, cand_mz)

            decoy_orig_idx[i] = cand_idx
            decoy_feat_mz[i] = cand_mz
            decoy_actual_delta[i] = cand_mz - orig
            break
        else:
            failed_counter += 1
            logger.warning(
                "mz_shift: no valid decoy found for '%s' (mh_mz=%.4f) after 50 attempts",
                unique_pep.at[i, "peptide"], orig,
            )

    valid_mask = decoy_orig_idx >= 0
    n_valid = int(valid_mask.sum())
    logger.info("mz_shift: %d/%d target peptides have valid decoy features", n_valid, n_unique)
    logger.info("mz_shift: %d target peptides failed to find valid decoy features (%.2f%%)", failed_counter, failed_counter / n_unique * 100)

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
    valid_feat_mz = decoy_feat_mz[valid_mask]
    valid_delta = decoy_actual_delta[valid_mask]

    # --- Build decoy rows ---
    # LC-MS/MS evidence columns are intentionally preserved from the source target
    # peptide.  The decoy is the same sequence at a different MALDI feature;
    # wiping them would give decoys systematically worse priors, breaking TDC symmetry.
    decoy_df = valid_pep_rows.copy()
    decoy_df["is_decoy"] = True
    decoy_df["source"] = "decoy_mz_shift"
    # Separate protein namespace: protein-level features (protein_colocalization,
    # protein_n_features, protein_coverage, ...) must be computed WITHIN class.
    # Keeping the real protein name would pool the decoy with its source target's
    # peptides, contaminating those features and breaking the TDC null.
    decoy_df["protein"] = "DECOY_" + decoy_df["protein"].astype(str)
    decoy_df["feature_mz"] = valid_feat_mz
    decoy_df["feature_idx"] = valid_orig_idx.astype(int)
    # ppm_error copied from the target match — not computed from the decoy feature,
    # because that would be ~delta/mz * 1e6 (thousands of ppm) and leak the label.
    decoy_df["ppm_error"] = decoy_df["peptide"].map(pep_ppm_map).fillna(0.0)
    decoy_df["ppm_error_abs"] = decoy_df["ppm_error"].abs()
    # actual offset from peptide mass to chosen decoy feature (diagnostic only)
    decoy_df["decoy_delta_da"] = valid_delta

    # Intensity lookup by grid index is only valid when decoys were snapped to grid
    # features.  In no-snap (raw-query) mode feature_idx is past the grid range and
    # intensities are attached later in the pipeline by feature_mz (arrays are None
    # here anyway), so skip the index-based assignment.
    if snap_to_features:
        fi_vals = valid_orig_idx.astype(int)
        if maldi_intensities_p90 is not None:
            decoy_df["feature_intensity_p90"] = maldi_intensities_p90[fi_vals]
        if maldi_intensities_sum is not None:
            decoy_df["feature_intensity_sum"] = maldi_intensities_sum[fi_vals]
        if maldi_intensities is not None:
            decoy_df["feature_intensity"] = maldi_intensities[fi_vals]

    kendrick = decoy_df["feature_mz"].values * (14.0 / 14.01565)
    decoy_df["kendrick_mass_defect"] = kendrick - np.round(kendrick)

    # --- Combine and recompute per-feature / per-protein statistics ---
    result = pd.concat([target_candidates, decoy_df], ignore_index=True)
    result["is_decoy"] = result["is_decoy"].astype(bool)
    result["n_candidates"] = result.groupby("feature_mz")["feature_mz"].transform("count")
    prot_feat_count = result.groupby("protein")["feature_mz"].nunique()
    result["protein_n_features"] = result["protein"].map(prot_feat_count).fillna(0).astype(int)
    # protein_tryptic_count is the full-digest peptide count per protein; decoys carry
    # a DECOY_-prefixed protein, so strip the prefix to inherit the source protein's
    # count (keeps protein_coverage symmetric between a protein and its decoy).
    if "protein_tryptic_count" in target_candidates.columns:
        prot_tryptic = (
            target_candidates.drop_duplicates(subset=["protein"])
            .set_index("protein")["protein_tryptic_count"]
            .to_dict()
        )
        _base_prot = result["protein"].astype(str).str.replace(r"^DECOY_", "", regex=True)
        result["protein_tryptic_count"] = (
            _base_prot.map(prot_tryptic).fillna(0).astype(int)
        )

    logger.info(
        "mz_shift: %d features → %d target + %d decoy candidates",
        result["feature_mz"].nunique(),
        int((~result["is_decoy"]).sum()),
        int(result["is_decoy"].sum()),
    )
    return result


def generate_mz_shuffle_candidates(
    target_df: pd.DataFrame,
    feature_mzs: np.ndarray,
    matching_ppm: float = 20.0,
    random_state: int = 42,
    maldi_intensities: np.ndarray | None = None,
    maldi_intensities_p90: np.ndarray | None = None,
    maldi_intensities_sum: np.ndarray | None = None,
) -> pd.DataFrame:
    """
    Generate m/z-assignment-shuffle decoys: a derangement of the target
    peptide -> feature assignment.

    Each unique target peptide is matched to its representative MALDI feature
    (the matched feature with the smallest ``ppm_error_abs``).  Decoys are formed
    by permuting which peptide is assigned to which feature, so every decoy is a
    REAL target peptide relocated onto a DIFFERENT real feature (the one belonging
    to another peptide).  Consequences, which make this a good TDC null:

    - Decoy features are the SAME set as target features (1 target + 1 decoy per
      feature, co-located on the identical ion image), so feature-quality features
      (intensity, fraction_detected, spatial autocorrelation, colocalization) are
      *identical* between the target and the decoy at a feature and contribute
      nothing to the target/decoy separation.  Discrimination is forced onto the
      peptide-specific predicted-vs-observed match (CCS, isotope pattern).
    - The permutation is built on a mass-sorted rotation, so a peptide is never
      assigned to its own feature (no fixed point) and never to a near-isobaric
      feature (the rotation magnitude spans a large mass-rank gap).

    ``ppm_error`` on decoy rows is copied from the peptide's best target match
    (non-discriminative) — in raw-query mode it is later recomputed from the
    observed peak centroid at the assigned feature (symmetric, ~0).  Mass accuracy
    must NOT be computed against the decoy peptide's own mass: that mismatch does
    not exist for real false positives (which match a feature within tolerance), so
    using it would make the null anti-conservative.

    ``decoy_delta_da`` stores assigned_feature_mz - peptide_mh_mz (diagnostic).
    ``source = "decoy_mz_shuffle"``.  Returns a combined target+decoy DataFrame
    with the same schema as ``match_to_maldi_features()`` plus ``decoy_delta_da``.
    """
    feature_mzs = np.asarray(feature_mzs, dtype=np.float64)

    # --- Match targets against MALDI features (normal path) ---
    target_candidates = match_to_maldi_features(
        feature_mzs, target_df, matching_ppm,
        maldi_intensities=maldi_intensities,
        maldi_intensities_p90=maldi_intensities_p90,
        maldi_intensities_sum=maldi_intensities_sum,
    )
    if len(target_candidates) == 0:
        logger.warning("mz_shuffle: no target candidates matched — returning empty")
        return target_candidates
    target_candidates["decoy_delta_da"] = np.nan
    if "source" not in target_candidates.columns:
        target_candidates["source"] = "target"

    # Representative feature per unique target peptide = its best (lowest |ppm|) match.
    best_idx = target_candidates.groupby("peptide")["ppm_error_abs"].idxmin()
    best = target_candidates.loc[best_idx].reset_index(drop=True)
    n = len(best)
    if n < 2:
        logger.warning("mz_shuffle: <2 unique target peptides — returning target-only")
        return target_candidates

    mh = best["mh_mz"].to_numpy(dtype=np.float64)
    feat_mz = best["feature_mz"].to_numpy(dtype=np.float64)
    feat_idx = best["feature_idx"].to_numpy()

    # Mass-sorted rotation derangement: in mass-rank space assign each peptide to
    # the one `k` ranks away (cyclic).  k in [n/4, 3n/4) guarantees both no fixed
    # point and a large mass gap (never near-isobaric).
    rng = np.random.default_rng(random_state)
    order = np.argsort(mh)
    if n > 3:
        k = int(rng.integers(max(1, n // 4), max(2, 3 * n // 4)))
    else:
        k = 1
    rolled = np.roll(order, k)
    sigma = np.empty(n, dtype=np.int64)
    sigma[order] = rolled  # sigma[i] = index of the peptide whose feature i is assigned to

    # ppm inherited from each peptide's own best target match (non-discriminative).
    pep_ppm = best["ppm_error"].to_numpy(dtype=np.float64)

    # --- Build decoy rows: peptide i relocated onto feature of peptide sigma[i] ---
    decoy_df = best.copy()
    decoy_df["is_decoy"] = True
    decoy_df["source"] = "decoy_mz_shuffle"
    # Separate protein namespace so protein-level features are computed within class
    # (a decoy must not be pooled with its source target's protein peptides).
    decoy_df["protein"] = "DECOY_" + decoy_df["protein"].astype(str)
    decoy_df["feature_mz"] = feat_mz[sigma]
    decoy_df["feature_idx"] = feat_idx[sigma]
    decoy_df["decoy_delta_da"] = feat_mz[sigma] - mh
    decoy_df["ppm_error"] = pep_ppm
    decoy_df["ppm_error_abs"] = np.abs(pep_ppm)

    fi = feat_idx[sigma]
    if maldi_intensities_p90 is not None:
        decoy_df["feature_intensity_p90"] = maldi_intensities_p90[fi.astype(int)]
    if maldi_intensities_sum is not None:
        decoy_df["feature_intensity_sum"] = maldi_intensities_sum[fi.astype(int)]
    if maldi_intensities is not None:
        decoy_df["feature_intensity"] = maldi_intensities[fi.astype(int)]

    kendrick = decoy_df["feature_mz"].to_numpy() * (14.0 / 14.01565)
    decoy_df["kendrick_mass_defect"] = kendrick - np.round(kendrick)

    # --- Combine and recompute per-feature / per-protein statistics ---
    # Use the representative-feature target set (`best`, one row per unique
    # peptide), NOT the full multiplicity `target_candidates`. A target peptide
    # whose m/z falls within `matching_ppm` of several MALDI peaks otherwise
    # yields multiple target rows while its single decoy yields one, producing a
    # ~(mean features/peptide):1 target:decoy imbalance (e.g. 5901:2895) and
    # leaving most target rows without a co-located decoy. Deduplicating to
    # `best` realises the mz_shuffle design — exactly 1 target + 1 decoy per
    # peptide, co-located on the identical feature — and loses no unique peptide
    # identifications (only redundant near-isobaric secondary matches; the
    # lowest-|ppm| match is kept).
    result = pd.concat([best, decoy_df], ignore_index=True)
    result["is_decoy"] = result["is_decoy"].astype(bool)
    result["n_candidates"] = result.groupby("feature_mz")["feature_mz"].transform("count")
    prot_feat_count = result.groupby("protein")["feature_mz"].nunique()
    result["protein_n_features"] = result["protein"].map(prot_feat_count).fillna(0).astype(int)
    if "protein_tryptic_count" in target_candidates.columns:
        prot_tryptic = (
            target_candidates.drop_duplicates(subset=["protein"])
            .set_index("protein")["protein_tryptic_count"]
            .to_dict()
        )
        _base_prot = result["protein"].astype(str).str.replace(r"^DECOY_", "", regex=True)
        result["protein_tryptic_count"] = (
            _base_prot.map(prot_tryptic).fillna(0).astype(int)
        )

    logger.info(
        "mz_shuffle: %d features → %d target + %d decoy candidates "
        "(every decoy co-located with a target on a real feature)",
        result["feature_mz"].nunique(),
        int((~result["is_decoy"]).sum()),
        int(result["is_decoy"].sum()),
    )
    return result


def load_entrapment_candidates(
    entrapment_fasta: str,
    target_df: pd.DataFrame,
    feature_mzs: np.ndarray,
    matching_ppm: float = 20.0,
    missed_cleavages: int = 2,
    min_length: int = 7,
    max_length: int = 30,
    enzyme: str = "trypsin",
    maldi_intensities: np.ndarray | None = None,
    maldi_intensities_p90: np.ndarray | None = None,
    maldi_intensities_sum: np.ndarray | None = None,
) -> pd.DataFrame:
    """
    Generate entrapment decoys from a foreign-organism FASTA.

    The entrapment FASTA is digested with the same trypsin rules as the targets.
    Entrapment peptides whose [M+H]+ m/z falls within ``matching_ppm`` of ANY
    target peptide m/z are removed as a *contamination filter* (not a decoy
    selection step): an isobaric entrapment peptide would inherit the real
    biological signal present at that m/z, making the null artificially good.
    The collision rate is logged; a rate > 10% warns that the entrapment organism
    and the sample proteome overlap heavily in m/z space.

    Surviving entrapment peptides are matched to ``feature_mzs`` exactly as
    targets are (``match_to_maldi_features``).  All rows are flagged
    ``is_decoy=True``, ``source="entrapment"``, ``protein="ENTRAPMENT_{accession}"``.

    Parameters
    ----------
    target_df
        Matched TARGET candidate DataFrame (must contain ``mh_mz``).  Used only
        for the contamination filter.
    feature_mzs
        MALDI feature m/z array (same array used to match the targets).

    Returns the matched entrapment DECOY rows only (schema identical to
    ``match_to_maldi_features`` output).  LC-MS/MS ID-derived columns are absent
    at this stage and are populated as NaN downstream, exactly as for shuffle
    decoys — entrapment peptides are not present in the LC-MS/MS data.
    """
    feature_mzs = np.asarray(feature_mzs, dtype=np.float64)

    # Phase 1: digest the entrapment FASTA (targets-only digest, no shuffle).
    rows = []  # (peptide, protein)
    for desc, seq in fasta.read(entrapment_fasta):
        protein_id = desc.split("|")[1] if "|" in desc else desc.split()[0]
        cleaved = sorted(parser.cleave(
            seq,
            parser.expasy_rules.get(enzyme, enzyme),
            missed_cleavages=missed_cleavages,
        ))
        for pep in cleaved:
            if min_length <= len(pep) <= max_length:
                rows.append((pep, protein_id))

    ent_db = pd.DataFrame(rows, columns=["peptide", "protein"])
    # Keep the first protein per unique peptide (entrapment is a foreign organism;
    # peptide-level uniqueness mirrors how targets are deduplicated).
    ent_db = ent_db.drop_duplicates(subset="peptide").reset_index(drop=True)
    if len(ent_db) == 0:
        logger.warning("entrapment: no peptides produced from %s", entrapment_fasta)
        return pd.DataFrame()

    # Phase 2: masses + elemental composition (Rust if available, else pyteomics).
    sequences = ent_db["peptide"].tolist()
    try:
        from ms1rescore_rs import compute_peptide_masses

        masses, mh_mzs, n_cs, n_hs, n_ns, n_os, n_ss = compute_peptide_masses(sequences)
        ent_db["mass"] = masses
        ent_db["mh_mz"] = mh_mzs
        ent_db["n_C"] = n_cs
        ent_db["n_H"] = n_hs
        ent_db["n_N"] = n_ns
        ent_db["n_O"] = n_os
        ent_db["n_S"] = n_ss
    except ImportError:
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
            ent_db[col] = mass_df[col].values

    ent_db = ent_db[ent_db["mass"] > 0].reset_index(drop=True)
    n_total = len(ent_db)

    # Contamination filter: drop entrapment peptides isobaric with any target.
    target_mzs = np.asarray(target_df["mh_mz"].values, dtype=np.float64)
    entrap_mzs = ent_db["mh_mz"].values.astype(np.float64)
    collided_pep_idx: set[int] = set()
    try:
        from ms1rescore_rs import match_mz

        _f, pep_idx, _e = match_mz(
            target_mzs.tolist(), entrap_mzs.tolist(), matching_ppm
        )
        collided_pep_idx = set(int(i) for i in pep_idx)
    except ImportError:
        ent_sorted_idx = np.argsort(entrap_mzs)
        ent_sorted = entrap_mzs[ent_sorted_idx]
        for tmz in target_mzs:
            tol = tmz * matching_ppm / 1e6
            lo = np.searchsorted(ent_sorted, tmz - tol, side="left")
            hi = np.searchsorted(ent_sorted, tmz + tol, side="right")
            for j in range(lo, hi):
                collided_pep_idx.add(int(ent_sorted_idx[j]))

    n_collided = len(collided_pep_idx)
    collision_rate = n_collided / n_total if n_total else 0.0
    logger.info(
        "entrapment: contamination filter removed %d/%d peptides (%.1f%% isobaric with a target)",
        n_collided, n_total, 100.0 * collision_rate,
    )
    if collision_rate > 0.10:
        logger.warning(
            "entrapment: collision rate %.1f%% > 10%% — the entrapment organism and the "
            "sample proteome overlap substantially in m/z space; the null may be biased.",
            100.0 * collision_rate,
        )

    keep_mask = ~ent_db.index.isin(collided_pep_idx)
    ent_db = ent_db[keep_mask].reset_index(drop=True)
    if len(ent_db) == 0:
        logger.warning("entrapment: all peptides removed by contamination filter")
        return pd.DataFrame()

    ent_db["is_decoy"] = True
    ent_db["protein"] = "ENTRAPMENT_" + ent_db["protein"].astype(str)
    ent_db["source"] = "entrapment"

    # Phase 3: match surviving entrapment peptides to MALDI features.
    decoy_candidates = match_to_maldi_features(
        feature_mzs, ent_db, matching_ppm,
        maldi_intensities=maldi_intensities,
        maldi_intensities_p90=maldi_intensities_p90,
        maldi_intensities_sum=maldi_intensities_sum,
    )
    if len(decoy_candidates) == 0:
        logger.warning("entrapment: no entrapment peptides matched any MALDI feature")
        return decoy_candidates

    decoy_candidates["is_decoy"] = decoy_candidates["is_decoy"].astype(bool)
    if "source" not in decoy_candidates.columns:
        decoy_candidates["source"] = "entrapment"
    logger.info(
        "entrapment: %d decoy candidates across %d features",
        len(decoy_candidates), decoy_candidates["feature_mz"].nunique(),
    )
    return decoy_candidates


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
    selection_mode: str = "length",
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

    ``selection_mode`` controls how the collected pool is subsampled:

    - ``"length"`` (default, ``balanced_shuffle``): length-stratified subsample
      to a global ``target_ratio * N_target`` count. Decoy per-feature occupancy
      is independent of target occupancy, so many MALDI features end up with only
      targets ("target-only") or only decoys ("decoy-only").
    - ``"feature"`` (``paired_shuffle``): feature-occupancy-matched selection.
      Decoys are first paired to the same MALDI features the targets occupy
      (maximising head-to-head competition and making decoy m/z density track
      target m/z density), then the pool is topped up to the same global
      ``target_ratio * N_target`` count from the remaining decoys. The global
      target:decoy ratio (and thus the FDR null mass) is identical to
      ``"length"`` mode; only the per-feature allocation differs. Selection is
      keyed purely on ``feature_idx`` (a mass property), never on scores or
      decoy correctness, so TDC validity is preserved.

    Returns a DataFrame with the same schema as match_to_maldi_features()
    plus decoy_delta_da (NaN for all rows) and source columns.
    """
    if selection_mode not in ("length", "feature"):
        raise ValueError(
            f"selection_mode must be 'length' or 'feature', got {selection_mode!r}"
        )
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
        from msi_picasso.lcms_ids import filter_fasta_to_proteins
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
    # Early stopping depends on selection_mode:
    #  - "length": continue until every target length bin that can produce
    #    MALDI-matching decoys has at least target_ratio * tgt_count entries in the
    #    pool, AND the total pool already has enough to subsample.
    #  - "feature": continue until a fraction (FEATURE_COVERAGE_TARGET) of the
    #    target-occupied features have at least one pool decoy at their m/z, AND
    #    the total pool already has enough to subsample.
    # In both modes, bins/features for which the pool never produces a decoy are
    # implicitly excluded (they cannot be satisfied regardless of round count); the
    # max_shuffle_rounds cap is the hard backstop.
    tgt_len_counts = target_candidates["peptide"].str.len().value_counts()
    pool_len_counts: dict[int, int] = {}
    target_feat_ids = set(target_candidates["feature_idx"].unique())
    covered_feats: set[int] = set()
    decoy_pool_parts: list[pd.DataFrame] = []
    n_pool = 0

    for r in range(max_shuffle_rounds):
        if n_pool >= n_decoys_needed:
            if selection_mode == "feature":
                # Feature-coverage-aware early stop.
                if target_feat_ids:
                    coverage = len(covered_feats & target_feat_ids) / len(target_feat_ids)
                else:
                    coverage = 1.0
                if coverage >= FEATURE_COVERAGE_TARGET:
                    logger.info(
                        "paired_shuffle: %.1f%% of target features covered after %d "
                        "rounds (pool %d)",
                        100 * coverage, r, n_pool,
                    )
                    break
            else:
                # Length-aware early stop: every length bin that has ever produced
                # pool entries must now have at least n_need entries.
                all_covered = all(
                    pool_len_counts.get(llen, 0)
                    >= int(round(target_ratio * tgt_len_counts.get(llen, 0)))
                    for llen in tgt_len_counts.index
                    if pool_len_counts.get(llen, 0) > 0
                )
                if all_covered:
                    logger.info(
                        "balanced_shuffle: all reachable length bins covered after %d "
                        "rounds (pool %d)",
                        r, n_pool,
                    )
                    break

        round_rows = []
        for acc, seq in sorted(protein_seqs.items()):
            shuffled = _shuffle_protein(seq, random_state=random_state + r)
            for pep in sorted(parser.cleave(shuffled, enzyme_rule, missed_cleavages=missed_cleavages)):
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

        if selection_mode == "feature":
            covered_feats.update(round_matched["feature_idx"].unique())
        else:
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

    # --- Step 5: Subsample the collected pool ---
    rng = np.random.default_rng(random_state)

    if selection_mode == "length":
        # Length-stratified subsample. For each target length bin take exactly
        # min(pool_available, n_need) decoys.  No fill from other lengths: adding
        # decoys of the wrong length to compensate for truly unreachable lengths
        # would introduce a length bias worse than the slight T:D count deficit.
        dec_lengths = decoy_pool["peptide"].str.len()
        keep_indices: list[int] = []
        unfilled: list[tuple[int, int, int]] = []  # (length, needed, available)
        for length, tgt_count in sorted(tgt_len_counts.items()):
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
                "(format: length:needed/available) — consider increasing "
                "--max-shuffle-rounds",
                ", ".join(f"{l}:{n}/{a}" for l, n, a in unfilled),
            )

        decoy_pool = decoy_pool.loc[np.sort(keep_indices)].reset_index(drop=True)
        logger.info(
            "balanced_shuffle: subsampled decoy pool %d → %d (length-stratified, no fill)",
            n_pool, len(decoy_pool),
        )
    else:
        # Feature-occupancy-matched selection (paired_shuffle).
        # (1) Pair: for each target-occupied feature, take up to
        #     round(target_ratio * n_targets_at_feature) pool decoys that match the
        #     SAME feature_idx, so target-only features become contested wherever a
        #     decoy exists at that m/z.
        # (2) Top up: draw the remaining shortfall from the rest of the pool
        #     (decoy-only features + surplus contested decoys) to reach the same
        #     global count as length mode, int(target_ratio * n_target).  This keeps
        #     the FDR null mass identical to balanced_shuffle while maximising the
        #     contested fraction first.
        # Selection is keyed only on feature_idx (a mass property); it never reads
        # scores or decoy correctness, so TDC validity is preserved.
        tgt_feat_counts = target_candidates.groupby("feature_idx").size()
        pool_by_feat = decoy_pool.groupby("feature_idx").indices  # {feat_idx: ndarray}

        keep_set: set[int] = set()
        n_residual_target_only = 0
        for feat_id, n_tgt in tgt_feat_counts.items():
            pool_rows = pool_by_feat.get(feat_id)
            if pool_rows is None or len(pool_rows) == 0:
                n_residual_target_only += 1  # no pool decoy at this m/z (unfillable)
                continue
            n_take = min(int(round(target_ratio * n_tgt)), len(pool_rows))
            if n_take >= len(pool_rows):
                chosen = np.asarray(pool_rows)
            else:
                chosen = rng.choice(pool_rows, size=n_take, replace=False)
            keep_set.update(int(i) for i in chosen)

        n_contested_decoys = len(keep_set)

        # Top up to the global count from the remaining pool rows.
        shortfall = n_decoys_needed - len(keep_set)
        if shortfall > 0:
            remaining = np.array(
                [i for i in range(len(decoy_pool)) if i not in keep_set], dtype=int
            )
            if len(remaining) > 0:
                n_extra = min(shortfall, len(remaining))
                extra = rng.choice(remaining, size=n_extra, replace=False)
                keep_set.update(int(i) for i in extra)

        if n_residual_target_only:
            logger.warning(
                "paired_shuffle: %d target-occupied features have no pool decoy at "
                "their m/z (residual target-only; increase --max-shuffle-rounds to "
                "reduce)",
                n_residual_target_only,
            )

        keep_indices = sorted(keep_set)
        decoy_pool = decoy_pool.iloc[keep_indices].reset_index(drop=True)
        logger.info(
            "paired_shuffle: subsampled decoy pool %d → %d "
            "(feature-paired=%d, topped up=%d, target=%d)",
            n_pool, len(decoy_pool), n_contested_decoys,
            len(decoy_pool) - n_contested_decoys, n_decoys_needed,
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

    # Feature-occupancy diagnostic (emitted in both modes for direct comparison).
    feat_has_tgt = result.groupby("feature_idx")["is_decoy"].agg(lambda s: (~s).any())
    feat_has_dec = result.groupby("feature_idx")["is_decoy"].agg("any")
    n_contested = int((feat_has_tgt & feat_has_dec).sum())
    n_tgt_only = int((feat_has_tgt & ~feat_has_dec).sum())
    n_dec_only = int((~feat_has_tgt & feat_has_dec).sum())
    n_feat_total = int(result["feature_idx"].nunique())
    if n_feat_total:
        logger.info(
            "%s feature occupancy: %d contested (%.1f%%), %d target-only (%.1f%%), "
            "%d decoy-only (%.1f%%) of %d features",
            "paired_shuffle" if selection_mode == "feature" else "balanced_shuffle",
            n_contested, 100 * n_contested / n_feat_total,
            n_tgt_only, 100 * n_tgt_only / n_feat_total,
            n_dec_only, 100 * n_dec_only / n_feat_total,
            n_feat_total,
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
        from msi_picasso.lcms_ids import filter_fasta_to_proteins

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
            for acc, seq in sorted(protein_seqs.items()):
                cleaved = sorted(parser.cleave(
                    seq,
                    parser.expasy_rules.get(enzyme, enzyme),
                    missed_cleavages=missed_cleavages,
                ))
                for pep in cleaved:
                    if min_length <= len(pep) <= max_length:
                        rows.append((pep, acc, False))

                if generate_decoys:
                    decoy_seq = _shuffle_protein(seq)
                    cleaved_d = sorted(parser.cleave(
                        decoy_seq,
                        parser.expasy_rules.get(enzyme, enzyme),
                        missed_cleavages=missed_cleavages,
                    ))
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
                    raw_decoys = sorted(parser.cleave(
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
