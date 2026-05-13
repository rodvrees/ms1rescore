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
    missed_cleavages: int = 2,
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

    # Candidates per feature
    result["n_candidates"] = result.groupby("feature_mz")["feature_mz"].transform("count")

    # A10 — Kendrick mass defect (CH₂ reference unit: 14 / 14.01565)
    kendrick_mass = result["feature_mz"] * (14.0 / 14.01565)
    result["kendrick_mass_defect"] = np.floor(kendrick_mass) - kendrick_mass

    logger.info(
        f"Matched {result['feature_mz'].nunique()}/{len(maldi_mzs)} features → "
        f"{(~result['is_decoy']).sum()} target + {result['is_decoy'].sum()} decoy candidates"
    )
    return result


def digest_identified_proteins(
    fasta_path: str,
    lcms_ids,
    enzyme: str = "trypsin",
    missed_cleavages: int = 2,
    min_length: int = 7,
    max_length: int = 30,
) -> pd.DataFrame:
    """
    Strategy C hybrid candidate generation.

    Builds a candidate DataFrame from:
    1. In-silico digest of LC-MS/MS-identified proteins (target + K/R-preserving
       shuffled decoy, via ``_shuffle_protein``).
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
        Path to the protein FASTA file.
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
    Returns an empty DataFrame (with those columns) if no identified
    proteins are found in the FASTA.
    """
    from ms1rescore.lcms_ids import filter_fasta_to_proteins

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
    _EMPTY_COLS = [
        "peptide", "protein", "is_decoy", "mass", "mh_mz",
        "n_C", "n_H", "n_N", "n_O", "n_S", "source",
    ] + list(_EV_COLS.values())

    # --- Step 1: Filter FASTA to identified proteins ---
    protein_seqs = filter_fasta_to_proteins(fasta_path, lcms_ids.proteins)

    if not protein_seqs:
        logger.warning(
            "No identified proteins found in FASTA — check accession format. "
            "Returning empty DataFrame."
        )
        return pd.DataFrame(columns=_EMPTY_COLS)

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

    # --- Step 3: Compute masses (Rust if available, else pyteomics) ---
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

    df = df[df["mass"] > 0].reset_index(drop=True)

    # --- Step 4: Label source ---
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
        novel_seqs = [s for s in confirmed_seqs if s not in existing_targets]

        if novel_seqs:
            logger.info(
                f"  {len(novel_seqs)} LC-MS/MS-confirmed peptides not in protein digest "
                f"— adding as lcms_confirmed targets"
            )
            novel_rows = []
            for seq in novel_seqs:
                prot_row = lcms_pep_df[lcms_pep_df["sequence"] == seq].iloc[0]
                prot = str(prot_row.get("protein", ""))
                novel_rows.append((seq, prot, False))
                # K/R-preserving peptide-level decoy
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
