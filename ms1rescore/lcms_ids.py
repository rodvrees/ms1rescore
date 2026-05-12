"""
Parse LC-MS/MS identification results for Strategy C candidate generation.

Supports Percolator output (_proteins.tsv + _peptides.tsv), mzIdentML, and
any format supported by psm_utils.io. Returns a set of identified protein
accessions and a DataFrame of identified peptides for use in
``digest_identified_proteins()``.
"""

import logging
import re
from collections import namedtuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Named tuple returned by parse_lcms_ids()
LCMSIds = namedtuple("LCMSIds", ["proteins", "peptides"])

# Columns always present in the peptide DataFrame
_PEP_COLS = [
    "sequence", "peptidoform", "protein",
    "q_value", "pep", "score", "n_psms",
    "charge", "rt_mean", "lcms_intensity",
]


# ---------------------------------------------------------------------------
# Accession normalisation
# ---------------------------------------------------------------------------

def _normalize_accession(header: str) -> str:
    """
    Normalise common FASTA / Percolator header formats to a bare accession.

    Examples
    --------
    ``sp|P12345|GENE_HUMAN``  → ``P12345``
    ``tr|A0A000|GENE_HUMAN``  → ``A0A000``
    ``P12345 some description`` → ``P12345``
    ``P12345``                 → ``P12345``
    ``>P12345``                → ``P12345``
    """
    header = header.strip().lstrip(">")
    if "|" in header:
        parts = header.split("|")
        if len(parts) >= 2 and parts[0] in ("sp", "tr", "ref"):
            return parts[1]
    return header.split()[0]


# ---------------------------------------------------------------------------
# Percolator peptide string helpers
# ---------------------------------------------------------------------------

def _strip_percolator_peptide(pep_str: str) -> str:
    """
    Strip flanking residues from a Percolator peptide string.

    ``-.PEPTIDEK.-``  → ``PEPTIDEK``
    ``A.PEPTIDEK.R``  → ``PEPTIDEK``
    """
    pep_str = pep_str.strip()
    if "." in pep_str:
        parts = pep_str.split(".")
        if len(parts) >= 3:
            return parts[1]
        if len(parts) == 2:
            return parts[1]
    return pep_str


def _strip_modifications(seq: str) -> str:
    """Remove modification annotations and return bare uppercase amino acids."""
    bare = re.sub(r"\[.*?\]", "", seq)   # [+16], [U:Oxidation], etc.
    bare = re.sub(r"\(.*?\)", "", bare)  # (ox), (cam), etc.
    return bare.upper()


def _bare_sequence(pep_str: str) -> str:
    """Strip flanks and modifications from any common peptide string format."""
    return _strip_modifications(_strip_percolator_peptide(pep_str))


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def parse_lcms_ids(
    proteins_path: str | None = None,
    peptides_path: str | None = None,
    psms_path: str | None = None,
    protein_fdr: float = 0.01,
    peptide_fdr: float = 0.01,
    format: str = "percolator",
    psm_utils_reader: str | None = None,
) -> LCMSIds:
    """
    Parse LC-MS/MS identification results.

    Parameters
    ----------
    proteins_path
        Path to protein-level results (Percolator ``_proteins.tsv``,
        mzIdentML, etc.). May be None — proteins will be derived from the
        peptide table in that case.
    peptides_path
        Path to peptide-level results. For ``format="msf"``, pass the
        ``.msf`` file path here (or via the dedicated ``msf_path`` argument
        in :func:`rescore`).
    psms_path
        Optional PSM-level file (Percolator ``_psms.tsv``) used to compute
        ``rt_mean``, ``lcms_intensity``, and ``charge`` via aggregation.
        Ignored for ``"msf"`` and ``"mzidentml"`` formats.
    protein_fdr
        Protein FDR threshold (default 0.01).
    peptide_fdr
        Peptide FDR threshold (default 0.01).
    format
        Input format: ``"percolator"``, ``"mzidentml"``, ``"psm_utils"``, or
        ``"msf"`` (ProteomeDiscoverer ``.msf`` SQLite database).
    psm_utils_reader
        Only used when ``format="psm_utils"``. Either a psm_utils filetype
        key (e.g. ``"maxquant"``, ``"tsv"``) or a reader class name (e.g.
        ``"MSMSReader"``, ``"TSVReader"``). When ``None``, auto-detection
        from filename is attempted.

    Returns
    -------
    LCMSIds(proteins=set[str], peptides=pd.DataFrame)
        ``proteins``: set of accessions passing protein FDR.
        ``peptides``: DataFrame with columns defined by ``_PEP_COLS``.
    """
    if format == "percolator":
        return _parse_percolator(
            proteins_path, peptides_path, psms_path, protein_fdr, peptide_fdr
        )
    elif format == "mzidentml":
        return _parse_mzidentml(
            proteins_path or peptides_path, protein_fdr, peptide_fdr
        )
    elif format == "psm_utils":
        return _parse_psm_utils(
            peptides_path, protein_fdr, peptide_fdr, psm_utils_reader
        )
    elif format == "msf":
        return _parse_msf(peptides_path, protein_fdr, peptide_fdr)
    else:
        raise ValueError(
            f"Unknown format {format!r}. "
            f"Choose 'percolator', 'mzidentml', 'psm_utils', or 'msf'."
        )


# ---------------------------------------------------------------------------
# Percolator parser
# ---------------------------------------------------------------------------

def _find_col(df: pd.DataFrame, *candidates: str) -> str | None:
    """Return the first column name that matches one of the lowercase candidates."""
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in lower:
            return lower[cand]
    # Partial match
    for cand in candidates:
        for col_lower, col_orig in lower.items():
            if cand in col_lower:
                return col_orig
    return None


def _parse_percolator(
    proteins_path: str | None,
    peptides_path: str | None,
    psms_path: str | None,
    protein_fdr: float,
    peptide_fdr: float,
) -> LCMSIds:
    """Parse Percolator _proteins.tsv and _peptides.tsv (+ optional _psms.tsv)."""

    # --- Proteins ---
    proteins_set: set[str] = set()
    if proteins_path is not None:
        try:
            prot_df = pd.read_csv(proteins_path, sep="\t")
            qcol = _find_col(prot_df, "q-value", "qvalue", "q_value")
            idcol = _find_col(prot_df, "proteinid", "protein_id", "proteinids", "protein")
            if qcol and idcol:
                passing = prot_df[prot_df[qcol].astype(float) <= protein_fdr]
                for raw_acc in passing[idcol].dropna():
                    for acc in str(raw_acc).split(","):
                        proteins_set.add(_normalize_accession(acc.strip()))
                logger.info(
                    f"  Proteins at {protein_fdr*100:.0f}% FDR: {len(proteins_set)}"
                )
            else:
                logger.warning(
                    f"Could not find q-value / protein-id columns in {proteins_path}. "
                    f"Columns found: {list(prot_df.columns)}"
                )
        except Exception as exc:
            logger.warning(f"Could not parse proteins file {proteins_path}: {exc}")

    if peptides_path is None:
        raise ValueError("peptides_path is required for Percolator format")

    pep_df = pd.read_csv(peptides_path, sep="\t")

    qcol = _find_col(pep_df, "q-value", "qvalue", "q_value")
    seqcol = _find_col(pep_df, "peptide", "sequence", "peptidoform")
    scorecol = _find_col(pep_df, "score", "percolator score")
    pepcol = _find_col(pep_df, "posterior_error_prob", "pep", "posterior")
    protcol = _find_col(pep_df, "proteinids", "protein_ids", "proteins", "protein")

    if qcol is None:
        raise ValueError(
            f"Could not find q-value column in {peptides_path}. "
            f"Columns: {list(pep_df.columns)}"
        )
    if seqcol is None:
        raise ValueError(
            f"Could not find peptide sequence column in {peptides_path}. "
            f"Columns: {list(pep_df.columns)}"
        )

    pep_df = pep_df[pep_df[qcol].astype(float) <= peptide_fdr].copy()

    pep_df["sequence"] = pep_df[seqcol].apply(_bare_sequence)
    pep_df["peptidoform"] = pep_df[seqcol].apply(_strip_percolator_peptide)
    pep_df["q_value"] = pep_df[qcol].astype(float)
    pep_df["score"] = pep_df[scorecol].astype(float) if scorecol else np.nan
    pep_df["pep"] = pep_df[pepcol].astype(float) if pepcol else np.nan

    if protcol:
        pep_df["protein"] = pep_df[protcol].apply(
            lambda x: _normalize_accession(str(x).split(",")[0].strip())
        )
    else:
        pep_df["protein"] = ""

    pep_df["n_psms"] = 1
    pep_df["charge"] = np.nan
    pep_df["rt_mean"] = np.nan
    pep_df["lcms_intensity"] = np.nan

    # Derive protein set from peptides if no proteins file provided
    if not proteins_set:
        for acc in pep_df["protein"].unique():
            if acc:
                proteins_set.add(acc)
        logger.info(
            f"  Derived {len(proteins_set)} unique proteins from peptide table"
        )

    # Deduplicate by sequence (keep best q-value)
    pep_df = (
        pep_df.sort_values("q_value")
        .drop_duplicates(subset="sequence", keep="first")
        .reset_index(drop=True)
    )

    # --- Optional PSM-level join ---
    if psms_path is not None:
        pep_df = _join_psm_rt_intensity(pep_df, psms_path)

    logger.info(
        f"  Peptides at {peptide_fdr*100:.0f}% FDR: {len(pep_df)} unique sequences"
    )
    return LCMSIds(proteins=proteins_set, peptides=pep_df[_PEP_COLS])


def _join_psm_rt_intensity(pep_df: pd.DataFrame, psms_path: str) -> pd.DataFrame:
    """Aggregate PSM-level RT and intensity and left-join onto the peptide DF."""
    try:
        psm_df = pd.read_csv(psms_path, sep="\t")
        seqcol = _find_col(psm_df, "peptide", "sequence")
        rtcol = _find_col(psm_df, "rt", "retention_time", "retentiontime")
        intcol = _find_col(psm_df, "intensity", "ms1_intensity", "precursorintensity")
        chargecol = _find_col(psm_df, "charge", "precursor_charge")

        if seqcol is None:
            logger.warning(f"No sequence column in {psms_path} — skipping RT/intensity join")
            return pep_df

        psm_df["_seq"] = psm_df[seqcol].apply(_bare_sequence)

        agg: dict = {}
        if rtcol:
            agg["rt_mean"] = pd.NamedAgg(column=rtcol, aggfunc="mean")
        if intcol:
            agg["lcms_intensity"] = pd.NamedAgg(column=intcol, aggfunc="sum")
        if chargecol:
            agg["charge"] = pd.NamedAgg(
                column=chargecol,
                aggfunc=lambda x: x.mode().iloc[0] if len(x) > 0 else np.nan,
            )
        agg["n_psms"] = pd.NamedAgg(column="_seq", aggfunc="count")

        psm_agg = psm_df.groupby("_seq").agg(**agg).reset_index()
        psm_agg = psm_agg.rename(columns={"_seq": "sequence"})

        pep_df = pep_df.merge(psm_agg, on="sequence", how="left", suffixes=("", "_psm"))
        for col in ["rt_mean", "lcms_intensity", "charge", "n_psms"]:
            psm_col = f"{col}_psm"
            if psm_col in pep_df.columns:
                mask = pep_df[col].isna()
                pep_df.loc[mask, col] = pep_df.loc[mask, psm_col]
                pep_df.drop(columns=[psm_col], inplace=True)

    except Exception as exc:
        logger.warning(f"Could not parse PSM file {psms_path}: {exc}")

    return pep_df


# ---------------------------------------------------------------------------
# mzIdentML parser
# ---------------------------------------------------------------------------

def _parse_mzidentml(path: str, protein_fdr: float, peptide_fdr: float) -> LCMSIds:
    """Parse mzIdentML via pyteomics."""
    try:
        from pyteomics import mzid as pyteomics_mzid
    except ImportError as exc:
        raise ImportError(
            "pyteomics is required for mzIdentML parsing. "
            "Install with: pip install pyteomics"
        ) from exc

    records = []
    with pyteomics_mzid.MzIdentML(path) as mzid:
        for sir in mzid:
            rt = float(sir.get("retentionTime", np.nan))
            for sii in sir.get("SpectrumIdentificationItem", []):
                qval = np.nan
                pep_val = np.nan
                for cv in sii.get("cvParam", []):
                    acc = cv.get("accession", "")
                    val = cv.get("value", np.nan)
                    if acc == "MS:1002354":      # PSM q-value
                        qval = float(val)
                    elif acc == "MS:1002356":    # PSM PEP
                        pep_val = float(val)

                for pe in sii.get("PeptideEvidenceRef", []):
                    seq = pe.get("peptide_ref", {}).get("PeptideSequence", "")
                    db = pe.get("dBSequence_ref", {})
                    prot = _normalize_accession(db.get("accession", ""))
                    records.append({
                        "sequence": _strip_modifications(seq),
                        "peptidoform": seq,
                        "protein": prot,
                        "q_value": qval,
                        "pep": pep_val,
                        "score": float(sii.get("score", np.nan)
                                       if isinstance(sii.get("score"), (int, float)) else np.nan),
                        "charge": int(sii.get("chargeState", 0)) or np.nan,
                        "rt": rt,
                    })

    if not records:
        return LCMSIds(proteins=set(), peptides=pd.DataFrame(columns=_PEP_COLS))

    df = pd.DataFrame(records)
    pep_agg = (
        df.sort_values("q_value")
        .groupby("sequence")
        .agg(
            peptidoform=("peptidoform", "first"),
            protein=("protein", "first"),
            q_value=("q_value", "min"),
            pep=("pep", "min"),
            score=("score", "max"),
            n_psms=("sequence", "count"),
            charge=("charge", lambda x: x.mode().iloc[0] if len(x) > 0 else np.nan),
            rt_mean=("rt", "mean"),
        )
        .reset_index()
    )
    pep_agg["lcms_intensity"] = np.nan
    pep_df = pep_agg[pep_agg["q_value"] <= peptide_fdr].copy()

    proteins_set = set(pep_df["protein"].unique())
    logger.info(
        f"  mzIdentML: {len(proteins_set)} proteins, {len(pep_df)} peptides at FDR thresholds"
    )
    return LCMSIds(proteins=proteins_set, peptides=pep_df[_PEP_COLS].reset_index(drop=True))


# ---------------------------------------------------------------------------
# psm_utils parser
# ---------------------------------------------------------------------------

def _parse_psm_utils(
    path: str,
    protein_fdr: float,
    peptide_fdr: float,
    psm_utils_reader: str | None = None,
) -> LCMSIds:
    """Parse any psm_utils-supported format."""
    try:
        from psm_utils.io import READERS, read_file
    except ImportError as exc:
        raise ImportError(
            "psm_utils is required for format='psm_utils'. "
            "Install with: pip install psm_utils"
        ) from exc

    # Resolve reader: accept either a filetype key ("tsv") or class name ("TSVReader")
    filetype = "infer"
    if psm_utils_reader is not None:
        cls_to_key = {cls.__name__: key for key, cls in READERS.items()}
        if psm_utils_reader in READERS:
            filetype = psm_utils_reader
        elif psm_utils_reader in cls_to_key:
            filetype = cls_to_key[psm_utils_reader]
        else:
            available = sorted(set(list(READERS.keys()) + list(cls_to_key.keys())))
            raise ValueError(
                f"Unknown psm_utils reader {psm_utils_reader!r}. "
                f"Available filetype keys and class names: {available}"
            )

    psm_list = read_file(path, filetype=filetype)

    records = []
    seq_proteins: dict[str, list[str]] = {}  # sequence → all normalised protein accessions
    for psm in psm_list:
        try:
            if psm.peptidoform is None:
                continue
            sequence = psm.peptidoform.sequence
            prot_first = (
                _normalize_accession(psm.protein_list[0]) if psm.protein_list else ""
            )
            if psm.protein_list:
                bucket = seq_proteins.setdefault(sequence, [])
                for acc in psm.protein_list:
                    norm = _normalize_accession(acc)
                    if norm and norm not in bucket:
                        bucket.append(norm)

            charge = psm.get_precursor_charge()
            intensity = np.nan
            if psm.metadata and "precursor_intensity" in psm.metadata:
                try:
                    intensity = float(psm.metadata["precursor_intensity"])
                except (TypeError, ValueError):
                    pass

            records.append({
                "sequence": sequence,
                "peptidoform": str(psm.peptidoform),
                "protein": prot_first,
                "q_value": float(psm.qvalue) if psm.qvalue is not None else np.nan,
                "pep": float(psm.pep) if psm.pep is not None else np.nan,
                "score": float(psm.score) if psm.score is not None else np.nan,
                "charge": int(charge) if charge is not None else np.nan,
                "rt": float(psm.retention_time) if psm.retention_time is not None else np.nan,
                "lcms_intensity": intensity,
            })
        except Exception:
            continue

    if not records:
        return LCMSIds(proteins=set(), peptides=pd.DataFrame(columns=_PEP_COLS))

    df = pd.DataFrame(records)

    pep_agg = (
        df.sort_values("q_value")
        .groupby("sequence")
        .agg(
            peptidoform=("peptidoform", "first"),
            protein=("protein", "first"),
            q_value=("q_value", "min"),
            pep=("pep", "min"),
            score=("score", "max"),
            n_psms=("sequence", "count"),
            charge=("charge", lambda x: x.mode().iloc[0] if len(x) > 0 else np.nan),
            rt_mean=("rt", "mean"),
            lcms_intensity=("lcms_intensity", "max"),
        )
        .reset_index()
    )

    # When q_value is absent for all PSMs (raw search output), skip FDR filter
    all_nan_q = pep_agg["q_value"].isna().all()
    if all_nan_q:
        logger.warning(
            "No q_value available for any PSM — skipping FDR filter and returning all peptides."
        )
        pep_df = pep_agg.copy()
    else:
        pep_df = pep_agg[pep_agg["q_value"] <= peptide_fdr].copy()
        if pep_df.empty:
            raise ValueError(
                f"No peptides remain after FDR filter (q_value <= {peptide_fdr}). "
                f"Check that your input file contains FDR-controlled identifications."
            )

    # Collect all protein accessions for sequences that pass FDR
    passing_seqs = set(pep_df["sequence"])
    proteins_set: set[str] = set()
    for seq, accs in seq_proteins.items():
        if seq in passing_seqs:
            proteins_set.update(accs)

    logger.info(
        f"  psm_utils: {len(proteins_set)} proteins, {len(pep_df)} peptides "
        f"at {peptide_fdr*100:.0f}% FDR"
    )
    return LCMSIds(
        proteins=proteins_set,
        peptides=pep_df[_PEP_COLS].reset_index(drop=True),
    )


# ---------------------------------------------------------------------------
# ProteomeDiscoverer .msf parser
# ---------------------------------------------------------------------------

def _parse_msf(msf_path: str, protein_fdr: float, peptide_fdr: float) -> LCMSIds:
    """
    Parse peptide identifications from a ProteomeDiscoverer ``.msf`` SQLite database.

    Queries ``TargetPsms`` (filtered by ``PercolatorqValue <= peptide_fdr``) joined
    with ``TargetProteins`` for accessions, then aggregates to peptide level.
    The protein set is derived from peptides passing the FDR threshold; PD does not
    store a separate protein-level q-value in the ``.msf`` format.
    ``protein_fdr`` is accepted for API consistency but is not applied separately.
    """
    import sqlite3

    if msf_path is None:
        raise ValueError("msf_path must be provided for format='msf'")

    conn = sqlite3.connect(msf_path)
    try:
        df = pd.read_sql_query(
            """
            SELECT
                tp.Sequence              AS sequence,
                tp.PercolatorqValue      AS q_value,
                tp.PercolatorSVMScore    AS score,
                CAST(tp.Charge AS INTEGER) AS charge,
                tp.RetentionTime         AS rt,
                tprot.Accession          AS protein
            FROM TargetPsms tp
            LEFT JOIN TargetProteinsTargetPsms tptp
                ON tp.PeptideID = tptp.TargetPsmsPeptideID
            LEFT JOIN TargetProteins tprot
                ON tptp.TargetProteinsUniqueSequenceID = tprot.UniqueSequenceID
            WHERE tp.PercolatorqValue <= :fdr
              AND tp.Sequence IS NOT NULL
            """,
            conn,
            params={"fdr": peptide_fdr},
        )
    except Exception as exc:
        logger.error(f"Could not query MSF file {msf_path!r}: {exc}")
        return LCMSIds(proteins=set(), peptides=pd.DataFrame(columns=_PEP_COLS))
    finally:
        conn.close()

    if df.empty:
        logger.warning(
            f"No peptides found at {peptide_fdr*100:.0f}% FDR in {msf_path!r}"
        )
        return LCMSIds(proteins=set(), peptides=pd.DataFrame(columns=_PEP_COLS))

    df["sequence"] = df["sequence"].apply(_strip_modifications)
    df["protein"] = (
        df["protein"].fillna("").apply(lambda x: _normalize_accession(x) if x else "")
    )
    for col in ("q_value", "score", "charge", "rt"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Aggregate PSMs to peptide level
    pep_df = (
        df.sort_values("q_value")
        .groupby("sequence")
        .agg(
            peptidoform=("sequence", "first"),
            protein=("protein", "first"),
            q_value=("q_value", "min"),
            score=("score", "max"),
            n_psms=("sequence", "count"),
            charge=("charge", lambda x: x.mode().iloc[0] if len(x) > 0 else np.nan),
            rt_mean=("rt", "mean"),
        )
        .reset_index()
    )
    # MSF stores no separate PEP column; leave as NaN
    pep_df["pep"] = np.nan
    pep_df["lcms_intensity"] = np.nan

    proteins_set = {acc for acc in pep_df["protein"].unique() if acc}
    logger.info(
        f"  MSF: {len(proteins_set)} proteins, {len(pep_df)} peptides "
        f"at {peptide_fdr*100:.0f}% FDR"
    )
    return LCMSIds(
        proteins=proteins_set, peptides=pep_df[_PEP_COLS].reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# FASTA filtering
# ---------------------------------------------------------------------------

def filter_fasta_to_proteins(fasta_path: str, protein_accessions: set) -> dict[str, str]:
    """
    Read a FASTA file and return ``{accession: sequence}`` for entries whose
    normalised accession is in ``protein_accessions``.

    Logs a warning if fewer than 50% of the requested accessions are found
    (likely accession format mismatch between the LC-MS/MS search DB and the
    FASTA supplied here).

    Parameters
    ----------
    fasta_path
        Path to the FASTA file.
    protein_accessions
        Set of accessions to keep (as returned by ``parse_lcms_ids().proteins``).

    Returns
    -------
    dict mapping accession → protein sequence.
    """
    from pyteomics import fasta

    found: dict[str, str] = {}
    for desc, seq in fasta.read(fasta_path):
        acc = _normalize_accession(desc)
        if acc in protein_accessions:
            found[acc] = seq

    n_found = len(found)
    n_wanted = len(protein_accessions)
    pct = n_found / max(n_wanted, 1) * 100.0

    if pct < 50.0:
        example_wanted = next(iter(protein_accessions), "N/A")
        example_found = next(iter(found), "N/A")
        logger.warning(
            f"Only {n_found}/{n_wanted} ({pct:.0f}%) LC-MS/MS protein accessions were "
            f"found in the FASTA. This likely indicates an accession format mismatch. "
            f"Example wanted: {example_wanted!r}, example found in FASTA: {example_found!r}. "
            f"Check that both use the same UniProt / RefSeq / custom accession format."
        )
    else:
        logger.info(f"  Found {n_found}/{n_wanted} ({pct:.0f}%) proteins in FASTA")

    return found
