/// Fast peptide mass calculation and m/z matching.
use std::collections::HashMap;

/// Standard amino acid monoisotopic masses (residue masses).
fn amino_acid_masses() -> HashMap<u8, f64> {
    let mut m = HashMap::new();
    m.insert(b'G', 57.02146);
    m.insert(b'A', 71.03711);
    m.insert(b'V', 99.06841);
    m.insert(b'L', 113.08406);
    m.insert(b'I', 113.08406);
    m.insert(b'P', 97.05276);
    m.insert(b'F', 147.06841);
    m.insert(b'W', 186.07931);
    m.insert(b'M', 131.04049);
    m.insert(b'S', 87.03203);
    m.insert(b'T', 101.04768);
    m.insert(b'C', 103.00919);
    m.insert(b'Y', 163.06333);
    m.insert(b'H', 137.05891);
    m.insert(b'D', 115.02694);
    m.insert(b'E', 129.04259);
    m.insert(b'N', 114.04293);
    m.insert(b'Q', 128.05858);
    m.insert(b'K', 128.09496);
    m.insert(b'R', 156.10111);
    m
}

/// Standard amino acid elemental compositions: (C, H, N, O, S)
fn amino_acid_compositions() -> HashMap<u8, (i32, i32, i32, i32, i32)> {
    let mut m = HashMap::new();
    m.insert(b'G', (2, 3, 1, 1, 0));
    m.insert(b'A', (3, 5, 1, 1, 0));
    m.insert(b'V', (5, 9, 1, 1, 0));
    m.insert(b'L', (6, 11, 1, 1, 0));
    m.insert(b'I', (6, 11, 1, 1, 0));
    m.insert(b'P', (5, 7, 1, 1, 0));
    m.insert(b'F', (9, 9, 1, 1, 0));
    m.insert(b'W', (11, 10, 2, 1, 0));
    m.insert(b'M', (5, 9, 1, 1, 1));
    m.insert(b'S', (3, 5, 1, 2, 0));
    m.insert(b'T', (4, 7, 1, 2, 0));
    m.insert(b'C', (3, 5, 1, 1, 1));
    m.insert(b'Y', (9, 9, 1, 2, 0));
    m.insert(b'H', (6, 7, 3, 1, 0));
    m.insert(b'D', (4, 5, 1, 3, 0));
    m.insert(b'E', (5, 7, 1, 3, 0));
    m.insert(b'N', (4, 6, 2, 2, 0));
    m.insert(b'Q', (5, 8, 2, 2, 0));
    m.insert(b'K', (6, 12, 2, 1, 0));
    m.insert(b'R', (6, 12, 4, 1, 0));
    m
}

const WATER_MASS: f64 = 18.010565;  // H2O added to peptide
const PROTON: f64 = 1.007276;

/// Result for a single peptide: mass, mh_mz, and elemental composition.
pub struct PeptideInfo {
    pub mass: f64,
    pub mh_mz: f64,
    pub n_c: i32,
    pub n_h: i32,
    pub n_n: i32,
    pub n_o: i32,
    pub n_s: i32,
}

/// Compute monoisotopic mass and elemental composition for a peptide sequence.
/// Returns None if the sequence contains unknown amino acids.
pub fn compute_peptide_info(sequence: &str) -> Option<PeptideInfo> {
    let aa_masses = amino_acid_masses();
    let aa_comps = amino_acid_compositions();

    let mut mass = WATER_MASS;
    // Water composition: H2O
    let mut n_c: i32 = 0;
    let mut n_h: i32 = 2;
    let mut n_n: i32 = 0;
    let mut n_o: i32 = 1;
    let mut n_s: i32 = 0;

    for &aa in sequence.as_bytes() {
        mass += aa_masses.get(&aa)?;
        let (c, h, n, o, s) = aa_comps.get(&aa)?;
        n_c += c;
        n_h += h;
        n_n += n;
        n_o += o;
        n_s += s;
    }

    Some(PeptideInfo {
        mass,
        mh_mz: mass + PROTON,
        n_c,
        n_h,
        n_n,
        n_o,
        n_s,
    })
}

/// Batch compute peptide info for many sequences.
/// Returns parallel vectors: masses, mh_mzs, n_C, n_H, n_N, n_O, n_S.
/// Sequences with unknown amino acids get mass=0, mh_mz=0, composition=0.
pub fn compute_peptide_info_batch(
    sequences: &[String],
) -> (Vec<f64>, Vec<f64>, Vec<i32>, Vec<i32>, Vec<i32>, Vec<i32>, Vec<i32>) {
    let n = sequences.len();
    let mut masses = vec![0.0f64; n];
    let mut mh_mzs = vec![0.0f64; n];
    let mut n_cs = vec![0i32; n];
    let mut n_hs = vec![0i32; n];
    let mut n_ns = vec![0i32; n];
    let mut n_os = vec![0i32; n];
    let mut n_ss = vec![0i32; n];

    for (i, seq) in sequences.iter().enumerate() {
        if let Some(info) = compute_peptide_info(seq) {
            masses[i] = info.mass;
            mh_mzs[i] = info.mh_mz;
            n_cs[i] = info.n_c;
            n_hs[i] = info.n_h;
            n_ns[i] = info.n_n;
            n_os[i] = info.n_o;
            n_ss[i] = info.n_s;
        }
    }

    (masses, mh_mzs, n_cs, n_hs, n_ns, n_os, n_ss)
}

/// Match peptide m/z values against MALDI feature m/z values within ppm tolerance.
/// Returns vectors of (feature_index, peptide_index, ppm_error) for all matches.
pub fn match_mz_batch(
    maldi_mzs: &[f64],
    peptide_mzs: &[f64],
    ppm_tolerance: f64,
) -> (Vec<u32>, Vec<u32>, Vec<f64>) {
    // Sort peptide m/z for binary search
    let mut sorted_indices: Vec<usize> = (0..peptide_mzs.len()).collect();
    sorted_indices.sort_by(|&a, &b| peptide_mzs[a].partial_cmp(&peptide_mzs[b]).unwrap());
    let sorted_mzs: Vec<f64> = sorted_indices.iter().map(|&i| peptide_mzs[i]).collect();

    let mut feat_indices = Vec::new();
    let mut pep_indices = Vec::new();
    let mut ppm_errors = Vec::new();

    for (feat_idx, &maldi_mz) in maldi_mzs.iter().enumerate() {
        let tol = maldi_mz * ppm_tolerance / 1e6;
        let lo = sorted_mzs.partition_point(|&x| x < maldi_mz - tol);
        let hi = sorted_mzs.partition_point(|&x| x <= maldi_mz + tol);

        for j in lo..hi {
            let pep_idx = sorted_indices[j];
            let ppm = (maldi_mz - peptide_mzs[pep_idx]) / peptide_mzs[pep_idx] * 1e6;
            feat_indices.push(feat_idx as u32);
            pep_indices.push(pep_idx as u32);
            ppm_errors.push(ppm);
        }
    }

    (feat_indices, pep_indices, ppm_errors)
}
