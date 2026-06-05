/// Per-sequence peptide feature computation (ionization, physicochemistry, pI).
///
/// Implements the same logic as `maldi_features.py` but in parallel Rust,
/// replacing Python loops over ~395K unique sequences.
use rayon::prelude::*;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Kyte-Doolittle hydropathy scale (matches Python _KD_SCALE).
#[inline(always)]
fn kd_value(aa: u8) -> f64 {
    match aa {
        b'A' =>  1.8, b'R' => -4.5, b'N' => -3.5, b'D' => -3.5, b'C' =>  2.5,
        b'Q' => -3.5, b'E' => -3.5, b'G' => -0.4, b'H' => -3.2, b'I' =>  4.5,
        b'L' =>  3.8, b'K' => -3.9, b'M' =>  1.9, b'F' =>  2.8, b'P' => -1.6,
        b'S' => -0.8, b'T' => -0.7, b'W' => -0.9, b'Y' => -1.3, b'V' =>  4.2,
        _    =>  0.0,
    }
}

/// Lehninger pKa values — must match Python _PKA exactly.
const PKA_NTERM: f64 = 8.0;
const PKA_CTERM: f64 = 3.1;
const PKA_D: f64 = 3.9;
const PKA_E: f64 = 4.1;
const PKA_H: f64 = 6.0;
const PKA_C: f64 = 8.3;
const PKA_Y: f64 = 10.1;
const PKA_K: f64 = 10.5;
const PKA_R: f64 = 12.5;

// ---------------------------------------------------------------------------
// Per-sequence helpers
// ---------------------------------------------------------------------------

/// pI by bisection (50 iterations, Lehninger pKa).
///
/// Reformulated with x = 10^ph so only one powf call per iteration
/// instead of 9, matching `_compute_pi_batch` in Python.
#[inline]
fn compute_pi(nd: f64, ne: f64, nh: f64, nc: f64, ny: f64, nk: f64, nr: f64) -> f64 {
    // Precompute 10^pKa scalars once per sequence.
    let pk_nt = 10f64.powf(PKA_NTERM);
    let pk_ct = 10f64.powf(PKA_CTERM);
    let pk_d  = 10f64.powf(PKA_D);
    let pk_e  = 10f64.powf(PKA_E);
    let pk_h  = 10f64.powf(PKA_H);
    let pk_c  = 10f64.powf(PKA_C);
    let pk_y  = 10f64.powf(PKA_Y);
    let pk_k  = 10f64.powf(PKA_K);
    let pk_r  = 10f64.powf(PKA_R);

    let mut lo = 0.0f64;
    let mut hi = 14.0f64;
    for _ in 0..50 {
        let mid = (lo + hi) * 0.5;
        let x = 10f64.powf(mid);   // 1 powf per iteration
        let q = pk_nt / (pk_nt + x)
            - x / (x + pk_ct)
            - nd * x / (x + pk_d)
            - ne * x / (x + pk_e)
            - nc * x / (x + pk_c)
            - ny * x / (x + pk_y)
            + nh * pk_h / (pk_h + x)
            + nk * pk_k / (pk_k + x)
            + nr * pk_r / (pk_r + x);
        if q > 0.0 { lo = mid; } else { hi = mid; }
    }
    (lo + hi) * 0.5
}

/// Missed cleavages: K/R not followed by P, excluding the last residue.
#[inline]
fn count_missed_cleavages(bytes: &[u8]) -> i32 {
    if bytes.len() < 2 { return 0; }
    let mut n = 0i32;
    for i in 0..bytes.len() - 1 {
        if (bytes[i] == b'K' || bytes[i] == b'R') && bytes[i + 1] != b'P' {
            n += 1;
        }
    }
    n
}

// ---------------------------------------------------------------------------
// Batch functions exposed to Python
// ---------------------------------------------------------------------------

/// Ionization features: (n_R, n_K, n_H, n_F, n_W, n_Y, gravy, peptide_pi).
///
/// All sequences processed in parallel. Returns 8 arrays aligned with input.
pub fn ionization_features_batch(
    sequences: &[String],
) -> (Vec<i32>, Vec<i32>, Vec<i32>, Vec<i32>, Vec<i32>, Vec<i32>, Vec<f64>, Vec<f64>) {
    let results: Vec<_> = sequences
        .par_iter()
        .map(|seq| {
            let bytes = seq.as_bytes();
            let mut nr = 0i32; let mut nk = 0i32; let mut nh = 0i32;
            let mut nf = 0i32; let mut nw = 0i32; let mut ny = 0i32;
            let mut nd = 0i32; let mut ne = 0i32; let mut nc = 0i32;
            let mut kd_sum = 0.0f64;
            for &aa in bytes {
                match aa {
                    b'R' => nr += 1, b'K' => nk += 1, b'H' => nh += 1,
                    b'F' => nf += 1, b'W' => nw += 1, b'Y' => ny += 1,
                    b'D' => nd += 1, b'E' => ne += 1, b'C' => nc += 1,
                    _ => {}
                }
                kd_sum += kd_value(aa);
            }
            let gravy = if bytes.is_empty() { 0.0 } else { kd_sum / bytes.len() as f64 };
            let pi = compute_pi(nd as f64, ne as f64, nh as f64, nc as f64,
                                ny as f64, nk as f64, nr as f64);
            (nr, nk, nh, nf, nw, ny, gravy, pi)
        })
        .collect();

    let mut n_r = Vec::with_capacity(results.len()); let mut n_k = Vec::with_capacity(results.len());
    let mut n_h = Vec::with_capacity(results.len()); let mut n_f = Vec::with_capacity(results.len());
    let mut n_w = Vec::with_capacity(results.len()); let mut n_y = Vec::with_capacity(results.len());
    let mut gravy   = Vec::with_capacity(results.len());
    let mut pi_vals = Vec::with_capacity(results.len());
    for (nr, nk, nh, nf, nw, ny, gv, pi) in results {
        n_r.push(nr); n_k.push(nk); n_h.push(nh);
        n_f.push(nf); n_w.push(nw); n_y.push(ny);
        gravy.push(gv); pi_vals.push(pi);
    }
    (n_r, n_k, n_h, n_f, n_w, n_y, gravy, pi_vals)
}

/// Property features: (n_D, n_E, n_C, n_P, n_M, n_W, n_Y, seq_len, nterm_code, peptide_pi).
///
/// nterm_code: ASCII code of first residue as i32 (0 for empty sequence).
/// All sequences processed in parallel. Returns 10 arrays aligned with input.
pub fn property_features_batch(
    sequences: &[String],
) -> (Vec<i32>, Vec<i32>, Vec<i32>, Vec<i32>, Vec<i32>, Vec<i32>, Vec<i32>, Vec<i32>, Vec<i32>, Vec<f64>) {
    let results: Vec<_> = sequences
        .par_iter()
        .map(|seq| {
            let bytes = seq.as_bytes();
            let mut nd = 0i32; let mut ne = 0i32; let mut nc = 0i32;
            let mut np = 0i32; let mut nm = 0i32; let mut nw = 0i32;
            let mut nh = 0i32; let mut ny = 0i32;
            let mut nk = 0i32; let mut nr = 0i32;
            for &aa in bytes {
                match aa {
                    b'D' => nd += 1, b'E' => ne += 1, b'C' => nc += 1,
                    b'P' => np += 1, b'M' => nm += 1, b'W' => nw += 1,
                    b'H' => nh += 1, b'Y' => ny += 1,
                    b'K' => nk += 1, b'R' => nr += 1,
                    _ => {}
                }
            }
            let pi = compute_pi(nd as f64, ne as f64, nh as f64, nc as f64,
                                ny as f64, nk as f64, nr as f64);
            let nterm = bytes.first().copied().unwrap_or(0) as i32;
            (nd, ne, nc, np, nm, nw, ny, bytes.len() as i32, nterm, pi)
        })
        .collect();

    let cap = results.len();
    let mut n_d = Vec::with_capacity(cap); let mut n_e = Vec::with_capacity(cap);
    let mut n_c = Vec::with_capacity(cap); let mut n_p = Vec::with_capacity(cap);
    let mut n_m = Vec::with_capacity(cap); let mut n_w = Vec::with_capacity(cap);
    let mut n_y = Vec::with_capacity(cap); let mut slen = Vec::with_capacity(cap);
    let mut nt  = Vec::with_capacity(cap); let mut pi_v = Vec::with_capacity(cap);
    for (nd, ne, nc, np, nm, nw, ny, sl, nterm, pi) in results {
        n_d.push(nd); n_e.push(ne); n_c.push(nc); n_p.push(np); n_m.push(nm);
        n_w.push(nw); n_y.push(ny); slen.push(sl); nt.push(nterm); pi_v.push(pi);
    }
    (n_d, n_e, n_c, n_p, n_m, n_w, n_y, slen, nt, pi_v)
}

/// Missed cleavage count for each sequence in a list (parallel).
pub fn missed_cleavages_batch(sequences: &[String]) -> Vec<i32> {
    sequences
        .par_iter()
        .map(|s| count_missed_cleavages(s.as_bytes()))
        .collect()
}
