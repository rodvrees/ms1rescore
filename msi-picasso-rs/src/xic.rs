/// Batch XIC extraction: for multiple target m/z values across all MS1 scans.
use rayon::prelude::*;

/// Binary search for the first index where arr[idx] >= target.
fn lower_bound(arr: &[f64], target: f64) -> usize {
    arr.partition_point(|&x| x < target)
}

/// Binary search for the first index where arr[idx] > target.
fn upper_bound(arr: &[f64], target: f64) -> usize {
    arr.partition_point(|&x| x <= target)
}

/// Extract XIC for a single target m/z across all MS1 scans.
fn extract_xic_single(
    ms1_rts: &[f64],
    ms1_mz_arrays: &[Vec<f64>],
    ms1_int_arrays: &[Vec<f64>],
    target_mz: f64,
    ppm_tolerance: f64,
) -> (Vec<f64>, Vec<f64>) {
    let tol = target_mz * ppm_tolerance / 1e6;
    let lo_mz = target_mz - tol;
    let hi_mz = target_mz + tol;

    let mut rts = Vec::new();
    let mut intensities = Vec::new();

    for scan_idx in 0..ms1_rts.len() {
        let mz_arr = &ms1_mz_arrays[scan_idx];
        let int_arr = &ms1_int_arrays[scan_idx];

        let lo = lower_bound(mz_arr, lo_mz);
        let hi = upper_bound(mz_arr, hi_mz);

        if lo < hi {
            // Sum all peaks in the window: accumulates signal across ion mobility scans
            // (timsTOF frames contain multiple mobility scans at the same m/z after
            // sorting; summing gives the conventional 2-D LC-MS projection). For
            // mzML centroid data this equals the single peak intensity.
            let total: f64 = int_arr[lo..hi].iter().sum();
            rts.push(ms1_rts[scan_idx]);
            intensities.push(total);
        }
    }

    (rts, intensities)
}

/// Extract XICs for multiple target m/z values in parallel.
/// Returns Vec of (rts, intensities) tuples, one per target m/z.
pub fn extract_xics_batch(
    ms1_rts: &[f64],
    ms1_mz_arrays: &[Vec<f64>],
    ms1_int_arrays: &[Vec<f64>],
    target_mzs: &[f64],
    ppm_tolerance: f64,
) -> Vec<(Vec<f64>, Vec<f64>)> {
    target_mzs
        .par_iter()
        .map(|&mz| extract_xic_single(ms1_rts, ms1_mz_arrays, ms1_int_arrays, mz, ppm_tolerance))
        .collect()
}
