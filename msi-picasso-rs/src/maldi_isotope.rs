/// Compute mean intensities at a set of target m/z values across all MALDI pixels.
///
/// Input uses CSR (compressed sparse row) layout to avoid per-pixel Vec allocation:
///   flat_mzs[pixel_offsets[px]..pixel_offsets[px+1]]  -- sorted m/z for pixel px
///   flat_ints[pixel_offsets[px]..pixel_offsets[px+1]] -- matching intensities
///
/// The algorithm is O(n_peaks + n_targets) per pixel (amortised) using a
/// two-pointer sweep, since both the pixel m/z array and target_mzs are sorted.
/// Rayon parallelises over pixels; results are reduced by summation then divided
/// by n_pixels to yield per-target means.
use rayon::prelude::*;

pub fn compute_isotope_means_flat(
    flat_mzs: &[f64],
    flat_ints: &[f32],
    pixel_offsets: &[usize],
    target_mzs: &[f64],
    ppm_tolerance: f64,
) -> Vec<f64> {
    let n_pixels = pixel_offsets.len().saturating_sub(1);
    let n_targets = target_mzs.len();

    if n_pixels == 0 || n_targets == 0 {
        return vec![0.0; n_targets];
    }

    // Sort targets while remembering original positions so the output is in the
    // original (unsorted) order.
    let mut sorted_pairs: Vec<(usize, f64)> = target_mzs.iter().cloned().enumerate().collect();
    sorted_pairs
        .sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    let sorted_mzs: Vec<f64> = sorted_pairs.iter().map(|(_, mz)| *mz).collect();
    let orig_idx: Vec<usize> = sorted_pairs.iter().map(|(i, _)| *i).collect();

    // Parallel map: each pixel produces a per-target sum vector.
    // Reduce: element-wise addition across pixels.
    let sums: Vec<f64> = (0..n_pixels)
        .into_par_iter()
        .map(|px| {
            let start = pixel_offsets[px];
            let end = pixel_offsets[px + 1];
            let mz_arr = &flat_mzs[start..end];
            let int_arr = &flat_ints[start..end];

            let mut pixel_sums = vec![0.0f64; n_targets];
            if mz_arr.is_empty() {
                return pixel_sums;
            }

            // Two-pointer sweep over sorted targets.
            // pi only advances forward (amortised O(n_peaks) total across all targets).
            let mut pi = 0usize;

            for ti in 0..sorted_mzs.len() {
                let tmz = sorted_mzs[ti];
                let tol = tmz * ppm_tolerance * 1e-6;
                let lo_mz = tmz - tol;
                let hi_mz = tmz + tol;

                // Advance pi to the first peak that could fall inside this window.
                while pi < mz_arr.len() && mz_arr[pi] < lo_mz {
                    pi += 1;
                }

                // Sum all peaks in the window (matches the RAM path which uses
                // np.bincount weighted sum via _extract_centroid_fast).
                let mut window_sum = 0.0f64;
                let mut scan = pi;
                while scan < mz_arr.len() && mz_arr[scan] <= hi_mz {
                    window_sum += int_arr[scan] as f64;
                    scan += 1;
                }

                pixel_sums[orig_idx[ti]] = window_sum;
            }

            pixel_sums
        })
        .reduce(
            || vec![0.0f64; n_targets],
            |mut a, b| {
                for i in 0..a.len() {
                    a[i] += b[i];
                }
                a
            },
        );

    // Divide accumulated sums by the number of pixels to get means.
    sums.into_iter().map(|s| s / n_pixels as f64).collect()
}
