/// Batch MS1 isotope envelope extraction at specific scans.

const NEUTRON: f64 = 1.003355;

/// Extract isotope envelope [M0, M+1, ..., M+n_peaks-1] from an MS1 scan.
/// Returns normalized intensities.
fn extract_envelope_single(
    mz_arr: &[f64],
    int_arr: &[f64],
    target_mz: f64,
    charge: usize,
    n_peaks: usize,
    ppm_tolerance: f64,
) -> Vec<f64> {
    let spacing = NEUTRON / charge as f64;
    let mut intensities = vec![0.0f64; n_peaks];

    for k in 0..n_peaks {
        let expected_mz = target_mz + k as f64 * spacing;
        let tol = expected_mz * ppm_tolerance / 1e6;

        let lo = mz_arr.partition_point(|&x| x < expected_mz - tol);
        let hi = mz_arr.partition_point(|&x| x <= expected_mz + tol);

        if lo < hi {
            // Closest to expected m/z
            let mut best_dist = f64::MAX;
            let mut best_int = 0.0f64;
            for i in lo..hi {
                let dist = (mz_arr[i] - expected_mz).abs();
                if dist < best_dist {
                    best_dist = dist;
                    best_int = int_arr[i];
                }
            }
            intensities[k] = best_int;
        }
    }

    // Normalize
    let total: f64 = intensities.iter().sum();
    if total > 0.0 {
        for v in &mut intensities {
            *v /= total;
        }
    }

    intensities
}

/// Batch extract isotope envelopes at specific scans for specific m/z values.
pub fn extract_envelopes_batch(
    ms1_mz_arrays: &[Vec<f64>],
    ms1_int_arrays: &[Vec<f64>],
    scan_indices: &[usize],
    target_mzs: &[f64],
    charge: usize,
    n_peaks: usize,
    ppm_tolerance: f64,
) -> Vec<Vec<f64>> {
    scan_indices
        .iter()
        .zip(target_mzs.iter())
        .map(|(&scan_idx, &mz)| {
            if scan_idx < ms1_mz_arrays.len() {
                extract_envelope_single(
                    &ms1_mz_arrays[scan_idx],
                    &ms1_int_arrays[scan_idx],
                    mz,
                    charge,
                    n_peaks,
                    ppm_tolerance,
                )
            } else {
                vec![0.0; n_peaks]
            }
        })
        .collect()
}
