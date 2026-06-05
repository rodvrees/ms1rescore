/// Parallel ion image extraction from profile-mode MALDI pixel spectra.
///
/// Instead of the cumsum trick (O(n_mz) per pixel, requires a large temp
/// array) this uses direct window summation: for each feature, sum
/// spectrum[lo..hi] directly.  This skips the bulk of the m/z axis when
/// the total window coverage is small relative to n_mz — typical for
/// MALDI with ~1323 features at ±25 ppm on a 0.001 Da grid where each
/// window is ~50 bins:  66 K ops/pixel vs 100 K–1 M for the full cumsum.
///
/// Rayon parallelises over pixels within each chunk; the Python caller
/// feeds chunks of pre-read spectra to keep peak RAM bounded.
use rayon::prelude::*;

/// Process a batch of pixel spectra and extract feature window integrals.
///
/// # Arguments
/// * `pixel_matrix` – row-major float32 slice, shape `(n_pixels, n_mz)`
/// * `n_mz`         – number of m/z points per spectrum (stride)
/// * `lo`           – inclusive start index of each feature window
/// * `hi`           – exclusive end index of each feature window
///
/// # Returns
/// Flat Vec<f32> of shape `(n_pixels, n_features)`, row-major.
pub fn accumulate_profile_images(
    pixel_matrix: &[f32],
    n_mz: usize,
    lo: &[usize],
    hi: &[usize],
) -> Vec<f32> {
    let n_features = lo.len();
    assert_eq!(hi.len(), n_features);
    if pixel_matrix.is_empty() || n_features == 0 || n_mz == 0 {
        return Vec::new();
    }
    let n_pixels = pixel_matrix.len() / n_mz;

    let mut output = vec![0.0f32; n_pixels * n_features];

    output
        .par_chunks_mut(n_features)
        .enumerate()
        .for_each(|(px, row)| {
            let spectrum = &pixel_matrix[px * n_mz..(px + 1) * n_mz];
            for fi in 0..n_features {
                let l = lo[fi];
                let h = hi[fi].min(spectrum.len()); // guard against out-of-bounds
                if l < h {
                    row[fi] = spectrum[l..h].iter().sum();
                }
            }
        });

    output
}
