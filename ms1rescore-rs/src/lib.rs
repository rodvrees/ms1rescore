mod digest;
mod features;
mod isotope;
mod maldi_isotope;
mod spectral;
mod xic;

use numpy::PyReadonlyArray1;
use pyo3::prelude::*;

/// Extract XICs for multiple target m/z values across all MS1 scans.
///
/// Args:
///     ms1_rts: Retention times for each MS1 scan.
///     ms1_mz_arrays: m/z arrays for each MS1 scan (sorted).
///     ms1_int_arrays: Intensity arrays for each MS1 scan.
///     target_mzs: Target m/z values to extract XICs for.
///     ppm_tolerance: Mass tolerance in ppm.
///
/// Returns:
///     List of (rts, intensities) tuples, one per target m/z.
#[pyfunction]
#[pyo3(signature = (ms1_rts, ms1_mz_arrays, ms1_int_arrays, target_mzs, ppm_tolerance=20.0))]
fn extract_xics_batch(
    ms1_rts: Vec<f64>,
    ms1_mz_arrays: Vec<Vec<f64>>,
    ms1_int_arrays: Vec<Vec<f64>>,
    target_mzs: Vec<f64>,
    ppm_tolerance: f64,
) -> Vec<(Vec<f64>, Vec<f64>)> {
    xic::extract_xics_batch(&ms1_rts, &ms1_mz_arrays, &ms1_int_arrays, &target_mzs, ppm_tolerance)
}

/// Batch compute spectral angles between predicted and observed MS2 spectra.
///
/// Args:
///     pred_mzs: Predicted fragment m/z arrays (sorted).
///     pred_ints: Predicted fragment intensity arrays.
///     obs_mzs: Observed fragment m/z arrays (sorted).
///     obs_ints: Observed fragment intensity arrays.
///     fragment_tol_da: Fragment matching tolerance in Da.
///
/// Returns:
///     List of spectral angles (0-1, 1=identical).
#[pyfunction]
#[pyo3(signature = (pred_mzs, pred_ints, obs_mzs, obs_ints, fragment_tol_da=0.02))]
fn spectral_angles_batch(
    pred_mzs: Vec<Vec<f64>>,
    pred_ints: Vec<Vec<f64>>,
    obs_mzs: Vec<Vec<f64>>,
    obs_ints: Vec<Vec<f64>>,
    fragment_tol_da: f64,
) -> Vec<f64> {
    spectral::spectral_angles_batch(&pred_mzs, &pred_ints, &obs_mzs, &obs_ints, fragment_tol_da)
}

/// Batch extract MS1 isotope envelopes at specific scans.
///
/// Args:
///     ms1_mz_arrays: m/z arrays for each MS1 scan.
///     ms1_int_arrays: Intensity arrays for each MS1 scan.
///     scan_indices: Which MS1 scan to extract from (one per target).
///     target_mzs: Monoisotopic m/z values.
///     charge: Charge state.
///     n_peaks: Number of isotope peaks to extract.
///     ppm_tolerance: Mass tolerance in ppm.
///
/// Returns:
///     List of normalized isotope envelopes.
#[pyfunction]
#[pyo3(signature = (ms1_mz_arrays, ms1_int_arrays, scan_indices, target_mzs, charge=1, n_peaks=3, ppm_tolerance=20.0))]
fn extract_ms1_envelopes_batch(
    ms1_mz_arrays: Vec<Vec<f64>>,
    ms1_int_arrays: Vec<Vec<f64>>,
    scan_indices: Vec<usize>,
    target_mzs: Vec<f64>,
    charge: usize,
    n_peaks: usize,
    ppm_tolerance: f64,
) -> Vec<Vec<f64>> {
    isotope::extract_envelopes_batch(
        &ms1_mz_arrays,
        &ms1_int_arrays,
        &scan_indices,
        &target_mzs,
        charge,
        n_peaks,
        ppm_tolerance,
    )
}

/// Batch compute monoisotopic mass and elemental composition for peptide sequences.
///
/// Returns tuple of 7 parallel lists:
///     (masses, mh_mzs, n_C, n_H, n_N, n_O, n_S)
#[pyfunction]
fn compute_peptide_masses(sequences: Vec<String>) -> (Vec<f64>, Vec<f64>, Vec<i32>, Vec<i32>, Vec<i32>, Vec<i32>, Vec<i32>) {
    digest::compute_peptide_info_batch(&sequences)
}

/// Match peptide [M+H]+ m/z values against MALDI feature m/z values.
///
/// Returns tuple of 3 parallel lists:
///     (feature_indices, peptide_indices, ppm_errors)
#[pyfunction]
#[pyo3(signature = (maldi_mzs, peptide_mzs, ppm_tolerance=20.0))]
fn match_mz(
    maldi_mzs: Vec<f64>,
    peptide_mzs: Vec<f64>,
    ppm_tolerance: f64,
) -> (Vec<u32>, Vec<u32>, Vec<f64>) {
    digest::match_mz_batch(&maldi_mzs, &peptide_mzs, ppm_tolerance)
}

/// Compute ionization features for unique peptide sequences (parallel).
///
/// Args:
///     sequences: List of unique peptide sequences.
///
/// Returns:
///     (n_R, n_K, n_H, n_F, n_W, n_Y, gravy, peptide_pi) — 8 arrays aligned with input.
#[pyfunction]
fn compute_ionization_features(
    sequences: Vec<String>,
) -> (Vec<i32>, Vec<i32>, Vec<i32>, Vec<i32>, Vec<i32>, Vec<i32>, Vec<f64>, Vec<f64>) {
    features::ionization_features_batch(&sequences)
}

/// Compute property features for unique peptide sequences (parallel).
///
/// Args:
///     sequences: List of unique peptide sequences.
///
/// Returns:
///     (n_D, n_E, n_C, n_P, n_M, n_W, n_Y, seq_len, nterm_code, peptide_pi) — 10 arrays.
///     nterm_code: ASCII code of first residue as i32 (0 for empty sequence).
#[pyfunction]
fn compute_property_features(
    sequences: Vec<String>,
) -> (Vec<i32>, Vec<i32>, Vec<i32>, Vec<i32>, Vec<i32>, Vec<i32>, Vec<i32>, Vec<i32>, Vec<i32>, Vec<f64>) {
    features::property_features_batch(&sequences)
}

/// Count missed cleavages (K/R not followed by P, excluding last residue) for
/// a list of sequences in parallel.
#[pyfunction]
fn count_missed_cleavages_batch(sequences: Vec<String>) -> Vec<i32> {
    features::missed_cleavages_batch(&sequences)
}

/// Compute mean intensities at target m/z values across all MALDI pixels.
///
/// Uses a single streaming pass (CSR layout) and rayon parallelism over pixels.
/// Replaces two ``reader.get_ion_images()`` calls for M+1 / M+2 isotope extraction,
/// avoiding two full streaming passes and two large temporary 3-D arrays.
///
/// Args:
///     flat_mzs: Concatenated sorted m/z arrays for all pixels (float64 numpy array).
///     flat_ints: Concatenated intensity arrays for all pixels (float32 numpy array).
///     pixel_offsets: CSR-style start index for each pixel (list of int, length n_pixels+1).
///     target_mzs: Target m/z values in any order (list of float).
///     ppm_tolerance: Mass tolerance in ppm (default 25.0).
///
/// Returns:
///     List of mean intensities, one per target m/z, in the same order as target_mzs.
#[pyfunction]
#[pyo3(signature = (flat_mzs, flat_ints, pixel_offsets, target_mzs, ppm_tolerance=25.0))]
fn compute_maldi_isotope_means(
    py: Python<'_>,
    flat_mzs: PyReadonlyArray1<'_, f64>,
    flat_ints: PyReadonlyArray1<'_, f32>,
    pixel_offsets: Vec<usize>,
    target_mzs: Vec<f64>,
    ppm_tolerance: f64,
) -> Vec<f64> {
    let mzs_slice = flat_mzs.as_slice().expect("flat_mzs must be C-contiguous");
    let ints_slice = flat_ints.as_slice().expect("flat_ints must be C-contiguous");

    // Cast pointers to usize to cross the allow_threads (Ungil) boundary.
    // usize is Send + Sync; we re-cast to *const inside the closure.
    // Safety: the numpy arrays remain alive (PyReadonlyArray1 holds a GIL-
    // protected reference); PyReadonlyArray1 guarantees no mutation while
    // we hold the borrow; rayon threads only read.
    let mzs_addr = mzs_slice.as_ptr() as usize;
    let mzs_len = mzs_slice.len();
    let ints_addr = ints_slice.as_ptr() as usize;
    let ints_len = ints_slice.len();

    py.allow_threads(move || {
        let mzs = unsafe { std::slice::from_raw_parts(mzs_addr as *const f64, mzs_len) };
        let ints = unsafe { std::slice::from_raw_parts(ints_addr as *const f32, ints_len) };
        maldi_isotope::compute_isotope_means_flat(mzs, ints, &pixel_offsets, &target_mzs, ppm_tolerance)
    })
}

#[pymodule]
fn ms1rescore_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(extract_xics_batch, m)?)?;
    m.add_function(wrap_pyfunction!(spectral_angles_batch, m)?)?;
    m.add_function(wrap_pyfunction!(extract_ms1_envelopes_batch, m)?)?;
    m.add_function(wrap_pyfunction!(compute_peptide_masses, m)?)?;
    m.add_function(wrap_pyfunction!(match_mz, m)?)?;
    m.add_function(wrap_pyfunction!(compute_ionization_features, m)?)?;
    m.add_function(wrap_pyfunction!(compute_property_features, m)?)?;
    m.add_function(wrap_pyfunction!(count_missed_cleavages_batch, m)?)?;
    m.add_function(wrap_pyfunction!(compute_maldi_isotope_means, m)?)?;
    Ok(())
}
