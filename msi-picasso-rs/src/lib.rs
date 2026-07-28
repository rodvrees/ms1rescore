mod digest;
mod features;
mod ion_image;
mod isotope;
mod maldi_isotope;
mod mob_coloc;
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

/// Extract feature window integrals from a batch of profile-mode pixel spectra.
///
/// Processes pixels in parallel (rayon) using direct window summation
/// instead of the cumsum trick, which is faster when n_features × window_width
/// << n_mz (typical for MALDI with narrow extraction windows).
///
/// Args:
///     pixel_matrix: 2-D float32 numpy array of shape (n_pixels, n_mz).
///                   Must be C-contiguous (row-major).
///     lo_indices:   Inclusive start index for each feature window (list of int).
///     hi_indices:   Exclusive end index for each feature window (list of int).
///
/// Returns:
///     1-D float32 numpy array of length n_pixels * n_features (row-major).
///     Reshape to (n_pixels, n_features) in Python.
#[pyfunction]
fn accumulate_profile_chunk<'py>(
    py: Python<'py>,
    pixel_matrix: numpy::PyReadonlyArray2<'_, f32>,
    lo_indices: Vec<usize>,
    hi_indices: Vec<usize>,
) -> pyo3::PyResult<pyo3::Bound<'py, numpy::PyArray1<f32>>> {
    use numpy::{IntoPyArray, PyUntypedArrayMethods};

    let shape = pixel_matrix.shape();
    let n_mz = shape[1];

    let mat_slice = pixel_matrix
        .as_slice()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("pixel_matrix must be C-contiguous"))?;

    let mat_addr = mat_slice.as_ptr() as usize;
    let mat_len = mat_slice.len();

    let flat_output: Vec<f32> = py.allow_threads(move || {
        let mat = unsafe { std::slice::from_raw_parts(mat_addr as *const f32, mat_len) };
        ion_image::accumulate_profile_images(mat, n_mz, &lo_indices, &hi_indices)
    });

    Ok(numpy::ndarray::Array1::from_vec(flat_output)
        .into_pyarray(py)
        .into())
}

/// Compute per-candidate mobility scalar features (M0 window only, no image building).
///
/// Much faster than mob_coloc_features: single m/z window per feature, scalar
/// accumulators only, no Pearson r or Moran's I computation.
///
/// Args:
///     flat_mzs:        Sorted m/z for all pixels (float32 numpy, CSR).
///     flat_scans:      Scan indices aligned with flat_mzs (uint32 numpy).
///     flat_ints:       Intensities aligned with flat_mzs (float32 numpy).
///     pixel_offsets:   CSR start offsets, length n_pixels + 1 (uint64 list).
///     mob_values:      1/K0 per scan index (float64 list).
///     feat_m0_lo/hi:   M0 m/z window bounds per feature (float32 numpy, each length n_features).
///     cand_ptr:        CSR candidate pointer per feature (uint32 list), len n_feat+1.
///     cand_k0_lo/hi:   Per-candidate 1/K0 bounds (float64 list).
///
/// Returns:
///     1-D float32 numpy array of length n_total_cands × 2.
///     Reshape to (n_total_cands, 2) in Python:
///       col 0 = mob_intensity_fraction, col 1 = mob_fraction_detected.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn mob_scalar_features<'py>(
    py: Python<'py>,
    flat_mzs: PyReadonlyArray1<'_, f32>,
    flat_scans: PyReadonlyArray1<'_, u32>,
    flat_ints: PyReadonlyArray1<'_, f32>,
    pixel_offsets: Vec<u64>,
    mob_values: Vec<f64>,
    feat_m0_lo: PyReadonlyArray1<'_, f32>,
    feat_m0_hi: PyReadonlyArray1<'_, f32>,
    cand_ptr: Vec<u32>,
    cand_k0_lo: Vec<f64>,
    cand_k0_hi: Vec<f64>,
) -> pyo3::PyResult<pyo3::Bound<'py, numpy::PyArray1<f32>>> {
    use numpy::IntoPyArray;

    let mzs_s = flat_mzs.as_slice().map_err(|_| pyo3::exceptions::PyValueError::new_err("flat_mzs not C-contiguous"))?;
    let scans_s = flat_scans.as_slice().map_err(|_| pyo3::exceptions::PyValueError::new_err("flat_scans not C-contiguous"))?;
    let ints_s = flat_ints.as_slice().map_err(|_| pyo3::exceptions::PyValueError::new_err("flat_ints not C-contiguous"))?;
    let lo_s = feat_m0_lo.as_slice().map_err(|_| pyo3::exceptions::PyValueError::new_err("feat_m0_lo not C-contiguous"))?;
    let hi_s = feat_m0_hi.as_slice().map_err(|_| pyo3::exceptions::PyValueError::new_err("feat_m0_hi not C-contiguous"))?;

    let mzs_addr = mzs_s.as_ptr() as usize; let mzs_len = mzs_s.len();
    let scans_addr = scans_s.as_ptr() as usize; let scans_len = scans_s.len();
    let ints_addr = ints_s.as_ptr() as usize; let ints_len = ints_s.len();
    let lo_addr = lo_s.as_ptr() as usize; let lo_len = lo_s.len();
    let hi_addr = hi_s.as_ptr() as usize; let hi_len = hi_s.len();

    let flat_out: Vec<f32> = py.allow_threads(move || {
        let mzs = unsafe { std::slice::from_raw_parts(mzs_addr as *const f32, mzs_len) };
        let scans = unsafe { std::slice::from_raw_parts(scans_addr as *const u32, scans_len) };
        let ints = unsafe { std::slice::from_raw_parts(ints_addr as *const f32, ints_len) };
        let lo = unsafe { std::slice::from_raw_parts(lo_addr as *const f32, lo_len) };
        let hi = unsafe { std::slice::from_raw_parts(hi_addr as *const f32, hi_len) };

        mob_coloc::compute_mob_scalars(
            mzs, scans, ints,
            &pixel_offsets, &mob_values,
            lo, hi,
            &cand_ptr, &cand_k0_lo, &cand_k0_hi,
        )
    });

    Ok(numpy::ndarray::Array1::from_vec(flat_out)
        .into_pyarray(py)
        .into())
}

/// Compute per-candidate mobility-filtered colocalization and spatial features.
///
/// Processes all MALDI features in parallel (rayon).  The flat CSR arrays
/// must be built ONCE in Python from alphatims (49 500 calls) and reused here.
///
/// Args:
///     flat_mzs:            Concatenated sorted m/z arrays for all pixels (float32 numpy).
///     flat_scans:          Scan indices aligned with flat_mzs (uint32 numpy).
///     flat_ints:           Intensities aligned with flat_mzs (float32 numpy).
///     pixel_offsets:       CSR start offsets, length n_pixels + 1 (uint64 list).
///     pixel_xi:            X coordinates per pixel (uint32 list).
///     pixel_yi:            Y coordinates per pixel (uint32 list).
///     mob_values:          1/K0 value for each scan index (float64 list).
///     feature_mz_windows:  Flat float32 array, shape (n_features, 6, 2): [lo, hi] per offset.
///     cand_ptr:            CSR pointer over candidates per feature (uint32 list), len n_feat+1.
///     cand_k0_lo:          Lower 1/K0 bound per candidate (float64 list).
///     cand_k0_hi:          Upper 1/K0 bound per candidate (float64 list).
///     max_x:               Image width.
///     max_y:               Image height.
///
/// Returns:
///     1-D float32 numpy array of length n_total_cands * 10.
///     Reshape to (n_total_cands, 10) in Python.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn mob_coloc_features<'py>(
    py: Python<'py>,
    flat_mzs: PyReadonlyArray1<'_, f32>,
    flat_scans: PyReadonlyArray1<'_, u32>,
    flat_ints: PyReadonlyArray1<'_, f32>,
    pixel_offsets: Vec<u64>,
    pixel_xi: Vec<u32>,
    pixel_yi: Vec<u32>,
    mob_values: Vec<f64>,
    feature_mz_windows: PyReadonlyArray1<'_, f32>,
    cand_ptr: Vec<u32>,
    cand_k0_lo: Vec<f64>,
    cand_k0_hi: Vec<f64>,
    max_x: usize,
    max_y: usize,
) -> pyo3::PyResult<pyo3::Bound<'py, numpy::PyArray1<f32>>> {
    use numpy::{IntoPyArray, PyUntypedArrayMethods};

    let mzs_s = flat_mzs.as_slice().map_err(|_| pyo3::exceptions::PyValueError::new_err("flat_mzs not C-contiguous"))?;
    let scans_s = flat_scans.as_slice().map_err(|_| pyo3::exceptions::PyValueError::new_err("flat_scans not C-contiguous"))?;
    let ints_s = flat_ints.as_slice().map_err(|_| pyo3::exceptions::PyValueError::new_err("flat_ints not C-contiguous"))?;
    let wins_s = feature_mz_windows.as_slice().map_err(|_| pyo3::exceptions::PyValueError::new_err("feature_mz_windows not C-contiguous"))?;

    // Cast to usize pointers so they cross the allow_threads (Ungil) boundary.
    // Safety: PyReadonlyArray1 holds a GIL-protected reference; arrays are
    // immutable while we hold the borrow; rayon threads only read.
    let mzs_addr = mzs_s.as_ptr() as usize;
    let mzs_len = mzs_s.len();
    let scans_addr = scans_s.as_ptr() as usize;
    let scans_len = scans_s.len();
    let ints_addr = ints_s.as_ptr() as usize;
    let ints_len = ints_s.len();
    let wins_addr = wins_s.as_ptr() as usize;
    let wins_len = wins_s.len();

    let flat_out: Vec<f32> = py.allow_threads(move || {
        let mzs = unsafe { std::slice::from_raw_parts(mzs_addr as *const f32, mzs_len) };
        let scans = unsafe { std::slice::from_raw_parts(scans_addr as *const u32, scans_len) };
        let ints = unsafe { std::slice::from_raw_parts(ints_addr as *const f32, ints_len) };
        let wins = unsafe { std::slice::from_raw_parts(wins_addr as *const f32, wins_len) };

        mob_coloc::compute_mob_coloc(
            mzs, scans, ints,
            &pixel_offsets, &pixel_xi, &pixel_yi,
            &mob_values,
            wins,
            &cand_ptr, &cand_k0_lo, &cand_k0_hi,
            max_x, max_y,
        )
    });

    Ok(numpy::ndarray::Array1::from_vec(flat_out)
        .into_pyarray(py)
        .into())
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
    m.add_function(wrap_pyfunction!(accumulate_profile_chunk, m)?)?;
    m.add_function(wrap_pyfunction!(mob_coloc_features, m)?)?;
    Ok(())
}
