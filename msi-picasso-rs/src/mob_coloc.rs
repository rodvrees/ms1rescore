/// Compute per-candidate mobility scalar features from M0 window only.
///
/// No image building. Two scalars per candidate:
///   [0] mob_intensity_fraction: sum(M0 intensity in k0 window) / sum(M0 intensity any k0)
///   [1] mob_fraction_detected:  (n pixels with M0 signal in k0 win) / (n pixels with M0 signal)
///
/// Rayon-parallel over features. O(n_pixels × n_hits_in_M0_window × n_cands) per feature.
///
/// # Arguments
/// * `flat_mzs`         – Sorted m/z for all pixels (CSR layout, f32).
/// * `flat_scans`       – Scan indices aligned with flat_mzs (u32).
/// * `flat_ints`        – Intensities aligned with flat_mzs (f32).
/// * `pixel_offsets`    – CSR start offsets (u64), length n_pixels + 1.
/// * `mob_values`       – 1/K0 per scan index (f64).
/// * `feat_m0_lo/hi`    – M0 m/z window per feature (f32, length n_features each).
/// * `cand_ptr`         – CSR candidate pointer per feature (u32), length n_features + 1.
/// * `cand_k0_lo/hi`    – Per-candidate 1/K0 window (f64, length n_total_cands each).
///
/// # Returns
/// Flat f32 Vec of length n_total_cands × 2 (row-major). Reshape in Python.
pub fn compute_mob_scalars(
    flat_mzs: &[f32],
    flat_scans: &[u32],
    flat_ints: &[f32],
    pixel_offsets: &[u64],
    mob_values: &[f64],
    feat_m0_lo: &[f32],
    feat_m0_hi: &[f32],
    cand_ptr: &[u32],
    cand_k0_lo: &[f64],
    cand_k0_hi: &[f64],
) -> Vec<f32> {
    let n_pixels = pixel_offsets.len().saturating_sub(1);
    let n_features = cand_ptr.len().saturating_sub(1);
    let n_total = cand_k0_lo.len();

    let mut result = vec![f32::NAN; n_total * 2];

    if n_pixels == 0 || n_features == 0 || n_total == 0 {
        return result;
    }

    let feature_results: Vec<(usize, Vec<f32>)> = (0..n_features)
        .into_par_iter()
        .filter_map(|f| {
            let c_start = cand_ptr[f] as usize;
            let c_end = cand_ptr[f + 1] as usize;
            let n_cands = c_end - c_start;
            if n_cands == 0 {
                return None;
            }

            let lo = feat_m0_lo[f];
            let hi = feat_m0_hi[f];

            let k0_lo = &cand_k0_lo[c_start..c_end];
            let k0_hi = &cand_k0_hi[c_start..c_end];

            let mut total_int = 0.0f64;
            let mut total_px = 0u32;
            let mut cand_int = vec![0.0f64; n_cands];
            let mut cand_px = vec![0u32; n_cands];

            for px in 0..n_pixels {
                let start = pixel_offsets[px] as usize;
                let end = pixel_offsets[px + 1] as usize;
                if start >= end {
                    continue;
                }

                let mz_arr = &flat_mzs[start..end];
                let idx_start = mz_arr.partition_point(|&m| m < lo);
                let mut pix_total = 0.0f64;
                let mut pix_cand = [0.0f64; 64]; // stack buffer; n_cands <= 64 typical
                // fall back to heap for very wide features
                let use_stack = n_cands <= 64;
                let mut heap_cand: Vec<f64> = if use_stack { Vec::new() } else { vec![0.0f64; n_cands] };

                let mut idx = idx_start;
                while idx < mz_arr.len() && mz_arr[idx] <= hi {
                    let abs = start + idx;
                    let scan_id = flat_scans[abs] as usize;
                    let intensity = flat_ints[abs] as f64;
                    if scan_id < mob_values.len() {
                        let k0 = mob_values[scan_id];
                        pix_total += intensity;
                        for c in 0..n_cands {
                            if k0 >= k0_lo[c] && k0 <= k0_hi[c] {
                                if use_stack {
                                    pix_cand[c] += intensity;
                                } else {
                                    heap_cand[c] += intensity;
                                }
                            }
                        }
                    }
                    idx += 1;
                }

                if pix_total > 0.0 {
                    total_int += pix_total;
                    total_px += 1;
                    for c in 0..n_cands {
                        let ci = if use_stack { pix_cand[c] } else { heap_cand[c] };
                        if ci > 0.0 {
                            cand_int[c] += ci;
                            cand_px[c] += 1;
                        }
                    }
                }
            }

            let mut out = vec![f32::NAN; n_cands * 2];
            for c in 0..n_cands {
                out[c * 2]     = if total_int > 1e-12 { (cand_int[c] / total_int) as f32 } else { 0.0 };
                out[c * 2 + 1] = if total_px > 0 { cand_px[c] as f32 / total_px as f32 } else { 0.0 };
            }
            Some((c_start, out))
        })
        .collect();

    for (c_start, out) in feature_results {
        let n_cands = out.len() / 2;
        for c in 0..n_cands {
            result[(c_start + c) * 2]     = out[c * 2];
            result[(c_start + c) * 2 + 1] = out[c * 2 + 1];
        }
    }

    result
}


/// Per-candidate mobility-filtered ion image colocalization features.
///
/// All 925 MALDI features are processed in parallel (rayon). Each feature reads
/// through the full pixel CSR once, applying 6 narrow m/z windows and — per
/// candidate — a 1/K0 window. Outputs 10 float32 features per candidate:
///
///   [0] isotope_colocalization_m1_mob   Pearson r(M0_mob, M1_mob)
///   [1] isotope_colocalization_m2_mob   Pearson r(M0_mob, M2_mob)
///   [2] isotope_colocalization_mean_mob mean of [0] and [1] (skipping NaN)
///   [3] adduct_colocalization_na_mob    Pearson r(M0_mob, Na_mob)
///   [4] adduct_colocalization_k_mob     Pearson r(M0_mob, K_mob)
///   [5] adduct_colocalization_chca_mob  Pearson r(M0_mob, CHCA_mob)
///   [6] fraction_detected_mob           fraction of pixels with M0 > 0
///   [7] intensity_cv_mob                CV of non-zero M0 pixels (0 if <2)
///   [8] log_mean_intensity_mob          log1p(mean(M0)) over all pixels
///   [9] spatial_morans_i_mob            Moran's I (8-connectivity) of M0 image
///
/// The flat CSR arrays are built ONCE in Python (49 500 alphatims calls), then
/// this function processes everything in Rust without Python overhead.
use rayon::prelude::*;

const N_OFFSETS: usize = 6; // m0, m1, m2, na, k, chca
const N_OUT: usize = 10;

/// Pearson r from running sums over n pixels.
///
/// Uses the one-pass formula:
///   r = (n·Σab − Σa·Σb) / sqrt((n·Σa² − (Σa)²)(n·Σb² − (Σb)²))
#[inline]
fn pearson_r(sum_a: f64, sum_b: f64, sum_ab: f64, sum_a2: f64, sum_b2: f64, n: f64) -> f32 {
    let num = n * sum_ab - sum_a * sum_b;
    let da = n * sum_a2 - sum_a * sum_a;
    let db = n * sum_b2 - sum_b * sum_b;
    if da <= 1e-20 || db <= 1e-20 {
        return f32::NAN;
    }
    (num / (da.sqrt() * db.sqrt())) as f32
}

/// Moran's I with 8-connectivity (queen contiguity).
///
/// I = (N / W) · Σᵢⱼ zᵢzⱼ / Σᵢ zᵢ²
/// where zᵢ = xᵢ − mean(x), W = total number of neighbour-pixel pairs
/// (counting both directions, so W ≈ 8 × N for interior pixels).
fn morans_i(img: &[f32], max_x: usize, max_y: usize) -> f32 {
    let n = (max_x * max_y) as f64;

    let sum: f64 = img.iter().map(|&v| v as f64).sum();
    let mean = sum / n;

    let mut numer = 0.0f64;
    let mut denom = 0.0f64;
    let mut w_count = 0u64;

    for y in 0..max_y {
        let y_start = y.saturating_sub(1);
        let y_end = (y + 1).min(max_y - 1);

        for x in 0..max_x {
            let zi = img[y * max_x + x] as f64 - mean;
            denom += zi * zi;

            let x_start = x.saturating_sub(1);
            let x_end = (x + 1).min(max_x - 1);

            for ny in y_start..=y_end {
                for nx in x_start..=x_end {
                    if ny == y && nx == x {
                        continue;
                    }
                    let zj = img[ny * max_x + nx] as f64 - mean;
                    numer += zi * zj;
                    w_count += 1;
                }
            }
        }
    }

    if denom < 1e-12 || w_count == 0 {
        return f32::NAN;
    }

    (n / w_count as f64 * numer / denom) as f32
}

/// Compute per-candidate mobility colocalization and spatial features.
///
/// # Arguments
/// * `flat_mzs`            – Concatenated **sorted** m/z arrays (f32) for all pixels, CSR.
/// * `flat_scans`          – Scan indices matching flat_mzs (u32); used to look up 1/K0.
/// * `flat_ints`           – Intensities matching flat_mzs (f32).
/// * `pixel_offsets`       – CSR start index per pixel (u64), length n_pixels + 1.
/// * `pixel_xi` / `pixel_yi` – Spatial coordinates (u32) for each pixel.
/// * `mob_values`          – 1/K0 in V·s/cm² for each scan index (f64).
/// * `feature_mz_windows`  – Flat array length n_features × N_OFFSETS × 2:
///                           `[lo_m0, hi_m0, lo_m1, hi_m1, ..., lo_chca, hi_chca]` per feature.
/// * `cand_ptr`            – CSR pointer over candidates per feature (u32), length n_features+1.
/// * `cand_k0_lo/hi`       – Per-candidate 1/K0 window bounds (f64), length n_total_cands.
/// * `max_x`, `max_y`      – Image dimensions.
///
/// # Returns
/// Flat f32 Vec of shape `(n_total_cands, N_OUT)` (row-major).  Reshape in Python.
pub fn compute_mob_coloc(
    flat_mzs: &[f32],
    flat_scans: &[u32],
    flat_ints: &[f32],
    pixel_offsets: &[u64],
    pixel_xi: &[u32],
    pixel_yi: &[u32],
    mob_values: &[f64],
    feature_mz_windows: &[f32],
    cand_ptr: &[u32],
    cand_k0_lo: &[f64],
    cand_k0_hi: &[f64],
    max_x: usize,
    max_y: usize,
) -> Vec<f32> {
    let n_pixels = pixel_offsets.len().saturating_sub(1);
    let n_features = cand_ptr.len().saturating_sub(1);
    let n_total_cands = cand_k0_lo.len();
    let n_px = max_y * max_x;

    let mut result = vec![f32::NAN; n_total_cands * N_OUT];

    if n_pixels == 0 || n_features == 0 || n_total_cands == 0 || n_px == 0 {
        return result;
    }

    // Collect per-feature results: (candidate_start_index, output_slice)
    let feature_results: Vec<(usize, Vec<f32>)> = (0..n_features)
        .into_par_iter()
        .filter_map(|f| {
            let c_start = cand_ptr[f] as usize;
            let c_end = cand_ptr[f + 1] as usize;
            let n_cands = c_end - c_start;
            if n_cands == 0 {
                return None;
            }

            let win_off = f * N_OFFSETS * 2;

            // Local k0 windows for cache-friendly access
            let k0_lo: &[f64] = &cand_k0_lo[c_start..c_end];
            let k0_hi: &[f64] = &cand_k0_hi[c_start..c_end];

            // m/z window bounds for this feature (6 × 2 = 12 values)
            let mz_lo: [f32; N_OFFSETS] = std::array::from_fn(|oi| feature_mz_windows[win_off + oi * 2]);
            let mz_hi: [f32; N_OFFSETS] = std::array::from_fn(|oi| feature_mz_windows[win_off + oi * 2 + 1]);

            // M0 image per candidate: (n_cands, max_y * max_x)
            let mut m0_images: Vec<f32> = vec![0.0f32; n_cands * n_px];

            // Pearson running sums per candidate × 5 pairs (m0 vs m1/m2/na/k/chca)
            // Layout: [c][pair][stat]   stat = {sum_m0, sum_x, sum_m0x, sum_m0sq, sum_xsq}
            let mut p_sums: Vec<[[f64; 5]; 5]> = vec![[[0.0f64; 5]; 5]; n_cands];

            // Scratch buffer: per-candidate, per-offset intensity for current pixel
            let mut pix_ints: Vec<[f32; N_OFFSETS]> = vec![[0.0f32; N_OFFSETS]; n_cands];

            for px in 0..n_pixels {
                let start = pixel_offsets[px] as usize;
                let end = pixel_offsets[px + 1] as usize;
                if start >= end {
                    continue;
                }

                let xi = pixel_xi[px] as usize;
                let yi = pixel_yi[px] as usize;
                if xi >= max_x || yi >= max_y {
                    continue;
                }
                let px_flat = yi * max_x + xi;

                let mz_arr = &flat_mzs[start..end]; // sorted

                // Reset scratch
                for c in 0..n_cands {
                    pix_ints[c] = [0.0f32; N_OFFSETS];
                }

                // For each offset: binary search for m/z window, iterate over hits
                for oi in 0..N_OFFSETS {
                    let lo = mz_lo[oi];
                    let hi = mz_hi[oi];

                    // Binary search for first m/z >= lo
                    let idx_start = mz_arr.partition_point(|&m| m < lo);
                    let mut idx = idx_start;

                    while idx < mz_arr.len() && mz_arr[idx] <= hi {
                        let abs = start + idx;
                        let scan_id = flat_scans[abs] as usize;
                        let intensity = flat_ints[abs];

                        if scan_id < mob_values.len() {
                            let k0 = mob_values[scan_id];
                            for c in 0..n_cands {
                                if k0 >= k0_lo[c] && k0 <= k0_hi[c] {
                                    pix_ints[c][oi] += intensity;
                                }
                            }
                        }
                        idx += 1;
                    }
                }

                // Accumulate M0 images and Pearson running sums
                for c in 0..n_cands {
                    let m0 = pix_ints[c][0];
                    m0_images[c * n_px + px_flat] = m0;

                    let m0_f = m0 as f64;
                    let ps = &mut p_sums[c];
                    for pair in 0..5usize {
                        let x = pix_ints[c][pair + 1] as f64;
                        ps[pair][0] += m0_f;
                        ps[pair][1] += x;
                        ps[pair][2] += m0_f * x;
                        ps[pair][3] += m0_f * m0_f;
                        ps[pair][4] += x * x;
                    }
                }
            }

            // Compute 10 output features per candidate
            let n = n_pixels as f64;
            let mut out = vec![f32::NAN; n_cands * N_OUT];

            for c in 0..n_cands {
                let off = c * N_OUT;
                let ps = &p_sums[c];

                // Pearson r for 5 pairs
                let r_vals: [f32; 5] = std::array::from_fn(|pair| {
                    pearson_r(ps[pair][0], ps[pair][1], ps[pair][2], ps[pair][3], ps[pair][4], n)
                });

                out[off + 0] = r_vals[0]; // m1
                out[off + 1] = r_vals[1]; // m2
                // mean of m1, m2 (skip NaN)
                let (iso_sum, iso_cnt) = [(0usize), (1usize)]
                    .iter()
                    .fold((0.0f32, 0u32), |(s, k), &i| {
                        if !r_vals[i].is_nan() { (s + r_vals[i], k + 1) } else { (s, k) }
                    });
                out[off + 2] = if iso_cnt > 0 { iso_sum / iso_cnt as f32 } else { f32::NAN };
                out[off + 3] = r_vals[2]; // na
                out[off + 4] = r_vals[3]; // k
                out[off + 5] = r_vals[4]; // chca

                // Spatial metrics from M0 image
                let m0_img = &m0_images[c * n_px..(c + 1) * n_px];
                let total: f32 = m0_img.iter().sum();
                let mean_int = total / n_px as f32;

                // fraction_detected
                let n_nz = m0_img.iter().filter(|&&v| v > 0.0).count();
                out[off + 6] = n_nz as f32 / n_px as f32;

                // intensity CV (non-zero pixels)
                if n_nz > 1 {
                    let nz_sum: f32 = m0_img.iter().filter(|&&v| v > 0.0).sum();
                    let nz_mean = nz_sum / n_nz as f32;
                    let var: f32 = m0_img
                        .iter()
                        .filter(|&&v| v > 0.0)
                        .map(|&v| (v - nz_mean) * (v - nz_mean))
                        .sum::<f32>()
                        / n_nz as f32;
                    out[off + 7] = if nz_mean > 0.0 { var.sqrt() / nz_mean } else { 0.0 };
                } else {
                    out[off + 7] = 0.0;
                }

                // log1p(mean(M0))
                out[off + 8] = (1.0 + mean_int).ln();

                // Moran's I
                out[off + 9] = morans_i(m0_img, max_x, max_y);
            }

            Some((c_start, out))
        })
        .collect();

    // Write per-feature results into the global result slice
    for (c_start, out) in feature_results {
        let n_cands = out.len() / N_OUT;
        for c in 0..n_cands {
            let src = c * N_OUT;
            let dst = (c_start + c) * N_OUT;
            result[dst..dst + N_OUT].copy_from_slice(&out[src..src + N_OUT]);
        }
    }

    result
}
