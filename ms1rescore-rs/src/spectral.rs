/// Batch spectral angle computation between predicted and observed MS2 spectra.

/// Compute spectral angle between a single predicted and observed spectrum.
/// Returns 1 - arccos(cosine) / pi. Range [0, 1], 1 = identical.
fn spectral_angle_single(
    pred_mz: &[f64],
    pred_int: &[f64],
    obs_mz: &[f64],
    obs_int: &[f64],
    fragment_tol_da: f64,
) -> f64 {
    if pred_mz.is_empty() || obs_mz.is_empty() {
        return 0.0;
    }

    let mut matched_pred = Vec::new();
    let mut matched_obs = Vec::new();

    for j in 0..pred_mz.len() {
        let target = pred_mz[j];
        let idx = obs_mz.partition_point(|&x| x < target);

        let mut best_dist = fragment_tol_da + 1.0;
        let mut best_int = 0.0f64;

        // Check idx-1 and idx
        for &k in &[idx.wrapping_sub(1), idx] {
            if k < obs_mz.len() {
                let dist = (obs_mz[k] - target).abs();
                if dist < best_dist {
                    best_dist = dist;
                    best_int = obs_int[k];
                }
            }
        }

        if best_dist <= fragment_tol_da {
            matched_pred.push(pred_int[j]);
            matched_obs.push(best_int);
        }
    }

    if matched_pred.len() < 3 {
        return 0.0;
    }

    // Cosine similarity
    let dot: f64 = matched_pred
        .iter()
        .zip(matched_obs.iter())
        .map(|(a, b)| a * b)
        .sum();
    let norm_a: f64 = matched_pred.iter().map(|a| a * a).sum::<f64>().sqrt();
    let norm_b: f64 = matched_obs.iter().map(|b| b * b).sum::<f64>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    let cos = (dot / (norm_a * norm_b)).clamp(-1.0, 1.0);
    1.0 - cos.acos() / std::f64::consts::PI
}

/// Batch compute spectral angles.
/// Each element is a pair of (predicted, observed) spectra.
pub fn spectral_angles_batch(
    pred_mzs: &[Vec<f64>],
    pred_ints: &[Vec<f64>],
    obs_mzs: &[Vec<f64>],
    obs_ints: &[Vec<f64>],
    fragment_tol_da: f64,
) -> Vec<f64> {
    pred_mzs
        .iter()
        .zip(pred_ints.iter())
        .zip(obs_mzs.iter().zip(obs_ints.iter()))
        .map(|((pm, pi), (om, oi))| spectral_angle_single(pm, pi, om, oi, fragment_tol_da))
        .collect()
}
