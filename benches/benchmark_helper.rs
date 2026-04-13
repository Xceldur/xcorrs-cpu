use ndarray::Array1;
use rand::rngs::StdRng;
use rand::RngExt;
use rand_distr::{Exp, Distribution};

//TODO: Maybe use rustyms instead
pub(crate) fn generate_random_spectrum(
    rng: &mut StdRng,
    num_peaks: usize,
    precursor_mz: f64,
) -> (Array1<f64>, Array1<f64>) {

    // 1. m/z Generation: Bound by precursor mass. +20.0 buffer for isotopes)
    let max_mz = f64::min(2000.0, precursor_mz + 20.0);
    let mut mz_vec: Vec<f64> = (0..num_peaks)
        .map(|_| rng.random_range(100.0..max_mz))
        .collect();
    mz_vec.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

    // Intensity Generation: exponential distribution.
    let exp_dist = Exp::new(2.0).unwrap();
    let mut intensity_vec: Vec<f64> = (0..num_peaks)
        .map(|_| {
            let raw_intensity = exp_dist.sample(rng);
            (raw_intensity * 5_000.0) + rng.random_range(10.0..100.0)
        })
        .collect();

    // 3. Inject Base Peaks (Dominant Fragment Ions)
    let num_base_peaks = rng.random_range(1..=5);
    for _ in 0..num_base_peaks {
        let random_idx = rng.random_range(0..num_peaks);
        intensity_vec[random_idx] += rng.random_range(50_000.0..100_000.0);
    }

    (Array1::from_vec(mz_vec), Array1::from_vec(intensity_vec))
}