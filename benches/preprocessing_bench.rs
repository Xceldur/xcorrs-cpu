use criterion::{criterion_group, criterion_main, Criterion};
use ndarray::Array1;
use rand::RngExt;
use xcorrrs::configuration::{Configuration, FinalizedConfiguration};
use xcorrrs::preprocessed_spectrum::PreprocessedSpectrum;


fn generate_random_spectrum(num_peaks: usize) -> (Array1<f64>, Array1<f64>) {
    let mut rng = rand::rng();
    let mut mz_vec: Vec<f64> = (0..num_peaks).map(|_|rng.random_range(100.0 .. 2000.0)).collect();

    mz_vec.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

    let intensity_vec: Vec<f64> = (0..num_peaks).map(|_| rng.random_range(0.0..100_000.0)).collect();

    (Array1::from_vec(mz_vec), Array1::from_vec(intensity_vec))
}

fn bench_process_spectra(c: &mut Criterion) {
    let config: FinalizedConfiguration = Configuration {
        bin_size: 1.0005,
        bin_offset: 0.4,
        use_flanking_peaks: true,
        ..Default::default()
    }.into();

    let num_spectra = 8_000;
    let num_peaks_per_spectrum = 500;

    let spectra: Vec<(Array1<f64>, Array1<f64>)> = (0..num_spectra)
        .map(|_| generate_random_spectrum(num_peaks_per_spectrum))
        .collect();

    let mut spectra_iter = spectra.iter().cycle();



    c.bench_function("preprocess_single_spectrum", |b| {
        b.iter_batched(
            || spectra_iter.next().unwrap(),
            |spectrum| {
                std::hint::black_box(
                    PreprocessedSpectrum::process(
                        std::hint::black_box(&config),
                        (&spectrum.0, &spectrum.1),
                    ).unwrap()
                )
            },
            criterion::BatchSize::SmallInput,
        )
    });
}

criterion_group!(benches, bench_process_spectra);
criterion_main!(benches);