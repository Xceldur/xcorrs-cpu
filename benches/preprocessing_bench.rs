mod benchmark_helper;

use criterion::{criterion_group, criterion_main, Criterion, BatchSize};
use ndarray::Array1;
use rand::{SeedableRng};
use rand::rngs::StdRng;
use xcorrrs::configuration::{Configuration, FinalizedConfiguration};
use xcorrrs::preprocessed_spectrum::PreprocessedSpectrum;

fn bench_process_spectra(c: &mut Criterion) {
    let base = Configuration {
        bin_size: 1.0005,
        bin_offset: 0.4,
        use_flanking_peaks: true,
        sp_matrix_enable: true,
        ..Default::default()
    };

    let config_sp_enable: FinalizedConfiguration = base.clone().into();
    let config_sp_disable: FinalizedConfiguration = Configuration {
        sp_matrix_enable: false,
        ..base
    }.into();

    //TODO: This might measure the memory bandwidth instead of algorithmic performance?
    //therefore we should maybe reduce the amout of spectra?
    let num_spectra = 100_000;
    let num_peaks_per_spectrum = 500;
    let mut rng = StdRng::seed_from_u64(42);

    let spectra: Vec<(Array1<f64>, Array1<f64>)> = (0..num_spectra)
        .map(|_| benchmark_helper::generate_random_spectrum(&mut rng, num_peaks_per_spectrum, 800.0))
        .collect();

    let mut spectra_iter = spectra.iter().cycle();

    let mut group = c
        .benchmark_group("Preprocess Spectrum");

    group.bench_function("sp_enable", |b| {
        b.iter_batched(
            || spectra_iter.next().unwrap(),
            |spectrum| {
                std::hint::black_box(
                    PreprocessedSpectrum::process(
                        &config_sp_enable,
                        (&spectrum.0, &spectrum.1),
                    ).unwrap()
                )
            },
            BatchSize::SmallInput,
        )
    });

    group.bench_function("sp_disable", |b| {
        b.iter_batched(
            || spectra_iter.next().unwrap(),
            |spectrum| {
                std::hint::black_box(
                    PreprocessedSpectrum::process(
                        &config_sp_disable,
                        (&spectrum.0, &spectrum.1),
                    ).unwrap()
                )
            },
            BatchSize::SmallInput,
        )
    });

    group.finish();
}

criterion_group!(benches, bench_process_spectra);
criterion_main!(benches);