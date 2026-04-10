use ndarray::{s, Array1, Axis};
use crate::{configuration::FinalizedConfiguration, error::Error};
//use crate::fast_xcorr::FastXcorr;

/// +/- m/z shift for the xcorr calculation.
pub const BIN_SHIFT: usize = 75;
/// According to the original authors, after binning the experimental spectrum, the maximum intensity is normalized
/// over a number of fixed windows.
pub const NUM_WINDOWS_FOR_NORMALIZATION: u8 = 10;

/// Size of each chunk in the sparse SP score matrix.
const SP_MATRIX_SIZE: usize = 100;

/// A small value to consider as zero in the SP score matrix.
const NEARLY_ZERO: f64 = 1e-6;

pub struct PreprocessedSpectrum {
    /// Sparse SP score matrix
    pub sp_matrix: Vec<Option<Vec<f64>>>,
    /// y' prime from equation 6 in https://pubs.acs.org/doi/10.1021/pr800420s
    pub preprocessed_experimental_spectrum: Array1<f64>,
}

impl PreprocessedSpectrum {
    pub fn process(
        config: &FinalizedConfiguration,
        experimental_spectrum: (&Array1<f64>, &Array1<f64>),
    ) -> Result<Self, Error> {
        if experimental_spectrum.0.is_empty() {
            return Err(Error::EmptyExperimentalSpectrum);
        }

        if experimental_spectrum.0.len() != experimental_spectrum.1.len() {
            return Err(Error::ExperimentalSpectrumShape(
                experimental_spectrum.0.len(),
                experimental_spectrum.1.len(),
            ));
        }

        // Filter out peaks below the minimum intensity
        let considerable_peaks_indexes: Vec<usize> = experimental_spectrum
            .1
            .iter()
            .enumerate()
            .filter(|&(_, &intensity)| intensity >= config.minimum_intensity)
            .map(|(index, _)| index)
            .collect();

        let mut filtered_experimental_spectrum = (
            experimental_spectrum.0.select(Axis(0), &considerable_peaks_indexes),
            experimental_spectrum.1.select(Axis(0), &considerable_peaks_indexes),
        );

        // Clear m/z range if specified in the configuration
        if let Some((min_mz, max_mz)) = config.clear_mz_range {
            let considerable_peaks_indexes: Vec<usize> = experimental_spectrum
                .0
                .iter()
                .enumerate()
                .filter(|&(_, &mz)| mz <= min_mz && mz >= max_mz)
                .map(|(index, _)| index)
                .collect();

            filtered_experimental_spectrum = (
                experimental_spectrum.0.select(Axis(0), &considerable_peaks_indexes),
                experimental_spectrum.1.select(Axis(0), &considerable_peaks_indexes),
            );
        }

        // Get max m/z once for binning and normalization
        let mz_max = filtered_experimental_spectrum
            .0
            .iter()
            .fold(f64::NEG_INFINITY, |acc, &x| acc.max(x));

        // Binning
        let binned_experimental_spectrum = Self::experimental_spectrum_binning(
            &filtered_experimental_spectrum.0,
            &filtered_experimental_spectrum.1,
            mz_max,
            config.bin_size,
            config.bin_offset,
        )?;

        // Get max sqrt intensity for SP score matrix calculation
        let max_intensity_sqrt = binned_experimental_spectrum
            .iter()
            .fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        let sp_matrix = Self::build_sparse_sp_score(&binned_experimental_spectrum, max_intensity_sqrt);

        // Normalization of the binned experimental spectrum
        let binned_normalized_experimental_spectrum = Self::experimental_spectrum_normalization(
            mz_max,
            binned_experimental_spectrum,
            config.bin_size,
            config.bin_offset,
            config.use_flanking_peaks,
        )?;

        let preprocessed_experimental_spectrum =
            Self::xcorr_preprocessing(&binned_normalized_experimental_spectrum);

        Ok(Self {
            sp_matrix,
            preprocessed_experimental_spectrum,
        })
    }

    pub fn calc_number_of_bins(mz_max: f64, bin_size: f64, bin_offset: f64) -> usize {
        (mz_max / bin_size + 1.0 - bin_offset) as usize + 2 + BIN_SHIFT
    }

    pub fn calc_binned_position(mz: f64, bin_size: f64, bin_offset: f64) -> usize {
        (mz / bin_size + 1.0 - bin_offset) as usize
    }

    fn experimental_spectrum_binning(
        mz: &Array1<f64>,
        intensities: &Array1<f64>,
        mz_max: f64,
        bin_size: f64,
        bin_offset: f64,
    ) -> Result<Array1<f64>, Error> {
        let number_of_bins = Self::calc_number_of_bins(mz_max, bin_size, bin_offset);
        let mut binned_spectrum: Array1<f64> = Array1::zeros(number_of_bins);

        for (mz, intensity) in mz.iter().zip(intensities.iter()) {
            let index = Self::calc_binned_position(*mz, bin_size, bin_offset);
            binned_spectrum[index] = binned_spectrum[index].max(intensity.sqrt());
        }

        Ok(binned_spectrum)
    }

    fn build_sparse_sp_score(
        binned_theoretical_spectrum: &Array1<f64>,
        max_intensity_sqrt: f64,
    ) -> Vec<Option<Vec<f64>>> {
        let matrix_size = binned_theoretical_spectrum.len() / SP_MATRIX_SIZE + 1;
        let mut sparse: Vec<Option<Vec<f64>>> = vec![None; matrix_size];

        if max_intensity_sqrt <= 0.0 {
            return sparse;
        }

        for i in 0..binned_theoretical_spectrum.len() {
            let normalized = 100.0 * binned_theoretical_spectrum[i] / max_intensity_sqrt;
            if normalized > NEARLY_ZERO {
                let x = i / SP_MATRIX_SIZE;
                let y = i - (x * SP_MATRIX_SIZE);

                if sparse[x].is_none() {
                    sparse[x] = Some(vec![0.0_f64; SP_MATRIX_SIZE]);
                }

                if let Some(ref mut row) = sparse[x] {
                    row[y] = normalized;
                }
            }
        }
        sparse
    }

    fn experimental_spectrum_normalization(
        mz_max: f64,
        mut binned_theoretical_spectrum: Array1<f64>,
        bin_size: f64,
        bin_offset: f64,
        use_flanking_peaks: bool,
    ) -> Result<Array1<f64>, Error> {
        let highest_ion = Self::calc_binned_position(mz_max, bin_size, bin_offset);
        let windows_size = (highest_ion as f64 / NUM_WINDOWS_FOR_NORMALIZATION as f64) as usize + 1;

        for window_start in (0..binned_theoretical_spectrum.len()).step_by(windows_size) {
            let window_end = (window_start + windows_size).min(binned_theoretical_spectrum.len());
            let mut window = binned_theoretical_spectrum.slice_mut(s![window_start..window_end]);

            let window_max = window.iter().fold(f64::NEG_INFINITY, |acc, &x| acc.max(x));
            let window_intensity_cutoff = 0.05 * window_max;
            window.mapv_inplace(|value| {
                if value <= window_intensity_cutoff {
                    value
                } else {
                    value / window_max * 50.0
                }
            });
        }

        if !use_flanking_peaks {
            Ok(binned_theoretical_spectrum)
        } else {
            let mut flanked_binned_spectrum = Array1::zeros(binned_theoretical_spectrum.len());
            binned_theoretical_spectrum
                .into_iter()
                .enumerate()
                .for_each(|(i, value)| {
                    flanked_binned_spectrum[i] += value;
                    let half_peak = value * 0.5;

                    if i > 0 {
                        flanked_binned_spectrum[i - 1] += half_peak;
                    }
                    if i < flanked_binned_spectrum.len() - 1 {
                        flanked_binned_spectrum[i + 1] += half_peak;
                    }
                });
            Ok(flanked_binned_spectrum)
        }
    }

    fn xcorr_preprocessing(binned_normalized_experimental_spectrum: &Array1<f64>) -> Array1<f64> {
        let mut corrected_experimental_spectrum_shift =
            Array1::zeros(binned_normalized_experimental_spectrum.len());

        let mut sum_offsets = binned_normalized_experimental_spectrum.slice(s![1..=BIN_SHIFT]).sum();
        let mean_offset = sum_offsets / 150.0;
        corrected_experimental_spectrum_shift[0] =
            binned_normalized_experimental_spectrum[0] - mean_offset;

        let bin_shift_plus = BIN_SHIFT + 1;

        for i in 1..binned_normalized_experimental_spectrum.len() {
            if i >= bin_shift_plus {
                sum_offsets -= binned_normalized_experimental_spectrum[i - bin_shift_plus];
            }

            let add_idx = i + BIN_SHIFT;
            if add_idx < binned_normalized_experimental_spectrum.len() {
                sum_offsets += binned_normalized_experimental_spectrum[add_idx];
            }

            let old_center = i - 1;
            if old_center < binned_normalized_experimental_spectrum.len() {
                sum_offsets += binned_normalized_experimental_spectrum[old_center];
            }

            if i < binned_normalized_experimental_spectrum.len() {
                sum_offsets -= binned_normalized_experimental_spectrum[i];
            }

            let mean_offset = sum_offsets / 150.0;
            corrected_experimental_spectrum_shift[i] =
                binned_normalized_experimental_spectrum[i] - mean_offset;
        }

        corrected_experimental_spectrum_shift
    }
}

#[cfg(test)]
mod tests {
    use std::{env, io::Write};
    use super::*;
    use ndarray::{Axis, Array1};
    use crate::{
        configuration::{Configuration, FinalizedConfiguration},
        utils::tests::{get_eng_experimental_spectrum, get_eng_fast_xcorr_spectrum},
    };

    /// Checks the Xcorr spectrum (y-prime) against the data provided by Eng
    #[test]
    fn test_preprocessed_experimental_spectrum() {
        let expected_xcorr_spec = get_eng_fast_xcorr_spectrum();

        // Create config for low resolution data
        let config: FinalizedConfiguration = Configuration {
            bin_size: 1.0005,
            bin_offset: 0.4,
            use_flanking_peaks: true,
            ..Configuration::default()
        }
            .into();

        let experimental_spectrum = get_eng_experimental_spectrum();

        let spectrum = PreprocessedSpectrum::process(
            &config,
            (&experimental_spectrum.0, &experimental_spectrum.1),
        )
            .unwrap();

        let mut rouneded_xcorr_sped = spectrum
            .preprocessed_experimental_spectrum
            .iter()
            .map(|x| (x * 100.0).round() / 100.0)
            .collect::<Array1<f64>>();

        // Just select the values that are in the expected spectrum from Eng
        rouneded_xcorr_sped =
            rouneded_xcorr_sped.select(Axis(0), expected_xcorr_spec.0.as_slice().unwrap());

        let rounded_expected_xcorr_spec = expected_xcorr_spec
            .1
            .iter()
            .map(|x| (x * 100.0).round() / 100.0)
            .collect::<Array1<f64>>();

        if env::var("VERBOSE").is_ok() {
            let max_mz = experimental_spectrum
                .0
                .iter()
                .fold(f64::NEG_INFINITY, |acc, &x| acc.max(x));

            let binned_experimental_spectrum = PreprocessedSpectrum::experimental_spectrum_binning(
                &experimental_spectrum.0,
                &experimental_spectrum.1,
                max_mz,
                config.bin_size,
                config.bin_offset,
            )
                .unwrap();

            let mut binned_normalized_experimental_spectrum =
                PreprocessedSpectrum::experimental_spectrum_normalization(
                    max_mz,
                    binned_experimental_spectrum,
                    config.bin_size,
                    config.bin_offset,
                    config.use_flanking_peaks,
                )
                    .unwrap();

            binned_normalized_experimental_spectrum = binned_normalized_experimental_spectrum
                .select(Axis(0), expected_xcorr_spec.0.as_slice().unwrap());

            let output_file =
                std::fs::File::create("DIGSETK_fast_xcorr_preprocessing.tsv").unwrap();
            let mut output_writer = std::io::BufWriter::new(output_file);

            let _ = output_writer
                .write("bin\texpected\tcalc_binned\tcalc_prep\n".as_bytes())
                .unwrap();
            for (idx, (expected, (calc_binned, calc_prep))) in rounded_expected_xcorr_spec
                .iter()
                .zip(
                    binned_normalized_experimental_spectrum
                        .iter()
                        .zip(rouneded_xcorr_sped.iter()),
                )
                .enumerate()
            {
                let _ = output_writer
                    .write(format!("{idx}\t{expected}\t{calc_binned}\t{calc_prep}\n").as_bytes())
                    .unwrap();
            }
        }

        assert_eq!(rouneded_xcorr_sped, rounded_expected_xcorr_spec)
    }
}