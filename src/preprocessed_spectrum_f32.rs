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
const NEARLY_ZERO: f32 = 1e-6;

pub struct PreprocessedSpectrumF32 {
    /// Sparse SP score matrix
    pub sp_matrix: Vec<Option<Vec<f32>>>,
    /// y' prime from equation 6 in https://pubs.acs.org/doi/10.1021/pr800420s
    pub preprocessed_experimental_spectrum: Array1<f32>,
}

impl PreprocessedSpectrumF32 {
    pub fn process(
        config: &FinalizedConfiguration,
        experimental_spectrum: (&Array1<f32>, &Array1<f32>),
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
            .filter(|&(_, &intensity)| intensity >= config.minimum_intensity as f32)
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
                .filter(|&(_, &mz)| mz <= min_mz as f32 && mz >= max_mz as f32)
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
            .fold(f32::NEG_INFINITY, |acc, &x| acc.max(x));

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
            .fold(f32::NEG_INFINITY, |a, &b| a.max(b));

        let sp_matrix = Self::build_sparse_sp_score(
            &binned_experimental_spectrum,
            max_intensity_sqrt,
            config.sp_matrix_enable,
        );

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

    pub fn calc_number_of_bins(mz_max: f32, bin_size: f32, bin_offset: f32) -> usize {
        (mz_max / bin_size + 1.0 - bin_offset) as usize + 2 + BIN_SHIFT
    }

    pub fn calc_binned_position(mz: f32, bin_size: f32, bin_offset: f32) -> usize {
        (mz / bin_size + 1.0 - bin_offset) as usize
    }

    fn experimental_spectrum_binning(
        mz: &Array1<f32>,
        intensities: &Array1<f32>,
        mz_max: f32,
        bin_size: f64,
        bin_offset: f64,
    ) -> Result<Array1<f32>, Error> {
        let number_of_bins = Self::calc_number_of_bins(mz_max, bin_size as f32, bin_offset as f32);
        let mut binned_spectrum: Array1<f32> = Array1::zeros(number_of_bins);

        for (mz, intensity) in mz.iter().zip(intensities.iter()) {
            let index = Self::calc_binned_position(*mz, bin_size as f32, bin_offset as f32);
            binned_spectrum[index] = binned_spectrum[index].max(intensity.sqrt());
        }

        Ok(binned_spectrum)
    }

    fn build_sparse_sp_score(
        binned_theoretical_spectrum: &Array1<f32>,
        max_intensity_sqrt: f32,
        sp_matrix_enable: bool,
    ) -> Vec<Option<Vec<f32>>> {
        let matrix_size = binned_theoretical_spectrum.len() / SP_MATRIX_SIZE + 1;
        let mut sparse: Vec<Option<Vec<f32>>> = vec![None; matrix_size];

        if max_intensity_sqrt <= 0.0 || !sp_matrix_enable {
            return sparse; //TODO: Maybe replace with Vec::new() if that still compatible
        }

        for i in 0..binned_theoretical_spectrum.len() {
            let normalized = 100.0 * binned_theoretical_spectrum[i] / max_intensity_sqrt;
            if normalized > NEARLY_ZERO {
                let x = i / SP_MATRIX_SIZE;
                let y = i - (x * SP_MATRIX_SIZE);

                if sparse[x].is_none() {
                    sparse[x] = Some(vec![0.0_f32; SP_MATRIX_SIZE]);
                }

                if let Some(ref mut row) = sparse[x] {
                    row[y] = normalized;
                }
            }
        }
        sparse
    }

    fn experimental_spectrum_normalization(
        mz_max: f32,
        mut binned_theoretical_spectrum: Array1<f32>,
        bin_size: f64,
        bin_offset: f64,
        use_flanking_peaks: bool,
    ) -> Result<Array1<f32>, Error> {
        let highest_ion = Self::calc_binned_position(mz_max, bin_size as f32, bin_offset as f32);
        let windows_size = (highest_ion as f64 / NUM_WINDOWS_FOR_NORMALIZATION as f64) as usize + 1;

        for window_start in (0..binned_theoretical_spectrum.len()).step_by(windows_size) {
            let window_end = (window_start + windows_size).min(binned_theoretical_spectrum.len());
            let mut window = binned_theoretical_spectrum.slice_mut(s![window_start..window_end]);

            let window_max = window.iter().fold(f32::NEG_INFINITY, |acc, &x| acc.max(x));
            let window_intensity_cutoff = 0.05 * window_max;
            window.mapv_inplace(|value| {
                if value <= window_intensity_cutoff as f32 {
                    value
                } else {
                    value / window_max as f32 * 50.0
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

    fn xcorr_preprocessing(binned_normalized_experimental_spectrum: &Array1<f32>) -> Array1<f32> {
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
