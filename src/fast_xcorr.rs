use crate::preprocessed_spectrum::PreprocessedSpectrum;
use crate::{configuration::FinalizedConfiguration, error::Error};
use crate::{scoring_result::ScoringResult, utils::create_theoretical_fragments};
use ndarray::Array1;
use rustyms::{CompoundPeptidoformIon, Fragment};
//use crate::preprocessed_spectrum::PreprocessedSpectrum;

// +/- m/z shift for the xcorr calculation.
pub const BIN_SHIFT: usize = 75;
/// According to the original authors, after binning the experimental spectrum, the maximum intensity is normalized
/// over a number of fixed windows.
pub const NUM_WINDOWS_FOR_NORMALIZATION: u8 = 10;

/// Size of each chunk in the sparse SP score matrix.
const SP_MATRIX_SIZE: usize = 100;

/// A small value to consider as zero in the SP score matrix.
const NEARLY_ZERO: f64 = 1e-6;

pub struct FastXcorr<'a> {
    config: &'a FinalizedConfiguration,
    fragment_charge: usize,
    spectrum: PreprocessedSpectrum,
}

impl FastXcorr<'_> {
    pub fn new<'a>(
        config: &'a FinalizedConfiguration,
        experimental_spectrum: (&'a Array1<f64>, &'a Array1<f64>),
        charge: usize,
    ) -> Result<FastXcorr<'a>, Error> {
        let spectrum = PreprocessedSpectrum::process(config, experimental_spectrum)?;

        let mut fragment_charge = (charge - 1).max(1);
        if fragment_charge > config.max_fragment_charge {
            fragment_charge = config.max_fragment_charge;
        }

        Ok(FastXcorr {
            config,
            fragment_charge,
            spectrum,
        })
    }

    pub fn xcorr_spectra(
        theoretical_spectrum: &Array1<f64>,
        preprocessed_experimental_spectrum: &Array1<f64>,
        bin_size: f64,
        bin_offset: f64,
    ) -> f64 {
        let xcorr: f64 = theoretical_spectrum
            .iter()
            .map(|mz| {
                let index = PreprocessedSpectrum::calc_binned_position(*mz, bin_size, bin_offset);
                if index < preprocessed_experimental_spectrum.len() {
                    preprocessed_experimental_spectrum[index]
                } else {
                    0.0
                }
            })
            .sum();

        xcorr * 0.005
    }

    fn match_ions_sp_score_based(
        theoretical_spectrum: &Array1<f64>,
        sp_matrix: &[Option<Vec<f64>>],
        bin_size: f64,
        bin_offset: f64,
    ) -> usize {
        theoretical_spectrum
            .iter()
            .map(|mz| {
                let index = PreprocessedSpectrum::calc_binned_position(*mz, bin_size, bin_offset);
                let sp_score = Self::find_sp_score(sp_matrix, index);
                if sp_score > NEARLY_ZERO { 1 } else { 0 }
            })
            .sum()
    }

    pub fn find_sp_score(sparse: &[Option<Vec<f64>>], bin: usize) -> f64 {
        let x = bin / SP_MATRIX_SIZE;

        if x >= sparse.len() || bin == 0 || sparse[x].is_none() {
            return 0.0_f64;
        }

        let y = bin - (x * SP_MATRIX_SIZE);
        let row = sparse[x].as_ref().unwrap();
        row[y]
    }

    pub fn create_theoretical_fragments(
        &self,
        peptide: &CompoundPeptidoformIon,
    ) -> Result<Vec<Fragment>, Error> {
        create_theoretical_fragments(
            peptide,
            &self.config.fragmentation_model,
            self.fragment_charge,
        )
    }

    pub fn create_threoretical_mz(theoretical_fragments: &[Fragment]) -> Array1<f64> {
        theoretical_fragments
            .iter()
            .filter_map(|f| f.mz(rustyms::MassMode::Monoisotopic).map(|mz| mz.value))
            .collect::<Array1<f64>>()
    }

    /** extracted from xcorr_peptide to have more fine-grained profiling in benchmark */
    pub fn prepare_theoretical_peptide(&self, peptide: &str) -> Result<(f64, f64, Array1<f64>, usize), Error> {
        let peptide = CompoundPeptidoformIon::pro_forma(peptide, None)
            .map_err(Error::InvalidPeptideSequence)?;

        let (min_theoretical_mass, max_theoretical_mass) =
            match peptide.formulas().mass_bounds().into_option() {
                Some((min, max)) => (min.monoisotopic_mass().value, max.monoisotopic_mass().value),
                None => (-1.0, -1.0),
            };

        let theoretical_fragments = self.create_theoretical_fragments(&peptide)?;
        let theoretical_mz = Self::create_threoretical_mz(&theoretical_fragments);
        let ions_total = theoretical_mz.len();


        Ok((min_theoretical_mass, max_theoretical_mass, theoretical_mz, ions_total))
    }

    pub fn xcorr_peptide(&self, peptide: &str) -> Result<ScoringResult, Error> {
        let peptide = CompoundPeptidoformIon::pro_forma(peptide, None)
            .map_err(Error::InvalidPeptideSequence)?;

        let (min_theoretical_mass, max_theoretical_mass) =
            match peptide.formulas().mass_bounds().into_option() {
                Some((min, max)) => (min.monoisotopic_mass().value, max.monoisotopic_mass().value),
                None => (-1.0, -1.0),
            };

        let theoretical_fragments = self.create_theoretical_fragments(&peptide)?;
        let theoretical_mz = Self::create_threoretical_mz(&theoretical_fragments);
        let ions_total = theoretical_mz.len();

        //TODO: skip calucation if sp_score is disabled...
        let ions_matched = Self::match_ions_sp_score_based(
            &theoretical_mz,
            &self.spectrum.sp_matrix,
            self.config.bin_size,
            self.config.bin_offset,
        );

        let score = Self::xcorr_spectra(
            &theoretical_mz,
            &self.spectrum.preprocessed_experimental_spectrum,
            self.config.bin_size,
            self.config.bin_offset,
        );

        Ok(ScoringResult {
            score,
            min_theoretical_mass,
            max_theoretical_mass,
            ions_total,
            ions_matched,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{env, io::Write};
    use std::fs::File;
    use itertools::multiunzip;
    use ndarray_stats::DeviationExt;
    use polars::prelude::*;
    use rayon::prelude::*;
    use rustyms::MassMode;

    use crate::{
        configuration::{Configuration, FinalizedConfiguration},
        utils::tests::{get_spectrum, read_test_data},
    };

    /// Tests the xcorr calculation against data provided by J. Eng
    #[test]
    fn test_xcorr_eng_data() {
        // Load experimental spectrum from Parquet file
        let experimental_spectrum =
            ParquetReader::new(std::fs::File::open("test_files/eng/DIGSETK.parquet").unwrap())
                .read_parallel(ParallelStrategy::None)
                .finish()
                .unwrap();

        let experimental_spectrum = (
            experimental_spectrum["mz"]
                .f64()
                .unwrap()
                .to_ndarray()
                .unwrap()
                .to_owned(),
            experimental_spectrum["intensity"]
                .f64()
                .unwrap()
                .to_ndarray()
                .unwrap()
                .to_owned(),
        );

        let config: FinalizedConfiguration = Configuration {
            bin_size: 1.0005,
            bin_offset: 0.4,
            ..Configuration::default()
        }
            .into();

        let xcorr = FastXcorr::new(
            &config,
            (&experimental_spectrum.0, &experimental_spectrum.1),
            1,
        )
            .unwrap();

        if env::var("VERBOSE").is_ok() {
            let peptide = CompoundPeptidoformIon::pro_forma("DIGSETK", None).unwrap();

            let mut binned_theoretical_spectrum =
                Array1::zeros(xcorr.spectrum.preprocessed_experimental_spectrum.len());

            for fragment in &xcorr.create_theoretical_fragments(&peptide).unwrap() {
                let mz = fragment.mz(MassMode::Monoisotopic).unwrap().value;
                let index = PreprocessedSpectrum::calc_binned_position(mz, config.bin_size, config.bin_offset);
                if index < binned_theoretical_spectrum.len() {
                    binned_theoretical_spectrum[index] = 50.0;
                }
            }

            let output_file =
                std::fs::File::create("DIGSETK__fast_xcorr_bin___theoretical_bin.tsv").unwrap();
            let mut output_writer = std::io::BufWriter::new(output_file);

            let _ = output_writer
                .write("bin\texperimental_bin\ttheoretical_bin\n".as_bytes())
                .unwrap();
            for (idx, (fast_xcorr_bin, theoretical_bin)) in xcorr
                .spectrum
                .preprocessed_experimental_spectrum
                .iter()
                .zip(binned_theoretical_spectrum.iter())
                .enumerate()
            {
                let _ = output_writer
                    .write(format!("{idx}\t{fast_xcorr_bin}\t{theoretical_bin}\n").as_bytes())
                    .unwrap();
            }
        }

        let scoring = xcorr.xcorr_peptide("DIGSETK").unwrap();
        println!("{scoring}");
        assert_eq!((scoring.score * 100.0).round() / 100.0, 2.92);
    }

    // extract one iteration from test_xcorr to run in a minimal test case. Used as sanity check
    // for benchmark
    #[test]
    #[ignore = "only used to sanity check the benchmark"]
    pub fn sanity_check_benchmark() {
        let (mz_array, intensity_array) = get_spectrum("12745");

        let config: FinalizedConfiguration = Configuration {
            use_flanking_peaks: true,
            max_fragment_charge: 5,
            ..Configuration::default()
        }
            .into();

        let fast_xcorr = FastXcorr::new(&config, (&mz_array, &intensity_array), 2).unwrap();
        let score = fast_xcorr.xcorr_peptide("GPISMTK").unwrap();
        println!("{}", score.score)
    }

    #[test]
    pub fn vector_to_tsv() {
        let all_peptides = peptides_from_test_data();

        // Create output file
        let file = File::create("test_files/all_proforma_testdata.tsv").unwrap();

        let column = Column::new("proforma".into(), all_peptides);

        let mut df: DataFrame = DataFrame::new(vec!(column)).unwrap();

        // Write as TSV (CSV with tab delimiter)
        CsvWriter::new(file)
            .with_separator(b'\t')
            .include_header(true)
            .finish(&mut df).unwrap();
    }

    /** implemented with data from test_xcorr. match one experimental spectrum against all
     theoretical spectrums. Useful to benchmark/profile the xcorrrs-algorithm and compare which
     parts of its implementation use how much time. The 3 parts we want to compare are:
     - PreprocessedSpectrum::process
     - fast_xcorr.prepare_theoretical_peptide
     - FastXcorr::xcorr_spectra
    Can be started with: RUSTFLAGS="-C force-frame-pointers=yes" cargo flamegraph --profile profiling --no-inline --unit-test -- fast_xcorr::tests::benchmark --ignored */
    #[test]
    #[ignore]
    #[inline(never)]
    pub fn benchmark() {
        let all_peptides = peptides_from_test_data();
        let (mz_array, intensity_array) = get_spectrum("12745");

        // execute this code in a loop like for _i in 0..100 {} to ensure small calls are hit aswell
        benchmark_internal((&mz_array, &intensity_array), &all_peptides);

    }

    fn peptides_from_test_data() -> Vec<String> {
        let test_data_df = read_test_data();
        let all_peptides: Vec<String> = (0..test_data_df.height())
            .into_par_iter()
            .map(|idx| {
                let proforma_peptide = test_data_df["proforma_peptide"]
                    .str()
                    .unwrap()
                    .get(idx)
                    .unwrap();
                proforma_peptide.to_string()
            })
            .collect();
        all_peptides
    }

    #[inline(never)]
    fn benchmark_internal(experimental_spectrum: (&Array1<f64>, &Array1<f64>), all_peptides: &[String]) {
        let config: FinalizedConfiguration = Configuration {
            use_flanking_peaks: true,
            max_fragment_charge: 5,
            ..Configuration::default()
        }
            .into();

        let spectrum = PreprocessedSpectrum::process(&config, experimental_spectrum).unwrap();
        let empty_spec: PreprocessedSpectrum = PreprocessedSpectrum {
            sp_matrix: vec![],
            preprocessed_experimental_spectrum: Default::default(),
        };

        let fast_xcorr = FastXcorr {
            config: &config,
            fragment_charge: 1,
            spectrum: empty_spec,
        };

        for peptide in all_peptides {
            let (_min_theoretical_mass, _max_theoretical_mass, theoretical_mz, _ions_total) =
                fast_xcorr.prepare_theoretical_peptide(peptide).unwrap();

            println!("{}: {}", peptide, FastXcorr::xcorr_spectra(&theoretical_mz, &spectrum.preprocessed_experimental_spectrum, config.bin_size, config.bin_offset));
        }
    }

    // Test xcorr implementations against high-res MS data
    #[test]
    fn test_xcorr() {
        let comet_df = read_test_data();

        #[allow(clippy::type_complexity)]
        let results: Vec<(i64, String, f64, f64, u64, u64, u64, u64)> = (0..comet_df.height())
            .into_par_iter()
            .map(|idx| {
                let scan = comet_df["scan"].i64().unwrap().get(idx).unwrap();
                let comet_xcorr = comet_df["xcorr"].f64().unwrap().get(idx).unwrap();
                let comet_ions_total =
                    comet_df["ions_total"].i64().unwrap().get(idx).unwrap() as u64;
                let comet_ions_matched =
                    comet_df["ions_matched"].i64().unwrap().get(idx).unwrap() as u64;
                let proforma_peptide = comet_df["proforma_peptide"]
                    .str()
                    .unwrap()
                    .get(idx)
                    .unwrap();
                let charge = comet_df["charge"].i64().unwrap().get(idx).unwrap() as usize;

                let (mz_array, intensity_array) = get_spectrum(scan.to_string().as_str());

                let config: FinalizedConfiguration = Configuration {
                    use_flanking_peaks: true,
                    max_fragment_charge: 5,
                    ..Configuration::default()
                }
                    .into();

                // fast xcorr implementation
                let fast_xcorr =
                    FastXcorr::new(&config, (&mz_array, &intensity_array), charge).unwrap();

                let scoring = fast_xcorr.xcorr_peptide(proforma_peptide).unwrap();

                (
                    scan,
                    proforma_peptide.to_string(),
                    comet_xcorr,
                    scoring.round_score(3),
                    comet_ions_matched,
                    scoring.ions_matched as u64,
                    comet_ions_total,
                    scoring.ions_total as u64,
                )
            })
            .collect();

        let (
            scan_col,
            peptide_col,
            comet_xcorr_col,
            rs_xcorr_col,
            comet_ions_match_col,
            rs_ions_matched_col,
            comet_ions_total_col,
            rs_ions_total_col,
        ): (
            Vec<_>,
            Vec<_>,
            Vec<_>,
            Vec<_>,
            Vec<_>,
            Vec<_>,
            Vec<_>,
            Vec<_>,
        ) = multiunzip(results);

        let mut xcorrrs_df = DataFrame::new(vec![
            Column::new("scan".into(), scan_col),
            Column::new("modified_peptide".into(), peptide_col),
            Column::new("comet_xcorr".into(), comet_xcorr_col),
            Column::new("rs_xcorr".into(), rs_xcorr_col),
            Column::new("comet_ions_match".into(), comet_ions_match_col),
            Column::new("rs_ions_matched".into(), rs_ions_matched_col),
            Column::new("comet_ions_total".into(), comet_ions_total_col),
            Column::new("rs_ions_total".into(), rs_ions_total_col),
        ])
            .unwrap();

        CsvWriter::new(std::fs::File::create("comparison.tsv").unwrap())
            .with_separator(b'\t')
            .finish(&mut xcorrrs_df)
            .unwrap();

        if env::var("VERBOSE").is_ok() {
            let max_comet_xcorr = xcorrrs_df["comet_xcorr"].f64().unwrap().max().unwrap();

            let max_calculated_xcorr = xcorrrs_df["rs_xcorr"].f64().unwrap().max().unwrap();

            let mut plot = plotly::Plot::new();
            let diagonal_trace =
                plotly::Scatter::new(vec![0.0, max_comet_xcorr], vec![0.0, max_calculated_xcorr])
                    .mode(plotly::common::Mode::Lines)
                    .marker(plotly::common::Marker::default().color("red"))
                    .hover_info(plotly::common::HoverInfo::None)
                    .show_legend(false);

            let correlation_trace = plotly::Scatter::new(
                xcorrrs_df["comet_xcorr"].f64().unwrap().to_vec(),
                xcorrrs_df["rs_xcorr"].f64().unwrap().to_vec(),
            )
                .mode(plotly::common::Mode::Markers)
                .marker(plotly::common::Marker::default().color("blue"))
                .show_legend(false);

            plot.add_trace(diagonal_trace);
            plot.add_trace(correlation_trace);

            plot.set_layout(
                plotly::Layout::new()
                    .title("Comet xcorr vs rs_xcorr")
                    .x_axis(
                        plotly::layout::Axis::new()
                            .title("Comet xcorr")
                            .constrain(plotly::layout::AxisConstrain::Domain),
                    )
                    .y_axis(
                        plotly::layout::Axis::new()
                            .title("rs_xcorr")
                            .scale_anchor("x"),
                    ),
            );
            plot.write_html("99-fast_xcorrrs_vs_comet_xcorr.html");
        }

        // Normalize comet xcorrs and calculates xcorrs
        let comet_xcorr_max = xcorrrs_df
            .column("comet_xcorr")
            .unwrap()
            .f64()
            .unwrap()
            .max()
            .unwrap();

        let fast_xcorrrs_max = xcorrrs_df
            .column("rs_xcorr")
            .unwrap()
            .f64()
            .unwrap()
            .max()
            .unwrap();

        let max_score = comet_xcorr_max.max(fast_xcorrrs_max);

        let scaled_comet_xcorr = xcorrrs_df
            .column("comet_xcorr")
            .unwrap()
            .f64()
            .unwrap()
            .to_ndarray()
            .unwrap()
            .mapv(|x| x / max_score);

        let scaled_fast_xcorrrs = xcorrrs_df
            .column("rs_xcorr")
            .unwrap()
            .f64()
            .unwrap()
            .to_ndarray()
            .unwrap()
            .mapv(|x| x / max_score);

        let rmse_fast_xcorrs = scaled_comet_xcorr
            .mean_sq_err(&scaled_fast_xcorrrs)
            .unwrap();

        println!("RMSE comet xcorr vs fast xcorrrs: {rmse_fast_xcorrs}");

        assert!(
            rmse_fast_xcorrs < 0.0002,
            "fast xcorr RMSE {rmse_fast_xcorrs} >= 0.0002"
        );
    }
}