use std::{fmt::Display, ops::Deref};

use rustyms::{
    model::{FragmentationModel, PrimaryIonSeries},
    molecular_formula, NeutralLoss,
};
use serde::{Deserialize, Serialize};

use crate::error::Error;

/// Ion types used in configuration to select which ion series to use
/// for the fragmentation model.
///
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Ion {
    A,
    B,
    C,
    X,
    Y,
    Z,
}

impl Display for Ion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Ion::A => write!(f, "A"),
            Ion::B => write!(f, "B"),
            Ion::C => write!(f, "C"),
            Ion::X => write!(f, "X"),
            Ion::Y => write!(f, "Y"),
            Ion::Z => write!(f, "Z"),
        }
    }
}

impl TryFrom<String> for Ion {
    type Error = Error;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        match value.to_ascii_uppercase().as_str() {
            "A" => Ok(Ion::A),
            "B" => Ok(Ion::B),
            "C" => Ok(Ion::C),
            "X" => Ok(Ion::X),
            "Y" => Ok(Ion::Y),
            "Z" => Ok(Ion::Z),
            _ => Err(Self::Error::InvalidConfiguration(format!(
                "Unknown ion type: {value}",
            ))),
        }
    }
}

/// Configuration for the xcorr algorithm.
/// The configuration is similar to Comet's configuration parameters.
/// Although it is much simpler as it only contains parameters related to the
/// fragmentation model and the experimental spectrum processing.
///
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Configuration {
    /// Same as: https://uwpr.github.io/Comet/parameters/parameters_202501/fragment_bin_tol.html
    pub bin_size: f64,
    /// Same as: https://uwpr.github.io/Comet/parameters/parameters_202501/fragment_bin_offset.html
    pub bin_offset: f64,
    /// Same as: https://uwpr.github.io/Comet/parameters/parameters_202502/theoretical_fragment_ions.html
    pub use_flanking_peaks: bool,
    /// Same as: https://uwpr.github.io/Comet/parameters/parameters_202502/minimum_fragment_intensity.html
    pub minimum_intensity: f64,
    /// Same as: https://uwpr.github.io/Comet/parameters/parameters_202502/max_fragment_charge.html
    pub max_fragment_charge: usize,
    /// Ions to use in the fragmentation model.
    pub ions: Vec<Ion>,
    /// Same as: https://uwpr.github.io/Comet/parameters/parameters_202502/use_NL_ions.html
    pub use_neutral_loss_ions: bool,
    /// Same as: https://uwpr.github.io/Comet/parameters/parameters_202502/clear_mz_range.html
    pub clear_mz_range: Option<(f64, f64)>,
    // disable sp matrix calculation
    pub sp_matrix_enable: bool,
}

impl Default for Configuration {
    /// Default is for high-resolution MS2 data (e.g. Orbitrap) and CID/HCD
    fn default() -> Self {
        Self {
            bin_size: 0.02,
            bin_offset: 0.0,
            use_flanking_peaks: true,
            minimum_intensity: 0.0,
            max_fragment_charge: 3,
            ions: vec![Ion::B, Ion::Y],
            use_neutral_loss_ions: false,
            clear_mz_range: None,
            sp_matrix_enable: true
        }
    }
}

/// Finalized configuration for the xcorr algorithm.
/// This configuration is created from the `Configuration` struct and contains
/// a `FragmentationModel` that is built based on the selected ions.
///
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FinalizedConfiguration {
    /// Fragmentation model that is built based on the selected ions.
    pub fragmentation_model: FragmentationModel,
    /// Original configuration that was used to create this finalized configuration.
    configuration: Configuration,
}

impl Deref for FinalizedConfiguration {
    type Target = Configuration;

    fn deref(&self) -> &Self::Target {
        &self.configuration
    }
}

impl From<Configuration> for FinalizedConfiguration {
    fn from(configuration: Configuration) -> Self {
        let mut fragmentation_model = FragmentationModel::none().clone();

        if configuration.ions.contains(&Ion::A) {
            fragmentation_model.a = PrimaryIonSeries::default();
        };

        if configuration.ions.contains(&Ion::B) {
            fragmentation_model.b = PrimaryIonSeries::default();
        };

        if configuration.ions.contains(&Ion::C) {
            fragmentation_model.c = PrimaryIonSeries::default();
        };

        if configuration.ions.contains(&Ion::X) {
            fragmentation_model.x = PrimaryIonSeries::default();
        };

        if configuration.ions.contains(&Ion::Y) {
            fragmentation_model.y = PrimaryIonSeries::default();
        };

        if configuration.ions.contains(&Ion::Z) {
            fragmentation_model.z = PrimaryIonSeries::default();
        };

        if configuration.use_neutral_loss_ions {
            let mut b_ions = fragmentation_model.b.clone();
            b_ions = b_ions.neutral_losses(vec![
                NeutralLoss::Loss(molecular_formula!(H 2 O 1)),
                NeutralLoss::Loss(molecular_formula!(N 1 H 3)),
            ]);
            fragmentation_model.b = b_ions;

            let mut y_ions = fragmentation_model.y.clone();
            y_ions = y_ions.neutral_losses(vec![
                NeutralLoss::Loss(molecular_formula!(H 2 O 1)),
                NeutralLoss::Loss(molecular_formula!(N 1 H 3)),
            ]);
            fragmentation_model.y = y_ions //Fixed a bug? (Probably)
        }

        Self {
            fragmentation_model,
            configuration,
        }
    }
}

impl Default for FinalizedConfiguration {
    fn default() -> Self {
        Configuration::default().into()
    }
}
