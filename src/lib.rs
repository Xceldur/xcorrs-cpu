pub mod configuration;
pub mod error;
/// Fast xcorr implementation. Less accurate than the reported xcorr in the test data (RSME 0.00018), but faster.
pub mod fast_xcorr;
pub mod scoring_result;
// Various utilities
pub mod utils;
pub mod preprocessed_spectrum;
