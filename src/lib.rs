//! Random forest classifier and regressor.
//!
//! # Examples
//!
//! ```
//! use randomforest::criterion::Mse;
//! use randomforest::RandomForestRegressorOptions;
//! use randomforest::table::TableBuilder;
//!
//! # fn main() -> anyhow::Result<()> {
//! let features = [
//!     &[0.0, 2.0, 1.0, 0.0][..],
//!     &[0.0, 2.0, 1.0, 1.0][..],
//!     &[1.0, 2.0, 1.0, 0.0][..],
//!     &[2.0, 1.0, 1.0, 0.0][..],
//!     &[2.0, 0.0, 0.0, 0.0][..],
//!     &[2.0, 0.0, 0.0, 1.0][..],
//!     &[1.0, 0.0, 0.0, 1.0][..],
//!     &[0.0, 1.0, 1.0, 0.0][..],
//!     &[0.0, 0.0, 0.0, 0.0][..],
//!     &[2.0, 1.0, 0.0, 0.0][..],
//!     &[0.0, 1.0, 0.0, 1.0][..],
//!     &[1.0, 1.0, 1.0, 1.0][..],
//! ];
//! let target = [
//!     25.0, 30.0, 46.0, 45.0, 52.0, 23.0, 43.0, 35.0, 38.0, 46.0, 48.0, 52.0
//! ];
//!
//! let mut table_builder = TableBuilder::new();
//! for (xs, y) in features.iter().zip(target.iter()) {
//!    table_builder.add_row(xs, *y)?;
//! }
//! let table = table_builder.build()?;
//!
//! let regressor = RandomForestRegressorOptions::new()
//!     .seed(0)
//!     .fit(Mse, table);
//! assert_eq!(regressor.predict(&[1.0, 2.0, 0.0, 0.0]), 41.9785);
//! # Ok(())
//! # }
//! ```
#![warn(missing_docs)]
pub use self::random_forest::classifier::{RandomForestClassifier, RandomForestClassifierOptions};
pub use self::random_forest::regressor::{RandomForestRegressor, RandomForestRegressorOptions};

pub mod criterion;
pub mod table;

mod decision_tree;
mod functions;
mod random_forest;
