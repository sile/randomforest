use super::core::{RandomForest, RandomForestOptions};
use crate::criterion::RegressionCriterion;
use crate::functions;
use crate::table::Table;
use std::io::{Read, Write};
use std::num::NonZeroUsize;

/// Random forest options.
#[derive(Debug, Clone, Default)]
pub struct RandomForestRegressorOptions {
    inner: RandomForestOptions,
}

impl RandomForestRegressorOptions {
    /// Makes a `RandomForestRegressorOptions` instance with the default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the random generator seed.
    ///
    /// The default value is random.
    pub fn seed(&mut self, seed: u64) -> &mut Self {
        self.inner.seed(seed);
        self
    }

    /// Sets the number of decision trees.
    ///
    /// The default value is `100`.
    pub fn trees(&mut self, trees: NonZeroUsize) -> &mut Self {
        self.inner.trees(trees);
        self
    }

    /// Sets the maximum number of candidate features used to determine each decision tree node.
    ///
    /// The default value is `sqrt(the number of features)`.
    pub fn max_features(&mut self, max: NonZeroUsize) -> &mut Self {
        self.inner.max_features(max);
        self
    }

    /// Sets the maximum number of samples used to train each decision tree.
    ///
    /// The default value is the number of rows in the target table.
    pub fn max_samples(&mut self, max: NonZeroUsize) -> &mut Self {
        self.inner.max_samples(max);
        self
    }

    /// Enables parallel executions of `RandomForestRegressor::fit`.
    ///
    /// This library use `rayon` for parallel execution.
    /// Please see [the rayon document](https://docs.rs/rayon) if you want to configure the behavior
    /// (e.g., the number of worker threads).
    pub fn parallel(&mut self) -> &mut Self {
        self.inner.parallel();
        self
    }

    /// Builds a regressor model fitting the given table.
    pub fn fit<T: RegressionCriterion>(&self, criterion: T, table: Table) -> RandomForestRegressor {
        RandomForestRegressor {
            inner: self.inner.fit(criterion, true, table),
        }
    }
}

/// Random forest regressor.
#[derive(Debug)]
pub struct RandomForestRegressor {
    inner: RandomForest,
}

impl RandomForestRegressor {
    /// Builds a regressor model fitting the given table with the default settings.
    pub fn fit<T: RegressionCriterion>(criterion: T, table: Table) -> Self {
        RandomForestRegressorOptions::default().fit(criterion, table)
    }

    /// Predicts the target value for the given features.
    pub fn predict(&self, features: &[f64]) -> f64 {
        functions::mean(self.inner.predict(features))
    }

    /// Returns an iterator that iterates over a target value predicted by each decision tree.
    pub fn predict_individuals<'a>(
        &'a self,
        features: &'a [f64],
    ) -> impl 'a + Iterator<Item = f64> {
        self.inner.predict(features)
    }

    /// Writes this regressor to the given byte stream.
    pub fn serialize<W: Write>(&self, writer: W) -> std::io::Result<()> {
        self.inner.serialize(writer)
    }

    /// Reads a regressor from the given byte stream.
    pub fn deserialize<R: Read>(reader: R) -> std::io::Result<Self> {
        let inner = RandomForest::deserialize(reader)?;
        Ok(Self { inner })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::criterion::Mse;
    use crate::table::TableBuilder;

    #[test]
    fn regression_works() -> Result<(), anyhow::Error> {
        let features = [
            &[0.0, 2.0, 1.0, 0.0][..],
            &[0.0, 2.0, 1.0, 1.0][..],
            &[1.0, 2.0, 1.0, 0.0][..],
            &[2.0, 1.0, 1.0, 0.0][..],
            &[2.0, 0.0, 0.0, 0.0][..],
            &[2.0, 0.0, 0.0, 1.0][..],
            &[1.0, 0.0, 0.0, 1.0][..],
            &[0.0, 1.0, 1.0, 0.0][..],
            &[0.0, 0.0, 0.0, 0.0][..],
            &[2.0, 1.0, 0.0, 0.0][..],
            &[0.0, 1.0, 0.0, 1.0][..],
            &[1.0, 1.0, 1.0, 1.0][..],
            &[1.0, 2.0, 0.0, 0.0][..],
            &[2.0, 1.0, 1.0, 1.0][..],
        ];
        let target = [
            25.0, 30.0, 46.0, 45.0, 52.0, 23.0, 43.0, 35.0, 38.0, 46.0, 48.0, 52.0, 44.0, 30.0,
        ];
        let train_len = target.len() - 2;

        let mut table_builder = TableBuilder::new();
        for (xs, y) in features.iter().zip(target.iter()).take(train_len) {
            table_builder.add_row(xs, *y)?;
        }
        let table = table_builder.build()?;

        let regressor = RandomForestRegressorOptions::new()
            .seed(0)
            .fit(Mse, table.clone());
        assert_eq!(regressor.predict(&features[train_len]), 41.9785);
        assert_eq!(
            regressor.predict(&features[train_len + 1]),
            43.50333333333333
        );

        let regressor_parallel = RandomForestRegressorOptions::new()
            .seed(0)
            .parallel()
            .fit(Mse, table);
        assert_eq!(
            regressor.predict(&features[train_len]),
            regressor_parallel.predict(&features[train_len])
        );

        let mut bytes = Vec::new();
        regressor.serialize(&mut bytes)?;
        let regressor_deserialized = RandomForestRegressor::deserialize(&mut &bytes[..])?;
        assert_eq!(
            regressor.predict(&features[train_len]),
            regressor_deserialized.predict(&features[train_len])
        );

        Ok(())
    }
}
