use super::core::{RandomForest, RandomForestOptions};
use crate::criterion::RegressionCriterion;
use crate::functions;
use crate::table::Table;
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

    /// Sets the number of maximum candidate features used to determine each decision tree node.
    ///
    /// The default value is `sqrt(the number of features)`.
    pub fn max_features(&mut self, max: NonZeroUsize) -> &mut Self {
        self.inner.max_features(max);
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

    pub fn fit<T: RegressionCriterion>(&self, criterion: T, table: Table) -> RandomForestRegressor {
        RandomForestRegressor {
            inner: self.inner.fit(criterion, true, table),
        }
    }
}

// TODO: Support categorical features
#[derive(Debug)]
pub struct RandomForestRegressor {
    inner: RandomForest,
}

impl RandomForestRegressor {
    pub fn fit<T: RegressionCriterion>(criterion: T, table: Table) -> Self {
        RandomForestRegressorOptions::default().fit(criterion, table)
    }

    pub fn predict(&self, xs: &[f64]) -> f64 {
        functions::mean(self.inner.predict(xs))
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

        let regressor = RandomForestRegressorOptions::new().seed(0).fit(Mse, table);
        assert_eq!(regressor.predict(&features[train_len]), 41.9785);
        assert_eq!(
            regressor.predict(&features[train_len + 1]),
            43.50333333333333
        );

        Ok(())
    }
}
