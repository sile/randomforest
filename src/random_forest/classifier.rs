use super::core::{RandomForest, RandomForestOptions};
use crate::criterion::ClassificationCriterion;
use crate::functions;
use crate::table::Table;
use std::num::NonZeroUsize;

/// Random forest options.
#[derive(Debug, Clone, Default)]
pub struct RandomForestClassifierOptions {
    inner: RandomForestOptions,
}

impl RandomForestClassifierOptions {
    /// Makes a `RandomForestClassifierOptions` instance with the default settings.
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

    /// Enables parallel executions of `RandomForestClassifier::fit`.
    ///
    /// This library use `rayon` for parallel execution.
    /// Please see [the rayon document](https://docs.rs/rayon) if you want to configure the behavior
    /// (e.g., the number of worker threads).
    pub fn parallel(&mut self) -> &mut Self {
        self.inner.parallel();
        self
    }

    pub fn fit<T: ClassificationCriterion>(
        &self,
        criterion: T,
        table: Table,
    ) -> RandomForestClassifier {
        RandomForestClassifier {
            inner: self.inner.fit(criterion, false, table),
        }
    }
}

#[derive(Debug)]
pub struct RandomForestClassifier {
    inner: RandomForest,
}

impl RandomForestClassifier {
    pub fn fit<T: ClassificationCriterion>(criterion: T, table: Table) -> Self {
        RandomForestClassifierOptions::default().fit(criterion, table)
    }

    pub fn predict(&self, xs: &[f64]) -> f64 {
        functions::most_frequent(self.inner.predict(xs))
    }
}
