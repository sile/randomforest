use super::core::{RandomForest, RandomForestOptions};
use crate::criterion::ClassificationCriterion;
use crate::functions;
use crate::table::Table;
use std::io::{Read, Write};
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

    /// Sets the maximum number of samples used to train each decision tree.
    ///
    /// The default value is the number of rows in the target table.
    pub fn max_samples(&mut self, max: NonZeroUsize) -> &mut Self {
        self.inner.max_samples(max);
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

    /// Builds a classifier model fitting the given table.
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

/// Random forest classifier.
#[derive(Debug)]
pub struct RandomForestClassifier {
    inner: RandomForest,
}

impl RandomForestClassifier {
    /// Builds a classifier model fitting the given table with the default settings.
    pub fn fit<T: ClassificationCriterion>(criterion: T, table: Table) -> Self {
        RandomForestClassifierOptions::default().fit(criterion, table)
    }

    /// Predicts the target value for the given features.
    pub fn predict(&self, features: &[f64]) -> f64 {
        functions::most_frequent(self.inner.predict(features))
    }

    /// Returns an iterator that iterates over a target value predicted by each decision tree.
    pub fn predict_individuals<'a>(
        &'a self,
        features: &'a [f64],
    ) -> impl 'a + Iterator<Item = f64> {
        self.inner.predict(features)
    }

    /// Writes this classifier to the given byte stream.
    pub fn serialize<W: Write>(&self, writer: W) -> std::io::Result<()> {
        self.inner.serialize(writer)
    }

    /// Reads a classifier from the given byte stream.
    pub fn deserialize<R: Read>(reader: R) -> std::io::Result<Self> {
        let inner = RandomForest::deserialize(reader)?;
        Ok(Self { inner })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::criterion::Gini;
    use crate::table::{ColumnType, TableBuilder};

    #[test]
    fn classification_works() -> Result<(), anyhow::Error> {
        let features = [
            &[0.0, 1.0, 0.0][..],
            &[1.0, 1.0, 1.0][..],
            &[0.0, 0.0, 1.0][..],
            &[1.0, 0.0, 0.0][..],
            &[1.0, 0.0, 0.0][..],
            &[0.0, 1.0, 0.0][..],
            &[1.0, 0.0, 1.0][..],
        ];

        let target = [1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0];
        let train_len = target.len() - 1;

        let mut table_builder = TableBuilder::new();
        table_builder.set_feature_column_types(&[
            ColumnType::Categorical,
            ColumnType::Categorical,
            ColumnType::Categorical,
        ])?;

        for (xs, y) in features.iter().zip(target.iter()).take(train_len) {
            table_builder.add_row(xs, *y)?;
        }
        let table = table_builder.build()?;

        let classifier = RandomForestClassifierOptions::new()
            .seed(0)
            .fit(Gini, table.clone());
        assert_eq!(classifier.predict(&features[train_len]), 0.0);

        let classifier_parallel = RandomForestClassifierOptions::new()
            .seed(0)
            .parallel()
            .fit(Gini, table);
        assert_eq!(
            classifier.predict(&features[train_len]),
            classifier_parallel.predict(&features[train_len])
        );

        let mut bytes = Vec::new();
        classifier.serialize(&mut bytes)?;
        let classifier_deserialized = RandomForestClassifier::deserialize(&mut &bytes[..])?;
        assert_eq!(
            classifier.predict(&features[train_len]),
            classifier_deserialized.predict(&features[train_len])
        );

        Ok(())
    }
}
