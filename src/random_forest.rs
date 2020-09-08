use crate::decision_tree::{DecisionTreeOptions, DecisionTreeRegressor};
use crate::functions;
use crate::table::Table;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use std::num::NonZeroUsize;

/// Random forest options.
#[derive(Debug, Clone)]
pub struct RandomForestOptions {
    trees: NonZeroUsize,
    max_features: Option<NonZeroUsize>,
    seed: Option<u64>,
    parallel: bool,
}

impl RandomForestOptions {
    /// Makes a `RandomForestOptions` instance with the default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the random generator seed.
    ///
    /// The default value is random.
    pub fn seed(&mut self, seed: u64) -> &mut Self {
        self.seed = Some(seed);
        self
    }

    /// Sets the number of decision trees.
    ///
    /// The default value is `100`.
    pub fn trees(&mut self, trees: NonZeroUsize) -> &mut Self {
        self.trees = trees;
        self
    }

    /// Sets the number of maximum candidate features used to determine each decision tree node.
    ///
    /// The default value is `sqrt(the number of features)`.
    pub fn max_features(&mut self, max: NonZeroUsize) -> &mut Self {
        self.max_features = Some(max);
        self
    }

    /// Enables parallel executions of `RandomForest::fit`.
    ///
    /// This library use `rayon` for parallel execution.
    /// Please see [the rayon document](https://docs.rs/rayon) if you want to configure the behavior
    /// (e.g., the number of worker threads).
    pub fn parallel(&mut self) -> &mut Self {
        self.parallel = true;
        self
    }

    pub fn fit(&self, table: Table) -> RandomForest {
        let max_features = self.decide_max_features(&table);
        let forest = if self.parallel {
            self.tree_rngs()
                .collect::<Vec<_>>()
                .into_par_iter()
                .map(|mut rng| Self::tree_fit(&mut rng, &table, max_features))
                .collect::<Vec<_>>()
        } else {
            self.tree_rngs()
                .map(|mut rng| Self::tree_fit(&mut rng, &table, max_features))
                .collect::<Vec<_>>()
        };
        RandomForest { forest }
    }

    fn tree_fit<R: Rng + ?Sized>(
        rng: &mut R,
        table: &Table,
        max_features: usize,
    ) -> DecisionTreeRegressor {
        let table = table.bootstrap_sample(rng);
        let tree_options = DecisionTreeOptions {
            max_features: Some(max_features),
        };
        DecisionTreeRegressor::fit(rng, table, tree_options)
    }

    fn tree_rngs(&self) -> impl Iterator<Item = StdRng> {
        let seed_u64 = self.seed.unwrap_or_else(|| rand::thread_rng().gen());
        let mut seed = [0u8; 32];
        (&mut seed[0..8]).copy_from_slice(&seed_u64.to_be_bytes()[..]);
        let mut rng = StdRng::from_seed(seed);
        (0..self.trees.get()).map(move |_| {
            let mut seed = [0u8; 32];
            rng.fill(&mut seed);
            StdRng::from_seed(seed)
        })
    }

    fn decide_max_features(&self, table: &Table) -> usize {
        if let Some(n) = self.max_features {
            n.get()
        } else {
            (table.features_len() as f64).sqrt().ceil() as usize
        }
    }
}

impl Default for RandomForestOptions {
    fn default() -> Self {
        Self {
            trees: NonZeroUsize::new(100).expect("unreachable"),
            max_features: None,
            seed: None,
            parallel: false,
        }
    }
}

// TODO: Support categorical features
// TODO: Support classifier
#[derive(Debug)]
pub struct RandomForest {
    forest: Vec<DecisionTreeRegressor>,
}

impl RandomForest {
    pub fn fit(table: Table) -> Self {
        RandomForestOptions::default().fit(table)
    }

    pub fn predict(&self, xs: &[f64]) -> f64 {
        functions::mean(self.forest.iter().map(|tree| tree.predict(xs)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TableBuilder;

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

        let regressor = RandomForestOptions::new().seed(0).fit(table);
        assert_eq!(regressor.predict(&features[train_len]), 41.9785);
        assert_eq!(
            regressor.predict(&features[train_len + 1]),
            43.50333333333333
        );

        Ok(())
    }
}
