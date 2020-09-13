use crate::criterion::Criterion;
use crate::decision_tree::{DecisionTree, DecisionTreeOptions};
use crate::table::{ColumnType, Table};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use std::num::NonZeroUsize;

#[derive(Debug, Clone)]
pub struct RandomForestOptions {
    trees: NonZeroUsize,
    max_features: Option<NonZeroUsize>,
    max_samples: Option<NonZeroUsize>,
    seed: Option<u64>,
    parallel: bool,
}

impl RandomForestOptions {
    pub fn seed(&mut self, seed: u64) -> &mut Self {
        self.seed = Some(seed);
        self
    }

    pub fn trees(&mut self, trees: NonZeroUsize) -> &mut Self {
        self.trees = trees;
        self
    }

    pub fn max_features(&mut self, max: NonZeroUsize) -> &mut Self {
        self.max_features = Some(max);
        self
    }

    pub fn max_samples(&mut self, max: NonZeroUsize) -> &mut Self {
        self.max_samples = Some(max);
        self
    }

    pub fn parallel(&mut self) -> &mut Self {
        self.parallel = true;
        self
    }

    pub fn fit<T: Criterion>(
        &self,
        criterion: T,
        is_regression: bool,
        table: Table,
    ) -> RandomForest {
        let forest = if self.parallel {
            self.tree_rngs()
                .collect::<Vec<_>>()
                .into_par_iter()
                .map(|mut rng| self.tree_fit(&mut rng, criterion.clone(), is_regression, &table))
                .collect::<Vec<_>>()
        } else {
            self.tree_rngs()
                .map(|mut rng| self.tree_fit(&mut rng, criterion.clone(), is_regression, &table))
                .collect::<Vec<_>>()
        };
        RandomForest {
            columns: table.column_types().to_owned(),
            forest,
        }
    }

    fn tree_fit<R: Rng + ?Sized, T: Criterion>(
        &self,
        rng: &mut R,
        criterion: T,
        is_regression: bool,
        table: &Table,
    ) -> DecisionTree {
        let max_features = self.decide_max_features(table);
        let max_samples = self.max_samples.map_or(table.rows_len(), |n| n.get());
        let table = table.bootstrap_sample(rng, max_samples);
        let tree_options = DecisionTreeOptions {
            max_features: Some(max_features),
            is_regression,
        };
        DecisionTree::fit(rng, criterion, table, tree_options)
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
            max_samples: None,
            seed: None,
            parallel: false,
        }
    }
}

#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct RandomForest {
    columns: Vec<ColumnType>,
    forest: Vec<DecisionTree>,
}

impl RandomForest {
    pub fn predict<'a>(&'a self, xs: &'a [f64]) -> impl 'a + Iterator<Item = f64> {
        self.forest
            .iter()
            .map(move |tree| tree.predict(xs, &self.columns))
    }
}
