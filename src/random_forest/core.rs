use crate::criterion::Criterion;
use crate::decision_tree::{DecisionTree, DecisionTreeOptions};
use crate::table::{ColumnType, Table};
use byteorder::{BigEndian, ReadBytesExt, WriteBytesExt};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use std::io::{Read, Write};
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

    pub fn serialize<W: Write>(&self, mut writer: W) -> std::io::Result<()> {
        writer.write_u16::<BigEndian>(self.columns.len() as u16)?;
        for &c in &self.columns {
            writer.write_u8(c as u8)?;
        }

        writer.write_u16::<BigEndian>(self.forest.len() as u16)?;
        for tree in &self.forest {
            tree.serialize(&mut writer)?;
        }

        Ok(())
    }

    pub fn deserialize<R: Read>(mut reader: R) -> std::io::Result<Self> {
        let columns_len = reader.read_u16::<BigEndian>()?;
        let columns = (0..columns_len)
            .map(|_| match reader.read_u8()? {
                0 => Ok(ColumnType::Numerical),
                1 => Ok(ColumnType::Categorical),
                v => Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("unknown column type {}", v),
                )),
            })
            .collect::<std::io::Result<Vec<_>>>()?;

        let forest_len = reader.read_u16::<BigEndian>()?;
        let forest = (0..forest_len)
            .map(|_| DecisionTree::deserialize(&mut reader))
            .collect::<std::io::Result<Vec<_>>>()?;

        Ok(Self { columns, forest })
    }
}
