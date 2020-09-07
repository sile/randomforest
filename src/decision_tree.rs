use crate::functions;
use crate::table::Table;
use rand::seq::SliceRandom as _;
use rand::Rng;

const MIN_SAMPLES_SPLIT: usize = 2;
const MAX_DEPTH: usize = 64;

#[derive(Debug, Clone, Default)]
pub struct DecisionTreeOptions {
    pub max_features: Option<usize>,
}

#[derive(Debug)]
pub struct DecisionTreeRegressor {
    tree: Tree,
}

impl DecisionTreeRegressor {
    pub fn fit<'a, R: Rng + ?Sized>(
        rng: &mut R,
        table: Table<'a>,
        options: DecisionTreeOptions,
    ) -> Self {
        let tree = Tree::fit(rng, table, options);
        Self { tree }
    }

    pub fn predict(&self, xs: &[f64]) -> f64 {
        self.tree.predict(xs)
    }
}

#[derive(Debug)]
pub struct Tree {
    root: Node,
}

impl Tree {
    pub fn fit<'a, R: Rng + ?Sized>(
        rng: &mut R,
        mut table: Table<'a>,
        options: DecisionTreeOptions,
    ) -> Self {
        let max_features = options.max_features.unwrap_or_else(|| table.features_len());
        let mut builder = NodeBuilder { rng, max_features };
        let root = builder.build(&mut table, 1);
        Self { root }
    }

    fn predict(&self, xs: &[f64]) -> f64 {
        self.root.predict(xs)
    }
}

#[derive(Debug)]
pub enum Node {
    Leaf { value: f64 },
    Internal { children: Children },
}

impl Node {
    fn predict(&self, xs: &[f64]) -> f64 {
        match self {
            Self::Leaf { value } => *value,
            Self::Internal { children } => {
                if xs[children.split.column] <= children.split.threshold {
                    children.left.predict(xs)
                } else {
                    children.right.predict(xs)
                }
            }
        }
    }
}

#[derive(Debug)]
pub struct Children {
    split: SplitPoint,
    left: Box<Node>,
    right: Box<Node>,
}

#[derive(Debug)]
pub struct SplitPoint {
    pub column: usize,
    pub threshold: f64,
}

#[derive(Debug)]
struct NodeBuilder<R> {
    rng: R,
    max_features: usize,
}

impl<R: Rng> NodeBuilder<R> {
    fn build(&mut self, table: &mut Table, depth: usize) -> Node {
        if table.rows_len() < MIN_SAMPLES_SPLIT || depth > MAX_DEPTH {
            let value = functions::mean(table.target());
            return Node::Leaf { value };
        }

        let impurity = functions::mse(table.target());
        let valid_columns = (0..table.features_len())
            .filter(|&i| !table.column(i).any(|f| f.is_nan()))
            .collect::<Vec<_>>();

        let mut best_split: Option<SplitPoint> = None;
        let mut best_informatin_gain = std::f64::MIN;
        let max_features = std::cmp::min(valid_columns.len(), self.max_features);
        for &column in valid_columns.choose_multiple(&mut self.rng, max_features) {
            table.sort_rows_by_column(column);
            for (row, threshold) in table.thresholds(column) {
                let impurity_l = functions::mse(table.target().take(row));
                let impurity_r = functions::mse(table.target().skip(row));
                let ratio_l = row as f64 / table.rows_len() as f64;
                let ratio_r = 1.0 - ratio_l;

                let information_gain = impurity - (ratio_l * impurity_l + ratio_r * impurity_r);
                if best_informatin_gain < information_gain {
                    best_informatin_gain = information_gain;
                    best_split = Some(SplitPoint { column, threshold });
                }
            }
        }

        if let Some(split) = best_split {
            let children = self.build_children(table, split, depth);
            Node::Internal { children }
        } else {
            let value = functions::mean(table.target());
            Node::Leaf { value }
        }
    }

    fn build_children(&mut self, table: &mut Table, split: SplitPoint, depth: usize) -> Children {
        table.sort_rows_by_column(split.column);
        let split_row = table
            .column(split.column)
            .take_while(|&f| f <= split.threshold)
            .count();
        let (left, right) =
            table.with_split(split_row, |table| Box::new(self.build(table, depth + 1)));
        Children { split, left, right }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand;

    #[test]
    fn regression_works() -> Result<(), anyhow::Error> {
        let columns = vec![
            // Features
            &[
                0.0, 0.0, 1.0, 2.0, 2.0, 2.0, 1.0, 0.0, 0.0, 2.0, 0.0, 1.0, 1.0, 2.0,
            ],
            &[
                2.0, 2.0, 2.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 2.0, 1.0,
            ],
            &[
                1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0,
            ],
            &[
                0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0,
            ],
            // Target
            &[
                25.0, 30.0, 46.0, 45.0, 52.0, 23.0, 43.0, 35.0, 38.0, 46.0, 48.0, 52.0, 44.0, 30.0,
            ],
        ];
        let train_len = columns[0].len() - 2;
        let table = Table::new(columns.iter().map(|c| &c[..train_len]).collect())?;
        let regressor =
            DecisionTreeRegressor::fit(&mut rand::thread_rng(), table, Default::default());
        assert_eq!(
            regressor.predict(&columns.iter().map(|f| f[train_len]).collect::<Vec<_>>()),
            46.0
        );
        assert_eq!(
            regressor.predict(&columns.iter().map(|f| f[train_len + 1]).collect::<Vec<_>>()),
            52.0
        );

        Ok(())
    }
}
