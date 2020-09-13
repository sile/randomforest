use crate::criterion::Criterion;
use crate::functions;
use crate::table::{ColumnType, Table};
use rand::seq::SliceRandom as _;
use rand::Rng;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

const MIN_SAMPLES_SPLIT: usize = 2;
const MAX_DEPTH: usize = 64;

#[derive(Debug, Clone, Default)]
pub struct DecisionTreeOptions {
    pub max_features: Option<usize>,
    pub is_regression: bool,
}

#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct DecisionTree {
    root: Node,
}

impl DecisionTree {
    pub fn fit<R: Rng + ?Sized, T: Criterion>(
        rng: &mut R,
        criterion: T,
        mut table: Table,
        options: DecisionTreeOptions,
    ) -> Self {
        let max_features = options.max_features.unwrap_or_else(|| table.features_len());
        let mut builder = NodeBuilder {
            rng,
            max_features,
            is_regression: options.is_regression,
            criterion,
        };
        let root = builder.build(&mut table, 1);
        Self { root }
    }

    pub fn predict(&self, xs: &[f64], columns: &[ColumnType]) -> f64 {
        self.root.predict(xs, columns)
    }
}

#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(untagged))]
pub enum Node {
    Leaf {
        #[serde(rename = "v")]
        value: f64,
    },
    Internal {
        #[serde(rename = "c")]
        children: Children,
    },
}

impl Node {
    fn predict(&self, xs: &[f64], columns: &[ColumnType]) -> f64 {
        match self {
            Self::Leaf { value } => *value,
            Self::Internal { children } => {
                if columns[children.split.column]
                    .is_left(xs[children.split.column], children.split.value)
                {
                    children.left.predict(xs, columns)
                } else {
                    children.right.predict(xs, columns)
                }
            }
        }
    }
}

#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Children {
    #[serde(rename = "s")]
    split: SplitPoint,
    #[serde(rename = "l")]
    left: Box<Node>,
    #[serde(rename = "r")]
    right: Box<Node>,
}

#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SplitPoint {
    #[serde(rename = "c")]
    column: usize,
    #[serde(rename = "v")]
    value: f64,
}

#[derive(Debug)]
struct NodeBuilder<R, T> {
    rng: R,
    max_features: usize,
    is_regression: bool,
    criterion: T,
}

impl<R: Rng, T: Criterion> NodeBuilder<R, T> {
    fn build(&mut self, table: &mut Table, depth: usize) -> Node {
        if table.rows_len() < MIN_SAMPLES_SPLIT || depth > MAX_DEPTH {
            let value = self.average(table.target());
            return Node::Leaf { value };
        }

        let impurity = self.criterion.calculate(table.target());
        let valid_columns = (0..table.features_len())
            .filter(|&i| !table.column(i).any(|f| f.is_nan()))
            .collect::<Vec<_>>();

        let mut best_split: Option<SplitPoint> = None;
        let mut best_informatin_gain = std::f64::MIN;
        let max_features = std::cmp::min(valid_columns.len(), self.max_features);
        for &column in valid_columns.choose_multiple(&mut self.rng, max_features) {
            table.sort_rows_by_column(column);
            for (left_row, value) in table.split_points(column) {
                let rows_l = table.target().take(left_row.end).skip(left_row.start);
                let rows_r = table
                    .target()
                    .take(left_row.start)
                    .chain(table.target().skip(left_row.end));
                let impurity_l = self.criterion.calculate(rows_l);
                let impurity_r = self.criterion.calculate(rows_r);
                let ratio_l = (left_row.end - left_row.start) as f64 / table.rows_len() as f64;
                let ratio_r = 1.0 - ratio_l;

                let information_gain = impurity - (ratio_l * impurity_l + ratio_r * impurity_r);
                if best_informatin_gain < information_gain {
                    best_informatin_gain = information_gain;
                    best_split = Some(SplitPoint { column, value });
                }
            }
        }

        if let Some(split) = best_split {
            let children = self.build_children(table, split, depth);
            Node::Internal { children }
        } else {
            let value = self.average(table.target());
            Node::Leaf { value }
        }
    }

    fn build_children(&mut self, table: &mut Table, split: SplitPoint, depth: usize) -> Children {
        match table.column_types()[split.column] {
            ColumnType::Categorical => {
                table.sort_rows_by_categorical_column(split.column, split.value);
            }
            ColumnType::Numerical => {
                table.sort_rows_by_column(split.column);
            }
        }

        let split_row = table
            .column(split.column)
            .take_while(|&f| table.column_types()[split.column].is_left(f, split.value))
            .count();
        let (left, right) =
            table.with_split(split_row, |table| Box::new(self.build(table, depth + 1)));
        Children { split, left, right }
    }

    fn average(&self, ys: impl Iterator<Item = f64>) -> f64 {
        if self.is_regression {
            functions::mean(ys)
        } else {
            functions::most_frequent(ys)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::criterion::Mse;
    use crate::table::TableBuilder;
    use rand;

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

        let columns = [
            ColumnType::Numerical,
            ColumnType::Numerical,
            ColumnType::Numerical,
            ColumnType::Numerical,
        ];
        let regressor = DecisionTree::fit(&mut rand::thread_rng(), Mse, table, Default::default());
        assert_eq!(regressor.predict(&features[train_len], &columns), 46.0);
        assert_eq!(regressor.predict(&features[train_len + 1], &columns), 52.0);

        Ok(())
    }
}
