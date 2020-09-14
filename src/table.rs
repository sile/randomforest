//! Table data which contains features and a target columns.
use ordered_float::OrderedFloat;
use rand::seq::SliceRandom;
use rand::Rng;
use std::ops::Range;
use thiserror::Error;

/// Column type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ColumnType {
    /// Numerical column.
    Numerical = 0,

    /// Categorical column.
    Categorical = 1,
}

impl ColumnType {
    pub(crate) fn is_left(self, x: f64, split_value: f64) -> bool {
        match self {
            Self::Numerical => x <= split_value,
            Self::Categorical => (x - split_value).abs() < std::f64::EPSILON,
        }
    }
}

/// `Table` builder.
#[derive(Debug)]
pub struct TableBuilder {
    column_types: Vec<ColumnType>,
    columns: Vec<Vec<f64>>,
}

impl TableBuilder {
    /// Makes a new `TableBuilder` instance.
    pub fn new() -> Self {
        Self {
            column_types: Vec::new(),
            columns: Vec::new(),
        }
    }

    /// Sets the types of feature columns.
    ///
    /// In the default, the feature columns are regarded as numerical.
    pub fn set_feature_column_types(&mut self, types: &[ColumnType]) -> Result<(), TableError> {
        if self.columns.is_empty() {
            self.columns = vec![Vec::new(); types.len() + 1];
        }

        if self.columns.len() != types.len() + 1 {
            return Err(TableError::ColumnSizeMismatch);
        }

        self.column_types = types.to_owned();
        Ok(())
    }

    /// Adds a row to the table.
    pub fn add_row(&mut self, features: &[f64], target: f64) -> Result<(), TableError> {
        if self.columns.is_empty() {
            self.columns = vec![Vec::new(); features.len() + 1];
        }

        if self.columns.len() != features.len() + 1 {
            return Err(TableError::ColumnSizeMismatch);
        }

        if !target.is_finite() {
            return Err(TableError::NonFiniteTarget);
        }

        if self.column_types.is_empty() {
            self.column_types = (0..features.len()).map(|_| ColumnType::Numerical).collect();
        }

        for (column, value) in self
            .columns
            .iter_mut()
            .zip(features.iter().copied().chain(std::iter::once(target)))
        {
            column.push(value);
        }

        Ok(())
    }

    /// Builds a `Table` instance.
    pub fn build(&self) -> Result<Table, TableError> {
        if self.columns.is_empty() || self.columns[0].is_empty() {
            return Err(TableError::EmptyTable);
        }

        let rows_len = self.columns[0].len();
        Ok(Table {
            row_index: (0..rows_len).collect(),
            row_range: Range {
                start: 0,
                end: rows_len,
            },
            column_types: &self.column_types,
            columns: &self.columns,
        })
    }
}

impl Default for TableBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// A table.
#[derive(Debug, Clone)]
pub struct Table<'a> {
    row_index: Vec<usize>,
    row_range: Range<usize>,
    column_types: &'a [ColumnType],
    columns: &'a [Vec<f64>],
}

impl<'a> Table<'a> {
    /// Returns an iterator over all rows of the table.
    ///
    /// The last element of each row is the target value.
    pub fn rows<'b>(&'b self) -> impl 'b + Iterator<Item = Vec<f64>> + Clone {
        self.row_indices().map(move |i| {
            (0..self.columns.len())
                .map(|j| self.columns[j][i])
                .collect()
        })
    }

    /// Removes rows which don't match the given condition from the table.
    ///
    /// Note that after calling this method the order of rows isn't preserved.
    pub fn filter<F>(&mut self, f: F) -> usize
    where
        F: Fn(&[f64]) -> bool,
    {
        let mut n = 0;
        let mut i = self.row_range.start;
        while i < self.row_range.end {
            let row_i = self.row_index[i];
            let row = (0..self.columns.len())
                .map(|j| self.columns[j][row_i])
                .collect::<Vec<_>>();
            if f(&row) {
                i += 1;
            } else {
                self.row_index.swap(i, self.row_range.end - 1);
                self.row_range.end -= 1;
                n += 1;
            }
        }
        n
    }

    /// Splits the table into train and test datasets.
    pub fn train_test_split<R: Rng + ?Sized>(
        mut self,
        rng: &mut R,
        test_rate: f64,
    ) -> (Self, Self) {
        (&mut self.row_index[self.row_range.start..self.row_range.end]).shuffle(rng);
        let test_num = (self.rows_len() as f64 * test_rate).round() as usize;

        let mut train = self.clone();
        let mut test = self;
        test.row_range.end = test.row_range.start + test_num;
        train.row_range.start = test.row_range.end;

        (train, test)
    }

    pub(crate) fn target<'b>(&'b self) -> impl 'b + Iterator<Item = f64> + Clone {
        self.column(self.columns.len() - 1)
    }

    pub(crate) fn column<'b>(
        &'b self,
        column_index: usize,
    ) -> impl 'b + Iterator<Item = f64> + Clone {
        self.row_indices()
            .map(move |i| self.columns[column_index][i])
    }

    pub(crate) fn features_len(&self) -> usize {
        self.columns.len() - 1
    }

    pub(crate) fn rows_len(&self) -> usize {
        self.row_range.end - self.row_range.start
    }

    pub(crate) fn column_types(&self) -> &'a [ColumnType] {
        self.column_types
    }

    fn row_indices<'b>(&'b self) -> impl 'b + Iterator<Item = usize> + Clone {
        self.row_index[self.row_range.start..self.row_range.end]
            .iter()
            .copied()
    }

    pub(crate) fn sort_rows_by_column(&mut self, column: usize) {
        let columns = &self.columns;
        (&mut self.row_index[self.row_range.start..self.row_range.end])
            .sort_by_key(|&x| OrderedFloat(columns[column][x]))
    }

    pub(crate) fn sort_rows_by_categorical_column(&mut self, column: usize, value: f64) {
        let columns = &self.columns;
        (&mut self.row_index[self.row_range.start..self.row_range.end]).sort_by_key(|&x| {
            if (columns[column][x] - value).abs() < std::f64::EPSILON {
                0
            } else {
                1
            }
        })
    }

    pub(crate) fn bootstrap_sample<R: Rng + ?Sized>(
        &self,
        rng: &mut R,
        max_samples: usize,
    ) -> Self {
        let samples = std::cmp::min(max_samples, self.rows_len());
        let row_index = (0..samples)
            .map(|_| self.row_index[rng.gen_range(self.row_range.start, self.row_range.end)])
            .collect::<Vec<_>>();
        let row_range = Range {
            start: 0,
            end: samples,
        };
        Self {
            row_index,
            row_range,
            column_types: self.column_types,
            columns: self.columns,
        }
    }

    pub(crate) fn split_points<'b>(
        &'b self,
        column_index: usize,
    ) -> impl 'b + Iterator<Item = (Range<usize>, f64)> {
        // Assumption: `self.columns[column]` has been sorted.
        let column = &self.columns[column_index];
        let categorical = self.column_types[column_index] == ColumnType::Categorical;
        self.row_indices()
            .map(move |i| column[i])
            .enumerate()
            .scan(None, move |prev, (i, x)| {
                if prev.is_none() {
                    *prev = Some((x, i));
                    Some(None)
                } else if prev.map_or(false, |(y, _)| (y - x).abs() > std::f64::EPSILON) {
                    let (y, j) = prev.expect("never fails");
                    *prev = Some((x, i));
                    if categorical {
                        let r = Range { start: j, end: i };
                        Some(Some((r, y)))
                    } else {
                        let r = Range { start: 0, end: i };
                        Some(Some((r, (x + y) / 2.0)))
                    }
                } else {
                    Some(None)
                }
            })
            .filter_map(|t| t)
    }

    pub(crate) fn with_split<F, T>(&mut self, row: usize, mut f: F) -> (T, T)
    where
        F: FnMut(&mut Self) -> T,
    {
        let row = row + self.row_range.start;
        let original = self.row_range.clone();

        self.row_range.end = row;
        let left = f(self);
        self.row_range.end = original.end;

        self.row_range.start = row;
        let right = f(self);
        self.row_range.start = original.start;

        (left, right)
    }
}

/// Error kinds which could be returned during buidling a table.
#[derive(Debug, Error, Clone, PartialEq, Eq, Hash)]
pub enum TableError {
    /// Table must have at least one column and one row.
    #[error("table must have at least one column and one row")]
    EmptyTable,

    /// Some of rows have a different column count from others.
    #[error("some of rows have a different column count from others")]
    ColumnSizeMismatch,

    /// Target column contains non finite numbers.
    #[error("target column contains non finite numbers")]
    NonFiniteTarget,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_check_works() -> anyhow::Result<()> {
        assert_eq!(
            TableBuilder::default().build().err(),
            Some(TableError::EmptyTable)
        );

        let mut table = TableBuilder::default();
        table.set_feature_column_types(&[ColumnType::Numerical])?;
        assert_eq!(
            table.add_row(&[1.0, 1.0], 10.0).err(),
            Some(TableError::ColumnSizeMismatch)
        );

        assert_eq!(
            TableBuilder::default()
                .add_row(&[1.0], std::f64::INFINITY)
                .err(),
            Some(TableError::NonFiniteTarget)
        );

        Ok(())
    }

    #[test]
    fn train_test_split_works() -> anyhow::Result<()> {
        let mut builder = TableBuilder::new();
        for _ in 0..100 {
            builder.add_row(&[0.0], 1.0)?;
        }
        let table = builder.build()?;
        assert_eq!(table.rows_len(), 100);

        let (train, test) = table.train_test_split(&mut rand::thread_rng(), 0.25);
        assert_eq!(train.rows_len(), 75);
        assert_eq!(test.rows_len(), 25);

        Ok(())
    }

    #[test]
    fn filter_works() -> anyhow::Result<()> {
        let mut builder = TableBuilder::new();
        for i in 0..100 {
            builder.add_row(&[0.0], i as f64)?;
        }
        let mut table = builder.build()?;
        assert_eq!(table.rows_len(), 100);

        let removed = table.filter(|row| row[row.len() - 1] < 10.0);
        assert_eq!(removed, 90);
        assert_eq!(table.rows_len(), 10);
        Ok(())
    }
}
