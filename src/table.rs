//! Table data which contains features and a target columns.
use ordered_float::OrderedFloat;
use rand::Rng;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use std::ops::Range;
use thiserror::Error;

/// Column type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "UPPERCASE"))]
pub enum ColumnType {
    /// Numerical column.
    Numerical,

    /// Categorical column.
    Categorical,
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
    pub(crate) fn target<'b>(&'b self) -> impl 'b + Iterator<Item = f64> + Clone {
        self.column(self.columns.len() - 1)
    }

    pub(crate) fn column<'b>(
        &'b self,
        column_index: usize,
    ) -> impl 'b + Iterator<Item = f64> + Clone {
        self.rows().map(move |i| self.columns[column_index][i])
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

    fn rows<'b>(&'b self) -> impl 'b + Iterator<Item = usize> + Clone {
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

    pub(crate) fn bootstrap_sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Self {
        let row_index = (0..self.rows_len())
            .map(|_| self.row_index[rng.gen_range(self.row_range.start, self.row_range.end)])
            .collect::<Vec<_>>();
        let row_range = Range {
            start: 0,
            end: self.rows_len(),
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
        self.rows()
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
}
