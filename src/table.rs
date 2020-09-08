use ordered_float::OrderedFloat;
use rand::Rng;
use std::ops::Range;
use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ColumnType {
    Numerical,
    Categorical,
}

#[derive(Debug)]
pub struct TableBuilder {
    column_types: Vec<ColumnType>,
    columns: Vec<Vec<f64>>,
}

impl TableBuilder {
    pub fn new() -> Self {
        Self {
            column_types: Vec::new(),
            columns: Vec::new(),
        }
    }

    pub fn set_column_types(&mut self, column_types: &[ColumnType]) -> Result<(), TableError> {
        if column_types.is_empty() {
            return Err(TableError::EmptyTable);
        }

        if !self.columns.is_empty() && self.columns.len() != column_types.len() {
            return Err(TableError::ColumnSizeMismatch);
        }

        self.column_types = column_types.to_owned();
        Ok(())
    }

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
            self.column_types = (0..self.columns.len())
                .map(|_| ColumnType::Numerical)
                .collect();
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

    pub(crate) fn thresholds<'b>(
        &'b self,
        column: usize,
    ) -> impl 'b + Iterator<Item = (usize, f64)> {
        // Assumption: `self.columns[column]` has been sorted.
        let column = &self.columns[column];
        self.rows()
            .map(move |i| column[i])
            .enumerate()
            .scan(None, |prev, (i, x)| {
                if prev.is_none() {
                    *prev = Some(x);
                    Some(None)
                } else if *prev != Some(x) {
                    let y = prev.expect("never fails");
                    *prev = Some(x);
                    Some(Some((i, (x + y) / 2.0)))
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

#[derive(Debug, Error, Clone)]
pub enum TableError {
    #[error("table must have at least one column and one row")]
    EmptyTable,

    #[error("some of rows have a different column count from others")]
    ColumnSizeMismatch,

    #[error("target column contains non finite numbers")]
    NonFiniteTarget,
}
