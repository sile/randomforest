randomforest
============

[![randomforest](https://img.shields.io/crates/v/randomforest.svg)](https://crates.io/crates/randomforest)
[![Documentation](https://docs.rs/randomforest/badge.svg)](https://docs.rs/randomforest)
[![Actions Status](https://github.com/sile/randomforest/workflows/CI/badge.svg)](https://github.com/sile/randomforest/actions)
![License](https://img.shields.io/crates/l/randomforest)

A random forest implementation in Rust.


Examples
--------

```rust
use randomforest::criterion::Mse;
use randomforest::RandomForestRegressorOptions;
use randomforest::table::TableBuilder;

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

];
let target = [
    25.0, 30.0, 46.0, 45.0, 52.0, 23.0, 43.0, 35.0, 38.0, 46.0, 48.0, 52.0
];

let mut table_builder = TableBuilder::new();
for (xs, y) in features.iter().zip(target.iter()) {
   table_builder.add_row(xs, *y)?;
}
let table = table_builder.build()?;

let regressor = RandomForestRegressorOptions::new()
    .seed(0)
    .fit(Mse, table);
assert_eq!(regressor.predict(&[1.0, 2.0, 0.0, 0.0]), 42.16872222222222);
```
