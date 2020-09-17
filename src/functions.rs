use ordered_float::OrderedFloat;
use std::collections::HashMap;

pub fn mean(xs: impl Iterator<Item = f64>) -> (f64, usize) {
    let mut count = 0;
    let mut total = 0.0;
    for x in xs {
        count += 1;
        total += x;
    }
    assert_ne!(count, 0);
    (total / count as f64, count)
}

pub fn most_frequent(xs: impl Iterator<Item = f64>) -> (f64, usize) {
    let (histogram, n) = histogram(xs);
    let f = histogram
        .into_iter()
        .max_by_key(|t| t.1)
        .map(|t| (t.0).0)
        .expect("unreachable");
    (f, n)
}

pub fn histogram(xs: impl Iterator<Item = f64>) -> (HashMap<OrderedFloat<f64>, usize>, usize) {
    let mut histogram = HashMap::<_, usize>::new();
    let mut n = 0;
    for x in xs {
        *histogram.entry(OrderedFloat(x)).or_default() += 1;
        n += 1;
    }
    (histogram, n)
}
