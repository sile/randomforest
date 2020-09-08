use crate::functions;

pub trait Criterion: Send + Sync + Clone {
    fn calculate<T>(&self, xs: T) -> f64
    where
        T: Iterator<Item = f64> + Clone;
}

pub trait RegressionCriterion: Criterion {}

#[derive(Debug, Clone)]
pub struct Mse;

impl Criterion for Mse {
    fn calculate<T>(&self, ys: T) -> f64
    where
        T: Iterator<Item = f64> + Clone,
    {
        let n = ys.clone().count() as f64;
        let m = functions::mean(ys.clone());
        ys.map(|x| (x - m).powi(2)).sum::<f64>() / n
    }
}

impl RegressionCriterion for Mse {}

pub trait ClassificationCriterion: Criterion {}

#[derive(Debug, Clone)]
pub struct Gini;

impl Criterion for Gini {
    fn calculate<T>(&self, ys: T) -> f64
    where
        T: Iterator<Item = f64> + Clone,
    {
        let (histogram, n) = functions::histogram(ys);
        1.0 - histogram
            .into_iter()
            .map(|(_, count)| (count as f64 / n as f64).powi(2))
            .sum::<f64>()
    }
}

impl ClassificationCriterion for Gini {}

#[derive(Debug, Clone)]
pub struct Entropy;

impl Criterion for Entropy {
    fn calculate<T>(&self, ys: T) -> f64
    where
        T: Iterator<Item = f64> + Clone,
    {
        let (histogram, n) = functions::histogram(ys);
        histogram
            .into_iter()
            .map(|(_, count)| {
                let p = count as f64 / n as f64;
                -p * p.log2()
            })
            .sum()
    }
}
