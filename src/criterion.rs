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
