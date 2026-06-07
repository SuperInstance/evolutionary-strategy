//! Self-adaptive step sizes.

use crate::population::{Individual, Population};

/// Compute the 1/5 success rule factor.
/// If success rate > 1/5, increase step size; if < 1/5, decrease.
pub fn one_fifth_rule(successes: usize, total: usize) -> f64 {
    if total == 0 { return 1.0; }
    let success_rate = successes as f64 / total as f64;
    if success_rate > 0.2 {
        1.0 + 0.5 * (success_rate - 0.2)
    } else if success_rate < 0.2 {
        (1.0 - 0.5 * (0.2 - success_rate)).max(0.1)
    } else {
        1.0
    }
}

/// Adapt step sizes globally using 1/5 rule.
pub fn adapt_global_step_size(population: &mut Population, factor: f64) {
    for ind in &mut population.individuals {
        for s in &mut ind.step_sizes {
            *s *= factor;
            *s = (*s).clamp(1e-10, 1e10);
        }
    }
}

/// CMA-ES-like covariance adaptation (simplified).
/// Computes the mean and covariance of the best individuals.
pub fn compute_mean(individuals: &[Individual]) -> Option<Vec<f64>> {
    if individuals.is_empty() { return None; }
    let dim = individuals[0].dimension();
    let n = individuals.len() as f64;
    let mut mean = vec![0.0; dim];
    for ind in individuals {
        for (i, &g) in ind.genes.iter().enumerate() {
            mean[i] += g / n;
        }
    }
    Some(mean)
}

/// Compute variance per dimension.
pub fn compute_variances(individuals: &[Individual], mean: &[f64]) -> Vec<f64> {
    if individuals.is_empty() { return vec![]; }
    let dim = individuals[0].dimension();
    let n = individuals.len() as f64;
    (0..dim).map(|i| {
        let sum: f64 = individuals.iter().map(|ind| (ind.genes[i] - mean[i]).powi(2)).sum();
        sum / n
    }).collect()
}

/// Adapt step sizes from variances.
pub fn step_sizes_from_variances(variances: &[f64]) -> Vec<f64> {
    variances.iter().map(|&v| v.sqrt().max(1e-10)).collect()
}

/// Learning rate for cumulative step-size adaptation.
pub fn learning_rate(dim: usize) -> f64 {
    1.0 / ((dim as f64 + 2.0).sqrt())
}

/// Damping parameter for step-size control.
pub fn damping(dim: usize) -> f64 {
    1.0 + 2.0 / (dim as f64 + 1.0)
}

/// Count how many offspring are better than their parents.
pub fn count_improvements(parents: &[Individual], offspring: &[Individual]) -> usize {
    let parent_best = parents.iter().filter_map(|i| i.fitness).fold(f64::INFINITY, f64::min);
    offspring.iter().filter(|o| o.fitness.is_some_and(|f| f < parent_best)).count()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_one_fifth_above() {
        let factor = one_fifth_rule(5, 10); // 50% success
        assert!(factor > 1.0);
    }

    #[test]
    fn test_one_fifth_below() {
        let factor = one_fifth_rule(1, 10); // 10% success
        assert!(factor < 1.0);
    }

    #[test]
    fn test_one_fifth_at() {
        let factor = one_fifth_rule(2, 10); // 20% success
        assert!((factor - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_one_fifth_zero() {
        let factor = one_fifth_rule(0, 0);
        assert!((factor - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_adapt_global_step_size() {
        let mut pop = Population::new();
        pop.add(Individual::new(vec![0.0]).with_step_sizes(vec![1.0]));
        adapt_global_step_size(&mut pop, 2.0);
        assert!((pop.individuals[0].step_sizes[0] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_adapt_clamps_min() {
        let mut pop = Population::new();
        pop.add(Individual::new(vec![0.0]).with_step_sizes(vec![1e-20]));
        adapt_global_step_size(&mut pop, 0.5);
        assert!(pop.individuals[0].step_sizes[0] >= 1e-10);
    }

    #[test]
    fn test_adapt_clamps_max() {
        let mut pop = Population::new();
        pop.add(Individual::new(vec![0.0]).with_step_sizes(vec![1e20]));
        adapt_global_step_size(&mut pop, 2.0);
        assert!(pop.individuals[0].step_sizes[0] <= 1e10);
    }

    #[test]
    fn test_compute_mean() {
        let inds = vec![
            Individual::new(vec![0.0, 0.0]),
            Individual::new(vec![10.0, 20.0]),
        ];
        let mean = compute_mean(&inds).unwrap();
        assert!((mean[0] - 5.0).abs() < 1e-10);
        assert!((mean[1] - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_compute_mean_empty() {
        assert!(compute_mean(&[]).is_none());
    }

    #[test]
    fn test_compute_variances() {
        let inds = vec![
            Individual::new(vec![0.0]),
            Individual::new(vec![2.0]),
        ];
        let mean = vec![1.0];
        let vars = compute_variances(&inds, &mean);
        assert!((vars[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_step_sizes_from_variances() {
        let vars = vec![4.0, 9.0];
        let steps = step_sizes_from_variances(&vars);
        assert!((steps[0] - 2.0).abs() < 1e-10);
        assert!((steps[1] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_step_sizes_clamps_zero() {
        let vars = vec![0.0];
        let steps = step_sizes_from_variances(&vars);
        assert!(steps[0] >= 1e-10);
    }

    #[test]
    fn test_learning_rate() {
        let lr = learning_rate(10);
        assert!(lr > 0.0 && lr < 1.0);
    }

    #[test]
    fn test_damping() {
        let d = damping(10);
        assert!(d > 1.0);
    }

    #[test]
    fn test_count_improvements() {
        let mut parents = vec![Individual::new(vec![0.0])];
        parents[0].set_fitness(5.0);
        let mut offspring = vec![Individual::new(vec![1.0])];
        offspring[0].set_fitness(3.0);
        assert_eq!(count_improvements(&parents, &offspring), 1);
    }

    #[test]
    fn test_count_improvements_none() {
        let mut parents = vec![Individual::new(vec![0.0])];
        parents[0].set_fitness(1.0);
        let mut offspring = vec![Individual::new(vec![1.0])];
        offspring[0].set_fitness(10.0);
        assert_eq!(count_improvements(&parents, &offspring), 0);
    }
}
