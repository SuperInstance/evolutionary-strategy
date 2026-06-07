//! Selection: comma and plus strategies.

use crate::population::{Individual, Population};

/// Selection strategy.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SelectionStrategy {
    Comma, // only offspring
    Plus,  // parents + offspring
}

/// Select top μ individuals from candidates.
pub fn select_mu_best(candidates: &[Individual], mu: usize) -> Vec<Individual> {
    let mut sorted: Vec<&Individual> = candidates.iter().filter(|i| i.fitness.is_some()).collect();
    sorted.sort_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap());
    sorted.into_iter().take(mu).cloned().collect()
}

/// (μ, λ) selection: only from offspring.
pub fn comma_selection(parents: &[Individual], offspring: &[Individual], mu: usize) -> Vec<Individual> {
    let _ = parents; // comma ignores parents
    select_mu_best(offspring, mu)
}

/// (μ + λ) selection: from parents and offspring combined.
pub fn plus_selection(parents: &[Individual], offspring: &[Individual], mu: usize) -> Vec<Individual> {
    let mut combined: Vec<Individual> = parents.iter().chain(offspring.iter()).cloned().collect();
    combined.sort_by(|a, b| {
        let fa = a.fitness.unwrap_or(f64::INFINITY);
        let fb = b.fitness.unwrap_or(f64::INFINITY);
        fa.partial_cmp(&fb).unwrap()
    });
    combined.truncate(mu);
    combined
}

/// Run one generation of evolution strategy.
pub fn evolve(
    population: &Population,
    strategy: SelectionStrategy,
    mu: usize,
    lambda: usize,
    mutate_fn: &dyn Fn(&Individual) -> Individual,
) -> Population {
    let mut offspring = Vec::new();
    for _ in 0..lambda {
        let parent_idx = rand_idx(population.len());
        if let Some(parent) = population.individuals.get(parent_idx) {
            offspring.push(mutate_fn(parent));
        }
    }
    let selected = match strategy {
        SelectionStrategy::Comma => comma_selection(&population.individuals, &offspring, mu),
        SelectionStrategy::Plus => plus_selection(&population.individuals, &offspring, mu),
    };
    Population::from_vec(selected)
}

/// Simple deterministic "random" index (for testing without rand).
fn rand_idx(len: usize) -> usize {
    if len == 0 { return 0; }
    // Use a simple sequence
    static mut COUNTER: usize = 0;
    unsafe {
        COUNTER = (COUNTER + 1) % len.max(1);
        COUNTER % len
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_ind(genes: Vec<f64>, fitness: f64) -> Individual {
        let mut ind = Individual::new(genes);
        ind.set_fitness(fitness);
        ind
    }

    #[test]
    fn test_select_mu_best() {
        let inds = vec![make_ind(vec![1.0], 5.0), make_ind(vec![2.0], 1.0), make_ind(vec![3.0], 3.0)];
        let best = select_mu_best(&inds, 2);
        assert_eq!(best.len(), 2);
        assert!((best[0].get_fitness() - 1.0).abs() < 1e-10);
        assert!((best[1].get_fitness() - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_select_mu_best_fewer() {
        let inds = vec![make_ind(vec![1.0], 5.0)];
        let best = select_mu_best(&inds, 5);
        assert_eq!(best.len(), 1);
    }

    #[test]
    fn test_comma_selection() {
        let parents = vec![make_ind(vec![0.0], 0.1)];
        let offspring = vec![make_ind(vec![1.0], 1.0), make_ind(vec![2.0], 0.5)];
        let selected = comma_selection(&parents, &offspring, 1);
        assert_eq!(selected.len(), 1);
        // Should pick best offspring (fitness 0.5)
        assert!((selected[0].get_fitness() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_comma_ignores_parents() {
        let parents = vec![make_ind(vec![0.0], 0.01)]; // parent is best
        let offspring = vec![make_ind(vec![1.0], 5.0)];
        let selected = comma_selection(&parents, &offspring, 1);
        // Should pick offspring, not parent
        assert!((selected[0].get_fitness() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_plus_selection() {
        let parents = vec![make_ind(vec![0.0], 0.01)];
        let offspring = vec![make_ind(vec![1.0], 5.0), make_ind(vec![2.0], 0.5)];
        let selected = plus_selection(&parents, &offspring, 2);
        assert_eq!(selected.len(), 2);
        assert!((selected[0].get_fitness() - 0.01).abs() < 1e-10);
    }

    #[test]
    fn test_plus_includes_parents() {
        let parents = vec![make_ind(vec![0.0], 0.01)];
        let offspring = vec![make_ind(vec![1.0], 5.0)];
        let selected = plus_selection(&parents, &offspring, 2);
        assert_eq!(selected.len(), 2);
    }

    #[test]
    fn test_plus_selection_size() {
        let parents = vec![make_ind(vec![0.0], 1.0)];
        let offspring: Vec<Individual> = (0..10).map(|i| make_ind(vec![i as f64], i as f64)).collect();
        let selected = plus_selection(&parents, &offspring, 5);
        assert_eq!(selected.len(), 5);
    }

    #[test]
    fn test_selection_strategy_enum() {
        assert_ne!(SelectionStrategy::Comma, SelectionStrategy::Plus);
    }
}
