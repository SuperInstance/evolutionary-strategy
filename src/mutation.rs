//! Gaussian perturbation for mutation.

use crate::population::Individual;

/// A simple pseudo-random normal generator (Box-Muller-like with provided uniform).
pub fn box_muller(u1: f64, u2: f64) -> f64 {
    let u1c = u1.max(1e-10);
    (-2.0 * u1c.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

/// Mutate an individual using its step sizes and provided random values.
/// randoms should have 2*dimension elements (pairs for Box-Muller).
pub fn mutate(individual: &Individual, randoms: &[f64]) -> Individual {
    let mut new_genes = individual.genes.clone();
    for (i, gene) in new_genes.iter_mut().enumerate() {
        let u1 = randoms.get(i * 2).copied().unwrap_or(0.5);
        let u2 = randoms.get(i * 2 + 1).copied().unwrap_or(0.5);
        let noise = box_muller(u1, u2) * individual.step_sizes[i];
        *gene += noise;
    }
    Individual {
        genes: new_genes,
        fitness: None,
        step_sizes: individual.step_sizes.clone(),
        age: 0,
    }
}

/// Mutate step sizes by a log-normal factor.
pub fn mutate_step_sizes(step_sizes: &[f64], tau: f64, randoms: &[f64]) -> Vec<f64> {
    let global_noise = box_muller(randoms.first().copied().unwrap_or(0.5), randoms.get(1).copied().unwrap_or(0.5));
    step_sizes.iter().enumerate().map(|(i, &s)| {
        let local_noise = box_muller(
            randoms.get((i + 2) * 2).copied().unwrap_or(0.5),
            randoms.get((i + 2) * 2 + 1).copied().unwrap_or(0.5),
        );
        (s * (tau * global_noise + tau * local_noise).exp()).max(1e-10)
    }).collect()
}

/// Apply full mutation to both genes and step sizes.
pub fn full_mutate(individual: &Individual, tau: f64, randoms: &[f64]) -> Individual {
    let new_steps = mutate_step_sizes(&individual.step_sizes, tau, randoms);
    let mut new_ind = Individual {
        genes: individual.genes.clone(),
        fitness: None,
        step_sizes: new_steps,
        age: 0,
    };
    // Use different random portion for gene mutation
    let gene_randoms: Vec<f64> = randoms.iter().rev().copied().collect();
    for (i, gene) in new_ind.genes.iter_mut().enumerate() {
        let u1 = gene_randoms.get(i * 2).copied().unwrap_or(0.5);
        let u2 = gene_randoms.get(i * 2 + 1).copied().unwrap_or(0.5);
        let noise = box_muller(u1, u2) * new_ind.step_sizes[i];
        *gene += noise;
    }
    new_ind
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_box_muller_shape() {
        let v = box_muller(0.3, 0.7);
        assert!(v.is_finite());
    }

    #[test]
    fn test_box_muller_not_constant() {
        let v1 = box_muller(0.3, 0.5);
        let v2 = box_muller(0.8, 0.2);
        assert_ne!(v1, v2);
    }

    #[test]
    fn test_mutate_changes_genes() {
        let ind = Individual::new(vec![0.0, 0.0, 0.0]);
        let randoms = vec![0.3, 0.7, 0.2, 0.8, 0.5, 0.4];
        let mutated = mutate(&ind, &randoms);
        assert_ne!(mutated.genes, ind.genes);
    }

    #[test]
    fn test_mutate_preserves_fitness_none() {
        let ind = Individual::new(vec![1.0]);
        let mutated = mutate(&ind, &[0.3, 0.7]);
        assert!(mutated.fitness.is_none());
    }

    #[test]
    fn test_mutate_preserves_step_sizes() {
        let ind = Individual::new(vec![1.0]).with_step_sizes(vec![0.5]);
        let mutated = mutate(&ind, &[0.3, 0.7]);
        assert_eq!(mutated.step_sizes, vec![0.5]);
    }

    #[test]
    fn test_mutate_step_sizes() {
        let steps = vec![1.0, 1.0];
        let randoms = vec![0.3, 0.7, 0.5, 0.4, 0.2, 0.8];
        let new_steps = mutate_step_sizes(&steps, 0.1, &randoms);
        assert_eq!(new_steps.len(), 2);
        // Should generally be different
    }

    #[test]
    fn test_mutate_step_sizes_positive() {
        let steps = vec![1.0];
        let randoms = vec![0.5, 0.5, 0.5, 0.5];
        let new_steps = mutate_step_sizes(&steps, 0.1, &randoms);
        assert!(new_steps[0] > 0.0);
    }

    #[test]
    fn test_full_mutate() {
        let ind = Individual::new(vec![1.0, 2.0]).with_step_sizes(vec![0.5, 0.5]);
        let randoms = vec![0.3, 0.7, 0.2, 0.8, 0.5, 0.4, 0.1, 0.9];
        let mutated = full_mutate(&ind, 0.1, &randoms);
        assert!(mutated.fitness.is_none());
        assert_eq!(mutated.dimension(), 2);
    }

    #[test]
    fn test_full_mutate_resets_age() {
        let mut ind = Individual::new(vec![1.0]);
        ind.age = 10;
        let mutated = full_mutate(&ind, 0.1, &[0.3, 0.7, 0.5, 0.4]);
        assert_eq!(mutated.age, 0);
    }

    #[test]
    fn test_mutate_with_large_randoms() {
        let ind = Individual::new(vec![0.0]);
        let mutated = mutate(&ind, &[0.9, 0.9]);
        assert!(mutated.genes[0] != 0.0);
    }
}
