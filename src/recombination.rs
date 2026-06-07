//! Parent mixing (recombination).

use crate::population::Individual;

/// Recombination type.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RecombinationType {
    Discrete,   // randomly pick from either parent per gene
    Intermediate, // average of parents
    GlobalDiscrete, // pick from any parent in the pool
    GlobalIntermediate, // average across all parents
}

/// Discrete recombination: pick each gene from a random parent.
pub fn discrete(a: &Individual, b: &Individual, mask: &[bool]) -> Individual {
    let genes: Vec<f64> = a.genes.iter().zip(b.genes.iter()).zip(mask.iter())
        .map(|((&ga, &gb), &m)| if m { ga } else { gb })
        .collect();
    let steps: Vec<f64> = a.step_sizes.iter().zip(b.step_sizes.iter()).zip(mask.iter())
        .map(|((&sa, &sb), &m)| if m { sa } else { sb })
        .collect();
    Individual { genes, fitness: None, step_sizes: steps, age: 0 }
}

/// Intermediate recombination: average of two parents.
pub fn intermediate(a: &Individual, b: &Individual) -> Individual {
    let genes: Vec<f64> = a.genes.iter().zip(b.genes.iter())
        .map(|(&ga, &gb)| (ga + gb) / 2.0)
        .collect();
    let steps: Vec<f64> = a.step_sizes.iter().zip(b.step_sizes.iter())
        .map(|(&sa, &sb)| (sa + sb) / 2.0)
        .collect();
    Individual { genes, fitness: None, step_sizes: steps, age: 0 }
}

/// Weighted intermediate recombination.
pub fn weighted_intermediate(a: &Individual, b: &Individual, weight_a: f64) -> Individual {
    let weight_b = 1.0 - weight_a;
    let genes: Vec<f64> = a.genes.iter().zip(b.genes.iter())
        .map(|(&ga, &gb)| ga * weight_a + gb * weight_b)
        .collect();
    let steps: Vec<f64> = a.step_sizes.iter().zip(b.step_sizes.iter())
        .map(|(&sa, &sb)| sa * weight_a + sb * weight_b)
        .collect();
    Individual { genes, fitness: None, step_sizes: steps, age: 0 }
}

/// Global intermediate: average of all parents.
pub fn global_intermediate(parents: &[Individual]) -> Option<Individual> {
    if parents.is_empty() { return None; }
    let dim = parents[0].dimension();
    let mut genes = vec![0.0; dim];
    let mut steps = vec![0.0; dim];
    for p in parents {
        for (i, &g) in p.genes.iter().enumerate() { genes[i] += g; }
        for (i, &s) in p.step_sizes.iter().enumerate() { steps[i] += s; }
    }
    let n = parents.len() as f64;
    for g in &mut genes { *g /= n; }
    for s in &mut steps { *s /= n; }
    Some(Individual { genes, fitness: None, step_sizes: steps, age: 0 })
}

/// Global discrete: for each gene position, pick from a random parent.
pub fn global_discrete(parents: &[Individual], selections: &[usize]) -> Option<Individual> {
    if parents.is_empty() { return None; }
    let dim = parents[0].dimension();
    let genes: Vec<f64> = (0..dim).map(|i| {
        let parent_idx = selections.get(i).copied().unwrap_or(0) % parents.len();
        parents[parent_idx].genes[i]
    }).collect();
    let steps: Vec<f64> = (0..dim).map(|i| {
        let parent_idx = selections.get(i).copied().unwrap_or(0) % parents.len();
        parents[parent_idx].step_sizes[i]
    }).collect();
    Some(Individual { genes, fitness: None, step_sizes: steps, age: 0 })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_discrete_picks_a() {
        let a = Individual::new(vec![1.0, 2.0]);
        let b = Individual::new(vec![10.0, 20.0]);
        let child = discrete(&a, &b, &[true, true]);
        assert_eq!(child.genes, vec![1.0, 2.0]);
    }

    #[test]
    fn test_discrete_picks_b() {
        let a = Individual::new(vec![1.0, 2.0]);
        let b = Individual::new(vec![10.0, 20.0]);
        let child = discrete(&a, &b, &[false, false]);
        assert_eq!(child.genes, vec![10.0, 20.0]);
    }

    #[test]
    fn test_discrete_mixed() {
        let a = Individual::new(vec![1.0, 2.0]);
        let b = Individual::new(vec![10.0, 20.0]);
        let child = discrete(&a, &b, &[true, false]);
        assert_eq!(child.genes, vec![1.0, 20.0]);
    }

    #[test]
    fn test_intermediate() {
        let a = Individual::new(vec![0.0, 10.0]);
        let b = Individual::new(vec![10.0, 10.0]);
        let child = intermediate(&a, &b);
        assert!((child.genes[0] - 5.0).abs() < 1e-10);
        assert!((child.genes[1] - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_weighted_intermediate() {
        let a = Individual::new(vec![0.0]);
        let b = Individual::new(vec![10.0]);
        let child = weighted_intermediate(&a, &b, 0.8);
        assert!((child.genes[0] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_weighted_intermediate_equal() {
        let a = Individual::new(vec![0.0]);
        let b = Individual::new(vec![10.0]);
        let child = weighted_intermediate(&a, &b, 0.5);
        assert!((child.genes[0] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_global_intermediate() {
        let a = Individual::new(vec![0.0, 0.0]);
        let b = Individual::new(vec![10.0, 20.0]);
        let c = Individual::new(vec![5.0, 10.0]);
        let child = global_intermediate(&[a, b, c]).unwrap();
        assert!((child.genes[0] - 5.0).abs() < 1e-10);
        assert!((child.genes[1] - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_global_intermediate_empty() {
        assert!(global_intermediate(&[]).is_none());
    }

    #[test]
    fn test_global_discrete() {
        let a = Individual::new(vec![1.0, 2.0, 3.0]);
        let b = Individual::new(vec![10.0, 20.0, 30.0]);
        let child = global_discrete(&[a, b], &[0, 1, 0]).unwrap();
        assert_eq!(child.genes, vec![1.0, 20.0, 3.0]);
    }

    #[test]
    fn test_global_discrete_empty() {
        assert!(global_discrete(&[], &[]).is_none());
    }

    #[test]
    fn test_recombination_resets_fitness() {
        let mut a = Individual::new(vec![1.0]); a.set_fitness(5.0);
        let b = Individual::new(vec![2.0]);
        let child = intermediate(&a, &b);
        assert!(child.fitness.is_none());
    }

    #[test]
    fn test_recombination_resets_age() {
        let mut a = Individual::new(vec![1.0]); a.age = 10;
        let b = Individual::new(vec![2.0]);
        let child = intermediate(&a, &b);
        assert_eq!(child.age, 0);
    }
}
