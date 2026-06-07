//! Candidate solutions for evolution strategies.

/// An individual solution with strategy parameters.
#[derive(Clone, Debug)]
pub struct Individual {
    pub genes: Vec<f64>,
    pub fitness: Option<f64>,
    pub step_sizes: Vec<f64>,
    pub age: usize,
}

impl Individual {
    pub fn new(genes: Vec<f64>) -> Self {
        let dim = genes.len();
        Individual { genes, fitness: None, step_sizes: vec![1.0; dim], age: 0 }
    }

    pub fn with_step_sizes(mut self, steps: Vec<f64>) -> Self {
        self.step_sizes = steps;
        self
    }

    pub fn dimension(&self) -> usize { self.genes.len() }

    pub fn set_fitness(&mut self, f: f64) { self.fitness = Some(f); }

    pub fn get_fitness(&self) -> f64 { self.fitness.unwrap_or(f64::NAN) }

    pub fn age_one(&mut self) { self.age += 1; }
}

/// A population of individuals.
pub struct Population {
    pub individuals: Vec<Individual>,
}

impl Default for Population {
    fn default() -> Self {
        Self::new()
    }
}

impl Population {
    pub fn new() -> Self { Population { individuals: Vec::new() } }

    pub fn from_vec(individuals: Vec<Individual>) -> Self {
        Population { individuals }
    }

    pub fn len(&self) -> usize { self.individuals.len() }
    pub fn is_empty(&self) -> bool { self.individuals.is_empty() }

    pub fn add(&mut self, ind: Individual) { self.individuals.push(ind); }

    /// Create a random uniform population.
    pub fn random(dim: usize, size: usize, bounds: (f64, f64), rng_vals: &[f64]) -> Self {
        let range = bounds.1 - bounds.0;
        let mut inds = Vec::new();
        for i in 0..size {
            let genes: Vec<f64> = (0..dim).map(|j| {
                let idx = (i * dim + j) % rng_vals.len();
                bounds.0 + rng_vals[idx] * range
            }).collect();
            inds.push(Individual::new(genes));
        }
        Population { individuals: inds }
    }

    /// Best individual by fitness (lower is better).
    pub fn best(&self) -> Option<&Individual> {
        self.individuals.iter().filter(|i| i.fitness.is_some()).min_by(|a, b| {
            a.fitness.partial_cmp(&b.fitness).unwrap()
        })
    }

    /// Worst individual.
    pub fn worst(&self) -> Option<&Individual> {
        self.individuals.iter().filter(|i| i.fitness.is_some()).max_by(|a, b| {
            a.fitness.partial_cmp(&b.fitness).unwrap()
        })
    }

    /// Average fitness.
    pub fn avg_fitness(&self) -> f64 {
        let fit: Vec<f64> = self.individuals.iter().filter_map(|i| i.fitness).collect();
        if fit.is_empty() { return f64::NAN; }
        fit.iter().sum::<f64>() / fit.len() as f64
    }

    /// Fitness standard deviation.
    pub fn fitness_std(&self) -> f64 {
        let avg = self.avg_fitness();
        if avg.is_nan() { return f64::NAN; }
        let fit: Vec<f64> = self.individuals.iter().filter_map(|i| i.fitness).collect();
        let variance = fit.iter().map(|f| (f - avg).powi(2)).sum::<f64>() / fit.len() as f64;
        variance.sqrt()
    }

    /// Sort by fitness (ascending, best first).
    pub fn sort_by_fitness(&mut self) {
        self.individuals.sort_by(|a, b| {
            let fa = a.fitness.unwrap_or(f64::INFINITY);
            let fb = b.fitness.unwrap_or(f64::INFINITY);
            fa.partial_cmp(&fb).unwrap()
        });
    }

    pub fn age_all(&mut self) {
        for i in &mut self.individuals { i.age_one(); }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_individual_new() {
        let ind = Individual::new(vec![1.0, 2.0, 3.0]);
        assert_eq!(ind.dimension(), 3);
        assert!(ind.fitness.is_none());
        assert_eq!(ind.step_sizes, vec![1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_individual_with_step_sizes() {
        let ind = Individual::new(vec![1.0, 2.0]).with_step_sizes(vec![0.1, 0.2]);
        assert_eq!(ind.step_sizes, vec![0.1, 0.2]);
    }

    #[test]
    fn test_individual_fitness() {
        let mut ind = Individual::new(vec![1.0]);
        ind.set_fitness(3.14);
        assert!((ind.get_fitness() - 3.14).abs() < 1e-10);
    }

    #[test]
    fn test_individual_age() {
        let mut ind = Individual::new(vec![1.0]);
        ind.age_one();
        ind.age_one();
        assert_eq!(ind.age, 2);
    }

    #[test]
    fn test_population_new() {
        let pop = Population::new();
        assert!(pop.is_empty());
    }

    #[test]
    fn test_population_add() {
        let mut pop = Population::new();
        pop.add(Individual::new(vec![1.0]));
        assert_eq!(pop.len(), 1);
    }

    #[test]
    fn test_population_best() {
        let mut pop = Population::new();
        let mut a = Individual::new(vec![1.0]); a.set_fitness(5.0);
        let mut b = Individual::new(vec![2.0]); b.set_fitness(2.0);
        pop.add(a);
        pop.add(b);
        assert!((pop.best().unwrap().get_fitness() - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_population_worst() {
        let mut pop = Population::new();
        let mut a = Individual::new(vec![1.0]); a.set_fitness(5.0);
        let mut b = Individual::new(vec![2.0]); b.set_fitness(2.0);
        pop.add(a);
        pop.add(b);
        assert!((pop.worst().unwrap().get_fitness() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_population_avg_fitness() {
        let mut pop = Population::new();
        let mut a = Individual::new(vec![1.0]); a.set_fitness(4.0);
        let mut b = Individual::new(vec![2.0]); b.set_fitness(6.0);
        pop.add(a); pop.add(b);
        assert!((pop.avg_fitness() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_population_fitness_std() {
        let mut pop = Population::new();
        let mut a = Individual::new(vec![1.0]); a.set_fitness(3.0);
        let mut b = Individual::new(vec![2.0]); b.set_fitness(7.0);
        pop.add(a); pop.add(b);
        let std = pop.fitness_std();
        assert!(std > 0.0);
    }

    #[test]
    fn test_population_sort() {
        let mut pop = Population::new();
        let mut a = Individual::new(vec![1.0]); a.set_fitness(10.0);
        let mut b = Individual::new(vec![2.0]); b.set_fitness(1.0);
        pop.add(a); pop.add(b);
        pop.sort_by_fitness();
        assert!((pop.individuals[0].get_fitness() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_population_random() {
        let rng = vec![0.5; 100];
        let pop = Population::random(3, 10, (-5.0, 5.0), &rng);
        assert_eq!(pop.len(), 10);
        assert_eq!(pop.individuals[0].dimension(), 3);
    }

    #[test]
    fn test_population_age_all() {
        let mut pop = Population::new();
        pop.add(Individual::new(vec![1.0]));
        pop.add(Individual::new(vec![2.0]));
        pop.age_all();
        assert_eq!(pop.individuals[0].age, 1);
        assert_eq!(pop.individuals[1].age, 1);
    }

    #[test]
    fn test_best_empty_pop() {
        let pop = Population::new();
        assert!(pop.best().is_none());
    }
}
