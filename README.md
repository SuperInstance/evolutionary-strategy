# Evolutionary Strategy

[![crates.io](https://img.shields.io/crates/v/evolutionary-strategy.svg)](https://crates.io/crates/evolutionary-strategy)
[![docs.rs](https://docs.rs/evolutionary-strategy/badge.svg)](https://docs.rs/evolutionary-strategy)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

> **Evolution strategies (ES) for agent parameter optimization — population, mutation, recombination, and selection.**

---

## The Problem

Optimizing agent parameters is hard. Gradient-based methods require differentiable objectives. Random search is inefficient. Evolution strategies offer a middle ground: derivative-free optimization that works with any black-box objective, using biological evolution as a template.

## Why This Exists

Evolutionary Strategy implements the classic ES pipeline:
1. **Population**: Generate a population of candidate solutions
2. **Mutation**: Add noise to explore the parameter space
3. **Recombination**: Combine successful parents to create offspring
4. **Selection**: Keep the best performers
5. **Adaptation**: Adjust strategy parameters (mutation strength) over time

## Architecture

```
  ┌───────────────┐
  │  Population    │  ← Initial random solutions
  └───────┬───────┘
          │
  ┌───────▼───────┐
  │  Mutation     │  ← Add Gaussian noise
  └───────┬───────┘
          │
  ┌───────▼───────┐
  │  Evaluation   │  ← Score against objective
  └───────┬───────┘
          │
  ┌───────▼───────┐
  │  Selection    │  ← Keep top performers
  └───────┬───────┘
          │
  ┌───────▼───────┐
  │ Recombination │  ← Combine parents
  └───────┬───────┘
          │
  ┌───────▼───────┐
  │  Adaptation   │  ← Adjust mutation σ
  └───────┬───────┘
          │
          └──→ Repeat until convergence
```

## Installation

```toml
[dependencies]
evolutionary-strategy = "0.1"
```

## Modules

| Module | Description |
|--------|-------------|
| `population` | Population management and initialization |
| `mutation` | Mutation operators (Gaussian, uniform) |
| `recombination` | Parent combination strategies |
| `selection` | Selection operators (tournament, truncation) |
| `adaptation` | Strategy parameter adaptation (1/5 rule, CMA) |

## Usage Examples

### Example 1: Basic ES Optimization

```rust
use evolutionary_strategy::population::*;
use evolutionary_strategy::mutation::*;
use evolutionary_strategy::selection::*;

// Create initial population
// Apply mutations
// Select best performers
// Repeat
```

### Example 2: Adaptive Mutation Rate

```rust
use evolutionary_strategy::adaptation::*;

// 1/5 success rule: if > 1/5 of mutations improve, increase σ
// if < 1/5, decrease σ
```

## Theoretical Background

**(μ/ρ, λ)-ES**: μ parents, ρ recombination parents, λ offspring. The comma strategy uses only offspring for the next generation; the plus strategy also considers parents.

**1/5 Success Rule** (Rechenberg): After a fixed number of mutations, if more than 20% are successful, increase mutation strength; otherwise decrease it.

## Performance

ES is embarrassingly parallel — each individual can be evaluated independently, making it ideal for GPU/cluster deployment.

## License

Licensed under the [MIT License](LICENSE).

## Contributing

1. Fork the repository
2. Create a feature branch
3. Write tests
4. Push and open a Pull Request

## API Reference

### `population`

Population management and initialization:

```rust
use evolutionary_strategy::population::*;

// Create initial population of candidate solutions
// Support for various initialization strategies:
// - Random initialization
// - Seeded initialization from prior solutions
// - Biased initialization toward known-good regions
```

### `mutation`

Mutation operators for exploring the parameter space:

```rust
use evolutionary_strategy::mutation::*;

// Gaussian mutation: x' = x + N(0, σ)
// Uniform mutation: x' = x + U(-a, a)
// Adaptive mutation: σ adjusts based on success rate
```

### `recombination`

Parent combination strategies:

```rust
use evolutionary_strategy::recombination::*;

// Discrete recombination: select each gene from a random parent
// Intermediate recombination: average parent genes
// Global recombination: use entire population
```

### `selection`

Selection operators for survival:

```rust
use evolutionary_strategy::selection::*;

// (μ, λ): Select μ best from λ offspring only
// (μ + λ): Select μ best from parents + offspring
// Tournament: Pairwise comparison selection
// Roulette wheel: Fitness-proportional selection
```

### `adaptation`

Strategy parameter self-adaptation:

```rust
use evolutionary_strategy::adaptation::*;

// 1/5 Success Rule (Rechenberg):
//   If success_rate > 1/5: σ ← σ / 0.85 (increase exploration)
//   If success_rate < 1/5: σ ← σ × 0.85 (decrease exploration)
//
// CMA-ES (Covariance Matrix Adaptation):
//   Adapt full covariance matrix of mutation distribution
//   State-of-the-art for continuous optimization
```

## Mathematical Background

**Evolution Strategies** were introduced by Rechenberg (1973) and Schwefel (1975). The key insight is that strategy parameters (like mutation step size σ) should co-evolve with the solution parameters.

**Notation**: (μ/ρ, λ)-ES means:
- μ = number of parents
- ρ = number of parents involved in recombination
- λ = number of offspring per generation
- Comma (,) = only offspring survive to next generation
- Plus (+) = parents also compete for survival

**Mutation**: Each parameter is mutated by adding Gaussian noise:

```
x'_i = x_i + σ × N(0, 1)
```

**1/5 Success Rule**: After every k mutations, measure the success rate:

```
φ = successful_mutations / k

if φ > 1/5: σ ← σ / c    (c < 1, explore more)
if φ < 1/5: σ ← σ × c    (c < 1, explore less)
if φ = 1/5: σ unchanged   (optimal rate)
```

**CMA-ES**: Instead of a single σ, maintain a full covariance matrix C:

```
x' ~ N(m, σ² × C)
```

Where m is the mean, σ is the global step size, and C captures the correlation structure of the search distribution.

## Performance Characteristics

| Operation | Complexity |
|-----------|-----------|
| Population initialization | O(μ × d) |
| Mutation (Gaussian) | O(λ × d) |
| Recombination | O(μ × d) |
| Selection (truncation) | O(λ log λ) |
| Full generation | O(λ × d + λ log λ) |

Where μ = parents, λ = offspring, d = dimensionality.

ES is **embarrassingly parallel** — each individual can be evaluated independently, making it ideal for GPU/cluster deployment.

## Comparison with Alternatives

| Feature | evolutionary-strategy | genetic-algorithm | CMA-ES (full) |
|---------|----------------------|-------------------|---------------|
| Derivative-free | ✅ | ✅ | ✅ |
| Self-adaptive σ | ✅ | ❌ | ✅ |
| Real-valued optimization | ✅ Native | ❌ Bit-string | ✅ |
| Parallelizable | ✅ | ✅ | ✅ |
| No crossover needed | ✅ | ❌ | ✅ |
| Lightweight | ✅ | Varies | ❌ Heavy |

## Usage Examples

### Example 2: Sphere Function Optimization

```rust
use evolutionary_strategy::*;

// Classic benchmark: minimize f(x) = Σ x_i²
// Optimal solution: x = [0, 0, ..., 0]

// 1. Initialize population around [5.0, 5.0, 5.0]
// 2. Evolve with selection pressure toward lower f(x)
// 3. Population converges to [0, 0, 0]
```

### Example 3: Adaptive Mutation Rate

```rust
use evolutionary_strategy::adaptation::*;

// Start with large σ for broad exploration
// As optimization progresses, 1/5 rule reduces σ
// Final σ is small for fine-tuning
// This automatic annealing is the key advantage of ES
```
