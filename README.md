# Stochastic Attention via Langevin Dynamics on the Modern Hopfield Energy

This repository contains the code, data, and manuscript source for stochastic
attention: an energy-based sampling method that combines the modern Hopfield
attention drift with Langevin noise. The construction also supports composite
targets $E(x)+\lambda C(x)$, for which the added energy can encode a constraint,
likelihood, or design objective.

For the unmodified tied modern-Hopfield energy, the normalized Boltzmann density
is additionally an explicit finite isotropic Gaussian mixture centered on the
stored memories. Exact ancestral draws from this special case provide a
ground-truth calibration for ULA and MALA. The GMM and Langevin formulations
target the same base law; the stochastic-attention construction is more general
because it continues to apply when an added energy destroys the mixture form.

For memories `m_k`, inverse temperature `β`, and optional nonnegative
multiplicities `r_k`, the exact component probabilities are proportional to

```text
r_k * exp(β * ||m_k||² / 2),
```

and a sample from component `k` is `m_k + β^(-1/2) ε`, with standard Gaussian
`ε`. For equal-norm memories and equal multiplicities, the component law is
uniform.

## Repository layout

- `code/` contains the Julia implementation, tests, experiments, and cached
  data.
- `paper-arxiv/` contains the revised arXiv manuscript.
- `reviews/` contains local revision notes and experiment logs; it is excluded
  from version control.

The core implementation is in `code/src/Compute.jl`. It provides the analytic
mixture weights, posterior responsibilities, normalized log density, exact
score, exact independent sampler, ULA, and MALA. The tests cover unequal memory
norms, multiplicities, finite-difference agreement of the score, empirical
component frequencies, and seeded reproducibility.

## Reproducing the revised results

The code requires Julia 1.10 or later. From `code/`:

```bash
julia --project=. test/runtests.jl
julia --project=. run_exact_target_validation.jl
```

The analytic target validation compares exact draws, ULA, and MALA against
known moments for a deliberately unequal-norm mixture.

For the MNIST endpoint reanalysis:

```bash
cd code/mnist-experiment
julia --project=. run_exact_mixture_baseline.jl
```

This script compares the submitted ULA endpoint statistics with independent
draws from the exact all-component law, an initialization-matched diagnostic,
a bandwidth sweep, and a local-neighbor perturbation baseline.

For the PF00076 protein endpoint reanalysis:

```bash
cd code/sequence-experiment
julia --project=. run_exact_mixture_protein.jl
```

The protein script uses the cached Pfam alignment and HMM files. Recomputing
the HMM pass-rate diagnostic requires HMMER.

Legacy notebooks and scripts from the submitted study remain under `code/` for
auditability. Their original exploratory labels, such as “retrieval” and
“generation” regimes, should not be read as claims that a Langevin trajectory
defines a more expressive equilibrium distribution than the exact mixture.

## Building the paper

Building the manuscript requires `pdflatex` and `bibtex`. From
`paper-arxiv/`:

```bash
./Build.sh Paper_v1
```

This produces `Paper_v1.pdf`. The revision presents stochastic attention and
its constrained-energy extension first, retains the ULA/MALA theory and
pseudocode, derives the exact base-model law, validates the sampling
implementations against that analytic target, and reanalyzes the MNIST and
protein endpoints.

## License

This project is released under the [MIT License](LICENSE).
