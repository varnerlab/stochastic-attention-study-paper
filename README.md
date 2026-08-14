# Modern Hopfield Energy as a Memory-Centered Gaussian Mixture

This repository contains the code, data, and manuscript source for a revised
study of sampling from the tied modern-Hopfield energy. The central correction
is that the normalized Boltzmann density is an explicit finite isotropic
Gaussian mixture centered on the stored memories. It can therefore be sampled
exactly by drawing a component and adding Gaussian noise. Unadjusted Langevin
(ULA) and Metropolis-adjusted Langevin (MALA) remain available as finite-time
dynamics, but they are not required to obtain equilibrium samples from this
base model.

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

This produces `Paper_v1.pdf`. The revision derives the exact normalized law,
identifies attention weights as posterior component responsibilities, validates
the sampling implementations against an analytic target, reanalyzes the MNIST
and protein endpoints, and narrows claims whose submitted evidence did not
support model-quality or open-ended-generation conclusions.

## License

This project is released under the [MIT License](LICENSE).
