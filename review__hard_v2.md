## NeurIPS 2026 Review

**Paper:** "Stochastic Attention via Langevin Dynamics on the Modern Hopfield Energy"
**Rating:** 5 — Borderline reject (leaning toward weak accept after author response)
**Confidence:** 4

---

### Summary

The paper applies the unadjusted Langevin algorithm (ULA) to the modern Hopfield energy to derive a "stochastic attention" update — a three-term iterate comprising contraction, softmax attention pull, and isotropic noise. The authors show the sampler interpolates between deterministic retrieval (β→∞) and generation (small β), derive a critical inverse temperature β* ~ √d from an entropy inflection argument, and validate on synthetic data, MNIST, protein sequences, face images, and financial returns.

---

### Strengths

1. **Core idea is elegant.** The observation that ∇E(ξ) = ξ − T(ξ) yields a closed-form score function is correct, principled, and non-obvious to practitioners. Converting attention into a sampler without any score matching or training is genuinely useful.

2. **Proposition 2 (entropy inflection) is the best theoretical contribution.** The β* ~ √d scaling law with a data-dependent generalization via dH²/dβ² is crisp and practically actionable. The protein example (β* = 3.85 vs. √59 = 7.68 due to conserved residues) is a satisfying use of the theory.

3. **Honesty about the theory–practice gap.** The authors explicitly state that convergence guarantees only hold for βσ²_max < 2, which is far below the experimental regime (β ∈ {200, 2000}). This transparency is appreciated. The MALA comparison (99.2% acceptance at α = 0.01) at least empirically validates the approximation.

4. **Multi-domain experiments.** Testing across MNIST, protein sequences, face images, and S&P 500 returns, with the same algorithm and principled temperature selection, is a strength.

---

### Weaknesses

**W1 — Derivation depth.** Once one accepts ∇E = ξ − T(ξ) (proved by Ramsauer et al.) and ULA (standard), the stochastic attention update in Eq. (3) is three lines of algebra. Proposition 1 follows from softmax Lipschitz bounds; Corollary 1 is immediate from standard ULA theory. The paper frames these as contributions, but they are largely textbook assembly. The scientific content that would justify NeurIPS acceptance is the β* analysis and the empirical validation — but those need to be substantially strengthened.

**W2 — The convergence guarantee covers none of the experiments.** Corollary 1 requires βσ²_max < 2. With σ²_max ≈ 2–3, this binds at β ≲ 1. Every experiment uses β ∈ {8, 77, 200, 2000}. The guarantee the paper provides therefore says nothing about its actual operating regime. The geometric ergodicity of the continuous SDE (Section 4, citing Roberts & Tweedie) is qualitative only — no rate, no bound. This is a significant theoretical gap. The paper cannot both claim theoretical rigor and then only demonstrate it for a parameter range that is irrelevant to all applications.

**W3 — Baselines are designed to lose.** The core experimental claim — SA outperforms learned models — rests on:

- A VAE with latent dimension 8 trained on K = 100 examples (MNIST) or K = 68 (protein). This will overfit severely. The comparison should use a VAE pretrained on full MNIST and then fine-tuned or prompted with the K-example memory.
- A DDPM trained on 100 images for 5,000 epochs. No one expects a diffusion model to learn a data distribution from 100 examples. This is not a fair baseline; it demonstrates that training-based methods require data, which is already known.
- For proteins: no comparison to ESM-2, MSA Transformer, or ProtGPT2, all of which are zero-shot and would not require 68-sequence training.

The entire regime where SA dominates (K small, training-based methods fail) is by construction. The paper needs at least one comparison where training-based methods have adequate data.

**W4 — Metrics reward noise.** The paper's central metrics are:
- Novelty = 1 − max cosine similarity to stored patterns. Higher is "better."
- Diversity = mean pairwise cosine distance. Higher is "better."

Table 1 shows DDPM achieves N = 0.938, D = 0.991 — the highest of any method — because it generates pure noise. SA at β = 200 achieves N = 0.548, D = 0.885 with mean energy +1.467 (above zero, i.e., off the memory manifold). The "Gaussian noise control" in the appendix partially rehabilitates this, but the main table's framing is misleading. A method that generates isotropic noise should not appear to dominate the table. The authors should either change the metrics or make the energy criterion a first-class constraint (e.g., report only samples with E < 0, so they lie on or near the manifold).

**W5 — Protein experiment overclaims.** The "6.9× lower KL" headline claim compares amino acid marginal frequencies — a single-position, zero-order statistic that ignores sequence-level dependencies, secondary structure, and biological function. PCA to d = 59 from K = 68 sequences means SA generates samples in a very low-dimensional space nearly spanned by the training data. These are not "novel" sequences in any biologically meaningful sense. No validation against Pfam HMM membership, no structural modeling, no comparison to protein language models.

**W6 — The generation regime samples are acknowledged to be low quality.** The paper describes β = 200 samples as "blurry-but-recognizable." This is honest but not compelling for 2026. No FID, no CLIP score, no perceptual quality metric. For a generation paper to succeed at NeurIPS, the sample quality needs to be compelling, not merely "structured interpolations between stored patterns."

**W7 — MALA comparison is incomplete.** MALA is only compared at β = 2000 (retrieval regime). At β = 200 (the generation regime where SA claims its strongest results), there is no MALA comparison. This omission is conspicuous — it would directly confirm or refute whether the ULA bias matters in the generation regime.

---

### Questions for Authors

1. What happens if you run MALA at β = 200 with the same multi-chain protocol? Does novelty/diversity change materially?

2. Can you compare SA against a VAE and GMM trained on the full MNIST dataset and then restricted to generate digit-3 images? This would isolate whether SA's advantage is fundamental or merely an artifact of the limited training data.

3. For the protein experiment: what fraction of SA-generated sequences pass the Pfam RRM HMM filter at E-value < 0.01? This would give a biologically meaningful novelty/validity metric.

4. The β* ~ √d scaling assumes random unit-norm memories with Var(e_k) = 1/d. How sensitive is this estimate to memory normalization? Real MNIST images are not unit-sphere distributed.

5. Can the method be extended to multi-head or masked attention? The introduction suggests this as future work, but even a toy demonstration would significantly strengthen the practical relevance.

---

### Minor Issues

- The N/d nomenclature inconsistency (method.tex uses N, experiments.tex uses d for the same quantity) will confuse readers and should be fixed before submission.
- Eq. (snr-star) writes SNR* ~ √(α/(2√d)), which has unusual dimensional structure. The derivation should be verified.
- The paper claims "every pretrained attention head" can serve as a sampler. This ignores that real attention uses Q, K, V projections and operates on hidden states, not memory patterns. The connection is approximate, not exact, and should be stated more carefully.

---

### Recommendation

The core idea is sound and worth publishing somewhere. But in its current form, the paper oversells its theoretical guarantees (which cover none of the experiments), uses unfair baselines, employs metrics that conflate noise with generation, and makes headline claims on protein sequences that require stronger domain validation.

**Recommended revisions:** (1) Either tighten theoretical claims to what is actually proved or prove non-convex mixing bounds; (2) add at least one baseline comparison where training-based methods have adequate data; (3) add MALA at β = 200; (4) replace amino acid KL with a structural/HMM-based validity metric for protein claims; (5) standardize the N/d notation.

The retrieval–generation duality is real and the entropy inflection criterion is genuinely useful. With the above addressed, this would be a strong poster or short paper. As a full NeurIPS paper, it currently does not meet the bar.

---

**Score: 5** | **Confidence: 4**
