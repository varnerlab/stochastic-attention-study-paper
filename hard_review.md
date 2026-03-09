# NeurIPS Review: "Stochastic Attention via Langevin Dynamics on the Modern Hopfield Energy"

**Rating: 4 — Weak Reject**
**Confidence: 4 — Confident**

---

## Summary

The paper applies the Unadjusted Langevin Algorithm (ULA) to the modern Hopfield energy (Ramsauer et al., 2021) to yield a "stochastic attention" sampler. The energy gradient is exactly the identity minus the softmax attention map, so no learned score network is needed. A single inverse temperature β interpolates between deterministic retrieval and stochastic generation. Validation is on synthetic data, MNIST digits 1/3/8 (K=100), protein sequences (Pfam PF00076, K=68), and two appendix datasets (S&P 500 log-returns, Simpsons faces).

---

## Strengths

1. **Clean conceptual connection.** The assembly of attention-as-gradient-descent, Hopfield energy, and Langevin dynamics is presented clearly and the derivation is correct.

2. **Training-free.** The closed-form score function is a genuine practical advantage over EBMs requiring score matching or contrastive divergence.

3. **Honest about limitations.** The paper explicitly states the geometric-rate convergence guarantee breaks down in the operating regime (β ≫ 1), which is unusually candid.

4. **Algorithm is simple and reproducible.** The pseudocode is complete and the per-step cost (O(NK)) is clearly stated.

---

## Weaknesses

**W1 — Novelty is incremental, not mechanistic.** ✅ **ADDRESSED.**
The core mathematical steps are: (a) Ramsauer et al. already proved attention = one gradient step on the Hopfield energy; (b) Langevin dynamics on energy-based models is 35 years old. The contribution is mechanically applying ULA to an already-derived gradient. The paper does not reveal any new structure about the Hopfield energy — it uses what was already known. A hard NeurIPS bar requires a new *insight*, not a new *combination*. The authors should clearly articulate what was non-obvious or required non-trivial analysis beyond substituting the known gradient into the ULA template.

> **Author response:** The introduction now explicitly acknowledges that the individual ingredients are known, and articulates three aspects of the synthesis that are inaccessible from either ingredient alone: (1) **Training-free score function from architectural structure** — the Hopfield energy's gradient being exactly the identity minus the attention map yields a closed-form score that eliminates the need for score matching or contrastive divergence, converting any pretrained attention head into a sampler with zero additional training; (2) **Retrieval–generation phase transition** — Ramsauer et al. analyzed the energy only for retrieval (β→∞) and the Langevin literature doesn't know about attention; the entropy inflection analysis (Proposition 4) characterizes the full temperature spectrum and derives the critical β* with a scaling law β*~√N; (3) **Energy landscape analysis for sampling** — smoothness, dissipativity, and condition number of the Hopfield energy as a function of β and K/d had not been characterized for sampling guarantees; the paper provides these bounds and explicitly identifies where they break down.

**W2 — The convergence guarantee does not apply in any experiment.** ✅ **ADDRESSED (reframed).**
Corollary 1 guarantees geometric mixing only when βσ²_max < 2. The paper states σ²_max ≈ 2–3, meaning the guarantee holds only for β ≲ 0.67–1.0. Every experiment uses β ∈ {5, 200, 2000}. The paper acknowledges this and says "deriving explicit mixing-time bounds in this regime is an open problem." This is not a minor footnote — it means the paper's theoretical contribution does not apply to its own experiments. What remains is only the continuous-time ergodicity of the SDE (Roberts & Tweedie 1996), which holds for any well-behaved energy and adds nothing specific to the Hopfield setting.

> **Author response:** The paper has been restructured to honestly position the convergence corollary. The corollary is now presented as characterizing *where the convex regime ends* (the onset of non-convexity at βσ²_max = 2), not as the paper's main theoretical contribution. A new "Beyond the convex regime" paragraph in Section 4 explains why quantitative mixing bounds at high β are a fundamental open problem: for non-convex energies, mixing time scales exponentially in barrier height / temperature (Bovier et al. 2004), and in the Hopfield energy at high β the barriers are O(β) while the temperature is 1/β. The paper explains that this is consistent with the empirical observation that single chains at β=2000 do not cross basins (diversity 0.282). However, *intra-basin* mixing is fast (locally strongly convex), and the multi-chain protocol exploits this by initializing across basins and relying on fast local mixing within each. At β=200 the barriers are low enough that single chains do cross basins (diversity 0.796). The primary theoretical contributions are now the entropy inflection analysis (Proposition 4, which applies at all β) and the energy landscape characterization (Proposition 1), not the convex-regime corollary. The discussion has been updated correspondingly.

**W3 — The experimental baselines are far too weak.** ✅ **ADDRESSED.**
The "best learned baseline" being a VAE with latent dimension 8 trained on 100 MNIST images is not a competitive generative model by 2024 standards. Absent comparisons to:
- A simple DDPM or flow-matching model trained on the same 100 patterns,
- A properly tuned EBM (e.g., with SGLD),
- A modern VAE with better architecture and more capacity,

the claim "2.6× more novel than the best learned baseline" is misleading. The VAE baseline appears deliberately underspecified (latent dim 8 for 784-dimensional images is extremely low).

> **Author response:** The VAE baseline now runs on all MNIST digits (1, 3, 8) and on the protein sequence domain, using the same two-phase training protocol (AE warmup + KL annealing). The VAE consistently emerges as the strongest non-Langevin baseline across all domains, confirming the ranking is not cherry-picked. A DDPM baseline (Ho et al. 2020) has been added: MLP denoiser (784→512→256→512→784 with sinusoidal timestep embedding), T=200 diffusion steps, linear β schedule, 5,000 full-batch training epochs. The DDPM produced samples with max-cosine 0.062 to the nearest stored pattern (indistinguishable from isotropic Gaussian noise at 0.057), novelty 0.938, diversity 0.991. The training loss plateaued at ≈0.855 (near the random-prediction baseline), confirming the model failed to learn the distribution from K=100 images.
>
> **Scaling study (K=100 to 3,500):** To ensure a fair comparison, we ran SA vs DDPM at K ∈ {100, 500, 1000, 2000, 3500} digit-3 images (the full MNIST supply). At each K, β was set via entropy inflection (Proposition 4) for a principled, data-dependent operating point. Results: DDPM's max-cosine never exceeds 0.09 across the entire range (near the isotropic noise floor of ~0.06), while SA achieves 0.13–0.18 at β* (generation) and 0.24–0.33 at 5β* (retrieval). SA's structural advantage persists and grows with K. The entropy-inflection β* increases from 9 (K=100) to 18 (K=3500), consistent with the theoretical √N scaling. SA requires no training and no manual tuning: the entropy inflection provides automatic β selection at each K. Full results in Appendix (Table + 4-panel scaling figure).

**W4 — The generation metrics are trivially gamed by high temperature.** ⚠️ **PARTIALLY ADDRESSED.**
At β=200 (the "generation" row), the mean energy is +1.467 — positive, meaning samples lie *outside* the attractor manifold. The paper describes these as "blurry-but-recognizable," which is consistent with high-temperature noise rather than structured generation. In this regime, high novelty and diversity are achieved trivially: a Gaussian noise sampler would also score high on both metrics. The paper never shows that β=200 samples are semantically meaningful in a way that high-temperature Gaussian noise is not. There are no FID scores, IS scores, or human evaluations anywhere in the paper.

> **Author response (partial):** A Gaussian noise control experiment (Appendix, Section "Gaussian Noise Control at β=200") compares SA against two controls: (1) Gaussian noise matched to SA's per-pixel mean and variance, and (2) isotropic Gaussian noise matched to SA's norm. The key separating metric is max cosine similarity to the nearest stored pattern: SA retains 0.453, the matched Gaussian achieves only 0.328, and isotropic noise gets 0.057. Hopfield energy follows the same ordering: SA (1.47) < matched Gaussian (1.78) < isotropic (2.34). Novelty and diversity are similar between SA and matched Gaussian, confirming the reviewer's observation that these metrics alone are insufficient — but max-cos and energy demonstrate that SA samples are geometrically closer to stored patterns because the Langevin gradient biases the chain toward the memory manifold. Visually, SA samples show faint digit-3 structure (curved strokes) while both Gaussian controls show uniform static. The paper now honestly notes that β=200 is a high-temperature regime with limited sample quality, and that higher-fidelity generation requires operating closer to the transition band. FID/IS via InceptionV3 remain impractical for 28×28 grayscale images with K=100 training patterns; a diagonal pixel-space Fréchet distance is reported instead (SA: 2.81 vs matched Gaussian: 2.86 vs isotropic: 3.91).

**W5 — The SNR selection rule is empirical, not derived.** ✅ **ADDRESSED.**
Equation (snr) defines SNR = √(αβ/2d). The claim that "the transition occurs near SNR ≈ 0.025" is an empirical observation from the d=64, K=16 synthetic experiment. The rule is then inverted to prescribe β for new domains. But the threshold 0.025 is data-dependent and dimensionality-normalized without theoretical justification. The paper presents this as a "principled" selection rule, which overstates its theoretical status.

> **Author response:** The revised paper adds Proposition 4 (Section 4.3), which derives dH/dβ = −β Var_p(e) for the attention entropy H as a function of inverse temperature. The retrieval–generation phase boundary is defined as the inflection point of H(β), satisfying Var_p(e) = −β* μ₃ (Eq. 7). For random unit-norm memories, a scaling argument gives β* ~ √N, yielding SNR* = √(α/(2√N)). At N=64, α=0.01 this evaluates to exactly 0.025, matching the empirical observation. The threshold is now explicitly characterized as dimension-dependent (scaling as N^{-1/4}), not a universal constant, and Eq. 7 provides a fully data-dependent criterion for structured data. The "dimension-independent" claim has been removed from the discussion. A full text consistency pass replaced all occurrences of "transition band (0.02--0.03)" with dimension-qualified references to the entropy inflection and Proposition 4.

**W6 — The "four domains" claim is inflated.** ✅ **ADDRESSED.**
The main paper body has one real-data experiment (MNIST). The finance (S&P 500) and face (Simpsons) experiments are in appendices and involve no comparison to baselines — they only test SA itself and report the same metrics. This does not constitute validation in four independent domains.

> **Author response:** A protein sequence experiment (Pfam PF00076, RRM family) has been added as a fifth domain. Unlike the appendix experiments, it includes the full baseline comparison (bootstrap, Gaussian perturbation, convex combination, GMM-PCA, VAE, MALA) with protein-specific evaluation metrics (sequence identity to nearest stored member, amino acid composition KL divergence). The pipeline encodes 68 aligned sequences via one-hot + PCA (1420 → 59 dims), runs SA in PCA space, and decodes back to amino acid sequences. SA achieves the lowest amino acid KL divergence (0.107) — closest composition to the real family — while generating sequences ~62% identical to stored patterns (novel but family-consistent). The entropy inflection analysis (Section 4.3) is also validated at this new dimensionality: empirical β*=3.85 vs theoretical √d=7.68, with the discrepancy explained by the structured (non-random) pattern geometry. This brings the total to five domains with two main-body experiments including full baseline comparisons.

**W7 — Claims about RAG and ICL are unsubstantiated.** ✅ **ADDRESSED.**
The abstract states the method "extends naturally to retrieval-augmented generation and in-context learning settings." Neither is demonstrated experimentally. These are speculative future directions, not contributions, and should not appear in the abstract.

> **Author response:** The RAG/ICL claim has been removed from the abstract. In the introduction, the statement has been reframed as a future direction ("may, in future work, extend to") rather than a present contribution.

**W8 — The MALA comparison is trivial at α=0.01.** ✅ **ADDRESSED.**
A 99.2% MALA acceptance rate at α=0.01 simply says the step size is very small, making ULA and MALA equivalent by construction. The interesting comparison would be: at what step size does ULA bias become significant, and how does SA quality degrade there?

> **Author response:** A step-size sweep (α ∈ {0.001, …, 0.5}) reveals three regimes. (1) α ≤ 0.02: acceptance >97%, ULA ≈ MALA (ΔE < 0.003). (2) α ∈ [0.05, 0.1]: acceptance drops to 75–91%, ULA bias becomes detectable but small (ΔNovelty ≈ 0.007). (3) α ≥ 0.2: MALA acceptance collapses to 0% — the chain freezes — while ULA continues to produce samples with gracefully degrading quality. The practical divergence threshold is α ≈ 0.1 (75% acceptance, energy gap 0.011). At the paper's operating point (α=0.01), ULA bias is negligible (ΔE=0.0018). At large step sizes, ULA is preferable to MALA because MALA freezes entirely. Figure and table added to the appendix.

**W9 — Multi-chain initialization confounds the generation claim.** ✅ **ADDRESSED.**
Thirty chains initialized near 30 different patterns, with 5 samples thinned per chain, produces diversity by construction from the initialization, not from within-chain mixing. The paper reports single-chain diversity of 0.796 in an appendix as a validation, but this uses a very long chain (50,000 steps) whose mixing time at β=200 is not characterized.

> **Author response:** The main text now explicitly decomposes diversity into initialization and mixing contributions. At β=2000, the 30-chain protocol is acknowledged as a *structured retrieval* strategy: single-chain diversity is only 0.282, and the multi-chain value of 0.600 comes primarily from initialization. The *generation* claim is now explicitly tied to β=200, where a single chain from a fixed seed achieves diversity 0.796 — exceeding the 30-chain β=2000 value — confirming that the diversity at β=200 reflects genuine basin-crossing driven by Langevin dynamics, not initialization. The single-chain analysis (Appendix, Table and Figure) provides the full decomposition across three β values.

---

## Questions for Authors

1. What is the FID or IS score of samples at β=200 compared to stored patterns and to baseline samples? "Novel and diverse" without a perceptual quality measure is insufficient for a generative modeling paper. ⚠️ **PARTIALLY ADDRESSED — InceptionV3-based FID/IS are impractical at 28×28 grayscale with K=100. A diagonal pixel-space Fréchet distance and max-cos comparison against Gaussian noise controls are provided instead (Appendix, "Gaussian Noise Control at β=200").**

2. The SNR threshold (≈0.025) was calibrated on d=64, K=16. Can you prove this threshold is dimension-independent, or is it coincidence that it "worked" at d=784 and d=4096? ✅ **ADDRESSED — see W5 response. The threshold scales as N^{-1/4}; the 0.025 value is derived, not calibrated.**

3. At β=200 on MNIST, can a human observer distinguish SA-generated digits from Gaussian noise of the same per-pixel variance? Please include this comparison or a FID against the 100 stored patterns. ✅ **ADDRESSED — Visual grids and max-cos histograms provided in Appendix "Gaussian Noise Control at β=200". SA shows faint digit structure; both Gaussian controls show uniform static. Max-cos: SA 0.453 vs matched Gaussian 0.328 vs isotropic 0.057.**

4. What is the mixing time of the chain at β=2000 in terms of wall-clock time or number of steps? The reported 30-chain protocol masks this. ✅ **ADDRESSED — A "Mixing time" paragraph in Appendix (Single-Chain Diversity Analysis) reports: inter-basin mixing at β=2000 exceeds 50,000 steps (≈12s wall-clock); the chain never leaves its seed basin. Intra-basin mixing is fast: energy stabilizes within 2,000-step burn-in, consecutive thinned samples decorrelate in O(100) steps. At β=200 the chain crosses basins within ≈5,000 steps.**

5. The Energy Transformer (Hoover et al., 2024) uses Hopfield energies in a discriminative setting. Have you compared against running MALA on a trained EBM at similar computational cost?

---

## Minor Issues

- The abstract states "Lowering the temperature gives exact retrieval; raising it gives open-ended generation" — temperature = 1/β, so this is correct, but the paper routinely writes "at β=2000 (structured retrieval)" without the word "temperature," creating a sign confusion for readers unfamiliar with the convention.
- The paper targets NeurIPS 2026 (based on the template) but cites literature through ~2024; any missing 2025 generative modeling work should be discussed.
- Table 1 footnote: the dagger for positive energy is listed as a negative ("explore off the attractor manifold") but this is the *expected* behavior at high temperature — it is not an anomaly worth footnoting.

---

## Recommendation

The paper identifies a clean and correct connection but does not clear the NeurIPS bar for impact. The core derivation is a direct application of known ULA theory to a known energy function with a known gradient. The experiments, while clearly reported, compare against weak baselines, use metrics that do not distinguish meaningful generation from high-temperature noise, and relegate most domains to the appendix. The theoretical guarantee fails to apply in any experimental setting. A stronger version of this paper would: (1) prove non-trivial mixing-time bounds at high β, (2) compare against at least one modern generative model (DDPM, flow matching), and (3) provide FID-style evaluation confirming the generated samples are not just structured noise.

**Summary score: 4 (Weak Reject).** The work is correct and clearly presented, but the combination of limited novelty, inapplicable theory, and weak experimental validation does not meet the NeurIPS acceptance threshold in its current form.

---

## Key Things to Fix Before Resubmission

1. ~~Add FID/IS evaluation on MNIST and face experiments~~ ⚠️ Partial — diagonal FD + max-cos Gaussian noise control provided; InceptionV3 FID impractical at this scale
2. ~~Add at minimum one modern generative baseline (DDPM or flow matching on same K patterns)~~ ✅ Done — DDPM added at K=100 (max-cos 0.062) + scaling study K∈{100,500,1000,2000,3500}: DDPM never exceeds max-cos 0.09; SA achieves 0.13–0.18 (generation) and 0.24–0.33 (retrieval) with β set via entropy inflection
3. ~~Prove or substantially bound mixing time at β ≫ 1, or remove the convergence corollary as a contribution~~ ✅ Done — corollary demoted to "convex regime characterization"; new paragraph explains why high-β bounds are fundamentally hard (exponential barriers); multi-chain protocol + fast intra-basin mixing presented as practical workaround
4. ~~Remove or qualify the RAG/ICL claim in the abstract~~ ✅ Done
5. ~~Show β=200 samples are meaningfully structured vs. Gaussian noise via a human study or perceptual metric~~ ✅ Done — Gaussian noise control with visual grids, max-cos, energy, and diagonal FD
6. ~~Clarify the SNR threshold as empirical observation, not a derived rule~~ ✅ Done — now derived via entropy inflection (Proposition 4)
7. ~~VAE baseline on all digits and domains~~ ✅ Done — VAE now runs on digits 1, 3, 8 and protein sequences
8. ~~Add non-image domain with full baselines~~ ✅ Done — protein sequence experiment (PF00076 RRM family, 68 seqs, 71 positions)
