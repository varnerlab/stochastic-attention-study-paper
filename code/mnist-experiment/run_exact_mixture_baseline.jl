#!/usr/bin/env julia
# ──────────────────────────────────────────────────────────────────────────────
# Exact ancestral sampling from the Hopfield Boltzmann target (Reviewer AQY1).
#
# For E(ξ) = ½‖ξ‖² − lse_β(Xᵀξ) + const and p_β ∝ exp(−βE), completing the
# square gives
#
#     p_β(ξ) = Σ_i w_i N(ξ; m_i, β⁻¹I),   w_i ∝ exp(β‖m_i‖²/2).
#
# All memories are ℓ₂-normalised, so w_i = 1/K and an independent equilibrium
# draw is  i ~ Uniform{1..K},  ξ = m_i + β^{-1/2} ε.
#
# This script compares that exact sampler against the SA numbers reported in
# Table 1, and adds the memory-based baselines AQY1 asked for (Parzen/KDE
# bandwidth sweep, kNN perturbation).
#
# Protocol matches run_sa_beta200_digit3.jl / run_multidigit_experiment.jl
# exactly: same K=100 digit-3 memory, same S=150, same metrics, same SE.
# ──────────────────────────────────────────────────────────────────────────────

@info "Loading environment …"
include(joinpath(@__DIR__, "Include-MNIST.jl"))
using Printf
@info "Environment loaded."

const number_of_examples = 100
const number_of_rows     = 28
const number_of_cols     = 28
const number_of_pixels   = number_of_rows * number_of_cols
const α_step             = 0.01
const S                  = 150
const n_chains           = 30
const samples_per_chain  = 5
const DIGIT              = 3

# ── Memory matrix: identical construction to the SA scripts ──────────────────
@info "Loading MNIST …"
digits_dict = MyMNISTHandwrittenDigitImageDataset(number_of_examples = number_of_examples)
ϵ  = 1e-12
X  = zeros(Float64, number_of_pixels, number_of_examples)
X̂  = zeros(Float64, number_of_pixels, number_of_examples)
for i in 1:number_of_examples
    X[:, i] = reshape(transpose(digits_dict[DIGIT][:, :, i]) |> Matrix, number_of_pixels) |> vec
end
for i in 1:number_of_examples
    X̂[:, i] = X[:, i] ./ (norm(X[:, i]) + ϵ)
end
K = size(X̂, 2)
d = size(X̂, 1)
@info "Memory matrix: $(size(X̂))  (K=$K, d=$d)"

# ── SE over the same 30-group structure the paper uses ───────────────────────
function chain_metric_se(samps, metric_fn; nc=n_chains, spc=samples_per_chain)
    vals = [metric_fn(samps[(i-1)*spc+1:i*spc]) for i in 1:nc]
    return std(vals) / sqrt(nc)
end

# ── The exact ancestral sampler ──────────────────────────────────────────────
"""
Draw S independent samples from p_β = Σ_i (1/K) N(m_i, β⁻¹I).
If `restrict` is supplied, components are drawn from that index set only
(this mirrors SA's 30-chain warm-start protocol).
"""
function exact_ancestral(X̂::Matrix{Float64}, β::Float64, S::Int;
                         seed::Int=2026, restrict=nothing, per_component=nothing)
    Random.seed!(seed)
    d, K = size(X̂)
    σ = 1.0 / sqrt(β)
    pool = restrict === nothing ? collect(1:K) : collect(restrict)
    idx = if per_component === nothing
        [rand(pool) for _ in 1:S]
    else
        vcat([fill(k, per_component) for k in pool]...)
    end
    return [X̂[:, i] .+ σ .* randn(d) for i in idx]
end

# ── Parzen / KDE: identical family, bandwidth h ↔ β = h⁻² ────────────────────
parzen(X̂, h, S; seed=2026) = exact_ancestral(X̂, 1/h^2, S; seed=seed)

# ── kNN perturbation baseline ────────────────────────────────────────────────
function knn_perturbation(X̂::Matrix{Float64}, S::Int; k::Int=5, σ::Float64=0.0, seed::Int=2026)
    Random.seed!(seed)
    d, K = size(X̂)
    C = X̂' * X̂                       # columns are unit norm ⇒ cosine similarity
    out = Vector{Vector{Float64}}()
    for _ in 1:S
        i = rand(1:K)
        nbrs = sortperm(C[:, i], rev=true)[2:k+1]
        j = rand(nbrs)
        t = rand()
        v = (1 - t) .* X̂[:, i] .+ t .* X̂[:, j]
        σ > 0 && (v .+= σ .* randn(d))
        push!(out, v)
    end
    return out
end

# ── Reporting ────────────────────────────────────────────────────────────────
fmt(x, dg) = let v = round(x; digits=dg); abs(v) == 0.0 ? abs(v) : v end

function report(name, samps, β_eval)
    nov  = mean(sample_novelty(ξ, X̂) for ξ in samps)
    div  = sample_diversity(samps)
    en   = mean(hopfield_energy(ξ, X̂, β_eval) for ξ in samps)
    nse  = chain_metric_se(samps, g -> mean(sample_novelty(ξ, X̂) for ξ in g))
    dse  = chain_metric_se(samps, g -> sample_diversity(g))
    ese  = chain_metric_se(samps, g -> mean(hopfield_energy(ξ, X̂, β_eval) for ξ in g))
    @printf("%-46s | %.3f ± %.3f | %.3f ± %.3f | %+.3f\n", name, nov, nse, div, dse, en)
    return (; nov, div, en, nse, dse, ese)
end

println("\n" * "="^110)
println("EXACT ANCESTRAL SAMPLING FROM THE HOPFIELD BOLTZMANN TARGET  (MNIST digit 3, K=100, d=784)")
println("="^110)
println("High-dimensional approximation (norm concentration; generating component assumed nearest):")
println("  N ≈ 1 − 1/√(1 + d/β)   -- not an exact closed form; reported N is max-cos over all K")
for β in (2000.0, 200.0)
    @printf("   β=%-6.0f  σ=β^(−1/2)=%.5f   approx N = %.3f\n", β, 1/sqrt(β), 1 - 1/sqrt(1 + d/β))
end

println("\n" * "-"^110)
@printf("%-46s | %-13s | %-13s | %s\n", "Method", "Novelty N", "Diversity D̄", "Energy Ē")
println("-"^110)

results = Dict{String,Any}()

# SA's own warm-start indices, reproduced exactly from the SA scripts
Random.seed!(42)
sa_pattern_indices = StatsBase.sample(1:K, n_chains, replace = (n_chains > K))

for β in (2000.0, 200.0)
    βi = Int(β)
    # (a) exact ancestral over all K components
    s_full = exact_ancestral(X̂, β, S; seed=2026)
    results["exact_full_$βi"] = report("Exact ancestral (all K), β=$βi", s_full, β)

    # (b) exact ancestral restricted to SA's 30 warm-start components
    s_match = exact_ancestral(X̂, β, S; seed=2026,
                              restrict=sa_pattern_indices, per_component=samples_per_chain)
    results["exact_match_$βi"] = report("Exact ancestral (SA's 30 components), β=$βi", s_match, β)
end

println("-"^110)
println("Published SA rows from Table 1, for comparison:")
println("  SA (β=2000, retrieval)     N = 0.152 ± 0.001   D̄ = 0.600 ± 0.001   Ē = −0.303")
println("  SA (β=200,  generation)    N = 0.548 ± 0.002   D̄ = 0.885 ± 0.002   Ē = +1.467")
println("  Gaussian perturbation      N = 0.004 ± 0.000   D̄ = 0.450 ± 0.013   Ē = −0.496")
println("  Bootstrap (replay)         N = 0.000 ± 0.000   D̄ = 0.459 ± 0.011   Ē = −0.500")

# ── Parzen / KDE bandwidth sweep (AQY1 Q2) ───────────────────────────────────
println("\n" * "="^110)
println("PARZEN / KDE BANDWIDTH SWEEP  (Parzen with bandwidth h ≡ exact target at β = h⁻²)")
println("="^110)
@printf("%-46s | %-13s | %-13s | %s\n", "Bandwidth", "Novelty N", "Diversity D̄", "Energy Ē@β=200")
println("-"^110)
for h in (0.00316, 0.01, 0.0224, 0.05, 0.0707, 0.10, 0.15)
    s = parzen(X̂, h, S; seed=2026)
    report(@sprintf("Parzen h=%.5f  (β=h⁻²=%.0f)", h, 1/h^2), s, 200.0)
end
println("-"^110)
println("Note: h=0.00316 is the Table 1 'Gaussian perturbation' scale √(2α/β) at β=2000;")
println("      h=0.0224 = β^(−1/2) at β=2000;  h=0.0707 = β^(−1/2) at β=200.")
println("      The Table 1 Gaussian-perturbation baseline used √(2α) = $(round(sqrt(2*α_step),digits=4))× the exact")
println("      component width, i.e. it was $(round(1/sqrt(2*α_step),digits=2))× too narrow.")

# ── kNN perturbation (AQY1 Q2) ───────────────────────────────────────────────
println("\n" * "="^110)
println("kNN PERTURBATION BASELINE")
println("="^110)
@printf("%-46s | %-13s | %-13s | %s\n", "Method", "Novelty N", "Diversity D̄", "Energy Ē@β=200")
println("-"^110)
for k in (3, 5, 10)
    for σ in (0.0, 0.0707)
        s = knn_perturbation(X̂, S; k=k, σ=σ, seed=2026)
        report(@sprintf("kNN interpolation k=%d, σ=%.4f", k, σ), s, 200.0)
    end
end

println("\nDone.")
