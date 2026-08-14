#!/usr/bin/env julia

# Validate ULA and MALA against the exact modern-Hopfield Gaussian mixture on a
# tractable two-dimensional target. The memory norms are deliberately unequal,
# so a uniform component sampler would be wrong.

include(joinpath(@__DIR__, "Include.jl"))
using Printf

const X_VALIDATION = [-1.00  0.20  1.35;
                       0.00  1.10 -0.25]
const BETA_VALIDATION = 1.5
const EXACT_SAMPLES = 100_000
const N_CHAINS = 16
const T_STEPS = 100_000
const BURN_IN = 20_000
const THIN = 20

function analytic_moments(X, β)
    weights = hopfield_component_weights(X; β=β)
    mean_target = X * weights
    covariance_target = Matrix{Float64}(I, size(X, 1), size(X, 1)) / β
    for k in axes(X, 2)
        centered = X[:, k] .- mean_target
        covariance_target .+= weights[k] .* (centered * centered')
    end
    return (; weights, mean=mean_target, covariance=covariance_target)
end

function summarize_samples(samples, target)
    sample_mean = vec(mean(samples; dims=2))
    sample_covariance = cov(samples; dims=2)
    return (
        mean_error=norm(sample_mean - target.mean),
        covariance_error=norm(sample_covariance - target.covariance),
    )
end

function pooled_chain_samples(sampler, X, β, α)
    pooled = Vector{Vector{Float64}}()
    acceptance = Float64[]
    for chain in 1:N_CHAINS
        initial = exact_hopfield_sample(X, 1; β=β, seed=1_000 + chain).samples[:, 1]
        result = sampler(X, Vector(initial), T_STEPS;
                         β=Float64(β), α=Float64(α), seed=2_000 + chain)
        for t in (BURN_IN + 1):THIN:(T_STEPS + 1)
            push!(pooled, Vector(result.Ξ[t, :]))
        end
        hasproperty(result, :accept_rate) && push!(acceptance, result.accept_rate)
    end
    samples = reduce(hcat, pooled)
    return (; samples, acceptance=isempty(acceptance) ? NaN : mean(acceptance))
end

function main()
    target = analytic_moments(X_VALIDATION, BETA_VALIDATION)
    exact = exact_hopfield_sample(X_VALIDATION, EXACT_SAMPLES;
                                  β=BETA_VALIDATION, seed=42)
    exact_summary = summarize_samples(exact.samples, target)

    println("="^92)
    println("EXACT-TARGET VALIDATION (d=2, K=3, unequal memory norms, β=$BETA_VALIDATION)")
    println("="^92)
    println("Analytic component weights: ", round.(target.weights; digits=4))
    @printf("%-18s | %-12s | %-12s | %s\n",
            "Method", "step α", "mean error", "covariance error / acceptance")
    println("-"^92)
    @printf("%-18s | %-12s | %-12.5g | %.5g\n", "Exact independent", "--",
            exact_summary.mean_error, exact_summary.covariance_error)

    for α in (0.01, 0.05, 0.20)
        ula = pooled_chain_samples(sample, X_VALIDATION, BETA_VALIDATION, α)
        summary = summarize_samples(ula.samples, target)
        @printf("%-18s | %-12.2f | %-12.5g | %.5g\n", "ULA", α,
                summary.mean_error, summary.covariance_error)

        mala = pooled_chain_samples(mala_sample, X_VALIDATION, BETA_VALIDATION, α)
        summary = summarize_samples(mala.samples, target)
        @printf("%-18s | %-12.2f | %-12.5g | %.5g / %.3f\n", "MALA", α,
                summary.mean_error, summary.covariance_error, mala.acceptance)
    end
    println("="^92)
end

main()
