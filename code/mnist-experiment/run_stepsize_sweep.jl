#!/usr/bin/env julia
# ──────────────────────────────────────────────────────────────────────────────
# Step-Size Sweep: ULA vs MALA as a function of α  (addresses reviewer W8)
#
# At α=0.01, MALA acceptance is 99.2% — ULA and MALA are trivially equivalent.
# This experiment sweeps α to find where ULA bias becomes significant:
#   - At what α does the acceptance rate drop meaningfully?
#   - At what α do ULA and MALA sample quality diverge?
#   - What is the practical α range where ULA is a valid approximation?
#
# Uses MNIST digit 3, K=100, β=2000 (same as main experiment).
# ──────────────────────────────────────────────────────────────────────────────

@info "Loading environment …"
include(joinpath(@__DIR__, "Include-MNIST.jl"))
@info "Environment loaded."

# ── Experiment parameters ────────────────────────────────────────────────────
const DIGIT = 3
const K = 100
const D = 784
const β_inv_temp = 2000.0
const n_chains = 30
const T_per_chain = 5000
const T_burnin = 2000
const thin_interval = 100
const samples_per_chain = 5
const σ_init = 0.01

# Step sizes to sweep: from very small (trivially equivalent) to large (significant bias)
const α_values = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]

# ── Load MNIST digit 3 ──────────────────────────────────────────────────────
@info "Loading MNIST digit $DIGIT …"
digits_image_dictionary = MyMNISTHandwrittenDigitImageDataset(number_of_examples = K)

ϵ = 1e-12
X̂ = zeros(Float64, D, K)
for i in 1:K
    image_array = digits_image_dictionary[DIGIT][:, :, i]
    xᵢ = reshape(transpose(image_array) |> Matrix, D) |> vec
    lᵢ = norm(xᵢ)
    X̂[:, i] = xᵢ ./ (lᵢ + ϵ)
end
@info "Memory matrix: $(size(X̂))"

# ── Helper: run multi-chain sampling ─────────────────────────────────────────
function run_multichain(X̂, α; method=:sa)
    d, K = size(X̂)
    all_samples = Vector{Vector{Float64}}()
    accept_rates = Float64[]

    Random.seed!(42)
    pattern_indices = StatsBase.sample(1:K, n_chains, replace=(n_chains > K))

    for (c, k) in enumerate(pattern_indices)
        Random.seed!(12345 + c)
        sₒ = X̂[:, k] .+ σ_init .* randn(d)

        if method == :sa
            (_, Ξ) = sample(X̂, sₒ, T_per_chain; β=β_inv_temp, α=α, seed=12345+c)
            push!(accept_rates, 1.0)  # ULA always "accepts"
        else
            (_, Ξ, ar) = mala_sample(X̂, sₒ, T_per_chain; β=β_inv_temp, α=α, seed=12345+c)
            push!(accept_rates, ar)
        end

        # collect thinned post-burn-in samples
        chain_pool = Vector{Vector{Float64}}()
        for tᵢ in (T_burnin+1):thin_interval:T_per_chain
            push!(chain_pool, Ξ[tᵢ, :])
        end
        n_avail = length(chain_pool)
        idxs = round.(Int, range(1, n_avail, length=min(samples_per_chain, n_avail)))
        for idx in idxs
            push!(all_samples, copy(chain_pool[idx]))
        end
    end

    return all_samples, mean(accept_rates)
end

# ── Helper: compute metrics ──────────────────────────────────────────────────
function compute_metrics(samples, X̂, β)
    novelty_vals = [sample_novelty(ξ, X̂) for ξ in samples]
    energy_vals  = [hopfield_energy(ξ, X̂, β) for ξ in samples]
    return (
        novelty   = mean(novelty_vals),
        diversity = sample_diversity(samples),
        energy    = mean(energy_vals),
    )
end

# ── Run the sweep ────────────────────────────────────────────────────────────
results = []

for α in α_values
    @info "── α = $α ──"

    # ULA (SA)
    @info "  Running SA …"
    sa_samples, _ = run_multichain(X̂, α; method=:sa)
    sa_metrics = compute_metrics(sa_samples, X̂, β_inv_temp)

    # MALA
    @info "  Running MALA …"
    mala_samples, mala_ar = run_multichain(X̂, α; method=:mala)

    # If acceptance is essentially zero, MALA chain is frozen; metrics are meaningless
    if mala_ar < 1e-6
        mala_metrics = (novelty = NaN, diversity = NaN, energy = NaN)
    else
        mala_metrics = compute_metrics(mala_samples, X̂, β_inv_temp)
    end

    push!(results, (
        α = α,
        mala_accept = mala_ar,
        sa_novelty = sa_metrics.novelty,
        sa_diversity = sa_metrics.diversity,
        sa_energy = sa_metrics.energy,
        mala_novelty = mala_metrics.novelty,
        mala_diversity = mala_metrics.diversity,
        mala_energy = mala_metrics.energy,
    ))

    @info "  MALA accept = $(round(mala_ar, digits=4))"
    @info "  SA:   N=$(round(sa_metrics.novelty, digits=4))  D=$(round(sa_metrics.diversity, digits=4))  E=$(round(sa_metrics.energy, digits=4))"
    if !isnan(mala_metrics.novelty)
        @info "  MALA: N=$(round(mala_metrics.novelty, digits=4))  D=$(round(mala_metrics.diversity, digits=4))  E=$(round(mala_metrics.energy, digits=4))"
    else
        @info "  MALA: chain frozen (acceptance ≈ 0)"
    end
end

# ── Print results table ──────────────────────────────────────────────────────
println("\n" * "═"^80)
println("STEP-SIZE SWEEP RESULTS — ULA vs MALA (digit $DIGIT, β=$β_inv_temp)")
println("═"^80)

df = DataFrame(results)
println(df)

# ── LaTeX table ──────────────────────────────────────────────────────────────
println("\n% --- LaTeX table (copy into paper) ---")
println("\\begin{tabular}{@{}rcccccc@{}}")
println("\\toprule")
println("\$\\alpha\$ & Accept & \\multicolumn{2}{c}{Novelty} & \\multicolumn{2}{c}{Diversity} & \$\\Delta E\$ \\\\")
println(" & rate & ULA & MALA & ULA & MALA & (ULA\$-\$MALA) \\\\")
println("\\midrule")
for r in results
    ar_str = r.mala_accept > 0.99 ? "\$>\$0.99" : string(round(r.mala_accept, digits=3))
    if isnan(r.mala_novelty)
        println("$(r.α) & $(ar_str) & $(round(r.sa_novelty, digits=3)) & 0.000 & " *
                "$(round(r.sa_diversity, digits=3)) & 0.000 & --- \\\\")
    else
        ΔE = abs(r.sa_energy - r.mala_energy)
        ΔE_str = ΔE < 0.001 ? "\$<\$0.001" : string(round(ΔE, digits=3))
        println("$(r.α) & $(ar_str) & $(round(r.sa_novelty, digits=3)) & $(round(r.mala_novelty, digits=3)) & " *
                "$(round(r.sa_diversity, digits=3)) & $(round(r.mala_diversity, digits=3)) & $(ΔE_str) \\\\")
    end
end
println("\\bottomrule")
println("\\end{tabular}")

# ── Generate figure ──────────────────────────────────────────────────────────
@info "Generating figures …"
figpath = _PATH_TO_FIG
mkpath(figpath)

αs = [r.α for r in results]
accept_rates = [r.mala_accept for r in results]

# Filter to non-frozen entries for panel (b)
valid = [!isnan(r.mala_novelty) for r in results]
αs_valid = αs[valid]
Δnovelty   = [abs(r.sa_novelty - r.mala_novelty) for r in results[valid]]
Δdiversity = [abs(r.sa_diversity - r.mala_diversity) for r in results[valid]]
Δenergy    = [abs(r.sa_energy - r.mala_energy) for r in results[valid]]

# Panel (a): MALA acceptance rate
p1 = plot(αs, accept_rates .* 100,
    xlabel="Step size α", ylabel="MALA acceptance rate (%)",
    xscale=:log10, label=nothing, lw=2, marker=:circle, color=:steelblue,
    title="(a) Acceptance rate", ylims=(-5, 105),
    size=(400, 300))
hline!([95], ls=:dash, color=:gray, label="95%")

# Panel (b): |ULA − MALA| metric divergence (only where MALA is not frozen)
p2 = plot(αs_valid, Δnovelty, label="Novelty", lw=2, marker=:circle,
    xlabel="Step size α", ylabel="|ULA − MALA|",
    xscale=:log10, title="(b) ULA–MALA divergence",
    legend=:topleft, size=(400, 300))
plot!(αs_valid, Δdiversity, label="Diversity", lw=2, marker=:square)
plot!(αs_valid, Δenergy, label="Energy", lw=2, marker=:diamond)

p_combined = plot(p1, p2, layout=(1, 2), size=(850, 350), margin=5Plots.mm)
savefig(p_combined, joinpath(figpath, "Fig_stepsize_sweep_ula_vs_mala.pdf"))
@info "Saved step-size sweep figure"

# Also save PNG for quick inspection
savefig(p_combined, joinpath(figpath, "Fig_stepsize_sweep_ula_vs_mala.png"))
@info "Saved PNG version"

println("\nDone.")
