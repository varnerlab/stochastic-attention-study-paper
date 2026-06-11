#!/usr/bin/env julia
# ──────────────────────────────────────────────────────────────────────────────
# Phase 1a: MALA vs SA at β=200 (generation regime)
# Addresses Review v2 W7 / Q1: "What happens if you run MALA at β=200?"
# Also computes energy-filtered metrics (Phase 2: W4)
# ──────────────────────────────────────────────────────────────────────────────

@info "Loading environment …"
include(joinpath(@__DIR__, "Include-MNIST.jl"))
using Images, ImageIO, Printf
@info "Environment loaded."

# ── decode / grid helpers ─────────────────────────────────────────────────────
function decode(s::Vector{<:Number}; number_of_rows::Int=28, number_of_columns::Int=28)
    X = reshape(s, number_of_rows, number_of_columns) |> X -> transpose(X) |> Matrix
    return replace(X, -1 => 0)
end

function build_grid(samples; nrows=4, ncols=4, gap=2)
    H, W = 28, 28
    canvas_h = nrows * H + (nrows - 1) * gap
    canvas_w = ncols * W + (ncols - 1) * gap
    canvas = zeros(Float64, canvas_h, canvas_w)
    indices = round.(Int, range(1, length(samples), length=nrows*ncols))
    for idx in 1:(nrows*ncols)
        r = div(idx - 1, ncols)
        c = rem(idx - 1, ncols)
        y0 = r * (H + gap) + 1
        x0 = c * (W + gap) + 1
        img = decode(samples[indices[idx]])
        lo, hi = minimum(img), maximum(img)
        if hi > lo
            img = (img .- lo) ./ (hi - lo)
        end
        canvas[y0:y0+H-1, x0:x0+W-1] .= img
    end
    return canvas
end

# ── Parameters ────────────────────────────────────────────────────────────────
const number_of_examples = 100
const number_of_pixels = 784
const α_step = 0.01
const n_chains = 30
const T_per_chain = 5000
const T_burnin = 2000
const thin_interval = 100
const samples_per_chain = 5
const σ_init = 0.01
const S = 150

# β values: generation regime (200) + transition band (500, 1000)
const β_values = [200.0, 500.0, 1000.0]

# ── Load MNIST digit 3 ───────────────────────────────────────────────────────
@info "Loading MNIST data …"
digits_image_dictionary = MyMNISTHandwrittenDigitImageDataset(number_of_examples = number_of_examples)
@info "MNIST loaded."

# Build memory matrix
ϵ = 1e-12
X  = zeros(Float64, number_of_pixels, number_of_examples)
X̂  = zeros(Float64, number_of_pixels, number_of_examples)
for i in 1:number_of_examples
    image_array = digits_image_dictionary[3][:, :, i]
    xᵢ = reshape(transpose(image_array) |> Matrix, number_of_pixels) |> vec
    X[:, i] = xᵢ
end
for i in 1:number_of_examples
    xᵢ = X[:, i]
    lᵢ = norm(xᵢ)
    X̂[:, i] = xᵢ ./ (lᵢ + ϵ)
end
K = size(X̂, 2)
@info "Memory matrix: $(size(X̂)), K=$K"

# ── Multi-chain sampling function ─────────────────────────────────────────────
function run_multichain_sa(X̂, β; α=α_step, label="SA")
    samples = Vector{Vector{Float64}}()
    Random.seed!(42)
    pattern_indices = StatsBase.sample(1:size(X̂,2), n_chains, replace=false)
    for (c, k) in enumerate(pattern_indices)
        Random.seed!(12345 + c)
        sₒ = X̂[:, k] .+ σ_init .* randn(size(X̂, 1))
        (_, Ξ) = sample(X̂, sₒ, T_per_chain; β=β, α=α, seed=12345+c)
        chain_pool = Vector{Vector{Float64}}()
        for tᵢ in (T_burnin+1):thin_interval:T_per_chain
            push!(chain_pool, Ξ[tᵢ, :])
        end
        n_avail = length(chain_pool)
        idxs = round.(Int, range(1, n_avail, length=min(samples_per_chain, n_avail)))
        for idx in idxs
            push!(samples, chain_pool[idx])
        end
    end
    return samples
end

function run_multichain_mala(X̂, β; α=α_step, label="MALA")
    samples = Vector{Vector{Float64}}()
    accept_rates = Float64[]
    Random.seed!(42)
    pattern_indices = StatsBase.sample(1:size(X̂,2), n_chains, replace=false)
    for (c, k) in enumerate(pattern_indices)
        Random.seed!(12345 + c)
        sₒ = X̂[:, k] .+ σ_init .* randn(size(X̂, 1))
        (_, Ξ, ar) = mala_sample(X̂, sₒ, T_per_chain; β=β, α=α, seed=12345+c)
        push!(accept_rates, ar)
        chain_pool = Vector{Vector{Float64}}()
        for tᵢ in (T_burnin+1):thin_interval:T_per_chain
            push!(chain_pool, Ξ[tᵢ, :])
        end
        n_avail = length(chain_pool)
        idxs = round.(Int, range(1, n_avail, length=min(samples_per_chain, n_avail)))
        for idx in idxs
            push!(samples, chain_pool[idx])
        end
    end
    return samples, mean(accept_rates)
end

# ── Metric computation ────────────────────────────────────────────────────────
function compute_metrics(samps, X̂, β; label="")
    novelty_vals  = [sample_novelty(ξ, X̂) for ξ in samps]
    maxcos_vals   = [nearest_cosine_similarity(ξ, X̂) for ξ in samps]
    energy_vals   = [hopfield_energy(ξ, X̂, β) for ξ in samps]
    diversity_val = sample_diversity(samps)

    # Energy-filtered: only samples with E < 0 (on the manifold)
    on_manifold_idx = findall(e -> e < 0.0, energy_vals)
    n_on_manifold = length(on_manifold_idx)
    frac_on_manifold = n_on_manifold / length(samps)

    if n_on_manifold >= 2
        on_samps = samps[on_manifold_idx]
        nov_filtered = mean(sample_novelty(ξ, X̂) for ξ in on_samps)
        div_filtered = sample_diversity(on_samps)
        eng_filtered = mean(energy_vals[on_manifold_idx])
        maxcos_filtered = mean(nearest_cosine_similarity(ξ, X̂) for ξ in on_samps)
    else
        nov_filtered = NaN
        div_filtered = NaN
        eng_filtered = NaN
        maxcos_filtered = NaN
    end

    return (
        label = label,
        n_samples = length(samps),
        novelty = mean(novelty_vals),
        max_cos = mean(maxcos_vals),
        diversity = diversity_val,
        energy = mean(energy_vals),
        n_on_manifold = n_on_manifold,
        frac_on_manifold = frac_on_manifold,
        novelty_filtered = nov_filtered,
        max_cos_filtered = maxcos_filtered,
        diversity_filtered = div_filtered,
        energy_filtered = eng_filtered,
    )
end

# ── Run experiments ───────────────────────────────────────────────────────────
figpath = _PATH_TO_FIG
mkpath(figpath)

println("\n" * "="^80)
println("MALA vs SA COMPARISON ACROSS β VALUES (digit 3, K=100)")
println("="^80)

for β in β_values
    println("\n" * "─"^60)
    println("β = $β")
    println("─"^60)

    @info "Running SA at β=$β …"
    sa_samps = run_multichain_sa(X̂, β)

    @info "Running MALA at β=$β …"
    mala_samps, mala_ar = run_multichain_mala(X̂, β)

    sa_metrics = compute_metrics(sa_samps, X̂, β; label="SA (ULA)")
    mala_metrics = compute_metrics(mala_samps, X̂, β; label="MALA")

    println("\nMALA acceptance rate: $(round(mala_ar, digits=4))")

    println("\n--- All samples ---")
    println("Method       | Novelty | Max-Cos | Diversity | Energy  | N_samples")
    for m in [sa_metrics, mala_metrics]
        @printf("%-12s | %7.4f | %7.4f | %9.4f | %+7.3f | %d\n",
            m.label, m.novelty, m.max_cos, m.diversity, m.energy, m.n_samples)
    end

    println("\n--- Energy-filtered (E < 0, on manifold) ---")
    println("Method       | Novelty | Max-Cos | Diversity | Energy  | N_on_manifold | Frac")
    for m in [sa_metrics, mala_metrics]
        if !isnan(m.novelty_filtered)
            @printf("%-12s | %7.4f | %7.4f | %9.4f | %+7.3f | %d/%d          | %.3f\n",
                m.label, m.novelty_filtered, m.max_cos_filtered,
                m.diversity_filtered, m.energy_filtered,
                m.n_on_manifold, m.n_samples, m.frac_on_manifold)
        else
            @printf("%-12s | (too few on-manifold samples: %d/%d)\n",
                m.label, m.n_on_manifold, m.n_samples)
        end
    end

    # Compute deltas
    println("\n--- SA vs MALA deltas ---")
    @printf("ΔNovelty:   %+.5f\n", sa_metrics.novelty - mala_metrics.novelty)
    @printf("ΔMax-Cos:   %+.5f\n", sa_metrics.max_cos - mala_metrics.max_cos)
    @printf("ΔDiversity: %+.5f\n", sa_metrics.diversity - mala_metrics.diversity)
    @printf("ΔEnergy:    %+.5f\n", sa_metrics.energy - mala_metrics.energy)

    # Save grids for β=200 (the key comparison)
    if β == 200.0
        for (tag, samps) in [("sa_beta200", sa_samps), ("mala_beta200", mala_samps)]
            canvas = build_grid(samps)
            img = Gray.(canvas)
            fname = "Fig_mnist_grid_$(tag)_digit3.png"
            save_pub(joinpath(figpath, fname), img)
            @info "Saved $fname"
        end
    end
end

println("\n" * "="^80)
println("EXPERIMENT COMPLETE")
println("="^80)
println("\nKey question answered: Does MALA at β=200 differ materially from ULA (SA)?")
println("If deltas are small (< 0.01), ULA bias is negligible in the generation regime.")
