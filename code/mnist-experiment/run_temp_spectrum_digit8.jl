#!/usr/bin/env julia
# ──────────────────────────────────────────────────────────────────────────────
# Temperature Spectrum on MNIST Digit 8
# Shows how generated samples change across β ∈ {10, 50, 200, 2000}
# At low β: diffuse/noisy. At moderate β: interpolation. At high β: retrieval.
# ──────────────────────────────────────────────────────────────────────────────

@info "Loading environment …"
include(joinpath(@__DIR__, "Include-MNIST.jl"))
using Images, ImageIO
@info "Environment loaded."

# ── decode helper ─────────────────────────────────────────────────────────────
function decode(s::Vector{<:Number}; number_of_rows::Int=28, number_of_columns::Int=28)
    X = reshape(s, number_of_rows, number_of_columns) |> X -> transpose(X) |> Matrix
    X̂ = replace(X, -1 => 0)
    return X̂
end

# ── Parameters ────────────────────────────────────────────────────────────────
const digit = 8
const number_of_examples = 100
const number_of_pixels = 28 * 28
const α_step = 0.01
const n_chains = 30
const T_per_chain = 5000
const T_burnin = 2000
const thin_interval = 100
const samples_per_chain = 5
const σ_init = 0.01

const β_values = [10.0, 50.0, 200.0, 2000.0]

# ── Load MNIST ────────────────────────────────────────────────────────────────
@info "Loading MNIST data …"
digits_image_dictionary = MyMNISTHandwrittenDigitImageDataset(number_of_examples = number_of_examples)

# ── Build memory matrix for digit 8 ──────────────────────────────────────────
ϵ = 1e-12
X  = zeros(Float64, number_of_pixels, number_of_examples)
X̂  = zeros(Float64, number_of_pixels, number_of_examples)
for i in 1:number_of_examples
    image_array = digits_image_dictionary[digit][:, :, i]
    xᵢ = reshape(transpose(image_array) |> Matrix, number_of_pixels) |> vec
    X[:, i] = xᵢ
end
for i in 1:number_of_examples
    xᵢ = X[:, i]
    lᵢ = norm(xᵢ)
    X̂[:, i] = xᵢ ./ (lᵢ + ϵ)
end
K = size(X̂, 2)
@info "Memory matrix: $(size(X̂))"

# ── Helper: build a 4×4 grid ─────────────────────────────────────────────────
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

# ── Run at each temperature ──────────────────────────────────────────────────
figpath = _PATH_TO_FIG
mkpath(figpath)

for β in β_values
    @info "Running β = $β …"
    
    # Multi-chain SA
    sa_samples = Vector{Vector{Float64}}()
    Random.seed!(42)
    pattern_indices = StatsBase.sample(1:K, n_chains, replace = (n_chains > K))
    
    for (c, k) in enumerate(pattern_indices)
        Random.seed!(12345 + c)
        sₒ = X̂[:, k] .+ σ_init .* randn(number_of_pixels)
        (_, Ξ) = sample(X̂, sₒ, T_per_chain; β = β, α = α_step, seed = 12345 + c)
        chain_pool = Vector{Vector{Float64}}()
        for tᵢ in (T_burnin+1):thin_interval:T_per_chain
            push!(chain_pool, Ξ[tᵢ, :])
        end
        n_avail = length(chain_pool)
        idxs = round.(Int, range(1, n_avail, length = min(samples_per_chain, n_avail)))
        for idx in idxs
            push!(sa_samples, chain_pool[idx])
        end
    end
    
    # Compute metrics with per-chain SEs
    function chain_se(samps, metric_fn)
        group_vals = [metric_fn(samps[(i-1)*samples_per_chain+1:i*samples_per_chain]) for i in 1:n_chains]
        return std(group_vals) / sqrt(n_chains)
    end
    novelty      = mean(sample_novelty(ξ, X̂) for ξ in sa_samples)
    diversity    = sample_diversity(sa_samples)
    energy       = sample_quality(sa_samples, X̂, β)
    mean_max_cos = mean(nearest_cosine_similarity(ξ, X̂) for ξ in sa_samples)
    novelty_se      = chain_se(sa_samples, g -> mean(sample_novelty(ξ, X̂) for ξ in g))
    diversity_se    = chain_se(sa_samples, g -> sample_diversity(g))
    energy_se       = chain_se(sa_samples, g -> mean(hopfield_energy(ξ, X̂, β) for ξ in g))
    mean_max_cos_se = chain_se(sa_samples, g -> mean(nearest_cosine_similarity(ξ, X̂) for ξ in g))

    fmt(x, d) = let v = round(x; digits=d); abs(v) == 0.0 ? abs(v) : v end
    fmtpm(v, se, dv, dse) = "\$$(fmt(v,dv)) \\pm $(fmt(se,dse))\$"
    println("β=$β: Novelty=$(fmt(novelty,3))±$(fmt(novelty_se,3)), Diversity=$(fmt(diversity,3))±$(fmt(diversity_se,3)), Energy=$(fmt(energy,1))±$(fmt(energy_se,2)), MeanMaxCos=$(fmt(mean_max_cos,3))±$(fmt(mean_max_cos_se,3))")
    println("  LaTeX: $(fmtpm(novelty,novelty_se,3,3)) & $(fmtpm(diversity,diversity_se,3,3)) & $(fmtpm(energy,energy_se,1,2)) & $(fmtpm(mean_max_cos,mean_max_cos_se,3,3))")
    
    # Save grid
    β_tag = replace(string(Int(β)), "." => "")
    canvas = build_grid(sa_samples)
    fname = "Fig_mnist_tempspectrum_digit8_beta$(β_tag).png"
    save_pub(joinpath(figpath, fname), Gray.(canvas))
    @info "  Saved $fname"
end

# ── Also build a stored-patterns reference grid ──────────────────────────────
stored_samples = [X̂[:, k] for k in 1:16]
canvas = build_grid(stored_samples)
save(joinpath(figpath, "Fig_mnist_tempspectrum_digit8_stored.png"), Gray.(canvas))
@info "Saved stored patterns reference grid"

println("\nDone.")
