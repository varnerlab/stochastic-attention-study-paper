#!/usr/bin/env julia
# ──────────────────────────────────────────────────────────────────────────────
# Paired before/after grid for the MNIST digit-3 figure in the paper.
#
# Goal: instead of N independent panels of "a bunch of 3s", produce three
# row-aligned panels — Warm-start | SA output | MALA output — so the reader
# can scan one row and see exactly what each method does to the same starting
# memory. Each row is one chain. Same warm-start drives the SA chain and the
# MALA chain so the comparison is direct.
#
# Outputs (paper-neurips/figs/):
#   Fig_mnist_pairs_warmstart.png   — 4x4 grid of warm-starts (the input)
#   Fig_mnist_pairs_sa.png          — 4x4 grid of SA chain endpoints
#   Fig_mnist_pairs_mala.png        — 4x4 grid of MALA chain endpoints
# Each grid is row-aligned: row r in all three panels comes from the same
# warm-start memory and the same chain seed.
# ──────────────────────────────────────────────────────────────────────────────

@info "Loading environment …"
include(joinpath(@__DIR__, "Include-MNIST.jl"))
@info "Environment loaded."

using FileIO, Images, LinearAlgebra, Random, NNlib

# ── helpers ─────────────────────────────────────────────────────────────────
function decode(s::Vector{<:Number}; H::Int=28, W::Int=28)
    X = reshape(s, H, W) |> X -> transpose(X) |> Matrix
    return replace(X, -1 => 0)
end

function build_grid(samples; nrows=4, ncols=4, gap=2)
    H, W = 28, 28
    canvas_h = nrows * H + (nrows - 1) * gap
    canvas_w = ncols * W + (ncols - 1) * gap
    canvas = zeros(Float64, canvas_h, canvas_w)
    @assert length(samples) == nrows * ncols
    for idx in 1:(nrows * ncols)
        r = div(idx - 1, ncols)
        c = rem(idx - 1, ncols)
        y0 = r * (H + gap) + 1
        x0 = c * (W + gap) + 1
        img = decode(samples[idx])
        lo, hi = minimum(img), maximum(img)
        hi > lo && (img = (img .- lo) ./ (hi - lo))
        canvas[y0:y0 + H - 1, x0:x0 + W - 1] .= img
    end
    return canvas
end

save_png(path, canvas) = save(path, Gray.(clamp.(canvas, 0.0, 1.0)))

# ── parameters (matched to the paper's MNIST experiment) ────────────────────
const DIGIT     = 3
const K         = 100
const D         = 784
const β         = 2000.0       # retrieval regime, matches Table 1
const α         = 0.01
const T         = 5000
const σ_init    = 0.01
const NCHAINS   = 16           # 4×4 grid of paired chains

# ── load memories ───────────────────────────────────────────────────────────
@info "Loading MNIST digit $DIGIT memories (K=$K)…"
digits = MyMNISTHandwrittenDigitImageDataset(number_of_examples = K)
ϵ = 1e-12
X = zeros(Float64, D, K)
for i in 1:K
    img = digits[DIGIT][:, :, i]
    x = reshape(transpose(img) |> Matrix, D) |> vec
    X[:, i] = x ./ (norm(x) + ϵ)
end

# ── pick 16 distinct warm-start indices (visually varied 3s) ────────────────
Random.seed!(7)
warm_idxs = StatsBase.sample(1:K, NCHAINS; replace = false)

# ── run paired chains ───────────────────────────────────────────────────────
warm_samples = Vector{Vector{Float64}}(undef, NCHAINS)
sa_samples   = Vector{Vector{Float64}}(undef, NCHAINS)
mala_samples = Vector{Vector{Float64}}(undef, NCHAINS)

@info "Running $NCHAINS paired chains (SA + MALA) at β=$β …"
for i in 1:NCHAINS
    k = warm_idxs[i]
    warm_samples[i] = X[:, k]

    Random.seed!(20000 + i)
    ξ0 = X[:, k] .+ σ_init .* randn(D)

    # SA (ULA)
    sa_res = sample(X, copy(ξ0), T; β = β, α = α, seed = 30000 + i)
    sa_samples[i] = collect(sa_res.Ξ[end, :])

    # MALA (same Langevin proposal + Metropolis correction)
    mala_res = mala_sample(X, copy(ξ0), T; β = β, α = α, seed = 30000 + i)
    mala_samples[i] = collect(mala_res.Ξ[end, :])
end
@info "Chains complete."

# ── render and save ─────────────────────────────────────────────────────────
fig_dir = abspath(joinpath(@__DIR__, "..", "..", "paper-neurips", "figs"))
@info "Saving paired grids to $fig_dir …"
save_png(joinpath(fig_dir, "Fig_mnist_pairs_warmstart.png"),
         build_grid(warm_samples; nrows = 4, ncols = 4))
save_png(joinpath(fig_dir, "Fig_mnist_pairs_sa.png"),
         build_grid(sa_samples;   nrows = 4, ncols = 4))
save_png(joinpath(fig_dir, "Fig_mnist_pairs_mala.png"),
         build_grid(mala_samples; nrows = 4, ncols = 4))
@info "Done."
