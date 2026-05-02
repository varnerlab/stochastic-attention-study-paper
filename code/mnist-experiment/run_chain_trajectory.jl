#!/usr/bin/env julia
# ──────────────────────────────────────────────────────────────────────────────
# Chain trajectory visualization: a SINGLE Langevin chain on MNIST digit 3 at
# β=200 (generation regime) snapshotted at six timesteps. Output is a 1-row
# strip the appendix can include alongside the Olivetti trajectory.
#
# Output: paper-neurips/figs/Fig_mnist_trajectory.png
# ──────────────────────────────────────────────────────────────────────────────

@info "Loading environment …"
include(joinpath(@__DIR__, "Include-MNIST.jl"))
@info "Environment loaded."

using FileIO, Images, LinearAlgebra, Random, NNlib, Printf

function decode(s::Vector{<:Number}; H::Int=28, W::Int=28)
    X = reshape(s, H, W) |> X -> transpose(X) |> Matrix
    return replace(X, -1 => 0)
end

function strip_grid(samples; gap=2, H=28, W=28)
    n = length(samples)
    canvas = zeros(Float64, H, n * W + (n - 1) * gap)
    for (k, s) in enumerate(samples)
        x0 = (k - 1) * (W + gap) + 1
        img = decode(s)
        lo, hi = minimum(img), maximum(img)
        hi > lo && (img = (img .- lo) ./ (hi - lo))
        canvas[1:H, x0:x0 + W - 1] .= img
    end
    return canvas
end

const DIGIT  = 3
const K      = 100
const D      = 784
const β      = 50.0       # well below retrieval — chain reliably crosses basins
const α      = 0.01
const T_max  = 50000
const σ_init = 0.01
const SEED   = 12345

# Snapshots to display (in chain steps); chosen to match the timescale of
# Appendix F's single-chain analysis (basin-crossing on ~5000-step timescale).
const SNAPSHOTS = [0, 1000, 5000, 10000, 25000, 50000]

@info "Loading MNIST digit $DIGIT memories (K=$K)…"
digits = MyMNISTHandwrittenDigitImageDataset(number_of_examples = K)
ϵ = 1e-12
X = zeros(Float64, D, K)
for i in 1:K
    img = digits[DIGIT][:, :, i]
    x = reshape(transpose(img) |> Matrix, D) |> vec
    X[:, i] = x ./ (norm(x) + ϵ)
end

# Pick a starting memory; warm-start with a small kick.
Random.seed!(SEED)
start_idx = StatsBase.sample(1:K, 1)[1]
ξ0 = X[:, start_idx] .+ σ_init .* randn(D)

@info "Running one chain at β=$β for $T_max steps…"
res = sample(X, copy(ξ0), T_max; β = β, α = α, seed = SEED)
Ξ = res.Ξ  # (T_max+1) × D, row t+1 holds ξ_t

# At β=200 (generation) the raw chain state lives off the digit manifold; apply
# a sharp denoise readout (one noise-free attention update at β_read=2000) at
# each checkpoint so the panels visualize WHICH stored basin the chain is in.
const β_read = 2000.0
function denoise_step(s::Vector{Float64}, X::Matrix{Float64}, β::Float64)
    logits = β .* (X' * s)
    w = NNlib.softmax(logits)
    return X * w
end

snapshots = [denoise_step(collect(Ξ[t + 1, :]), X, β_read) for t in SNAPSHOTS]

# Identify which stored "3" basin each snapshot occupies (= argmax cosine sim
# to any stored memory). Reported so the caption can name the basin sequence.
function nearest_stored(s::Vector{Float64}, X::Matrix{Float64})
    K = size(X, 2)
    sims = [dot(s, X[:, k]) / (norm(s) * norm(X[:, k]) + 1e-12) for k in 1:K]
    return argmax(sims)
end
@info "Basin-id at each snapshot:"
for (t, s) in zip(SNAPSHOTS, snapshots)
    @info @sprintf("  T=%6d  →  stored 3 #%d", t, nearest_stored(s, X))
end

fig_dir = abspath(joinpath(@__DIR__, "..", "..", "paper-neurips", "figs"))
out_path = joinpath(fig_dir, "Fig_mnist_trajectory.png")
canvas = strip_grid(snapshots)
save(out_path, Gray.(clamp.(canvas, 0.0, 1.0)))
@info "Wrote $out_path  (β=$β, snapshots at T=$SNAPSHOTS)"
