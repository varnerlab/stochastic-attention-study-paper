#!/usr/bin/env julia
# ──────────────────────────────────────────────────────────────────────────────
# Chain trajectory visualization on Olivetti faces: a SINGLE unconditional
# Langevin chain at β_chain=200 over T=3000 steps, snapshotted at six time
# steps with a sharp readout at β_read=10,000 applied at each checkpoint to
# project the chain endpoint onto the convex hull of the memory bank.
#
# The output illustrates that without a mask the chain visibly drifts across
# subjects — the visual counterpart of the 10.4% subject-recovery number.
#
# Output: paper-neurips/figs/Fig_olivetti_trajectory.png
# ──────────────────────────────────────────────────────────────────────────────

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", "code"))

using LinearAlgebra, Statistics, Random, DelimitedFiles, NNlib, StatsBase, Printf
using Images, FileIO

const _DATA_DIR = joinpath(@__DIR__, "data")
const _PAPER_FIG_DIR = abspath(joinpath(@__DIR__, "..", "..", "paper-neurips", "figs"))

# ── Olivetti loader (matches the masking driver) ────────────────────────────
@info "Loading Olivetti…"
faces_raw   = readdlm(joinpath(_DATA_DIR, "faces.csv"),   ',', Float64)
targets_raw = readdlm(joinpath(_DATA_DIR, "targets.csv"), ',', Int)

const N_SUBJECTS = 10
keep = vec(targets_raw .< N_SUBJECTS)
faces_raw = faces_raw[keep, :]
targets   = vec(targets_raw)[keep]

const d = size(faces_raw, 2)
K = size(faces_raw, 1)
X = Matrix(transpose(faces_raw))
for j in 1:K
    nj = norm(X[:, j])
    nj > 0 && (X[:, j] ./= nj)
end
y = targets
@info "Memory matrix: $(size(X)), subjects: $(length(unique(y)))"

# ── Vendored SA + helpers ───────────────────────────────────────────────────
function masked_sa_sample_with_history(X::Matrix{Float64}, ξ₀::Vector{Float64}, T::Int;
        β::Float64, α::Float64,
        snapshot_steps::Vector{Int},
        seed::Union{Int,Nothing} = nothing)
    seed === nothing || Random.seed!(seed)
    d, K = size(X)
    ξ = copy(ξ₀)
    noise = sqrt(2.0 * α / β)
    snaps = Dict{Int, Vector{Float64}}()
    if 0 in snapshot_steps
        snaps[0] = copy(ξ)
    end
    for t in 1:T
        logits = β .* (X' * ξ)
        w = NNlib.softmax(logits)
        ξ .= (1.0 - α) .* ξ .+ α .* (X * w) .+ noise .* randn(d)
        if t in snapshot_steps
            snaps[t] = copy(ξ)
        end
    end
    return snaps
end

function denoise(s::Vector{Float64}, X::Matrix{Float64}, β::Float64)
    logits = β .* (X' * s)
    w = NNlib.softmax(logits)
    return X * w
end

function class_means(X::Matrix{Float64}, y::Vector{Int})
    classes = sort(unique(y))
    Dict(c => vec(mean(X[:, findall(==(c), y)], dims=2)) for c in classes)
end

function classify_nearest_mean(s::Vector{Float64}, means::Dict{Int, Vector{Float64}})
    best_c, best_d = first(keys(means)), Inf
    for (c, m) in means
        dd = norm(s .- m)
        if dd < best_d
            best_c, best_d = c, dd
        end
    end
    return best_c
end

# ── Run one chain ────────────────────────────────────────────────────────────
const β_chain = 200.0
const β_read  = 10000.0
const α       = 0.01
const T_max   = 3000
const σ_init  = 0.05
const SNAPSHOTS = [0, 100, 500, 1000, 2000, 3000]
const SEED      = 4242

# Pick a warm-start in subject 0 to make the drift visible.
Random.seed!(SEED)
pool0 = findall(==(0), y)
kpat = pool0[1]
ξ0 = X[:, kpat] .+ σ_init .* randn(d)

@info "Running one unconditional chain (β_chain=$β_chain, T=$T_max) starting from subject 0…"
snaps_raw = masked_sa_sample_with_history(X, copy(ξ0), T_max;
    β = β_chain, α = α, snapshot_steps = SNAPSHOTS, seed = SEED)

# Sharp denoise readout at each checkpoint so the panels are readable as faces.
snaps_clean = Dict(t => denoise(snaps_raw[t], X, β_read) for t in SNAPSHOTS)

# Subject identity at each checkpoint, for the caption.
means = class_means(X, y)
ids = [(t, classify_nearest_mean(snaps_clean[t], means)) for t in SNAPSHOTS]
@info "Subject ID per snapshot:"
for (t, c) in ids
    @info @sprintf("  T=%5d  →  subject %d", t, c)
end

# ── Render strip ─────────────────────────────────────────────────────────────
function render_strip(samples::Vector{Vector{Float64}}; H::Int=64, W::Int=64, gap::Int=2)
    n = length(samples)
    canvas = zeros(Float64, H, n * W + (n - 1) * gap)
    for (k, s) in enumerate(samples)
        x0 = (k - 1) * (W + gap) + 1
        img = transpose(reshape(s, W, H))  # row-major CSV → display orientation
        lo, hi = minimum(img), maximum(img)
        hi > lo && (img = (img .- lo) ./ (hi - lo))
        canvas[1:H, x0:x0 + W - 1] .= Matrix(img)
    end
    return canvas
end

samples_in_order = [snaps_clean[t] for t in SNAPSHOTS]
canv = render_strip(samples_in_order)
out_path = joinpath(_PAPER_FIG_DIR, "Fig_olivetti_trajectory.png")
save(out_path, Gray.(clamp.(canv, 0.0, 1.0)))
@info "Wrote $out_path"

# Also dump the per-checkpoint subject IDs to a small text file the caption
# can cite.
open(joinpath(@__DIR__, "olivetti_trajectory_ids.txt"), "w") do io
    println(io, "T,subject_id")
    for (t, c) in ids
        println(io, "$t,$c")
    end
end
@info "Done."
