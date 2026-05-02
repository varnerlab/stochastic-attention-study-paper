#!/usr/bin/env julia
# ──────────────────────────────────────────────────────────────────────────────
# Olivetti Faces Experiment: unconditional + masked Stochastic Attention
#
# Runs three experimental blocks on the Olivetti faces memory matrix:
#   1. Unconditional SA — Langevin chains over all stored portraits, no mask.
#   2. Hard-masked SA   — set logit_bias[j] = -Inf for j outside the chosen
#                         subject; chain stays inside that subject's basin.
#   3. β_cond sweep     — fix the subject mask, sweep β in {2, 8, 32, 128},
#                         track novelty (median nearest-stored distance) and
#                         hit rate (nearest-mean classifier).
#
# Outputs into ../paper-neurips/figs/ :
#   Fig_olivetti_unconditional.png      — 2x5 grid of unconditional samples
#   Fig_olivetti_masked.png             — 5 subjects x 5 samples each (hard mask)
#   Fig_olivetti_beta_sweep.pdf         — novelty vs. hit-rate tradeoff
# Numbers are written to ../paper-neurips/figs/olivetti_metrics.csv
# ──────────────────────────────────────────────────────────────────────────────

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", "code"))

using LinearAlgebra
using Statistics
using Random
using DelimitedFiles
using Plots
using NNlib
using StatsBase
using Printf
using Images
using FileIO

const _DATA_DIR = joinpath(@__DIR__, "data")
const _PAPER_FIG_DIR = joinpath(@__DIR__, "..", "..", "paper-neurips", "figs")
isdir(_PAPER_FIG_DIR) || mkpath(_PAPER_FIG_DIR)

# ── Load Olivetti ────────────────────────────────────────────────────────────
@info "Loading Olivetti faces …"
faces_raw   = readdlm(joinpath(_DATA_DIR, "faces.csv"), ',', Float64)   # 400 x 4096
targets_raw = readdlm(joinpath(_DATA_DIR, "targets.csv"), ',', Int)     # 400 x 1

# Use first N_SUBJECTS people, all 10 portraits each.
const N_SUBJECTS = 10
keep = vec(targets_raw .< N_SUBJECTS)
faces_raw = faces_raw[keep, :]
targets   = vec(targets_raw)[keep]

# Pattern matrix: d x K, columns are unit-norm portraits.
const d = size(faces_raw, 2)         # 4096
K = size(faces_raw, 1)               # 100
X = Matrix(transpose(faces_raw))     # d x K
for j in 1:K
    nj = norm(X[:, j])
    nj > 0 && (X[:, j] ./= nj)
end
y = targets                          # length-K subject labels in 0:N_SUBJECTS-1
@info "Memory matrix: $(size(X)), subjects: $(length(unique(y)))"

# ── Vendored masked SA sampler ───────────────────────────────────────────────
# Identical to Algorithm 1 in the paper, with one extra line:
#   logits = β X' s + logit_bias
# where logit_bias[j] = -Inf hard-excludes pattern j (softmax weight = 0).

function masked_sa_sample(X::Matrix{Float64}, ξ₀::Vector{Float64}, T::Int;
        β::Float64,
        α::Float64,
        logit_bias::Union{Nothing, Vector{Float64}} = nothing,
        seed::Union{Int, Nothing} = nothing)
    seed === nothing || Random.seed!(seed)
    d, K = size(X)
    ξ = copy(ξ₀)
    noise = sqrt(2.0 * α / β)
    bias = logit_bias === nothing ? zeros(K) : logit_bias
    for t in 1:T
        logits = β .* (X' * ξ) .+ bias
        w = NNlib.softmax(logits)
        ξ .= (1.0 - α) .* ξ .+ α .* (X * w) .+ noise .* randn(d)
    end
    return ξ  # only the chain endpoint is needed
end

function hard_mask_bias(keep::Vector{Bool})::Vector{Float64}
    return [k ? 0.0 : -Inf for k in keep]
end

# ── Helpers ──────────────────────────────────────────────────────────────────
function denoise_step(s::Vector{Float64}, X::Matrix{Float64}, β::Float64;
        bias::Vector{Float64} = zeros(size(X, 2)))
    logits = β .* (X' * s) .+ bias
    w = NNlib.softmax(logits)
    return X * w
end

function nearest_stored_distance(s::Vector{Float64}, X::Matrix{Float64})::Float64
    d = [norm(s .- X[:, j]) for j in 1:size(X, 2)]
    return minimum(d)
end

function pairwise_median_distance(X::Matrix{Float64})::Float64
    K = size(X, 2)
    ds = Float64[]
    for i in 1:K, j in (i+1):K
        push!(ds, norm(X[:, i] .- X[:, j]))
    end
    return median(ds)
end

# Class means (training-free nearest-mean classifier per the practicum).
function class_means(X::Matrix{Float64}, y::Vector{Int})
    classes = sort(unique(y))
    means = Dict{Int, Vector{Float64}}()
    for c in classes
        cols = findall(==(c), y)
        means[c] = vec(mean(X[:, cols], dims=2))
    end
    return means
end

function classify_nearest_mean(s::Vector{Float64}, means::Dict{Int, Vector{Float64}})::Int
    best_c, best_d = first(keys(means)), Inf
    for (c, m) in means
        dd = norm(s .- m)
        if dd < best_d
            best_c, best_d = c, dd
        end
    end
    return best_c
end

# Stack a list of d-vectors into a single grid image (rows × cols), 64x64 cells.
function image_grid(samples::Vector{Vector{Float64}}; nrows::Int, ncols::Int,
        H::Int = 64, W::Int = 64, gap::Int = 2)
    n = length(samples)
    @assert n == nrows * ncols
    canvas = fill(0.0, nrows * H + (nrows - 1) * gap, ncols * W + (ncols - 1) * gap)
    for k in 1:n
        r = div(k - 1, ncols)
        c = rem(k - 1, ncols)
        y0 = r * (H + gap) + 1
        x0 = c * (W + gap) + 1
        img = reshape(samples[k], H, W)  # column-major: rows = first dim
        # Olivetti CSV is row-major (each row is a flattened image read left-to-right);
        # reshape to (H, W) then transpose for correct orientation.
        img = transpose(reshape(samples[k], W, H))  # H x W
        # Min-max normalize per image for display.
        lo, hi = minimum(img), maximum(img)
        hi > lo && (img = (img .- lo) ./ (hi - lo))
        canvas[y0:y0 + H - 1, x0:x0 + W - 1] .= Matrix(img)
    end
    return canvas
end

function save_grid(path::String, canvas::Matrix{Float64})
    img = Gray.(clamp.(canvas, 0.0, 1.0))
    save(path, img)
end

# ── Block 1: Unconditional SA ────────────────────────────────────────────────
# Two-phase β schedule: chain runs at β_chain = 200 (generation regime, SNR
# matches MNIST generation experiment when scaled by d), then a single noise-
# free denoise step at β_read = 10000 produces a sharp MAP-style readout. This
# mirrors how Langevin samplers are used in practice: low β for mixing, high β
# for the final read.
const α_step  = 0.01
const β_chain = 200.0      # SA dynamics (low → chain wanders between subjects)
const β_read  = 10000.0    # final denoise (high → sharp readout on memory hull)
const T_chain = 3000
const σ_init  = 0.05
const n_chains_uncond = 25  # 5x5 grid for visual symmetry with stored/masked panels

@info "Block 1: unconditional SA, β_chain=$β_chain, β_read=$β_read, $(n_chains_uncond) chains, T=$T_chain …"
Random.seed!(42)
init_pat_idxs = StatsBase.sample(1:K, n_chains_uncond; replace = (n_chains_uncond > K))
uncond_samples = Vector{Vector{Float64}}()
uncond_raw     = Vector{Vector{Float64}}()  # pre-denoise (for novelty)
for (c, kpat) in enumerate(init_pat_idxs)
    Random.seed!(1000 + c)
    ξ0 = X[:, kpat] .+ σ_init .* randn(d)
    ξT = masked_sa_sample(X, ξ0, T_chain; β = β_chain, α = α_step, seed = 1000 + c)
    push!(uncond_raw, copy(ξT))
    s = denoise_step(ξT, X, β_read)
    push!(uncond_samples, s)
end

# ── Block 2: Paired before/after — masked AND unconditional from same starts ──
# Each cell (i, j) of all three panels comes from the SAME warm-start: the
# j-th stored portrait of subject i. This makes the figure a true row-aligned
# comparison: row r in (a, b, c) is the same subject; cell (r, j) in (a) is
# the literal warm-start for (b, j) and (c, j).
const SUBJECTS_TO_SHOW = collect(0:4)         # 5 subjects in the figure
const samples_per_subject = 5

@info "Block 2: paired chains (stored | masked | unconditional) for $(length(SUBJECTS_TO_SHOW)) × $samples_per_subject = $(length(SUBJECTS_TO_SHOW)*samples_per_subject) cells …"
warmstart_by_subject = Dict{Int, Vector{Vector{Float64}}}()
masked_samples_by_subject = Dict{Int, Vector{Vector{Float64}}}()
masked_raw_by_subject     = Dict{Int, Vector{Vector{Float64}}}()
uncond_paired_by_subject  = Dict{Int, Vector{Vector{Float64}}}()
for c in SUBJECTS_TO_SHOW
    keep_c = [yj == c for yj in y]
    bias_c = hard_mask_bias(keep_c)
    pool_c = findall(keep_c)
    @assert length(pool_c) >= samples_per_subject "subject $c has only $(length(pool_c)) portraits"

    warm_c   = Vector{Vector{Float64}}()
    masked_c = Vector{Vector{Float64}}()
    raw_c    = Vector{Vector{Float64}}()
    uncond_c = Vector{Vector{Float64}}()
    for j in 1:samples_per_subject
        kpat = pool_c[j]                       # j-th stored portrait of subject c
        push!(warm_c, X[:, kpat])

        # paired warm-start (same kick + seed shared between masked and unmasked)
        Random.seed!(2000 + c * 100 + j)
        ξ0 = X[:, kpat] .+ σ_init .* randn(d)

        # (b) masked SA from this warm-start
        ξT_m = masked_sa_sample(X, copy(ξ0), T_chain; β = β_chain, α = α_step,
            logit_bias = bias_c, seed = 2000 + c * 100 + j)
        push!(raw_c, copy(ξT_m))
        push!(masked_c, denoise_step(ξT_m, X, β_read; bias = bias_c))

        # (c) UNCONDITIONAL SA from the SAME warm-start, no mask
        ξT_u = masked_sa_sample(X, copy(ξ0), T_chain; β = β_chain, α = α_step,
            seed = 2000 + c * 100 + j)
        push!(uncond_c, denoise_step(ξT_u, X, β_read))
    end
    warmstart_by_subject[c]      = warm_c
    masked_samples_by_subject[c] = masked_c
    masked_raw_by_subject[c]     = raw_c
    uncond_paired_by_subject[c]  = uncond_c
end

# ── Subject-recovery rate (nearest-mean classifier on Olivetti class means) ──
@info "Block 2 metrics: subject-recovery rate …"
means = class_means(X, y)

# Hard-mask hit rate.
hard_hits = 0
hard_total = 0
for c in SUBJECTS_TO_SHOW
    for s in masked_samples_by_subject[c]
        global hard_total += 1
        classify_nearest_mean(s, means) == c && (global hard_hits += 1)
    end
end
hard_rate = hard_hits / hard_total

# Unconditional baseline: for the SAME target subjects we masked on, run
# unconditional chains warm-started UNIFORMLY across all 10 subjects (the
# realistic baseline an end-user would face when sampling without conditioning)
# and ask what fraction land in each target subject. With 10 subjects, the
# expected uniform-coverage rate is ~1/10 per target.
const n_chains_baseline = 50
@info "Computing unconditional baseline subject-recovery rate ($n_chains_baseline chains)…"
Random.seed!(99)
baseline_pat_idxs = StatsBase.sample(1:K, n_chains_baseline; replace = (n_chains_baseline > K))
baseline_classes = Int[]
for (i, kpat) in enumerate(baseline_pat_idxs)
    Random.seed!(3000 + i)
    ξ0 = X[:, kpat] .+ σ_init .* randn(d)
    ξT = masked_sa_sample(X, ξ0, T_chain; β = β_chain, α = α_step, seed = 3000 + i)
    s = denoise_step(ξT, X, β_read)
    push!(baseline_classes, classify_nearest_mean(s, means))
end
# Per-target-subject hit rate: average over the 5 SUBJECTS_TO_SHOW.
uncond_rate = mean(mean(baseline_classes .== c) for c in SUBJECTS_TO_SHOW)

# Random-guess rate: 1 / N_SUBJECTS.
random_rate = 1.0 / N_SUBJECTS

@info @sprintf("Subject-recovery rate: hard-mask=%.3f, unconditional=%.3f, random=%.3f",
    hard_rate, uncond_rate, random_rate)

# Novelty: median nearest-stored distance, computed on the PRE-denoise chain
# endpoints (post-denoise the soft attention readout snaps to the convex hull
# at β_read=10000, which would zero-out novelty by construction).
function median_nearest(samples::Vector{Vector{Float64}}, X::Matrix{Float64})
    return median(nearest_stored_distance(s, X) for s in samples)
end
pwise = pairwise_median_distance(X)
novelty_uncond = median_nearest(uncond_raw, X) / pwise
novelty_masked = median(median_nearest(masked_raw_by_subject[c],
                                        X[:, findall(==(c), y)]) for c in SUBJECTS_TO_SHOW) / pwise

@info @sprintf("Median nearest-stored / pairwise-median: unconditional=%.3f, masked=%.3f",
    novelty_uncond, novelty_masked)

# ── Block 3: single-β masked sampling — Hopfield → Hebbian transition ────────
# Use the SAME β for chain dynamics AND for the (one-shot) softmax readout.
# Subject mask stays fixed. As β grows, the readout sharpens from "soft blend
# of all in-class portraits" (novel synthetic faces) to "exact recall of one
# stored portrait" (zero novelty). Hit rate is high throughout because the
# mask hard-excludes out-of-class memories. This recovers the modern-Hopfield
# → Hebbian-recall transition empirically inside the masked sampler.
const β_sweep = [2.0, 8.0, 32.0, 128.0, 512.0, 2048.0]
const sweep_subject = 0
const n_chains_sweep = 20

@info "Block 3: single-β masked sweep on subject $sweep_subject ($(length(β_sweep)) points × $n_chains_sweep chains)…"
keep_sweep = [yj == sweep_subject for yj in y]
bias_sweep = hard_mask_bias(keep_sweep)
pool_sweep = findall(keep_sweep)
Xc = X[:, pool_sweep]
sweep_data = Dict{Float64, NamedTuple{(:novelty, :hit_rate), Tuple{Float64, Float64}}}()
for βc in β_sweep
    samples_β = Vector{Vector{Float64}}()
    for r in 1:n_chains_sweep
        Random.seed!(5000 + Int(round(βc * 10)) + r)
        ξ0 = X[:, rand(pool_sweep)] .+ σ_init .* randn(d)
        ξT = masked_sa_sample(X, ξ0, T_chain; β = βc, α = α_step,
            logit_bias = bias_sweep, seed = 5000 + Int(round(βc * 10)) + r)
        # Single-β denoise (read at the same β as the chain).
        s = denoise_step(ξT, X, βc; bias = bias_sweep)
        push!(samples_β, s)
    end
    # Novelty: minimum cosine distance to any in-class portrait. 1 = orthogonal,
    # 0 = identical to a stored in-class portrait.
    function min_cosine_dist(s::Vector{Float64}, M::Matrix{Float64})
        ns = norm(s)
        ns < 1e-12 && return 1.0
        sims = [dot(s, M[:, j]) / (ns * norm(M[:, j])) for j in 1:size(M, 2)]
        return 1.0 - maximum(sims)
    end
    nov = median(min_cosine_dist(s, Xc) for s in samples_β)
    hr  = mean(classify_nearest_mean(s, means) == sweep_subject for s in samples_β)
    sweep_data[βc] = (novelty = nov, hit_rate = hr)
    @info @sprintf("  β=%.1f: novelty=%.3f, hit_rate=%.3f", βc, nov, hr)
end

# ── Save figures ─────────────────────────────────────────────────────────────
@info "Saving figures …"

# Three row-aligned 5×5 panels for the paper figure. Cell (i, j) of every
# panel corresponds to the same warm-start: the j-th stored portrait of
# subject i. Reader can scan one row left-to-right and see what the masked
# and unmasked SA chains do to the same starting memory.
warm_flat   = Vector{Vector{Float64}}()
masked_flat = Vector{Vector{Float64}}()
uncond_flat = Vector{Vector{Float64}}()
for c in SUBJECTS_TO_SHOW
    append!(warm_flat,   warmstart_by_subject[c])
    append!(masked_flat, masked_samples_by_subject[c])
    append!(uncond_flat, uncond_paired_by_subject[c])
end
nrows_panel = length(SUBJECTS_TO_SHOW)
ncols_panel = samples_per_subject
save_grid(joinpath(_PAPER_FIG_DIR, "Fig_olivetti_stored.png"),
          image_grid(warm_flat;   nrows = nrows_panel, ncols = ncols_panel))
save_grid(joinpath(_PAPER_FIG_DIR, "Fig_olivetti_masked.png"),
          image_grid(masked_flat; nrows = nrows_panel, ncols = ncols_panel))
save_grid(joinpath(_PAPER_FIG_DIR, "Fig_olivetti_unconditional.png"),
          image_grid(uncond_flat; nrows = nrows_panel, ncols = ncols_panel))

# β-sweep numbers go to CSV only; the curves are too flat for a useful figure
# (within-subject portraits are highly similar so single-β masked novelty is
# always small). The numerical table in the appendix conveys the same content.
βs_sorted = sort(collect(keys(sweep_data)))

# ── Save metrics CSV ─────────────────────────────────────────────────────────
metrics_path = joinpath(@__DIR__, "olivetti_metrics.csv")
open(metrics_path, "w") do io
    println(io, "metric,value")
    println(io, "K,$K")
    println(io, "d,$d")
    println(io, "n_subjects,$(length(unique(y)))")
    println(io, "beta_chain,$β_chain")
    println(io, "beta_read,$β_read")
    println(io, "alpha,$α_step")
    println(io, "T_chain,$T_chain")
    println(io, @sprintf("hard_mask_subject_recovery,%.4f", hard_rate))
    println(io, @sprintf("uncond_subject_recovery,%.4f", uncond_rate))
    println(io, @sprintf("random_guess_rate,%.4f", random_rate))
    println(io, @sprintf("novelty_uncond_normalized,%.4f", novelty_uncond))
    println(io, @sprintf("novelty_masked_normalized,%.4f", novelty_masked))
    println(io, @sprintf("pairwise_median_distance,%.4f", pwise))
    for β in βs_sorted
        println(io, @sprintf("sweep_beta_chain=%g_novelty,%.4f", β, sweep_data[β].novelty))
        println(io, @sprintf("sweep_beta_chain=%g_hit_rate,%.4f", β, sweep_data[β].hit_rate))
    end
end
@info "Metrics written to $metrics_path"
@info "Done."
