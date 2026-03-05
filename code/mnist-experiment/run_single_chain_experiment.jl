#!/usr/bin/env julia
# ──────────────────────────────────────────────────────────────────────────────
# Single-Chain Diversity Experiment  (digit 3, multiple β values)
#
# PURPOSE: Address reviewer P2 — "report single-chain diversity."
# The main-text Table 1 uses 30 chains each initialized near a different stored
# pattern; the reported diversity is dominated by that initialization spread,
# not by within-chain mixing.  This script answers the question directly:
# starting from ONE fixed initialization, what diversity does a long chain
# actually achieve at β = 2000, 200, and 50?
#
# PROTOCOL:
#   - Digit 3, K = 100, d = 784 (identical memory matrix to main table)
#   - Fixed initialization: stored pattern #1 + small Gaussian noise (σ = 0.01)
#   - β ∈ {2000, 200, 50}; α = 0.01 throughout
#   - T = 50_000 steps; burn-in = 10_000; thin every 100 → 400 samples per β
#   - Metrics: novelty N, diversity D̄, mean energy Ē, mean max-cosine
#
# OUTPUTS:
#   - Console: LaTeX-ready table rows
#   - code/figs/Fig_single_chain_grid_beta{β}.png  (4×4 sample grids)
# ──────────────────────────────────────────────────────────────────────────────

@info "Loading environment …"
include(joinpath(@__DIR__, "Include-MNIST.jl"))
using Images, ImageIO
@info "Environment loaded."

# ── decode / grid helpers (identical to run_multidigit_experiment.jl) ─────────
function decode(s::Vector{<:Number}; number_of_rows::Int=28, number_of_columns::Int=28)
    X = reshape(s, number_of_rows, number_of_columns) |> X -> transpose(X) |> Matrix
    return replace(X, -1 => 0)
end

function build_grid(samples; nrows=6, ncols=6, gap=2)
    H, W = 28, 28
    canvas_h = nrows * H + (nrows - 1) * gap
    canvas_w = ncols * W + (ncols - 1) * gap
    canvas = zeros(Float64, canvas_h, canvas_w)
    n = nrows * ncols
    indices = round.(Int, range(1, length(samples), length=n))
    for idx in 1:n
        r, c = divrem(idx - 1, ncols)
        y0 = r * (H + gap) + 1
        x0 = c * (W + gap) + 1
        img = decode(samples[indices[idx]])
        lo, hi = minimum(img), maximum(img)
        hi > lo && (img = (img .- lo) ./ (hi - lo))
        canvas[y0:y0+H-1, x0:x0+W-1] .= img
    end
    return canvas
end

# ── Experiment parameters ─────────────────────────────────────────────────────
const number_of_examples = 100
const number_of_pixels   = 28 * 28
const α_step  = 0.01
const T_run   = 50_000     # total chain length
const T_burnin = 10_000    # steps discarded as burn-in
const thin    = 100        # thinning interval  →  400 samples per chain
const σ_init  = 0.01       # initialization noise
const seed_init = 42       # fixed seed for initialization
const β_values = [2000.0, 200.0, 50.0]

# ── Load MNIST digit 3 ────────────────────────────────────────────────────────
@info "Loading MNIST …"
digits_dict = MyMNISTHandwrittenDigitImageDataset(number_of_examples = number_of_examples)

ϵ = 1e-12
X_raw = zeros(Float64, number_of_pixels, number_of_examples)
for i in 1:number_of_examples
    img = digits_dict[3][:, :, i]
    xᵢ = reshape(transpose(img) |> Matrix, number_of_pixels) |> vec
    X_raw[:, i] = xᵢ
end
X̂ = similar(X_raw)
for i in 1:number_of_examples
    l = norm(X_raw[:, i])
    X̂[:, i] = X_raw[:, i] ./ (l + ϵ)
end
K = size(X̂, 2)
@info "Memory matrix: $(size(X̂))  (K=$K, d=$(size(X̂,1)))"

# ── Fixed initialization: stored pattern #1 + noise ─────────────────────────
Random.seed!(seed_init)
ξ_init = X̂[:, 1] .+ σ_init .* randn(number_of_pixels)

# ── Run single-chain experiment for each β ───────────────────────────────────
figpath = _PATH_TO_FIG
mkpath(figpath)

println("\n% ── Single-Chain Experiment Results (digit 3, one fixed init) ──")
println("% Columns: β | N (novelty) | D̄ (diversity) | Ē (energy) | max-cos")
println("% T=$T_run, burn-in=$T_burnin, thin=$thin → $(div(T_run-T_burnin, thin)) samples per run\n")

results = Vector{NamedTuple}()

for β in β_values
    @info "Running single chain at β=$β …"

    (_, Ξ) = sample(X̂, copy(ξ_init), T_run; β=β, α=α_step, seed=seed_init + Int(β))

    # collect thinned post-burn-in samples
    chain_samples = Vector{Vector{Float64}}()
    for t in (T_burnin + 1):thin:T_run
        push!(chain_samples, Ξ[t, :])
    end
    S = length(chain_samples)
    @info "  β=$β: collected $S samples"

    # metrics
    novelty_vals  = [sample_novelty(ξ, X̂)          for ξ in chain_samples]
    energy_vals   = [hopfield_energy(ξ, X̂, β)       for ξ in chain_samples]
    maxcos_vals   = [nearest_cosine_similarity(ξ, X̂) for ξ in chain_samples]

    nov_mean  = mean(novelty_vals)
    nov_se    = std(novelty_vals) / sqrt(S)
    div_mean  = sample_diversity(chain_samples)
    # diversity SE: block into groups of 40 (10 blocks)
    block_size = 40
    n_blocks   = div(S, block_size)
    div_blocks = [sample_diversity(chain_samples[(b-1)*block_size+1:b*block_size])
                  for b in 1:n_blocks]
    div_se     = std(div_blocks) / sqrt(n_blocks)
    eng_mean  = mean(energy_vals)
    eng_se    = std(energy_vals) / sqrt(S)
    cos_mean  = mean(maxcos_vals)
    cos_se    = std(maxcos_vals) / sqrt(S)

    push!(results, (
        β         = β,
        N_mean    = nov_mean,  N_se    = nov_se,
        D_mean    = div_mean,  D_se    = div_se,
        E_mean    = eng_mean,  E_se    = eng_se,
        cos_mean  = cos_mean,  cos_se  = cos_se,
        n_samples = S,
    ))

    # print LaTeX row
    fmt3(v, se) = "\$$(round(v; digits=3)) \\pm $(round(se; digits=3))\$"
    fmt1(v, se) = "\$$(round(v; digits=1)) \\pm $(round(se; digits=1))\$"
    println("β=$(Int(β)) & $(fmt3(nov_mean, nov_se)) & $(fmt3(div_mean, div_se)) & $(fmt1(eng_mean, eng_se)) & $(fmt3(cos_mean, cos_se)) \\\\")

    # save grid figure
    fname = "Fig_single_chain_grid_beta$(Int(β)).png"
    canvas = build_grid(chain_samples[1:min(36, S)])
    save(joinpath(figpath, fname), Gray.(canvas))
    @info "  Saved $fname"
end

# ── Summary comparison ────────────────────────────────────────────────────────
println("\n% ── Summary: single-chain diversity vs multi-chain diversity ──")
println("% Multi-chain diversity at β=2000 (from Table 1): 0.600 ± 0.001")
println("% Single-chain diversity at β=2000 (this script): $(round(results[1].D_mean; digits=3)) ± $(round(results[1].D_se; digits=3))")
println("% → Gap = $(round(0.600 - results[1].D_mean; digits=3)) — this is the initialization-diversity confound")

println("\n% ── Interpretation ──")
for r in results
    β_label = Int(r.β)
    if r.D_mean < 0.05
        note = "near-zero → chain stays in its initial basin; diversity in Table 1 comes entirely from multi-chain initialization"
    elseif r.D_mean < 0.3
        note = "moderate → some basin-crossing; genuine but limited single-chain exploration"
    else
        note = "high → chain mixes across basins; genuine single-chain generation"
    end
    println("% β=$β_label: D̄=$(round(r.D_mean; digits=3)), max-cos=$(round(r.cos_mean; digits=3)) → $note")
end

println("\nDone. Grid figures saved to: $figpath")
