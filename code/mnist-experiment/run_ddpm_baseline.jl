#!/usr/bin/env julia
# ──────────────────────────────────────────────────────────────────────────────
# DDPM Baseline for MNIST  (addresses reviewer W3)
#
# Trains a simple Denoising Diffusion Probabilistic Model (Ho et al., 2020)
# on the same K=100 MNIST digit images used by SA, generates 150 samples,
# and computes the same metrics for a direct comparison.
#
# Architecture: MLP denoiser with sinusoidal timestep embedding.
# Runs on CPU (no GPU required).
# ──────────────────────────────────────────────────────────────────────────────

@info "Loading environment …"
include(joinpath(@__DIR__, "Include-MNIST.jl"))
using Flux
@info "Environment loaded."

# ── helpers ──────────────────────────────────────────────────────────────────
function decode(s::Vector{<:Number}; number_of_rows::Int=28, number_of_columns::Int=28)
    X = reshape(s, number_of_rows, number_of_columns) |> X -> transpose(X) |> Matrix
    X̂ = replace(X, -1 => 0)
    return X̂
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

# ── experiment parameters ────────────────────────────────────────────────────
const DIGIT = 3
const K_patterns = 100
const D = 784
const β_inv_temp = 2000.0       # for Hopfield energy computation
const S = 150                   # number of generated samples
const n_chains = 30             # for SE computation (group samples into 30 groups of 5)
const samples_per_chain = 5

# DDPM-specific
const DDPM_T = 200              # diffusion timesteps
const DDPM_EPOCHS = 5000        # training epochs
const DDPM_LR = 1e-3            # learning rate
const DDPM_EMB_DIM = 64         # sinusoidal embedding dimension

# ── load MNIST ───────────────────────────────────────────────────────────────
@info "Loading MNIST digit $DIGIT …"
digits_image_dictionary = MyMNISTHandwrittenDigitImageDataset(number_of_examples = K_patterns)

ϵ_small = 1e-12
X̂ = zeros(Float64, D, K_patterns)
for i in 1:K_patterns
    image_array = digits_image_dictionary[DIGIT][:, :, i]
    xᵢ = reshape(transpose(image_array) |> Matrix, D) |> vec
    lᵢ = norm(xᵢ)
    X̂[:, i] = xᵢ ./ (lᵢ + ϵ_small)
end
@info "Memory matrix: $(size(X̂))"

# ── DDPM forward process schedule ────────────────────────────────────────────
betas = Float32.(range(1f-4, 0.02f0, length=DDPM_T))
alphas = 1f0 .- betas
alpha_bars = cumprod(alphas)
sqrt_alpha_bars = sqrt.(alpha_bars)
sqrt_one_minus_alpha_bars = sqrt.(1f0 .- alpha_bars)

# ── sinusoidal timestep embedding (precomputed) ─────────────────────────────
function build_time_embeddings(T::Int, emb_dim::Int)
    embed = zeros(Float32, emb_dim, T)
    half = emb_dim ÷ 2
    freqs = exp.(Float32.(-(0:half-1)) .* (log(10000f0) / half))
    for t in 1:T
        embed[1:half, t]       = sin.(t .* freqs)
        embed[half+1:end, t]   = cos.(t .* freqs)
    end
    return embed
end

time_embeddings = build_time_embeddings(DDPM_T, DDPM_EMB_DIM)

# ── denoiser MLP ─────────────────────────────────────────────────────────────
# Input: [x_t (D); time_embed (EMB_DIM)] = D + EMB_DIM
# Output: predicted noise (D)
denoiser = Chain(
    Dense(D + DDPM_EMB_DIM => 512, relu),
    Dense(512 => 256, relu),
    Dense(256 => 512, relu),
    Dense(512 => D)
)

# forward pass: concatenate time embedding and predict noise
function denoise(model, x_t::AbstractMatrix{Float32}, t_indices::Vector{Int})
    # x_t: D x batch, t_indices: batch-length vector of timestep indices
    t_emb = time_embeddings[:, t_indices]  # EMB_DIM x batch
    input = vcat(x_t, t_emb)              # (D + EMB_DIM) x batch
    return model(input)
end

# ── training ─────────────────────────────────────────────────────────────────
@info "Training DDPM (T=$DDPM_T, epochs=$DDPM_EPOCHS) …"
X_train = Float32.(X̂)  # D x K
opt = Flux.setup(Adam(Float32(DDPM_LR)), denoiser)

t_start = time()
for epoch in 1:DDPM_EPOCHS
    # sample random timesteps for each training image
    t_idx = rand(1:DDPM_T, K_patterns)

    # sample noise
    ε = randn(Float32, D, K_patterns)

    # compute noisy images: x_t = sqrt(ᾱ_t) * x_0 + sqrt(1-ᾱ_t) * ε
    sab = reshape(sqrt_alpha_bars[t_idx], 1, K_patterns)
    somab = reshape(sqrt_one_minus_alpha_bars[t_idx], 1, K_patterns)
    x_t = sab .* X_train .+ somab .* ε

    # train: predict noise
    loss, grads = Flux.withgradient(denoiser) do m
        ε_pred = denoise(m, x_t, t_idx)
        Flux.mse(ε_pred, ε)
    end
    Flux.update!(opt, denoiser, grads[1])

    if epoch % 1000 == 0
        elapsed = round(time() - t_start, digits=1)
        @info "  Epoch $epoch/$DDPM_EPOCHS, loss=$(round(loss, digits=6)), elapsed=$(elapsed)s"
    end
end
train_time = round(time() - t_start, digits=1)
@info "Training complete in $(train_time)s"

# ── sampling (reverse process) ───────────────────────────────────────────────
@info "Generating $S samples via reverse diffusion …"
function ddpm_sample(model, n_samples::Int)
    Random.seed!(9999)
    x = randn(Float32, D, n_samples)
    for t in DDPM_T:-1:1
        t_idx = fill(t, n_samples)
        ε_pred = denoise(model, x, t_idx)
        alpha_t = alphas[t]
        beta_t = betas[t]
        alpha_bar_t = alpha_bars[t]
        coeff = beta_t / sqrt(1f0 - alpha_bar_t)
        mu = (1f0 / sqrt(alpha_t)) .* (x .- coeff .* ε_pred)
        if t > 1
            z = randn(Float32, D, n_samples)
            x = mu .+ sqrt(beta_t) .* z
        else
            x = mu
        end
    end
    return x
end

x = ddpm_sample(denoiser, S)

# L2 normalize (matching the convention for all other methods)
for i in 1:S
    ni = sqrt(sum(x[:, i] .^ 2)) + 1f-8
    x[:, i] ./= ni
end

ddpm_samples = [Float64.(x[:, i]) for i in 1:S]
@info "Generated $(length(ddpm_samples)) DDPM samples"

# ── compute metrics ──────────────────────────────────────────────────────────
@info "Computing metrics …"
novelty_vals  = [sample_novelty(ξ, X̂) for ξ in ddpm_samples]
energy_vals   = [hopfield_energy(ξ, X̂, β_inv_temp) for ξ in ddpm_samples]
novelty_mean  = mean(novelty_vals)
diversity_mean = sample_diversity(ddpm_samples)
energy_mean   = mean(energy_vals)
maxcos_vals   = [nearest_cosine_similarity(ξ, X̂) for ξ in ddpm_samples]
maxcos_mean   = mean(maxcos_vals)

# SE via chain grouping (split S samples into n_chains groups)
function chain_metric_se(samps, metric_fn)
    spc = samples_per_chain
    group_vals = [metric_fn(samps[(i-1)*spc+1:i*spc]) for i in 1:n_chains]
    return std(group_vals) / sqrt(n_chains)
end

novelty_se   = chain_metric_se(ddpm_samples, g -> mean(sample_novelty(ξ, X̂) for ξ in g))
diversity_se = chain_metric_se(ddpm_samples, g -> sample_diversity(g))
energy_se    = chain_metric_se(ddpm_samples, g -> mean(hopfield_energy(ξ, X̂, β_inv_temp) for ξ in g))

@info "DDPM results:"
@info "  Novelty:   $(round(novelty_mean, digits=3)) ± $(round(novelty_se, digits=3))"
@info "  Diversity:  $(round(diversity_mean, digits=3)) ± $(round(diversity_se, digits=3))"
@info "  Energy:    $(round(energy_mean, digits=2)) ± $(round(energy_se, digits=2))"
@info "  Max-cos:   $(round(maxcos_mean, digits=3))"

# ── also run SA at β=2000 for direct comparison ─────────────────────────────
@info "Running SA (β=$β_inv_temp) for comparison …"
sa_samples = Vector{Vector{Float64}}()
Random.seed!(42)
pattern_indices = StatsBase.sample(1:K_patterns, n_chains, replace=(n_chains > K_patterns))
for (c, k) in enumerate(pattern_indices)
    Random.seed!(12345 + c)
    sₒ = X̂[:, k] .+ 0.01 .* randn(D)
    (_, Ξ) = sample(X̂, sₒ, 5000; β=β_inv_temp, α=0.01, seed=12345+c)
    chain_pool = Vector{Vector{Float64}}()
    for tᵢ in 2001:100:5000
        push!(chain_pool, Ξ[tᵢ, :])
    end
    n_avail = length(chain_pool)
    idxs = round.(Int, range(1, n_avail, length=min(samples_per_chain, n_avail)))
    for idx in idxs
        push!(sa_samples, chain_pool[idx])
    end
end

sa_novelty  = mean(sample_novelty(ξ, X̂) for ξ in sa_samples)
sa_diversity = sample_diversity(sa_samples)
sa_energy   = mean(hopfield_energy(ξ, X̂, β_inv_temp) for ξ in sa_samples)

@info "SA results: N=$(round(sa_novelty, digits=3)), D=$(round(sa_diversity, digits=3)), E=$(round(sa_energy, digits=2))"

# ── print results ────────────────────────────────────────────────────────────
println("\n" * "═"^70)
println("DDPM BASELINE — MNIST digit $DIGIT")
println("═"^70)

fmt(x, d) = let v = round(x; digits=d); abs(v) == 0.0 ? abs(v) : v end
fmtpm(v, se, dv, dse) = "\$$(fmt(v,dv)) \\pm $(fmt(se,dse))\$"

println("\n% --- LaTeX table row (copy into paper) ---")
nv  = fmtpm(novelty_mean,   novelty_se,   3, 3)
dv  = fmtpm(diversity_mean, diversity_se, 3, 3)
en  = fmtpm(energy_mean,    energy_se,    1, 2)
println("DDPM (T=$DDPM_T) & $nv & $dv & $en \\\\")

println("\nComparison:")
println("  SA (β=2000):  N=$(round(sa_novelty,digits=3)), D=$(round(sa_diversity,digits=3)), E=$(round(sa_energy,digits=2))")
println("  DDPM (T=$DDPM_T): N=$(round(novelty_mean,digits=3)), D=$(round(diversity_mean,digits=3)), E=$(round(energy_mean,digits=2))")

# ── save grid figures ────────────────────────────────────────────────────────
@info "Saving grid figures …"
figpath = _PATH_TO_FIG
mkpath(figpath)

canvas_ddpm = build_grid(ddpm_samples)
img_ddpm = Images.Gray.(canvas_ddpm)
Images.save(joinpath(figpath, "Fig_mnist_grid_ddpm_digit$(DIGIT).png"), img_ddpm)
@info "  Saved DDPM grid"

canvas_sa = build_grid(sa_samples)
img_sa = Images.Gray.(canvas_sa)
Images.save(joinpath(figpath, "Fig_mnist_grid_sa_ddpm_comparison_digit$(DIGIT).png"), img_sa)
@info "  Saved SA grid (for comparison)"

println("\nDone. Training time: $(train_time)s")
