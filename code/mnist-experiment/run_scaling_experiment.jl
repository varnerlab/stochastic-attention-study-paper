#!/usr/bin/env julia
# ──────────────────────────────────────────────────────────────────────────────
# Scaling Experiment: SA vs DDPM as a function of K
#
# Tests both methods at K ∈ {100, 500, 1000, 3000, 6000} on MNIST digit 3.
# For SA, β is set via the entropy inflection (Proposition 4) at each K.
# For DDPM, the MLP denoiser is trained from scratch at each K.
#
# This provides a fair comparison: at small K, SA should win (DDPM can't learn);
# at large K, DDPM catches up. The crossover is the interesting finding.
# ──────────────────────────────────────────────────────────────────────────────

@info "Loading environment …"
include(joinpath(@__DIR__, "Include-MNIST.jl"))
using Flux
@info "Environment loaded."

# ── helpers ──────────────────────────────────────────────────────────────────
function decode(s::Vector{<:Number}; number_of_rows::Int=28, number_of_columns::Int=28)
    X_img = reshape(s, number_of_rows, number_of_columns) |> X_img -> transpose(X_img) |> Matrix
    X_out = replace(X_img, -1 => 0)
    return X_out
end

function build_grid(samples; nrows=4, ncols=4, gap=2)
    H, W = 28, 28
    canvas_h = nrows * H + (nrows - 1) * gap
    canvas_w = ncols * W + (ncols - 1) * gap
    canvas = zeros(Float64, canvas_h, canvas_w)
    indices = round.(Int, range(1, length(samples), length=nrows*ncols))
    for idx_i in 1:(nrows*ncols)
        r = Base.div(idx_i - 1, ncols)
        c = rem(idx_i - 1, ncols)
        y0 = r * (H + gap) + 1
        x0 = c * (W + gap) + 1
        img = decode(samples[indices[idx_i]])
        lo, hi = minimum(img), maximum(img)
        if hi > lo
            img = (img .- lo) ./ (hi - lo)
        end
        canvas[y0:y0+H-1, x0:x0+W-1] .= img
    end
    return canvas
end

# ── entropy inflection finder ────────────────────────────────────────────────
function find_entropy_inflection(X̂::Matrix{Float64};
                                  α::Float64=0.01,
                                  n_betas::Int=50,
                                  β_range::Tuple{Float64,Float64}=(0.1, 500.0))
    d, K = size(X̂)
    βs = 10 .^ range(log10(β_range[1]), log10(β_range[2]), length=n_betas)

    n_probes = min(K, 20)
    Hs = zeros(n_betas)
    for (bi, β) in enumerate(βs)
        H_sum = 0.0
        for k in 1:n_probes
            H_sum += attention_entropy(X̂[:, k], X̂, β)
        end
        Hs[bi] = H_sum / n_probes
    end

    log_βs = log.(βs)
    dH = diff(Hs) ./ diff(log_βs)
    d2H = diff(dH) ./ diff(log_βs[1:end-1])

    inflection_idx = 1
    min_d2H = Inf
    for i in 1:length(d2H)
        if d2H[i] < min_d2H
            min_d2H = d2H[i]
            inflection_idx = i + 1
        end
    end

    β_star = βs[inflection_idx]
    snr_star = sqrt(α * β_star / (2 * d))
    return (β_star=β_star, snr_star=snr_star, βs=βs, Hs=Hs)
end

# ── DDPM infrastructure ─────────────────────────────────────────────────────
const DDPM_T = 200
const DDPM_EMB_DIM = 64
const D = 784

# precompute schedule
const ddpm_betas = Float32.(range(1f-4, 0.02f0, length=DDPM_T))
const ddpm_alphas = 1f0 .- ddpm_betas
const ddpm_alpha_bars = cumprod(ddpm_alphas)
const ddpm_sqrt_alpha_bars = sqrt.(ddpm_alpha_bars)
const ddpm_sqrt_one_minus_alpha_bars = sqrt.(1f0 .- ddpm_alpha_bars)

# sinusoidal timestep embedding
function build_time_embeddings(T::Int, emb_dim::Int)
    embed = zeros(Float32, emb_dim, T)
    half = emb_dim ÷ 2
    freqs = exp.(Float32.(-(0:half-1)) .* (log(10000f0) / half))
    for t in 1:T
        embed[1:half, t]     = sin.(t .* freqs)
        embed[half+1:end, t] = cos.(t .* freqs)
    end
    return embed
end
const time_embeddings = build_time_embeddings(DDPM_T, DDPM_EMB_DIM)

function denoise(model, x_t::AbstractMatrix{Float32}, t_indices::Vector{Int})
    t_emb = time_embeddings[:, t_indices]
    input = vcat(x_t, t_emb)
    return model(input)
end

function train_ddpm(X_train::Matrix{Float32}, n_epochs::Int; lr=1e-3)
    d_in, K = size(X_train)
    model = Chain(
        Dense(d_in + DDPM_EMB_DIM => 512, relu),
        Dense(512 => 256, relu),
        Dense(256 => 512, relu),
        Dense(512 => d_in)
    )
    opt = Flux.setup(Adam(Float32(lr)), model)

    for epoch in 1:n_epochs
        t_idx = rand(1:DDPM_T, K)
        ε = randn(Float32, d_in, K)
        sab = reshape(ddpm_sqrt_alpha_bars[t_idx], 1, K)
        somab = reshape(ddpm_sqrt_one_minus_alpha_bars[t_idx], 1, K)
        x_t = sab .* X_train .+ somab .* ε

        loss, grads = Flux.withgradient(model) do m
            ε_pred = denoise(m, x_t, t_idx)
            Flux.mse(ε_pred, ε)
        end
        Flux.update!(opt, model, grads[1])
    end
    return model
end

function ddpm_generate(model, n_samples::Int; seed=9999)
    Random.seed!(seed)
    x = randn(Float32, D, n_samples)
    for t in DDPM_T:-1:1
        t_idx = fill(t, n_samples)
        ε_pred = denoise(model, x, t_idx)
        alpha_t = ddpm_alphas[t]
        beta_t = ddpm_betas[t]
        alpha_bar_t = ddpm_alpha_bars[t]
        coeff = beta_t / sqrt(1f0 - alpha_bar_t)
        mu = (1f0 / sqrt(alpha_t)) .* (x .- coeff .* ε_pred)
        if t > 1
            z = randn(Float32, D, n_samples)
            x = mu .+ sqrt(beta_t) .* z
        else
            x = mu
        end
    end
    # L2 normalize
    for i in 1:n_samples
        ni = sqrt(sum(x[:, i] .^ 2)) + 1f-8
        x[:, i] ./= ni
    end
    return [Float64.(x[:, i]) for i in 1:n_samples]
end

# ── SA infrastructure ────────────────────────────────────────────────────────
const α_step = 0.01
const n_chains = 30
const T_per_chain = 5000
const T_burnin = 2000
const thin_interval = 100
const samples_per_chain = 5
const σ_init = 0.01
const S = 150

function run_sa(X̂::Matrix{Float64}, β::Float64)
    d, K = size(X̂)
    sa_samples = Vector{Vector{Float64}}()
    Random.seed!(42)
    pattern_indices = StatsBase.sample(1:K, n_chains, replace=(n_chains > K))
    for (c, k) in enumerate(pattern_indices)
        Random.seed!(12345 + c)
        sₒ = X̂[:, k] .+ σ_init .* randn(d)
        (_, Ξ) = sample(X̂, sₒ, T_per_chain; β=β, α=α_step, seed=12345+c)
        chain_pool = Vector{Vector{Float64}}()
        for tᵢ in (T_burnin+1):thin_interval:T_per_chain
            push!(chain_pool, Ξ[tᵢ, :])
        end
        n_avail = length(chain_pool)
        idxs = round.(Int, range(1, n_avail, length=min(samples_per_chain, n_avail)))
        for idx in idxs
            push!(sa_samples, chain_pool[idx])
        end
    end
    return sa_samples
end

# ── metrics ──────────────────────────────────────────────────────────────────
function compute_metrics(samples, X̂, β)
    novelty_vals = [sample_novelty(ξ, X̂) for ξ in samples]
    energy_vals  = [hopfield_energy(ξ, X̂, β) for ξ in samples]
    maxcos_vals  = [nearest_cosine_similarity(ξ, X̂) for ξ in samples]
    return (
        novelty   = mean(novelty_vals),
        diversity  = sample_diversity(samples),
        energy    = mean(energy_vals),
        maxcos    = mean(maxcos_vals),
    )
end

# ── load full MNIST digit 3 ─────────────────────────────────────────────────
@info "Loading MNIST digit 3 (full training set) …"
# MNIST has ~3795 digit-3 training images; use safe cap
const MAX_DIGIT3 = 3500
digits_image_dictionary = MyMNISTHandwrittenDigitImageDataset(number_of_examples = MAX_DIGIT3)

ϵ_small = 1e-12
all_images = zeros(Float64, D, MAX_DIGIT3)
for i in 1:MAX_DIGIT3
    image_array = digits_image_dictionary[3][:, :, i]
    xᵢ = reshape(transpose(image_array) |> Matrix, D) |> vec
    lᵢ = norm(xᵢ)
    all_images[:, i] = xᵢ ./ (lᵢ + ϵ_small)
end
@info "Loaded $(size(all_images, 2)) images"

# ── scaling experiment ───────────────────────────────────────────────────────
K_values = [100, 500, 1000, 2000, 3500]

# Scale DDPM epochs: more data → fewer epochs needed per sample
ddpm_epochs_for_K = Dict(100 => 5000, 500 => 3000, 1000 => 2000, 2000 => 1500, 3500 => 1000)

results = []

for K in K_values
    @info "══════════════════════════════════════════════════════"
    @info "K = $K  (K/d = $(round(K/D, digits=3)))"
    @info "══════════════════════════════════════════════════════"

    # build memory matrix
    X̂ = all_images[:, 1:K]

    # find β* via entropy inflection
    @info "  Finding entropy inflection …"
    # adjust β range based on K: larger K needs larger β range
    β_max = max(500.0, 5.0 * sqrt(D))
    ei = find_entropy_inflection(X̂; α=α_step, β_range=(0.1, β_max))
    β_star = ei.β_star
    snr_star = ei.snr_star
    @info "  β* = $(round(β_star, digits=2)), SNR* = $(round(snr_star, digits=4))"

    # SA at β* (generation regime)
    @info "  Running SA at β* = $(round(β_star, digits=1)) (generation) …"
    t0 = time()
    sa_gen_samples = run_sa(X̂, β_star)
    sa_gen_time = round(time() - t0, digits=1)
    sa_gen_metrics = compute_metrics(sa_gen_samples, X̂, β_star)
    @info "  SA gen: N=$(round(sa_gen_metrics.novelty,digits=3)), D=$(round(sa_gen_metrics.diversity,digits=3)), " *
          "E=$(round(sa_gen_metrics.energy,digits=2)), max-cos=$(round(sa_gen_metrics.maxcos,digits=3)), time=$(sa_gen_time)s"

    # SA at 5*β* (retrieval regime)
    β_ret = 5.0 * β_star
    @info "  Running SA at 5β* = $(round(β_ret, digits=1)) (retrieval) …"
    t0 = time()
    sa_ret_samples = run_sa(X̂, β_ret)
    sa_ret_time = round(time() - t0, digits=1)
    sa_ret_metrics = compute_metrics(sa_ret_samples, X̂, β_ret)
    @info "  SA ret: N=$(round(sa_ret_metrics.novelty,digits=3)), D=$(round(sa_ret_metrics.diversity,digits=3)), " *
          "E=$(round(sa_ret_metrics.energy,digits=2)), max-cos=$(round(sa_ret_metrics.maxcos,digits=3)), time=$(sa_ret_time)s"

    # DDPM
    n_epochs = ddpm_epochs_for_K[K]
    @info "  Training DDPM ($n_epochs epochs) …"
    t0 = time()
    ddpm_model = train_ddpm(Float32.(X̂), n_epochs)
    ddpm_train_time = round(time() - t0, digits=1)
    @info "  DDPM training: $(ddpm_train_time)s"

    @info "  Generating DDPM samples …"
    ddpm_samples = ddpm_generate(ddpm_model, S)
    ddpm_metrics = compute_metrics(ddpm_samples, X̂, β_star)
    @info "  DDPM:   N=$(round(ddpm_metrics.novelty,digits=3)), D=$(round(ddpm_metrics.diversity,digits=3)), " *
          "E=$(round(ddpm_metrics.energy,digits=2)), max-cos=$(round(ddpm_metrics.maxcos,digits=3))"

    push!(results, (
        K = K,
        load_ratio = K / D,
        β_star = β_star,
        snr_star = snr_star,
        # SA generation
        sa_gen_novelty = sa_gen_metrics.novelty,
        sa_gen_diversity = sa_gen_metrics.diversity,
        sa_gen_energy = sa_gen_metrics.energy,
        sa_gen_maxcos = sa_gen_metrics.maxcos,
        sa_gen_time = sa_gen_time,
        # SA retrieval
        sa_ret_novelty = sa_ret_metrics.novelty,
        sa_ret_diversity = sa_ret_metrics.diversity,
        sa_ret_energy = sa_ret_metrics.energy,
        sa_ret_maxcos = sa_ret_metrics.maxcos,
        # DDPM
        ddpm_novelty = ddpm_metrics.novelty,
        ddpm_diversity = ddpm_metrics.diversity,
        ddpm_energy = ddpm_metrics.energy,
        ddpm_maxcos = ddpm_metrics.maxcos,
        ddpm_train_time = ddpm_train_time,
        ddpm_epochs = n_epochs,
    ))
end

# ── print results table ──────────────────────────────────────────────────────
println("\n" * "═"^100)
println("SCALING EXPERIMENT: SA vs DDPM on MNIST digit 3")
println("═"^100)

df = DataFrame(results)
println(df)

# ── LaTeX table ──────────────────────────────────────────────────────────────
println("\n% --- LaTeX table ---")
println("\\begin{tabular}{@{}rrccccccccc@{}}")
println("\\toprule")
println("\$K\$ & \$K/d\$ & \$\\beta^*\$ & \\multicolumn{3}{c}{SA (generation, \$\\beta{=}\\beta^*\$)} & \\multicolumn{3}{c}{DDPM (\$T{=}$(DDPM_T)\$)} & DDPM \\\\")
println(" & & & \$\\mathcal{N}\$ & \$\\bar{\\mathcal{D}}\$ & max-\$\\cos\$ & \$\\mathcal{N}\$ & \$\\bar{\\mathcal{D}}\$ & max-\$\\cos\$ & train (s) \\\\")
println("\\midrule")
for r in results
    println("$(r.K) & $(round(r.load_ratio, digits=2)) & $(round(r.β_star, digits=0)) & " *
            "$(round(r.sa_gen_novelty, digits=3)) & $(round(r.sa_gen_diversity, digits=3)) & $(round(r.sa_gen_maxcos, digits=3)) & " *
            "$(round(r.ddpm_novelty, digits=3)) & $(round(r.ddpm_diversity, digits=3)) & $(round(r.ddpm_maxcos, digits=3)) & " *
            "$(round(r.ddpm_train_time, digits=0)) \\\\")
end
println("\\bottomrule")
println("\\end{tabular}")

# ── save grids at each K ─────────────────────────────────────────────────────
@info "Saving sample grids …"
figpath = _PATH_TO_FIG
mkpath(figpath)

# Re-run to save grids (use cached results where possible)
# Actually we already have the samples from the last K. Save grids for all K values
# by re-running just the generation (fast for SA, skip DDPM re-train by saving samples above)
# For simplicity, save grids only for first and last K
for (i, r) in enumerate(results)
    K = r.K
    X̂_k = all_images[:, 1:K]

    # regenerate SA at β* for this K (deterministic seed)
    sa_samps = run_sa(X̂_k, r.β_star)
    canvas = build_grid(sa_samps)
    img = Images.Gray.(canvas)
    Images.save(joinpath(figpath, "Fig_scaling_sa_K$(K).png"), img)

    # regenerate DDPM for this K
    ddpm_model_k = train_ddpm(Float32.(X̂_k), ddpm_epochs_for_K[K])
    ddpm_samps = ddpm_generate(ddpm_model_k, S)
    canvas = build_grid(ddpm_samps)
    img = Images.Gray.(canvas)
    Images.save(joinpath(figpath, "Fig_scaling_ddpm_K$(K).png"), img)

    @info "  Saved grids for K=$K"
end

# ── generate scaling figure ──────────────────────────────────────────────────
@info "Generating scaling figure …"
Ks = [r.K for r in results]

p1 = plot(Ks, [r.sa_gen_maxcos for r in results],
    label="SA (β=β*)", lw=2, marker=:circle, color=:steelblue,
    xlabel="K (number of stored patterns)", ylabel="Max cosine similarity",
    title="(a) Structural fidelity", xscale=:log10, legend=:right,
    ylims=(0, 1), size=(400, 300))
plot!(Ks, [r.ddpm_maxcos for r in results],
    label="DDPM (T=$DDPM_T)", lw=2, marker=:square, color=:coral)
plot!(Ks, [r.sa_ret_maxcos for r in results],
    label="SA (β=5β*, retrieval)", lw=2, marker=:diamond, color=:steelblue, ls=:dash)

p2 = plot(Ks, [r.sa_gen_novelty for r in results],
    label="SA (β=β*)", lw=2, marker=:circle, color=:steelblue,
    xlabel="K (number of stored patterns)", ylabel="Novelty",
    title="(b) Novelty", xscale=:log10, legend=:right,
    ylims=(0, 1), size=(400, 300))
plot!(Ks, [r.ddpm_novelty for r in results],
    label="DDPM (T=$DDPM_T)", lw=2, marker=:square, color=:coral)

p3 = plot(Ks, [r.sa_gen_diversity for r in results],
    label="SA (β=β*)", lw=2, marker=:circle, color=:steelblue,
    xlabel="K (number of stored patterns)", ylabel="Diversity",
    title="(c) Diversity", xscale=:log10, legend=:right,
    ylims=(0, 1), size=(400, 300))
plot!(Ks, [r.ddpm_diversity for r in results],
    label="DDPM (T=$DDPM_T)", lw=2, marker=:square, color=:coral)

p4 = plot(Ks, [r.β_star for r in results],
    label=nothing, lw=2, marker=:circle, color=:black,
    xlabel="K", ylabel="β*",
    title="(d) Entropy inflection β*", xscale=:log10,
    size=(400, 300))

p_combined = plot(p1, p2, p3, p4, layout=(2, 2), size=(850, 650), margin=5Plots.mm)
savefig(p_combined, joinpath(figpath, "Fig_scaling_sa_vs_ddpm.pdf"))
@info "Saved scaling figure"

println("\nDone.")
