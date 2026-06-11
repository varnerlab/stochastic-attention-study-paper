#!/usr/bin/env julia
# ──────────────────────────────────────────────────────────────────────────────
# Multi-Digit MNIST Experiment  (digits 1 and 8)
# Mirrors the protocol in StochasticAttention-Experiment-4-MNIST.ipynb (digit 3)
# Runs all 6 baselines (incl. GMM-PCA) + MALA for each digit, computes metrics,
# saves grid PNGs.
# ──────────────────────────────────────────────────────────────────────────────

@info "Loading environment …"
include(joinpath(@__DIR__, "Include-MNIST.jl"))
using Images, ImageIO, Flux
@info "Environment loaded."

# ── decode helper (same as notebook) ──────────────────────────────────────────
function decode(s::Vector{<:Number}; number_of_rows::Int=28, number_of_columns::Int=28)
    X = reshape(s, number_of_rows, number_of_columns) |> X -> transpose(X) |> Matrix
    X̂ = replace(X, -1 => 0)
    return X̂
end

# ── grid-building helper (same as notebook) ───────────────────────────────────
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

# ── GMM-PCA helpers ───────────────────────────────────────────────────────────
const R_pca    = 50     # PCA components
const C_gmm    = 10     # GMM mixture components
const EM_iters = 500
const EM_tol   = 1e-8

function _logsumexp_cols(A::Matrix{Float64})
    m = maximum(A, dims=1)
    return m .+ log.(sum(exp.(A .- m), dims=1))
end

function _fit_gmm(X::Matrix{Float64}, C::Int; n_iters=EM_iters, tol=EM_tol, seed=42)
    r, N = size(X)
    rng  = MersenneTwister(seed)
    # K-means++ init
    μ = zeros(r, C)
    μ[:, 1] = X[:, rand(rng, 1:N)]
    for k in 2:C
        dists = [minimum(sum((X[:, n] .- μ[:, j]).^2) for j in 1:(k-1)) for n in 1:N]
        μ[:, k] = X[:, StatsBase.sample(rng, 1:N, Weights(dists ./ sum(dists)))]
    end
    log_σ² = log.(repeat(max.(vec(var(X, dims=2)), 1e-6), 1, C))
    π_k    = ones(C) / C
    log_r  = zeros(C, N)
    log_lik_prev = -Inf
    for _ in 1:n_iters
        σ² = exp.(log_σ²)
        for k in 1:C
            diff = X .- μ[:, k]
            log_r[k, :] = log(π_k[k]) .-
                           0.5 .* vec(sum(diff.^2 ./ σ²[:, k], dims=1)) .-
                           0.5 .* sum(log.(2π .* σ²[:, k]))
        end
        lse     = _logsumexp_cols(log_r)
        log_lik = sum(lse)
        log_r .-= lse
        r_mat   = exp.(log_r)
        abs(log_lik - log_lik_prev) / (abs(log_lik_prev) + 1e-30) < tol && break
        log_lik_prev = log_lik
        N_k = vec(sum(r_mat, dims=2))
        π_k = max.(N_k / N, 1e-8);  π_k ./= sum(π_k)
        for k in 1:C
            μ[:, k]      = X * r_mat[k, :] / N_k[k]
            diff         = X .- μ[:, k]
            log_σ²[:, k] = log.(max.(vec(sum(diff.^2 .* r_mat[k, :]', dims=2)) / N_k[k], 1e-6))
        end
    end
    return μ, log_σ², π_k
end

function _sample_gmm(μ, log_σ², π_k, n_samples; seed=9999)
    rng = MersenneTwister(seed)
    r   = size(μ, 1)
    wts = Weights(π_k)
    [let k = StatsBase.sample(rng, 1:length(π_k), wts)
         μ[:, k] .+ sqrt.(exp.(log_σ²[:, k])) .* randn(rng, r)
     end for _ in 1:n_samples]
end

# ── Experiment parameters (identical to digit-3 experiment) ───────────────────
const number_of_examples = 100
const number_of_rows = 28
const number_of_cols = 28
const number_of_pixels = number_of_rows * number_of_cols
const α_step = 0.01
const β_inv_temp = 2000.0
const S = 150
const n_chains = 30
const T_per_chain = 5000
const T_burnin = 2000
const thin_interval = 100
const samples_per_chain = 5
const σ_init = 0.01

# ── VAE struct (defined once, used per-digit) ─────────────────────────────
struct _MnistVAE
    enc_shared; enc_μ; enc_logσ²; dec
end
Flux.@layer _MnistVAE

# ── Load MNIST ────────────────────────────────────────────────────────────────
@info "Loading MNIST data …"
digits_image_dictionary = MyMNISTHandwrittenDigitImageDataset(number_of_examples = number_of_examples)
@info "MNIST loaded."

# ── Run one digit end-to-end ──────────────────────────────────────────────────
function run_digit_experiment(digit::Int; digits_image_dictionary, figpath)

    @info "═══  Digit $digit  ═══"

    # ── Build memory matrix ───────────────────────────────────────────────────
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
    @info "  Memory matrix: $(size(X̂))"

    K = size(X̂, 2)

    # ── SA (multi-chain) ─────────────────────────────────────────────────────
    @info "  Running SA multi-chain …"
    sa_samples = Vector{Vector{Float64}}()
    Random.seed!(42)
    pattern_indices = StatsBase.sample(1:K, n_chains, replace = (n_chains > K))
    for (c, k) in enumerate(pattern_indices)
        Random.seed!(12345 + c)
        sₒ = X̂[:, k] .+ σ_init .* randn(number_of_pixels)
        (_, Ξ) = sample(X̂, sₒ, T_per_chain; β = β_inv_temp, α = α_step, seed = 12345 + c)
        # collect thinned post-burn-in
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
    @info "  SA: $(length(sa_samples)) samples"

    # ── MALA (multi-chain) ───────────────────────────────────────────────────
    @info "  Running MALA multi-chain …"
    mala_samples = Vector{Vector{Float64}}()
    mala_accept_rates = Float64[]
    Random.seed!(42)
    pattern_indices = StatsBase.sample(1:K, n_chains, replace = (n_chains > K))
    for (c, k) in enumerate(pattern_indices)
        Random.seed!(12345 + c)
        sₒ = X̂[:, k] .+ σ_init .* randn(number_of_pixels)
        (_, Ξ, ar) = mala_sample(X̂, sₒ, T_per_chain; β = β_inv_temp, α = α_step, seed = 12345 + c)
        push!(mala_accept_rates, ar)
        chain_pool = Vector{Vector{Float64}}()
        for tᵢ in (T_burnin+1):thin_interval:T_per_chain
            push!(chain_pool, Ξ[tᵢ, :])
        end
        n_avail = length(chain_pool)
        idxs = round.(Int, range(1, n_avail, length = min(samples_per_chain, n_avail)))
        for idx in idxs
            push!(mala_samples, chain_pool[idx])
        end
    end
    mala_mean_ar = round(mean(mala_accept_rates), digits=4)
    @info "  MALA: $(length(mala_samples)) samples, accept rate = $mala_mean_ar"

    # ── Gaussian perturbation ────────────────────────────────────────────────
    gp_samples = Vector{Vector{Float64}}()
    σ_noise = sqrt(2 * α_step / β_inv_temp)
    Random.seed!(12345)
    for _ in 1:S
        k = rand(1:K)
        ξ = X̂[:, k] .+ σ_noise .* randn(number_of_pixels)
        push!(gp_samples, ξ)
    end

    # ── Random convex combination ────────────────────────────────────────────
    rc_samples = Vector{Vector{Float64}}()
    dirichlet_dist = Dirichlet(K, 1.0)
    Random.seed!(12345)
    for _ in 1:S
        w = rand(dirichlet_dist)
        ξ = X̂ * w
        push!(rc_samples, ξ)
    end

    # ── Bootstrap (replay) ───────────────────────────────────────────────────
    bs_samples = Vector{Vector{Float64}}()
    Random.seed!(12345)
    for _ in 1:S
        k = rand(1:K)
        push!(bs_samples, copy(X̂[:, k]))
    end

    # ── GMM-PCA ──────────────────────────────────────────────────────────────
    @info "  Fitting GMM-PCA (r=$R_pca, C=$C_gmm) …"
    pca_model  = MultivariateStats.fit(PCA, X̂; maxoutdim = R_pca)
    Z          = MultivariateStats.transform(pca_model, X̂)   # R_pca × K
    μ_gmm, log_σ²_gmm, π_gmm = _fit_gmm(Z, C_gmm; seed = 42 + digit)
    pca_samps  = _sample_gmm(μ_gmm, log_σ²_gmm, π_gmm, S; seed = 7777 + digit)
    gmm_samples = [vec(MultivariateStats.reconstruct(pca_model, z)) for z in pca_samps]
    @info "  GMM-PCA: $(length(gmm_samples)) samples"

    # ── VAE baseline ──────────────────────────────────────────────────────────
    @info "  Training VAE (latent=8, two-phase) …"
    local VAE_LAT = 8
    local VAE_P1 = 2000
    local VAE_P2 = 2000

    mvae = _MnistVAE(
        Chain(Dense(number_of_pixels => 256, relu), Dense(256 => 128, relu)),
        Dense(128 => VAE_LAT),
        Dense(128 => VAE_LAT),
        Chain(Dense(VAE_LAT => 128, relu), Dense(128 => 256, relu), Dense(256 => number_of_pixels))
    )
    X_train_vae = Float32.(X̂)
    opt_vae = Flux.setup(Adam(1f-3), mvae)

    # Phase 1: autoencoder
    for epoch in 1:VAE_P1
        ε = randn(Float32, VAE_LAT, K)
        loss, grads = Flux.withgradient(mvae) do m
            h = m.enc_shared(X_train_vae)
            μ = m.enc_μ(h); lσ² = m.enc_logσ²(h)
            z = μ .+ exp.(0.5f0 .* lσ²) .* ε
            o = m.dec(z)
            x̂ = o ./ (sqrt.(sum(o .^ 2; dims=1)) .+ 1f-8)
            mean(sum((X_train_vae .- x̂) .^ 2; dims=1))
        end
        Flux.update!(opt_vae, mvae, grads[1])
    end
    # Phase 2: KL warmup
    for epoch in 1:VAE_P2
        kl_w = 0.0001f0 * Float32(epoch) / Float32(VAE_P2)
        ε = randn(Float32, VAE_LAT, K)
        loss, grads = Flux.withgradient(mvae) do m
            h = m.enc_shared(X_train_vae)
            μ = m.enc_μ(h); lσ² = m.enc_logσ²(h)
            z = μ .+ exp.(0.5f0 .* lσ²) .* ε
            o = m.dec(z)
            x̂ = o ./ (sqrt.(sum(o .^ 2; dims=1)) .+ 1f-8)
            recon = mean(sum((X_train_vae .- x̂) .^ 2; dims=1))
            kl = -0.5f0 * mean(sum(1f0 .+ lσ² .- μ .^ 2 .- exp.(lσ²); dims=1))
            recon + kl_w * kl
        end
        Flux.update!(opt_vae, mvae, grads[1])
    end
    Random.seed!(9999 + digit)
    Z_vae = randn(Float32, VAE_LAT, S)
    raw_vae = let o = mvae.dec(Z_vae); o ./ (sqrt.(sum(o .^ 2; dims=1)) .+ 1f-8) end
    vae_samples = [Float64.(raw_vae[:, i]) for i in 1:S]
    @info "  VAE: $(length(vae_samples)) samples"

    # ── Compute metrics with per-chain SEs ──────────────────────────────────
    methods = [
        "Bootstrap (replay)"       => bs_samples,
        "Gaussian perturbation"    => gp_samples,
        "Random convex combination"=> rc_samples,
        "GMM-PCA"                  => gmm_samples,
        "VAE (latent=8)"           => vae_samples,
        "MALA"                     => mala_samples,
        "Stochastic attention"     => sa_samples,
    ]

    # Helper: split S samples into n_chains groups of samples_per_chain
    function chain_metric_se(samps, metric_fn)
        nc = n_chains
        spc = samples_per_chain
        group_vals = [metric_fn(samps[(i-1)*spc+1:i*spc]) for i in 1:nc]
        return std(group_vals) / sqrt(nc)
    end

    rows = Vector{NamedTuple}()
    for (name, samps) in methods
        novelty_vals  = [sample_novelty(ξ, X̂) for ξ in samps]
        energy_vals   = [hopfield_energy(ξ, X̂, β_inv_temp) for ξ in samps]
        novelty_mean  = mean(novelty_vals)
        diversity_mean = sample_diversity(samps)
        energy_mean   = mean(energy_vals)
        novelty_se    = chain_metric_se(samps, g -> mean(sample_novelty(ξ, X̂) for ξ in g))
        diversity_se  = chain_metric_se(samps, g -> sample_diversity(g))
        energy_se     = chain_metric_se(samps, g -> mean(hopfield_energy(ξ, X̂, β_inv_temp) for ξ in g))
        push!(rows, (Method = name,
                     Novelty = novelty_mean,   Novelty_SE = novelty_se,
                     Diversity = diversity_mean, Diversity_SE = diversity_se,
                     Energy = energy_mean,      Energy_SE = energy_se))
    end
    df = DataFrame(rows)

    # ── Print LaTeX table rows ───────────────────────────────────────────────
    fmt(x, d) = let v = round(x; digits=d); abs(v) == 0.0 ? abs(v) : v end
    fmtpm(v, se, dv, dse) = "\$$(fmt(v,dv)) \\pm $(fmt(se,dse))\$"
    println("\n% --- Digit $digit: Table values (copy into appendix.tex) ---")
    for row in eachrow(df)
        name = row.Method
        if name == "Stochastic attention"
            name = "\\textbf{Stochastic attention}"
        end
        nv  = fmtpm(row.Novelty,   row.Novelty_SE,   3, 3)
        div = fmtpm(row.Diversity, row.Diversity_SE, 3, 3)
        en  = fmtpm(row.Energy,    row.Energy_SE,    1, 2)
        println("$(name) & $(nv) & $(div) & $(en) \\\\")
    end
    println("% MALA acceptance rate (digit $digit): $mala_mean_ar\n")

    # ── Save grid figures ────────────────────────────────────────────────────
    grid_methods = [
        ("bootstrap", bs_samples),
        ("gaussian",  gp_samples),
        ("convex",    rc_samples),
        ("gmm_pca",   gmm_samples),
        ("vae",       vae_samples),
        ("mala",      mala_samples),
        ("sa",        sa_samples),
    ]
    for (tag, samps) in grid_methods
        canvas = build_grid(samps)
        img = Gray.(canvas)
        fname = "Fig_mnist_grid_$(tag)_digit$(digit).png"
        save_pub(joinpath(figpath, fname), img)
        @info "  Saved $fname"
    end

    println()  # blank line between digits
    return df, mala_mean_ar
end

# ── Main ──────────────────────────────────────────────────────────────────────
figpath = _PATH_TO_FIG
mkpath(figpath)

df3, ar3 = run_digit_experiment(3; digits_image_dictionary, figpath)
df1, ar1 = run_digit_experiment(1; digits_image_dictionary, figpath)
df8, ar8 = run_digit_experiment(8; digits_image_dictionary, figpath)

println("\n══════════════════════════════════════════════════════")
println("FINAL RESULTS")
println("══════════════════════════════════════════════════════")
println("\nDigit 3:")
println(df3)
println("MALA acceptance rate: $ar3")
println("\nDigit 1:")
println(df1)
println("MALA acceptance rate: $ar1")
println("\nDigit 8:")
println(df8)
println("MALA acceptance rate: $ar8")
println("\nDone. Grid figures saved to: $figpath")
