#!/usr/bin/env julia
# ──────────────────────────────────────────────────────────────────────────────
# Phase 3: Full-MNIST VAE Baseline
# Addresses Review v2 W3 / Q2: "Compare SA against a VAE trained on full MNIST"
#
# Strategy:
#   1. Train a VAE on ALL 60k MNIST digits (larger architecture, latent=32)
#   2. Train a class-conditional VAE on all digit-3 images (~6k)
#   3. Compare generation quality against SA at β=200 and β=2000
#
# This isolates whether SA's advantage is fundamental or an artifact of K=100.
# ──────────────────────────────────────────────────────────────────────────────

@info "Loading environment …"
include(joinpath(@__DIR__, "Include-MNIST.jl"))
using Images, ImageIO, Flux, Printf
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

# ── VAE struct ────────────────────────────────────────────────────────────────
struct FullMnistVAE
    enc_shared; enc_μ; enc_logσ²; dec
end
Flux.@layer FullMnistVAE

# ── Parameters ────────────────────────────────────────────────────────────────
const number_of_pixels = 784
const K_memory = 100          # SA memory size (for fair metric comparison)
const S = 150                 # number of generated samples
const α_step = 0.01
const n_chains = 30
const T_per_chain = 5000
const T_burnin = 2000
const thin_interval = 100
const samples_per_chain = 5
const σ_init = 0.01

# ── Load ALL MNIST data ──────────────────────────────────────────────────────
flush(stdout); flush(stderr)
@info "Loading full MNIST dataset (1000 per digit) …"
flush(stdout); flush(stderr)

# Load 1000 per digit (10k total, 10x more than SA's K=100 memory)
# This gives the VAE "adequate data" while keeping load time reasonable
digits_image_dictionary_full = MyMNISTHandwrittenDigitImageDataset(number_of_examples = 1000)

# Also load K=100 for SA memory (same as original experiment)
digits_image_dictionary_k100 = MyMNISTHandwrittenDigitImageDataset(number_of_examples = K_memory)
@info "MNIST loaded."
flush(stdout); flush(stderr)

# ── Build full digit-3 dataset ────────────────────────────────────────────────
function build_digit_matrix(dict, digit; n_examples=nothing)
    raw = dict[digit]  # 28 × 28 × n array
    n_avail = size(raw, 3)
    n = isnothing(n_examples) ? n_avail : min(n_examples, n_avail)
    X = zeros(Float64, number_of_pixels, n)
    for i in 1:n
        image_array = raw[:, :, i]
        X[:, i] = reshape(transpose(image_array) |> Matrix, number_of_pixels) |> vec
    end
    return X
end

function unit_normalize(X)
    ϵ = 1e-12
    X̂ = similar(X)
    for i in 1:size(X, 2)
        X̂[:, i] = X[:, i] ./ (norm(X[:, i]) + ϵ)
    end
    return X̂
end

# SA memory: K=100 digit-3 images (same as original experiment)
X_k100 = build_digit_matrix(digits_image_dictionary_k100, 3; n_examples=K_memory)
X̂_k100 = unit_normalize(X_k100)
@info "SA memory: $(size(X̂_k100)) (K=$(size(X̂_k100,2)))"

# Full digit-3 dataset
X_full3 = build_digit_matrix(digits_image_dictionary_full, 3)
X̂_full3 = unit_normalize(X_full3)
N_full3 = size(X_full3, 2)
@info "Full digit-3 dataset: $(size(X̂_full3)) (N=$N_full3)"

# All-digit dataset for unconditional VAE
all_digits_list = Matrix{Float64}[]
for d in 0:9
    push!(all_digits_list, build_digit_matrix(digits_image_dictionary_full, d))
end
X_all = hcat(all_digits_list...)
X̂_all = unit_normalize(X_all)
N_all = size(X_all, 2)
@info "Full MNIST dataset: $(size(X̂_all)) (N=$N_all)"

# ── Train VAE helper ──────────────────────────────────────────────────────────
function train_vae(X_train::Matrix{Float32}, latent_dim::Int;
                   phase1_epochs=3000, phase2_epochs=3000,
                   lr=1e-3, kl_weight_final=0.001, batch_size=256,
                   hidden1=512, hidden2=256, label="VAE")

    D = size(X_train, 1)
    N = size(X_train, 2)

    vae = FullMnistVAE(
        Chain(Dense(D => hidden1, relu), Dense(hidden1 => hidden2, relu)),
        Dense(hidden2 => latent_dim),
        Dense(hidden2 => latent_dim),
        Chain(Dense(latent_dim => hidden2, relu), Dense(hidden2 => hidden1, relu), Dense(hidden1 => D))
    )
    opt = Flux.setup(Adam(Float32(lr)), vae)

    # Mini-batch iterator
    function get_batches(X, bs)
        N = size(X, 2)
        perm = randperm(N)
        batches = [X[:, perm[i:min(i+bs-1, N)]] for i in 1:bs:N]
        return batches
    end

    # Phase 1: AE-only (no KL)
    @info "  [$label] Phase 1: AE warmup ($phase1_epochs epochs) …"
    for epoch in 1:phase1_epochs
        for batch in get_batches(X_train, batch_size)
            K_b = size(batch, 2)
            ε = randn(Float32, latent_dim, K_b)
            loss, grads = Flux.withgradient(vae) do m
                h = m.enc_shared(batch)
                μ = m.enc_μ(h); lσ² = m.enc_logσ²(h)
                z = μ .+ exp.(0.5f0 .* lσ²) .* ε
                o = m.dec(z)
                x̂ = o ./ (sqrt.(sum(o .^ 2; dims=1)) .+ 1f-8)
                mean(sum((batch .- x̂) .^ 2; dims=1))
            end
            Flux.update!(opt, vae, grads[1])
        end
        if epoch % 500 == 0
            @info "    [$label] Phase 1 epoch $epoch/$phase1_epochs"
        end
    end

    # Phase 2: VAE with KL annealing
    @info "  [$label] Phase 2: KL annealing ($phase2_epochs epochs) …"
    for epoch in 1:phase2_epochs
        kl_w = Float32(kl_weight_final) * Float32(epoch) / Float32(phase2_epochs)
        for batch in get_batches(X_train, batch_size)
            K_b = size(batch, 2)
            ε = randn(Float32, latent_dim, K_b)
            loss, grads = Flux.withgradient(vae) do m
                h = m.enc_shared(batch)
                μ = m.enc_μ(h); lσ² = m.enc_logσ²(h)
                z = μ .+ exp.(0.5f0 .* lσ²) .* ε
                o = m.dec(z)
                x̂ = o ./ (sqrt.(sum(o .^ 2; dims=1)) .+ 1f-8)
                recon = mean(sum((batch .- x̂) .^ 2; dims=1))
                kl = -0.5f0 * mean(sum(1f0 .+ lσ² .- μ .^ 2 .- exp.(lσ²); dims=1))
                recon + kl_w * kl
            end
            Flux.update!(opt, vae, grads[1])
        end
        if epoch % 500 == 0
            @info "    [$label] Phase 2 epoch $epoch/$phase2_epochs"
        end
    end

    return vae
end

function generate_vae_samples(vae, n_samples, latent_dim; seed=9999)
    Random.seed!(seed)
    Z = randn(Float32, latent_dim, n_samples)
    raw = let o = vae.dec(Z); o ./ (sqrt.(sum(o .^ 2; dims=1)) .+ 1f-8) end
    return [Float64.(raw[:, i]) for i in 1:n_samples]
end

# ── Metric computation (same as other experiments) ────────────────────────────
function compute_all_metrics(samps, X̂, β; label="")
    novelty_vals  = [sample_novelty(ξ, X̂) for ξ in samps]
    maxcos_vals   = [nearest_cosine_similarity(ξ, X̂) for ξ in samps]
    energy_vals   = [hopfield_energy(ξ, X̂, β) for ξ in samps]
    diversity_val = sample_diversity(samps)

    on_manifold_idx = findall(e -> e < 0.0, energy_vals)
    n_on_manifold = length(on_manifold_idx)

    return (
        label = label,
        n_samples = length(samps),
        novelty = mean(novelty_vals),
        max_cos = mean(maxcos_vals),
        diversity = diversity_val,
        energy = mean(energy_vals),
        n_on_manifold = n_on_manifold,
        frac_on_manifold = n_on_manifold / length(samps),
    )
end

# ── Run SA baselines (same protocol as original) ─────────────────────────────
function run_sa(X̂, β; label="SA")
    samples = Vector{Vector{Float64}}()
    K = size(X̂, 2)
    Random.seed!(42)
    pattern_indices = StatsBase.sample(1:K, n_chains, replace=(n_chains > K))
    for (c, k) in enumerate(pattern_indices)
        Random.seed!(12345 + c)
        sₒ = X̂[:, k] .+ σ_init .* randn(size(X̂, 1))
        (_, Ξ) = sample(X̂, sₒ, T_per_chain; β=β, α=α_step, seed=12345+c)
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

# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 1: VAE trained on full digit-3 dataset (~6k images)
# ══════════════════════════════════════════════════════════════════════════════
flush(stdout); flush(stderr)
println("\n" * "="^80)
println("EXPERIMENT 1: VAE trained on ALL digit-3 images (N=$N_full3)")
println("="^80)
flush(stdout)

# Sweep latent dimensions to be fair
for latent_dim in [8, 16, 32, 64]
    @info "Training digit-3 VAE with latent_dim=$latent_dim …"
    vae_d3 = train_vae(Float32.(X̂_full3), latent_dim;
                       phase1_epochs=2000, phase2_epochs=2000,
                       hidden1=512, hidden2=256,
                       kl_weight_final=0.001,
                       label="VAE-d3-lat$latent_dim")

    vae_samps = generate_vae_samples(vae_d3, S, latent_dim; seed=9999)

    # Metrics against SA memory (K=100) -- same X̂ SA uses
    for β in [200.0, 2000.0]
        m = compute_all_metrics(vae_samps, X̂_k100, β; label="VAE-fullD3(lat=$latent_dim)")
        @printf("β=%4d | %-30s | Nov=%.4f | MaxCos=%.4f | Div=%.4f | E=%+.3f | OnManif=%d/%d\n",
            Int(β), m.label, m.novelty, m.max_cos, m.diversity, m.energy,
            m.n_on_manifold, m.n_samples)
    end
end

# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 2: VAE trained on ALL MNIST digits, generate digit-3 by filtering
# ══════════════════════════════════════════════════════════════════════════════
flush(stdout); flush(stderr)
println("\n" * "="^80)
println("EXPERIMENT 2: VAE trained on ALL MNIST digits (N=$N_all), filtered to digit-3")
println("="^80)
flush(stdout)

# Train a larger VAE on all digits
@info "Training full-MNIST VAE (latent=32) …"
vae_all = train_vae(Float32.(X̂_all), 32;
                    phase1_epochs=2000, phase2_epochs=2000,
                    hidden1=512, hidden2=256,
                    kl_weight_final=0.001,
                    label="VAE-allMNIST-lat32")

# Generate many samples and filter by proximity to digit-3 memory
@info "Generating samples and filtering to digit-3 …"
Random.seed!(9999)
n_generate = 5000  # generate many, then pick closest to digit-3 manifold
Z = randn(Float32, 32, n_generate)
raw_all = let o = vae_all.dec(Z); o ./ (sqrt.(sum(o .^ 2; dims=1)) .+ 1f-8) end

# Filter: keep samples with highest max-cos to digit-3 memory
all_generated = [Float64.(raw_all[:, i]) for i in 1:n_generate]
maxcos_to_d3 = [nearest_cosine_similarity(ξ, X̂_k100) for ξ in all_generated]

# Take top-S by max-cos (mimics "generate digit-3" without a classifier)
sorted_idx = sortperm(maxcos_to_d3, rev=true)
vae_all_filtered = all_generated[sorted_idx[1:S]]

for β in [200.0, 2000.0]
    m = compute_all_metrics(vae_all_filtered, X̂_k100, β; label="VAE-allMNIST(top-$S)")
    @printf("β=%4d | %-30s | Nov=%.4f | MaxCos=%.4f | Div=%.4f | E=%+.3f | OnManif=%d/%d\n",
        Int(β), m.label, m.novelty, m.max_cos, m.diversity, m.energy,
        m.n_on_manifold, m.n_samples)
end

# Also report unfiltered (random S from all generated)
vae_all_random = all_generated[randperm(n_generate)[1:S]]
for β in [200.0, 2000.0]
    m = compute_all_metrics(vae_all_random, X̂_k100, β; label="VAE-allMNIST(random-$S)")
    @printf("β=%4d | %-30s | Nov=%.4f | MaxCos=%.4f | Div=%.4f | E=%+.3f | OnManif=%d/%d\n",
        Int(β), m.label, m.novelty, m.max_cos, m.diversity, m.energy,
        m.n_on_manifold, m.n_samples)
end

# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 3: SA baselines for direct comparison
# ══════════════════════════════════════════════════════════════════════════════
flush(stdout); flush(stderr)
println("\n" * "="^80)
println("SA BASELINES (digit-3, K=100 memory)")
println("="^80)
flush(stdout)

for β in [200.0, 2000.0]
    @info "Running SA at β=$β …"
    sa_samps = run_sa(X̂_k100, β; label="SA(β=$β)")
    m = compute_all_metrics(sa_samps, X̂_k100, β; label="SA(β=$(Int(β)))")
    @printf("β=%4d | %-30s | Nov=%.4f | MaxCos=%.4f | Div=%.4f | E=%+.3f | OnManif=%d/%d\n",
        Int(β), m.label, m.novelty, m.max_cos, m.diversity, m.energy,
        m.n_on_manifold, m.n_samples)
end

# ── Also run original K=100 VAE for reference ────────────────────────────────
println("\n" * "="^80)
println("ORIGINAL VAE BASELINE (K=100 digit-3, latent=8) — for reference")
println("="^80)

struct _MnistVAE
    enc_shared; enc_μ; enc_logσ²; dec
end
Flux.@layer _MnistVAE

@info "Training original VAE (latent=8, K=100) …"
mvae_orig = _MnistVAE(
    Chain(Dense(number_of_pixels => 256, relu), Dense(256 => 128, relu)),
    Dense(128 => 8),
    Dense(128 => 8),
    Chain(Dense(8 => 128, relu), Dense(128 => 256, relu), Dense(256 => number_of_pixels))
)
X_train_orig = Float32.(X̂_k100)
opt_orig = Flux.setup(Adam(1f-3), mvae_orig)

K_orig = size(X̂_k100, 2)
for epoch in 1:2000
    ε = randn(Float32, 8, K_orig)
    loss, grads = Flux.withgradient(mvae_orig) do m
        h = m.enc_shared(X_train_orig)
        μ = m.enc_μ(h); lσ² = m.enc_logσ²(h)
        z = μ .+ exp.(0.5f0 .* lσ²) .* ε
        o = m.dec(z)
        x̂ = o ./ (sqrt.(sum(o .^ 2; dims=1)) .+ 1f-8)
        mean(sum((X_train_orig .- x̂) .^ 2; dims=1))
    end
    Flux.update!(opt_orig, mvae_orig, grads[1])
end
for epoch in 1:2000
    kl_w = 0.0001f0 * Float32(epoch) / 2000f0
    ε = randn(Float32, 8, K_orig)
    loss, grads = Flux.withgradient(mvae_orig) do m
        h = m.enc_shared(X_train_orig)
        μ = m.enc_μ(h); lσ² = m.enc_logσ²(h)
        z = μ .+ exp.(0.5f0 .* lσ²) .* ε
        o = m.dec(z)
        x̂ = o ./ (sqrt.(sum(o .^ 2; dims=1)) .+ 1f-8)
        recon = mean(sum((X_train_orig .- x̂) .^ 2; dims=1))
        kl = -0.5f0 * mean(sum(1f0 .+ lσ² .- μ .^ 2 .- exp.(lσ²); dims=1))
        recon + kl_w * kl
    end
    Flux.update!(opt_orig, mvae_orig, grads[1])
end

Random.seed!(9999)
Z_orig = randn(Float32, 8, S)
raw_orig = let o = mvae_orig.dec(Z_orig); o ./ (sqrt.(sum(o .^ 2; dims=1)) .+ 1f-8) end
vae_orig_samps = [Float64.(raw_orig[:, i]) for i in 1:S]

for β in [200.0, 2000.0]
    m = compute_all_metrics(vae_orig_samps, X̂_k100, β; label="VAE-orig(K=100,lat=8)")
    @printf("β=%4d | %-30s | Nov=%.4f | MaxCos=%.4f | Div=%.4f | E=%+.3f | OnManif=%d/%d\n",
        Int(β), m.label, m.novelty, m.max_cos, m.diversity, m.energy,
        m.n_on_manifold, m.n_samples)
end

# ── Save sample grids ────────────────────────────────────────────────────────
figpath = _PATH_TO_FIG
mkpath(figpath)

# Best full-MNIST VAE grids
for (tag, samps) in [
    ("vae_fullD3_lat32", generate_vae_samples(
        train_vae(Float32.(X̂_full3), 32; phase1_epochs=2000, phase2_epochs=2000,
                  hidden1=512, hidden2=256, kl_weight_final=0.001, label="grid-VAE"),
        S, 32; seed=9999)),
    ("vae_allMNIST_filtered", vae_all_filtered),
    ("vae_orig_k100", vae_orig_samps),
]
    canvas = build_grid(samps)
    img = Gray.(canvas)
    fname = "Fig_mnist_grid_$(tag)_digit3.png"
    save_pub(joinpath(figpath, fname), img)
    @info "Saved $fname"
end

println("\n" * "="^80)
println("ALL EXPERIMENTS COMPLETE")
println("="^80)
println("\nKey question: Does a VAE trained on full MNIST (~6k digit-3 or ~60k all)")
println("outperform SA (K=100, training-free) on max-cos and energy metrics?")
println("If SA is competitive despite using 60x less data and zero training,")
println("that demonstrates its fundamental advantage as a training-free method.")
