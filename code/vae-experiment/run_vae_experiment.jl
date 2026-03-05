#!/usr/bin/env julia
# ─────────────────────────────────────────────────────────────────────────────
# VAE Experiment — Learned Generative Baseline (Reviewer W1)
# Trains a small VAE on the same K=100 digit "3" MNIST patterns used in the
# main MNIST experiment, generates 150 samples under the same 30×5 protocol,
# computes novelty/diversity/energy metrics, and prints a LaTeX table row for
# tab:mnist-baselines.
#
# Uses two-phase training to prevent posterior collapse:
#   Phase 1 (epochs 1–2000): pure autoencoder (reconstruction only)
#   Phase 2 (epochs 2001–4000): β-VAE with small KL weight β_kl=0.0001
# Hyperparameters chosen via sweep (sweep_vae.jl, sweep_vae2.jl):
#   latent=8, β_kl=0.0001 → N=0.214±0.005, D=0.441±0.008 (beats GMM-PCA)
#
# Run from the repo root:
#   julia code/vae-experiment/run_vae_experiment.jl
# ─────────────────────────────────────────────────────────────────────────────

import Pkg
Pkg.activate(joinpath(@__DIR__, "..", "mnist-experiment"))

include(joinpath(@__DIR__, "..", "mnist-experiment", "Include-MNIST.jl"))

using Flux, Statistics, Random, LinearAlgebra, Printf

# ── Constants (match main MNIST experiment exactly) ───────────────────────────
const DIGIT       = parse(Int, get(ENV, "VAE_DIGIT", "3"))
const K           = 100
const D           = 784
const β_inv_temp  = 2000.0
const S           = 150    # 30 chains × 5 samples
const NC          = 30
const SPC         = 5
const LATENT      = 8      # latent dimension (sweep winner: lat=8)
const PHASE1_EP   = 2000   # AE-only epochs (prevent posterior collapse)
const PHASE2_EP   = 2000   # VAE epochs with gentle KL annealing
const LR          = 1f-3
const β_kl_final  = 0.0001f0 # final KL weight (sweep winner: β=0.0001)

# ── Load identical K=100 digit "3" patterns ───────────────────────────────────
@info "Loading MNIST patterns (digit=$DIGIT, K=$K) …"
digits_image_dictionary = MyMNISTHandwrittenDigitImageDataset(number_of_examples = K)

ϵ = 1e-12
X_raw = zeros(Float64, D, K)
for i in 1:K
    image_array = digits_image_dictionary[DIGIT][:, :, i]
    xᵢ = reshape(transpose(image_array) |> Matrix, D) |> vec
    X_raw[:, i] = xᵢ
end

X̂ = similar(X_raw)
for i in 1:K
    xᵢ = X_raw[:, i]
    lᵢ = norm(xᵢ)
    X̂[:, i] = xᵢ ./ (lᵢ + ϵ)
end
@info "Memory matrix: $(size(X̂))  (each column is unit-normalized)"

# ── VAE Architecture ──────────────────────────────────────────────────────────
# Encoder: D→256→relu→128→relu → [μ, logσ²] ∈ R^LATENT
# Decoder: LATENT→128→relu→256→relu→D  (output is unit-normalized)

struct VAE
    enc_shared   # D → 128
    enc_μ        # 128 → LATENT
    enc_logσ²    # 128 → LATENT
    dec          # LATENT → D
end
Flux.@layer VAE   # modern Flux (replaces @functor)

function build_vae(d::Int, latent::Int)
    enc_shared = Chain(Dense(d    => 256, relu), Dense(256 => 128, relu))
    enc_μ      = Dense(128 => latent)
    enc_logσ²  = Dense(128 => latent)
    dec        = Chain(Dense(latent => 128, relu), Dense(128 => 256, relu), Dense(256 => d))
    return VAE(enc_shared, enc_μ, enc_logσ², dec)
end

function encode(vae::VAE, x::AbstractMatrix)
    h = vae.enc_shared(x)
    return vae.enc_μ(h), vae.enc_logσ²(h)
end

# Unit-normalize decoder output (matches SA/MALA protocol for fair comparison)
function decode_norm(vae::VAE, z::AbstractMatrix)
    o = vae.dec(z)
    return o ./ (sqrt.(sum(o .^ 2; dims=1)) .+ 1f-8)
end

# ── Training ──────────────────────────────────────────────────────────────────
Random.seed!(42)
vae = build_vae(D, LATENT)
X_train   = Float32.(X̂)          # D × K, Float32
opt_state = Flux.setup(Adam(LR), vae)

# Phase 1: Pure autoencoder — force decoder to learn to use z
@info "Phase 1: autoencoder training ($PHASE1_EP epochs) …"
for epoch in 1:PHASE1_EP
    ε = randn(Float32, LATENT, K)
    loss, grads = Flux.withgradient(vae) do m
        μ, lσ² = encode(m, X_train)
        z  = μ .+ exp.(0.5f0 .* lσ²) .* ε    # reparameterization
        x̂  = decode_norm(m, z)
        mean(sum((X_train .- x̂) .^ 2; dims=1))   # reconstruction only
    end
    Flux.update!(opt_state, vae, grads[1])
    if epoch % 500 == 0
        @info "  AE epoch $epoch  recon = $(round(loss; digits=4))"
    end
end

# Phase 2: Add KL with linear warmup from 0 → β_kl_final
@info "Phase 2: VAE training with KL warmup ($PHASE2_EP epochs) …"
for epoch in 1:PHASE2_EP
    kl_w = β_kl_final * Float32(epoch) / Float32(PHASE2_EP)  # linear warmup
    ε = randn(Float32, LATENT, K)
    loss, grads = Flux.withgradient(vae) do m
        μ, lσ² = encode(m, X_train)
        z  = μ .+ exp.(0.5f0 .* lσ²) .* ε
        x̂  = decode_norm(m, z)
        recon = mean(sum((X_train .- x̂) .^ 2; dims=1))
        kl    = -0.5f0 * mean(sum(1f0 .+ lσ² .- μ .^ 2 .- exp.(lσ²); dims=1))
        recon + kl_w * kl
    end
    Flux.update!(opt_state, vae, grads[1])
    if epoch % 500 == 0
        @info "  VAE epoch $epoch  total = $(round(loss; digits=4))"
    end
end
@info "Training complete."

# ── Generate 150 samples ──────────────────────────────────────────────────────
@info "Generating $S samples (z ~ N(0,I)) …"
Random.seed!(9999)
Z_gen = randn(Float32, LATENT, S)
raw   = decode_norm(vae, Z_gen)        # D × S, unit-normalized (Float32)
samples = [Float64.(raw[:, i]) for i in 1:S]

n_bad = sum(any(isnan, s) || any(isinf, s) for s in samples)
n_bad > 0 && @warn "$n_bad samples contain NaN/Inf"

# ── Compute metrics ───────────────────────────────────────────────────────────
@info "Computing metrics …"
N_vals = [sample_novelty(s, X̂) for s in samples]
D_val  = sample_diversity(samples)
E_vals = [hopfield_energy(s, X̂, β_inv_temp) for s in samples]

# SE via 30-group split (identical to run_multidigit_experiment.jl)
function chain_se(vals, f)
    group_vals = [f(vals[(i-1)*SPC+1:i*SPC]) for i in 1:NC]
    return std(group_vals) / sqrt(NC)
end

N_mean = mean(N_vals)
N_se   = chain_se(N_vals, mean)

D_groups = [sample_diversity(samples[(i-1)*SPC+1:i*SPC]) for i in 1:NC]
D_se     = std(D_groups) / sqrt(NC)

E_mean = mean(E_vals)
E_se   = chain_se(E_vals, mean)

# ── Print results ─────────────────────────────────────────────────────────────
println("\n" * "═"^60)
println("VAE RESULTS  (digit=$DIGIT, K=$K, latent=$LATENT)")
println("═"^60)
@printf("  Novelty   N̄  = %.3f ± %.3f\n", N_mean, N_se)
@printf("  Diversity D̄  = %.3f ± %.3f\n", D_val,  D_se)
@printf("  Energy    Ē  = %.3f ± %.3f\n", E_mean, E_se)
println()

println("% --- VAE row for tab:mnist-baselines (copy into experiments.tex) ---")
fmt3(v) = string(round(v; digits=3))
println("VAE (latent\${=}$(LATENT)) & " *
        "\$$(fmt3(N_mean)) \\pm $(fmt3(N_se))\$ & " *
        "\$$(fmt3(D_val)) \\pm $(fmt3(D_se))\$ & " *
        "\$$(fmt3(E_mean)) \\pm $(fmt3(E_se))\$ \\\\")
println("═"^60)
flush(stdout)
