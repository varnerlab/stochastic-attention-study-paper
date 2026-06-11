#!/usr/bin/env julia
# ──────────────────────────────────────────────────────────────────────────────
# GMM-PCA Baseline for MNIST
# Protocol:
#   1. PCA: project K=100 normalised patterns X̂ ∈ ℝ⁷⁸⁴ → Z ∈ ℝʳ  (r=50)
#   2. GMM: fit diagonal-covariance GMM with C components via EM in ℝʳ
#   3. Sample S points from the GMM in ℝʳ, reconstruct to ℝ⁷⁸⁴
#   4. Compute novelty / diversity / Hopfield-energy (same functions as SA)
# ──────────────────────────────────────────────────────────────────────────────

@info "Loading environment …"
include(joinpath(@__DIR__, "Include-MNIST.jl"))
using Images, ImageIO
@info "Environment loaded."

# ── Shared experiment parameters (must match run_multidigit_experiment.jl) ────
const number_of_examples  = 100
const number_of_rows      = 28
const number_of_cols      = 28
const number_of_pixels    = number_of_rows * number_of_cols
const β_inv_temp          = 2000.0
const S                   = 150      # samples to draw from GMM
const R_pca               = 50       # PCA components  (r << d=784, r < K=100)
const C_gmm               = 10       # GMM mixture components
const EM_iters            = 500
const EM_tol              = 1e-8

# ── Decode / grid helpers (identical to run_multidigit_experiment.jl) ─────────
function decode(s::Vector{<:Number}; number_of_rows::Int=28, number_of_columns::Int=28)
    X = reshape(s, number_of_rows, number_of_columns) |> X -> transpose(X) |> Matrix
    return replace(X, -1 => 0)
end

function build_grid(samples; nrows=4, ncols=4, gap=2)
    H, W    = 28, 28
    canvas  = zeros(Float64, nrows*H + (nrows-1)*gap, ncols*W + (ncols-1)*gap)
    indices = round.(Int, range(1, length(samples), length=nrows*ncols))
    for idx in 1:(nrows*ncols)
        r  = div(idx-1, ncols);  c  = rem(idx-1, ncols)
        y0 = r*(H+gap)+1;        x0 = c*(W+gap)+1
        img = decode(samples[indices[idx]])
        lo, hi = minimum(img), maximum(img)
        hi > lo && (img = (img .- lo) ./ (hi - lo))
        canvas[y0:y0+H-1, x0:x0+W-1] .= img
    end
    return canvas
end

# ── Column-wise log-sum-exp ───────────────────────────────────────────────────
function logsumexp_cols(A::Matrix{Float64})
    m = maximum(A, dims=1)
    return m .+ log.(sum(exp.(A .- m), dims=1))
end

# ── Diagonal-covariance GMM via EM ────────────────────────────────────────────
# X : r×N  (PCA-projected data, columns = observations)
# Returns (μ, log_σ², π_k)  where μ is r×C, log_σ² is r×C, π_k is length-C
function fit_gmm(X::Matrix{Float64}, C::Int; n_iters=EM_iters, tol=EM_tol, seed=42)
    r, N = size(X)
    rng  = MersenneTwister(seed)

    # K-means++ initialisation
    μ = zeros(r, C)
    first_idx = rand(rng, 1:N)
    μ[:, 1]   = X[:, first_idx]
    for k in 2:C
        dists = [minimum(sum((X[:, n] .- μ[:, j]).^2) for j in 1:(k-1)) for n in 1:N]
        probs = dists ./ sum(dists)
        μ[:, k] = X[:, StatsBase.sample(rng, 1:N, Weights(probs))]
    end

    # Initialise variances from pooled data variance
    global_var = vec(var(X, dims=2))
    log_σ²     = log.(repeat(max.(global_var, 1e-6), 1, C))
    π_k        = ones(C) / C

    log_r        = zeros(C, N)
    log_lik_prev = -Inf

    for iter in 1:n_iters
        # ── E-step ──────────────────────────────────────────────────────────
        σ² = exp.(log_σ²)   # r×C
        for k in 1:C
            diff = X .- μ[:, k]   # r×N
            log_r[k, :] = log(π_k[k]) .-
                           0.5 .* vec(sum(diff.^2 ./ σ²[:, k], dims=1)) .-
                           0.5 .* sum(log.(2π .* σ²[:, k]))
        end
        lse     = logsumexp_cols(log_r)   # 1×N
        log_lik = sum(lse)
        log_r .-= lse
        r_mat   = exp.(log_r)             # C×N

        Δ = abs(log_lik - log_lik_prev) / (abs(log_lik_prev) + 1e-30)
        if Δ < tol
            @info "  EM converged at iteration $iter"
            break
        end
        log_lik_prev = log_lik

        # ── M-step ──────────────────────────────────────────────────────────
        N_k = vec(sum(r_mat, dims=2))   # C
        π_k = max.(N_k / N, 1e-8);  π_k ./= sum(π_k)
        for k in 1:C
            μ[:, k]      = X * r_mat[k, :] / N_k[k]
            diff         = X .- μ[:, k]
            σ²_k         = vec(sum(diff.^2 .* r_mat[k, :]', dims=2)) / N_k[k]
            log_σ²[:, k] = log.(max.(σ²_k, 1e-6))
        end
    end

    return μ, log_σ², π_k
end

# ── Sample from fitted GMM (in PCA space) ────────────────────────────────────
function sample_gmm(μ, log_σ², π_k, n_samples; seed=9999)
    rng  = MersenneTwister(seed)
    r    = size(μ, 1)
    wts  = Weights(π_k)
    samps = Vector{Vector{Float64}}()
    for _ in 1:n_samples
        k = StatsBase.sample(rng, 1:length(π_k), wts)
        push!(samps, μ[:, k] .+ sqrt.(exp.(log_σ²[:, k])) .* randn(rng, r))
    end
    return samps
end

# ── Load MNIST ────────────────────────────────────────────────────────────────
@info "Loading MNIST …"
digits_image_dictionary = MyMNISTHandwrittenDigitImageDataset(number_of_examples = number_of_examples)
@info "MNIST loaded."

# ── Run one digit ─────────────────────────────────────────────────────────────
function run_gmm_pca_digit(digit::Int; digits_image_dictionary, figpath)
    @info "═══  Digit $digit  ═══"

    # Build normalised memory matrix (identical to run_multidigit_experiment.jl)
    ϵ  = 1e-12
    X  = zeros(Float64, number_of_pixels, number_of_examples)
    X̂  = zeros(Float64, number_of_pixels, number_of_examples)
    for i in 1:number_of_examples
        xᵢ = reshape(transpose(digits_image_dictionary[digit][:, :, i]) |> Matrix,
                     number_of_pixels) |> vec
        X[:, i] = xᵢ
    end
    for i in 1:number_of_examples
        lᵢ = norm(X[:, i])
        X̂[:, i] = X[:, i] ./ (lᵢ + ϵ)
    end
    K = size(X̂, 2)
    @info "  Memory matrix: $(size(X̂))  K=$K"

    # ── Step 1: PCA ───────────────────────────────────────────────────────────
    # MultivariateStats.fit(PCA, ...) expects d×N, mean-centres by default
    pca_model = MultivariateStats.fit(PCA, X̂; maxoutdim = R_pca)
    Z = MultivariateStats.transform(pca_model, X̂)   # R_pca × K
    var_explained = round(100 * sum(principalvars(pca_model)) / tvar(pca_model), digits=1)
    @info "  PCA: r=$R_pca components, variance explained = $var_explained%"

    # ── Step 2: Fit GMM in PCA space ─────────────────────────────────────────
    @info "  Fitting GMM (C=$C_gmm) in ℝ^$R_pca …"
    μ_gmm, log_σ²_gmm, π_gmm = fit_gmm(Z, C_gmm; seed = 42 + digit)
    @info "  π_k = $(round.(π_gmm, digits=3))"

    # ── Step 3: Sample in PCA space, reconstruct to ℝ⁷⁸⁴ ────────────────────
    @info "  Sampling $S points …"
    pca_samples = sample_gmm(μ_gmm, log_σ²_gmm, π_gmm, S; seed = 7777 + digit)

    # reconstruct: MultivariateStats.reconstruct maps ℝʳ → ℝ⁷⁸⁴
    gmm_samples = [vec(MultivariateStats.reconstruct(pca_model, z)) for z in pca_samples]
    @info "  Reconstructed to ℝ^$(length(gmm_samples[1]))"

    # ── Step 4: Metrics (same structure as other baselines) ───────────────────
    n_chains          = 30
    samples_per_chain = 5
    function chain_metric_se(samps, metric_fn)
        spc = samples_per_chain
        vals = [metric_fn(samps[(i-1)*spc+1:i*spc]) for i in 1:n_chains]
        return std(vals) / sqrt(n_chains)
    end

    novelty_vals   = [sample_novelty(ξ, X̂) for ξ in gmm_samples]
    energy_vals    = [hopfield_energy(ξ, X̂, β_inv_temp) for ξ in gmm_samples]
    novelty_mean   = mean(novelty_vals)
    diversity_mean = sample_diversity(gmm_samples)
    energy_mean    = mean(energy_vals)
    novelty_se     = chain_metric_se(gmm_samples, g -> mean(sample_novelty(ξ, X̂) for ξ in g))
    diversity_se   = chain_metric_se(gmm_samples, g -> sample_diversity(g))
    energy_se      = chain_metric_se(gmm_samples, g -> mean(hopfield_energy(ξ, X̂, β_inv_temp) for ξ in g))

    fmt(x, d)  = let v = round(x; digits=d); abs(v) == 0.0 ? abs(v) : v end
    fmtpm(v, se, dv, dse) = "\$$(fmt(v,dv)) \\pm $(fmt(se,dse))\$"

    println("\n% --- Digit $digit: GMM-PCA table row (r=$R_pca, C=$C_gmm) ---")
    println("GMM-PCA & $(fmtpm(novelty_mean, novelty_se, 3, 3)) & " *
            "$(fmtpm(diversity_mean, diversity_se, 3, 3)) & " *
            "$(fmtpm(energy_mean, energy_se, 1, 2)) \\\\")
    println("%   var_explained=$(var_explained)%  novelty=$(round(novelty_mean,digits=4))" *
            "  diversity=$(round(diversity_mean,digits=4))  energy=$(round(energy_mean,digits=2))")

    # ── Save grid PNG ─────────────────────────────────────────────────────────
    canvas = build_grid(gmm_samples)
    fname  = "Fig_mnist_grid_gmm_pca_digit$(digit).png"
    save_pub(joinpath(figpath, fname), Gray.(canvas))
    @info "  Saved $fname"

    return (novelty=novelty_mean, novelty_se=novelty_se,
            diversity=diversity_mean, diversity_se=diversity_se,
            energy=energy_mean, energy_se=energy_se,
            var_explained=var_explained)
end

# ── Main ──────────────────────────────────────────────────────────────────────
figpath = _PATH_TO_FIG
mkpath(figpath)

r3 = run_gmm_pca_digit(3; digits_image_dictionary, figpath)
r1 = run_gmm_pca_digit(1; digits_image_dictionary, figpath)
r8 = run_gmm_pca_digit(8; digits_image_dictionary, figpath)

println("\n══════════════════════════════════════════════════════")
println("FINAL GMM-PCA RESULTS  (r=$R_pca, C=$C_gmm)")
println("══════════════════════════════════════════════════════")
for (d, r) in [(3,r3),(1,r1),(8,r8)]
    println("Digit $d  var_exp=$(r.var_explained)%  " *
            "novelty=$(round(r.novelty,digits=4))  " *
            "diversity=$(round(r.diversity,digits=4))  " *
            "energy=$(round(r.energy,digits=2))")
end
println("\nGrid PNGs saved to: $figpath")
