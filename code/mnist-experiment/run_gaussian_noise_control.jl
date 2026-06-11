#!/usr/bin/env julia
# ──────────────────────────────────────────────────────────────────────────────
# Gaussian Noise Control Experiment  (addresses reviewer W4)
#
# The reviewer asks: "At β=200, can a human observer distinguish SA-generated
# digits from Gaussian noise of the same per-pixel variance?"
#
# This script:
#   1. Runs SA at β=200 (the "generation" regime) on MNIST digit 3
#   2. Generates Gaussian noise matched to SA's per-pixel mean and variance
#   3. Generates isotropic Gaussian noise matched to SA's norm distribution
#   4. Computes all standard metrics (novelty, diversity, energy, max-cos)
#   5. Computes pixel-space Fréchet distance to stored patterns
#   6. Saves visual grids for side-by-side comparison
# ──────────────────────────────────────────────────────────────────────────────

@info "Loading environment …"
include(joinpath(@__DIR__, "Include-MNIST.jl"))
@info "Environment loaded."

# ── decode + grid helpers ────────────────────────────────────────────────────
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

# ── Experiment parameters ────────────────────────────────────────────────────
const DIGIT = 3
const K = 100
const D = 784
const β_generation = 200.0      # the "generation" regime from the paper
const β_retrieval  = 2000.0     # for energy computation at retrieval scale
const α_step = 0.01
const n_chains = 30
const T_per_chain = 5000
const T_burnin = 2000
const thin_interval = 100
const samples_per_chain = 5
const σ_init = 0.01
const S = 150                   # total samples for non-SA baselines

# ── Load MNIST digit 3 ──────────────────────────────────────────────────────
@info "Loading MNIST digit $DIGIT …"
digits_image_dictionary = MyMNISTHandwrittenDigitImageDataset(number_of_examples = K)

ϵ = 1e-12
X̂ = zeros(Float64, D, K)
for i in 1:K
    image_array = digits_image_dictionary[DIGIT][:, :, i]
    xᵢ = reshape(transpose(image_array) |> Matrix, D) |> vec
    lᵢ = norm(xᵢ)
    X̂[:, i] = xᵢ ./ (lᵢ + ϵ)
end
@info "Memory matrix: $(size(X̂))"

# ── 1. Run SA at β=200 (generation regime) ───────────────────────────────────
@info "Running SA at β=$β_generation (generation regime) …"
sa_samples = Vector{Vector{Float64}}()
Random.seed!(42)
pattern_indices = StatsBase.sample(1:K, n_chains, replace=(n_chains > K))
for (c, k) in enumerate(pattern_indices)
    Random.seed!(12345 + c)
    sₒ = X̂[:, k] .+ σ_init .* randn(D)
    (_, Ξ) = sample(X̂, sₒ, T_per_chain; β=β_generation, α=α_step, seed=12345+c)
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
@info "SA: $(length(sa_samples)) samples"

# ── 2. Compute SA sample statistics ─────────────────────────────────────────
sa_matrix = hcat(sa_samples...)  # D × n_samples
sa_pixel_mean = vec(mean(sa_matrix, dims=2))
sa_pixel_var  = vec(var(sa_matrix, dims=2))
sa_norms = [norm(s) for s in sa_samples]
sa_mean_norm = mean(sa_norms)
sa_std_norm  = std(sa_norms)
@info "SA pixel variance: mean=$(round(mean(sa_pixel_var), digits=6)), " *
      "min=$(round(minimum(sa_pixel_var), digits=6)), max=$(round(maximum(sa_pixel_var), digits=6))"
@info "SA norms: mean=$(round(sa_mean_norm, digits=4)), std=$(round(sa_std_norm, digits=4))"

# ── 3. Generate Gaussian noise controls ──────────────────────────────────────

# Control A: matched per-pixel mean and variance (strongest possible Gaussian)
@info "Generating matched Gaussian noise (per-pixel μ and σ²) …"
Random.seed!(7777)
gauss_matched_samples = Vector{Vector{Float64}}()
for _ in 1:S
    g = sa_pixel_mean .+ sqrt.(sa_pixel_var) .* randn(D)
    push!(gauss_matched_samples, g)
end

# Control B: isotropic Gaussian, matched norm (what you'd get from random directions)
@info "Generating isotropic Gaussian noise (matched norm) …"
Random.seed!(8888)
gauss_iso_samples = Vector{Vector{Float64}}()
for _ in 1:S
    g = randn(D)
    g = g ./ norm(g) .* sa_mean_norm  # match the norm of SA samples
    push!(gauss_iso_samples, g)
end

# Also get stored patterns as reference
Random.seed!(12345)
stored_samples = [copy(X̂[:, rand(1:K)]) for _ in 1:S]

# ── 4. Compute metrics ──────────────────────────────────────────────────────
function compute_all_metrics(samples, X̂, β; label="")
    novelty_vals  = [sample_novelty(ξ, X̂) for ξ in samples]
    energy_vals   = [hopfield_energy(ξ, X̂, β) for ξ in samples]
    maxcos_vals   = [nearest_cosine_similarity(ξ, X̂) for ξ in samples]
    norms         = [norm(ξ) for ξ in samples]

    n_mean = mean(novelty_vals)
    d_mean = sample_diversity(samples)
    e_mean = mean(energy_vals)
    mc_mean = mean(maxcos_vals)
    norm_mean = mean(norms)

    @info "  $label: N=$(round(n_mean,digits=3)), D=$(round(d_mean,digits=3)), " *
          "E=$(round(e_mean,digits=2)), max-cos=$(round(mc_mean,digits=3)), " *
          "‖ξ‖=$(round(norm_mean,digits=3))"

    return (novelty=n_mean, diversity=d_mean, energy=e_mean,
            maxcos=mc_mean, norm=norm_mean,
            novelty_vals=novelty_vals, energy_vals=energy_vals,
            maxcos_vals=maxcos_vals)
end

@info "Computing metrics …"
m_sa      = compute_all_metrics(sa_samples, X̂, β_generation; label="SA β=200")
m_matched = compute_all_metrics(gauss_matched_samples, X̂, β_generation; label="Gauss(matched)")
m_iso     = compute_all_metrics(gauss_iso_samples, X̂, β_generation; label="Gauss(iso)")
m_stored  = compute_all_metrics(stored_samples, X̂, β_generation; label="Stored")

# ── 5. Pixel-space Fréchet distance ─────────────────────────────────────────
# FD = ‖μ₁ - μ₂‖² + Tr(Σ₁ + Σ₂ - 2(Σ₁Σ₂)^{1/2})
# For high-d with limited samples, we use a diagonal approximation:
# FD_diag = ‖μ₁ - μ₂‖² + Σ(σ₁² + σ₂² - 2σ₁σ₂)
#         = ‖μ₁ - μ₂‖² + ‖σ₁ - σ₂‖²
function frechet_distance_diag(samples_a, samples_b)
    A = hcat(samples_a...)
    B = hcat(samples_b...)
    μ_a = vec(mean(A, dims=2)); μ_b = vec(mean(B, dims=2))
    σ_a = vec(std(A, dims=2));  σ_b = vec(std(B, dims=2))
    mean_term = sum((μ_a .- μ_b).^2)
    cov_term  = sum((σ_a .- σ_b).^2)
    return mean_term + cov_term
end

@info "Computing pixel-space Fréchet distances (diagonal approx.) …"
fd_sa      = frechet_distance_diag(stored_samples, sa_samples)
fd_matched = frechet_distance_diag(stored_samples, gauss_matched_samples)
fd_iso     = frechet_distance_diag(stored_samples, gauss_iso_samples)

@info "  FD(stored, SA):            $(round(fd_sa, digits=2))"
@info "  FD(stored, Gauss-matched): $(round(fd_matched, digits=2))"
@info "  FD(stored, Gauss-iso):     $(round(fd_iso, digits=2))"

# ── 6. Print results table ──────────────────────────────────────────────────
println("\n" * "═"^90)
println("GAUSSIAN NOISE CONTROL — MNIST digit $DIGIT, β=$β_generation (generation regime)")
println("═"^90)

header = ["Method", "Novelty", "Diversity", "Energy", "Max-cos", "‖ξ‖", "FD(diag)"]
rows_data = [
    ("Stored patterns",          m_stored.novelty,  m_stored.diversity,  m_stored.energy,  m_stored.maxcos,  m_stored.norm,  0.0),
    ("SA (β=200)",               m_sa.novelty,      m_sa.diversity,      m_sa.energy,      m_sa.maxcos,      m_sa.norm,      fd_sa),
    ("Gaussian (matched μ,σ²)",  m_matched.novelty, m_matched.diversity, m_matched.energy, m_matched.maxcos, m_matched.norm, fd_matched),
    ("Gaussian (iso, matched ‖·‖)", m_iso.novelty,  m_iso.diversity,     m_iso.energy,     m_iso.maxcos,     m_iso.norm,     fd_iso),
]

df = DataFrame(
    Method    = [r[1] for r in rows_data],
    Novelty   = [round(r[2], digits=3) for r in rows_data],
    Diversity = [round(r[3], digits=3) for r in rows_data],
    Energy    = [round(r[4], digits=2) for r in rows_data],
    MaxCos    = [round(r[5], digits=3) for r in rows_data],
    Norm      = [round(r[6], digits=3) for r in rows_data],
    FD_diag   = [round(r[7], digits=2) for r in rows_data],
)
println(df)

# ── 7. LaTeX table ──────────────────────────────────────────────────────────
println("\n% --- LaTeX table (copy into paper) ---")
println("\\begin{tabular}{@{}lcccccc@{}}")
println("\\toprule")
println("Method & \$\\mathcal{N}\$ \$\\uparrow\$ & \$\\bar{\\mathcal{D}}\$ \$\\uparrow\$ & " *
        "\$\\bar{E}\$ \$\\downarrow\$ & Max-\$\\cos\$ & \$\\|\\boldsymbol{\\xi}\\|\$ & " *
        "FD\$_{\\text{diag}}\$ \$\\downarrow\$ \\\\")
println("\\midrule")
for r in rows_data
    name = r[1]
    if contains(name, "SA")
        name = "\\textbf{SA (\$\\beta{=}200\$)}"
    end
    println("$(name) & $(round(r[2],digits=3)) & $(round(r[3],digits=3)) & " *
            "$(round(r[4],digits=2)) & $(round(r[5],digits=3)) & " *
            "$(round(r[6],digits=3)) & $(round(r[7],digits=1)) \\\\")
end
println("\\bottomrule")
println("\\end{tabular}")

# ── 8. Save visual grids ────────────────────────────────────────────────────
@info "Saving visual comparison grids …"
figpath = _PATH_TO_FIG
mkpath(figpath)

grids = [
    ("stored",        stored_samples),
    ("sa_beta200",    sa_samples),
    ("gauss_matched", gauss_matched_samples),
    ("gauss_iso",     gauss_iso_samples),
]
for (tag, samps) in grids
    canvas = build_grid(samps)
    img = Images.Gray.(canvas)
    fname = "Fig_noise_control_$(tag).png"
    save_pub(joinpath(figpath, fname), img)
    @info "  Saved $fname"
end

# ── 9. Combined comparison figure ────────────────────────────────────────────
@info "Generating combined comparison figure …"

# Panel: side-by-side grids
p_stored  = heatmap(build_grid(stored_samples)', c=:grays, yflip=true, axis=false,
                     title="Stored", titlefontsize=9, colorbar=false, aspect_ratio=:equal)
p_sa      = heatmap(build_grid(sa_samples)', c=:grays, yflip=true, axis=false,
                     title="SA (β=200)", titlefontsize=9, colorbar=false, aspect_ratio=:equal)
p_matched = heatmap(build_grid(gauss_matched_samples)', c=:grays, yflip=true, axis=false,
                     title="Gauss (matched)", titlefontsize=9, colorbar=false, aspect_ratio=:equal)
p_iso     = heatmap(build_grid(gauss_iso_samples)', c=:grays, yflip=true, axis=false,
                     title="Gauss (iso)", titlefontsize=9, colorbar=false, aspect_ratio=:equal)

p_grids = plot(p_stored, p_sa, p_matched, p_iso, layout=(1, 4), size=(900, 250), margin=2Plots.mm)
savefig(p_grids, joinpath(figpath, "Fig_noise_control_grids.pdf"))

# Panel: max-cos histograms (the key distinguishing metric)
p_hist = histogram(m_sa.maxcos_vals, bins=20, alpha=0.6, label="SA (β=200)",
                   xlabel="Max cosine similarity to nearest stored pattern",
                   ylabel="Count", title="Structural Similarity to Stored Patterns",
                   titlefontsize=10, legend=:topright, size=(500, 350))
histogram!(m_matched.maxcos_vals, bins=20, alpha=0.6, label="Gauss (matched)")
histogram!(m_iso.maxcos_vals, bins=20, alpha=0.6, label="Gauss (iso)")
vline!([mean(m_stored.maxcos_vals)], lw=2, ls=:dash, color=:black, label="Stored mean")
savefig(p_hist, joinpath(figpath, "Fig_noise_control_maxcos_hist.pdf"))

# Combined
p_combined = plot(p_grids, p_hist, layout=@layout([a{0.45h}; b]), size=(900, 650), margin=3Plots.mm)
savefig(p_combined, joinpath(figpath, "Fig_noise_control_combined.pdf"))

@info "All figures saved."
println("\nDone.")
