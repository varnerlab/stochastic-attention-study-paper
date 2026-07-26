#!/usr/bin/env julia
# ──────────────────────────────────────────────────────────────────────────────
# Exact ancestral sampling vs. SA on Pfam PF00076 (Reviewer AQY1).
#
# The Boltzmann target of the modern Hopfield energy is exactly
#
#     p_β(ξ) = Σ_i w_i N(ξ; m_i, β⁻¹I),   w_i ∝ exp(β‖m_i‖²/2),
#
# and the PCA memories are ℓ₂-normalised, so w_i = 1/K. This script asks the
# question the novelty/diversity closed form cannot answer: do the *decoded*
# sequence statistics (per-position KL, pairwise mutual information, HMM
# acceptance) also match, given that decoding is a nonlinear inverse-PCA +
# argmax that no analytic formula covers?
#
# Helper functions, memory construction, decoding, and metrics are taken
# verbatim from run_protein_hmm_validation.jl (the script that produced
# Table 2) so the comparison is exactly like-for-like.
# ──────────────────────────────────────────────────────────────────────────────

# ── Load the Table 2 script's helpers, but not its main() driver ─────────────
const _VALIDATION_SCRIPT = joinpath(@__DIR__, "run_protein_hmm_validation.jl")
let lines = readlines(_VALIDATION_SCRIPT)
    stop = findfirst(l -> startswith(l, "function main()"), lines)
    stop === nothing && error("could not locate main() boundary in $_VALIDATION_SCRIPT")
    include_string(Main, join(lines[1:stop-1], "\n"), _VALIDATION_SCRIPT)
end

"""
Draw S independent samples from p_β = Σ_i (1/K) N(m_i, β⁻¹I).
`restrict`/`per_component` mirror SA's 30-chain warm-start protocol.
"""
function exact_ancestral(X̂::Matrix{Float64}, β::Float64, S::Int;
                         seed::Int=2026, restrict=nothing, per_component=nothing)
    Random.seed!(seed)
    d, K = size(X̂)
    σ = 1.0 / sqrt(β)
    pool = restrict === nothing ? collect(1:K) : collect(restrict)
    idx = per_component === nothing ? [rand(pool) for _ in 1:S] :
                                      vcat([fill(k, per_component) for k in pool]...)
    return [X̂[:, i] .+ σ .* randn(d) for i in idx]
end

function knn_perturbation(X̂::Matrix{Float64}, S::Int; k::Int=5, σ::Float64=0.0, seed::Int=2026)
    Random.seed!(seed)
    d, K = size(X̂)
    C = X̂' * X̂
    out = Vector{Vector{Float64}}()
    for _ in 1:S
        i = rand(1:K)
        nbrs = sortperm(C[:, i], rev=true)[2:min(k+1, K)]
        j = rand(nbrs); t = rand()
        v = (1 - t) .* X̂[:, i] .+ t .* X̂[:, j]
        σ > 0 && (v .+= σ .* randn(d))
        push!(out, v)
    end
    return out
end

function driver()
    datapath = _PATH_TO_DATA
    mkpath(datapath)

    # ── Steps 1-2: identical to run_protein_hmm_validation.jl main() ──────────
    @info "Step 1: Loading Pfam $PFAM_ID alignment …"
    sto_file = download_pfam_seed(PFAM_ID)
    raw_seqs = parse_stockholm(sto_file)
    isempty(raw_seqs) && (raw_seqs = parse_fasta(sto_file))
    char_mat, seq_names = clean_alignment(raw_seqs)
    K_total, L = size(char_mat)
    if K_total > K_MAX
        Random.seed!(42)
        keep = StatsBase.sample(1:K_total, K_MAX, replace=false) |> sort
        char_mat = char_mat[keep, :]; seq_names = seq_names[keep]
    end
    K = size(char_mat, 1)
    stored_seqs = [String(char_mat[k, :]) for k in 1:K]
    @info "  Using K=$K sequences, L=$L positions"

    @info "Step 2: Encoding …"
    X_onehot = onehot_encode(char_mat)
    pca_model = MultivariateStats.fit(PCA, X_onehot; pratio=0.95)
    Z = MultivariateStats.transform(pca_model, X_onehot)
    ϵ = 1e-12
    X̂ = copy(Z)
    for k in 1:K
        X̂[:, k] ./= (norm(X̂[:, k]) + ϵ)
    end
    d = size(X̂, 1)
    @info "  PCA: $(size(X_onehot,1)) → $d dimensions"
    decode_sample(ξ) = decode_onehot(vec(MultivariateStats.reconstruct(pca_model, ξ)), L)

    # ── Step 3: same operating temperatures ──────────────────────────────────
    pt = find_entropy_inflection(X̂; α=α_step)
    β_ret = Float64(round(Int, 20 * pt.β_star))
    β_gen = Float64(round(Int, 2 * pt.β_star))
    @info "  β* = $(round(pt.β_star, digits=3)) → β_ret=$β_ret, β_gen=$β_gen"

    # ── Step 4: SA (reproduce Table 2 rows) + exact ancestral ────────────────
    function run_multichain(X̂, β)
        samples = Vector{Vector{Float64}}()
        Random.seed!(42)
        pattern_indices = StatsBase.sample(1:size(X̂,2), n_chains, replace=(n_chains > size(X̂,2)))
        for (c, k) in enumerate(pattern_indices)
            Random.seed!(12345 + c)
            sₒ = X̂[:, k] .+ σ_init .* randn(size(X̂,1))
            (_, Ξ) = sample(X̂, sₒ, T_per_chain; β=β, α=α_step, seed=12345+c)
            pool = [Ξ[tᵢ, :] for tᵢ in (T_burnin+1):thin_interval:T_per_chain]
            idxs = round.(Int, range(1, length(pool), length=min(samples_per_chain, length(pool))))
            for idx in idxs; push!(samples, pool[idx]); end
        end
        return samples, pattern_indices
    end

    @info "Step 4: Running samplers …"
    @info "  SA (retrieval, β=$β_ret) …"
    sa_ret, sa_idx = run_multichain(X̂, β_ret)
    @info "  SA (generation, β=$β_gen) …"
    sa_gen, _ = run_multichain(X̂, β_gen)

    @info "  Exact ancestral …"
    methods = [
        ("Bootstrap",                        [copy(X̂[:, rand(1:K)]) for _ in 1:S]),
        ("SA_ret_b$(Int(β_ret))",            sa_ret),
        ("SA_gen_b$(Int(β_gen))",            sa_gen),
        ("Exact_ret_b$(Int(β_ret))_allK",    exact_ancestral(X̂, β_ret, S; seed=2026)),
        ("Exact_ret_b$(Int(β_ret))_matched", exact_ancestral(X̂, β_ret, S; seed=2026,
                                                 restrict=sa_idx, per_component=samples_per_chain)),
        ("Exact_gen_b$(Int(β_gen))_allK",    exact_ancestral(X̂, β_gen, S; seed=2026)),
        ("Exact_gen_b$(Int(β_gen))_matched", exact_ancestral(X̂, β_gen, S; seed=2026,
                                                 restrict=sa_idx, per_component=samples_per_chain)),
        ("kNN_k5_sigma_bgen",                knn_perturbation(X̂, S; k=5, σ=1/sqrt(β_gen), seed=2026)),
    ]

    # ── Step 5: decode ───────────────────────────────────────────────────────
    @info "Step 5: Decoding …"
    decoded = Dict(name => [decode_sample(ξ) for ξ in samps] for (name, samps) in methods)

    # ── Step 6-7: HMMER ──────────────────────────────────────────────────────
    @info "Step 6: HMMER validation …"
    fasta_dir = joinpath(datapath, "exact_mixture_validation")
    mkpath(fasta_dir)
    hmm_file = download_pfam_hmm(PFAM_ID)
    hmm_results = Dict{String,Any}()
    for (name, _) in methods
        fpath = joinpath(fasta_dir, "$(name).fasta")
        write_fasta(fpath, decoded[name], name)
        hmm_results[name] = run_hmmsearch(hmm_file, fpath)
    end

    # ── Step 8: summary ──────────────────────────────────────────────────────
    MI_stored, _ = pairwise_mutual_information(stored_seqs, L)

    println("\n" * "="^118)
    println("EXACT ANCESTRAL SAMPLING vs. SA  —  Pfam PF00076 (K=$K, L=$L, d=$d)")
    println("="^118)
    @printf("%-34s | %-6s | %-6s | %-7s | %-8s | %-6s | %-6s | %s\n",
            "Method", "AA_KL", "Pos_KL", "MI_corr", "HMM_pass", "SeqID", "Nov", "Div")
    println("-"^118)
    for (name, samps) in methods
        seqs = decoded[name]
        kl_aa = aa_composition_kl(seqs, stored_seqs)
        mean_pos_kl, _ = position_specific_kl(seqs, stored_seqs, L)
        MI_gen, _ = pairwise_mutual_information(seqs, L)
        r_mi = mi_correlation(MI_gen, MI_stored)
        hmm = hmm_results[name]
        hmm_pct = 100.0 * hmm.n_hits / max(hmm.n_total, 1)
        seq_ids = [nearest_sequence_identity(s, stored_seqs) for s in seqs]
        @printf("%-34s | %.4f | %.4f | %.4f  | %6.1f%%  | %.3f  | %.4f | %.4f\n",
                name, kl_aa, mean_pos_kl, r_mi, hmm_pct, mean(seq_ids),
                mean(sample_novelty(ξ, X̂) for ξ in samps), sample_diversity(samps))
    end
    println("-"^118)
    println("Published Table 2 rows:")
    println("  Bootstrap (replay)          N=0.000  SeqID=0.644  KL=0.143  Pos-KL=7.52  MI r=0.692  HMM=100%")
    println("  VAE (latent=8)              N=0.621  SeqID=0.532  KL=0.416  Pos-KL=9.99  MI r=0.525  HMM=100%")
    println("  SA (β=77, retrieval)        N=0.243  SeqID=0.616  KL=0.107  Pos-KL=5.66  MI r=0.733  HMM=100%")
    println("  SA (β=8,  generation)       N=0.623  SeqID=0.538  KL=0.060  Pos-KL=2.92  MI r=0.871  HMM=100%")
    println("="^118)
end

driver()
