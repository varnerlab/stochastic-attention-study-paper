#!/usr/bin/env julia
# Diagnostic: what actually drives the protein decode at generation temperature?
#
# Two competing explanations for why raising the temperature increases novelty
# while improving per-position fidelity:
#   (A) samples "sit between several aligned family members" (convex-interpolation story)
#   (B) the PCA-space perturbation swamps the memory, so inverse PCA is dominated
#       by the empirical mean and per-position argmax falls back on consensus
# (B) is testable: measure how much of the reconstruction comes from the PCA mean,
# and how close decoded sequences land to the family consensus.

const _VALIDATION_SCRIPT = joinpath(@__DIR__, "run_protein_hmm_validation.jl")
let lines = readlines(_VALIDATION_SCRIPT)
    stop = findfirst(l -> startswith(l, "function main()"), lines)
    include_string(Main, join(lines[1:stop-1], "\n"), _VALIDATION_SCRIPT)
end

function driver()
    sto = download_pfam_seed(PFAM_ID)
    raw = parse_stockholm(sto); isempty(raw) && (raw = parse_fasta(sto))
    char_mat, _ = clean_alignment(raw)
    K_total, L = size(char_mat)
    if K_total > K_MAX
        Random.seed!(42)
        char_mat = char_mat[sort(StatsBase.sample(1:K_total, K_MAX, replace=false)), :]
    end
    K = size(char_mat, 1)
    stored_seqs = [String(char_mat[k, :]) for k in 1:K]

    X_onehot = onehot_encode(char_mat)
    pca_model = MultivariateStats.fit(PCA, X_onehot; pratio=0.95)
    Z = MultivariateStats.transform(pca_model, X_onehot)
    X̂ = copy(Z); for k in 1:K; X̂[:, k] ./= (norm(X̂[:, k]) + 1e-12); end
    d = size(X̂, 1)
    decode_sample(ξ) = decode_onehot(vec(MultivariateStats.reconstruct(pca_model, ξ)), L)

    # family consensus: most frequent residue at each position
    consensus = String([begin
        col = [stored_seqs[k][p] for k in 1:K if stored_seqs[k][p] != '-']
        isempty(col) ? '-' : sort(collect(Set(col)), by=c->-count(==(c), col))[1]
    end for p in 1:L])

    # what the decoder returns for a pure-mean input (zero in PCA space)
    mean_decode = decode_sample(zeros(d))
    μ = MultivariateStats.mean(pca_model)

    println("="^92)
    println("DECODE MECHANISM DIAGNOSTIC  (K=$K, L=$L, d=$d)")
    println("="^92)
    @printf("PCA mean norm ||mu|| = %.3f ; a unit-norm memory maps to a PCA-space vector of norm 1\n", norm(μ))
    @printf("Decoding the PCA-space origin (pure mean) gives a sequence %.1f%% identical to consensus\n",
            100*mean(collect(mean_decode) .== collect(consensus)))
    println()
    @printf("%-26s | %-9s | %-9s | %-11s | %s\n",
            "Sampler", "sigma", "||noise||", "to nearest", "to consensus")
    println("-"^92)

    for (name, β) in [("Exact ancestral, b=77", 77.0), ("Exact ancestral, b=8", 8.0)]
        σ = 1/sqrt(β)
        Random.seed!(2026)
        samps = [X̂[:, rand(1:K)] .+ σ .* randn(d) for _ in 1:S]
        seqs  = [decode_sample(ξ) for ξ in samps]
        id_near = mean(nearest_sequence_identity(s, stored_seqs) for s in seqs)
        id_cons = mean(mean(collect(s) .== collect(consensus)) for s in seqs)
        @printf("%-26s | %.4f    | %.2f      | %.3f       | %.3f\n",
                name, σ, σ*sqrt(d), id_near, id_cons)
    end

    # stored sequences themselves, for reference
    @printf("%-26s | %-9s | %-9s | %-11s | %.3f\n", "Stored sequences", "-", "-", "-",
            mean(mean(collect(s) .== collect(consensus)) for s in stored_seqs))
    println("-"^92)
    println("A unit-norm memory has PCA-space norm 1. At b=8 the noise norm is sigma*sqrt(d),")
    println("so if that exceeds 1 the memory identity is swamped and the reconstruction is")
    println("dominated by the mean, pushing argmax toward consensus.")
end

driver()
