#!/usr/bin/env julia
# Test suite for Stochastic Attention codebase
# Verifies core algorithms, utilities, and experiment entry points

println("=" ^ 60)
println("Stochastic Attention Test Suite")
println("=" ^ 60)

# ── Setup ──────────────────────────────────────────────────────────
println("\n[1/6] Loading core environment...")
t0 = time()
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
include(joinpath(@__DIR__, "..", "Include.jl"))
println("  Loaded in $(round(time() - t0, digits=1))s")

passed = 0
failed = 0

macro test(name, expr)
    quote
        try
            result = $(esc(expr))
            if result
                println("  PASS: ", $(esc(name)))
                global passed += 1
            else
                println("  FAIL: ", $(esc(name)), " (returned false)")
                global failed += 1
            end
        catch e
            println("  FAIL: ", $(esc(name)), " (", typeof(e), ": ", e, ")")
            global failed += 1
        end
    end
end

# ── Test 1: Data generation ────────────────────────────────────────
println("\n[2/6] Testing data generation...")

result = datagenerate(8, 4, 2; seed=42)
@test "datagenerate returns correct keys" haskey(result, "datasets") && haskey(result, "d")
@test "datagenerate returns correct dimensions" size(result["datasets"][1]) == (8, 4)
@test "datagenerate produces unit-norm columns" begin
    X = result["datasets"][1]
    all(abs(norm(X[:, k]) - 1.0) < 1e-10 for k in 1:4)
end
@test "datagenerate is reproducible with seed" begin
    r1 = datagenerate(8, 4, 1; seed=123)
    r2 = datagenerate(8, 4, 1; seed=123)
    r1["datasets"][1] == r2["datasets"][1]
end

# ── Test 2: ULA sampler (Algorithm 1) ─────────────────────────────
println("\n[3/6] Testing ULA sampler...")

X_test = result["datasets"][1]  # 8 x 4
xi0 = randn(8)

res_ula = sample(X_test, xi0, 100; β=5.0, α=0.01, seed=42)
@test "ULA returns correct trajectory shape" size(res_ula.Ξ) == (101, 8)
@test "ULA returns correct time indices" length(res_ula.t) == 101 && res_ula.t[1] == 0 && res_ula.t[end] == 100
@test "ULA initial state matches input" res_ula.Ξ[1, :] == xi0
@test "ULA final state differs from initial" res_ula.Ξ[end, :] != xi0

# Energy should generally decrease from random init at moderate beta
e_init = hopfield_energy(Vector(res_ula.Ξ[1, :]), X_test, 5.0)
e_final = hopfield_energy(Vector(res_ula.Ξ[end, :]), X_test, 5.0)
@test "ULA energy decreases from random init" e_final < e_init

# ── Test 3: MALA sampler ──────────────────────────────────────────
println("\n[4/6] Testing MALA sampler...")

res_mala = mala_sample(X_test, xi0, 200; β=5.0, α=0.01, seed=42)
@test "MALA returns correct trajectory shape" size(res_mala.Ξ) == (201, 8)
@test "MALA returns acceptance rate" 0.0 <= res_mala.accept_rate <= 1.0
@test "MALA acceptance rate > 0 at small step size" res_mala.accept_rate > 0.5

# At small alpha, MALA and ULA should give similar energy distributions
e_mala = hopfield_energy(Vector(res_mala.Ξ[end, :]), X_test, 5.0)
@test "MALA reaches low energy" e_mala < e_init

# ── Test 4: Utility functions ─────────────────────────────────────
println("\n[5/6] Testing utility functions...")

xi_sample = Vector(res_ula.Ξ[end, :])

@test "nearest_cosine_similarity in [-1,1]" begin
    sim = nearest_cosine_similarity(xi_sample, X_test)
    -1.0 <= sim <= 1.0
end

@test "hopfield_energy returns finite" isfinite(hopfield_energy(xi_sample, X_test, 5.0))

@test "attention_entropy is non-negative" attention_entropy(xi_sample, X_test, 5.0) >= 0.0

@test "attention_entropy <= log(K) at low beta" begin
    H = attention_entropy(xi_sample, X_test, 0.001)
    H <= log(4) + 0.01  # K=4
end

@test "sample_novelty in [0,2]" begin
    nov = sample_novelty(xi_sample, X_test)
    0.0 <= nov <= 2.0
end

samples = [Vector(res_ula.Ξ[end-i, :]) for i in 0:9]
@test "sample_diversity is non-negative" sample_diversity(samples) >= 0.0
@test "sample_quality returns finite" isfinite(sample_quality(samples, X_test, 5.0))

# ── Test 5: Phase transition behavior ─────────────────────────────
println("\n[6/6] Testing phase transition behavior...")

# At high beta, should converge near a stored pattern (high cosine similarity)
X_big = datagenerate(64, 16, 1; seed=99)["datasets"][1]
xi_init = X_big[:, 1] .+ 0.01 .* randn(64)  # start near pattern 1
res_high = sample(X_big, xi_init, 1000; β=500.0, α=0.01, seed=42)
xi_high = Vector(res_high.Ξ[end, :])
@test "High beta: converges near stored pattern" nearest_cosine_similarity(xi_high, X_big) > 0.9

# At low beta, entropy should be near log(K)
res_low = sample(X_big, randn(64), 500; β=0.1, α=0.01, seed=42)
xi_low = Vector(res_low.Ξ[end, :])
@test "Low beta: attention entropy near log(K)" begin
    H = attention_entropy(xi_low, X_big, 0.1)
    H > 0.8 * log(16)  # within 80% of uniform
end

# ── Summary ────────────────────────────────────────────────────────
println("\n" * "=" ^ 60)
println("Results: $passed passed, $failed failed, $(passed + failed) total")
println("=" ^ 60)

if failed > 0
    println("SOME TESTS FAILED")
    exit(1)
else
    println("ALL TESTS PASSED")
end
