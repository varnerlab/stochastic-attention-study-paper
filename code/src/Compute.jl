"""
    hopfield_component_weights(X; β=1.0, multiplicities=nothing) -> Vector{Float64}

Return the normalized Gaussian-mixture weights of the modern Hopfield
Boltzmann law. For memories `m_k = X[:, k]` and nonnegative multiplicities
`r_k`,

```math
w_k \\propto r_k\\exp\\!\\left(\\frac{\\beta}{2}\\|m_k\\|_2^2\\right).
```

The weights are uniform only when the memories have equal norm and the
multiplicities are equal.
"""
function hopfield_component_weights(X::AbstractMatrix{<:Real};
    β::Real=1.0, multiplicities::Union{Nothing,AbstractVector{<:Real}}=nothing)

    β > 0 || throw(ArgumentError("β must be positive, got β = $β"))
    _, K = size(X)
    K > 0 || throw(ArgumentError("X must contain at least one memory"))

    r = multiplicities === nothing ? ones(Float64, K) : Float64.(multiplicities)
    length(r) == K || throw(DimensionMismatch(
        "multiplicities has length $(length(r)) but X has $K memories"))
    all(isfinite, r) || throw(ArgumentError("multiplicities must be finite"))
    all(>=(0.0), r) || throw(ArgumentError("multiplicities must be nonnegative"))
    any(>(0.0), r) || throw(ArgumentError("at least one multiplicity must be positive"))

    log_weights = similar(r)
    for k in 1:K
        log_weights[k] = r[k] == 0.0 ? -Inf :
            log(r[k]) + 0.5 * Float64(β) * sum(abs2, @view X[:, k])
    end
    max_log_weight = maximum(log_weights)
    weights = exp.(log_weights .- max_log_weight)
    return weights ./ sum(weights)
end


"""
    hopfield_responsibilities(ξ, X; β=1.0, multiplicities=nothing)

Return the posterior mixture-component responsibilities at state `ξ`. These
are exactly the multiplicity-biased attention weights

```math
a_k(\\xi)=\\operatorname{softmax}_k(\\beta X^\\top\\xi+\\log r).
```
"""
function hopfield_responsibilities(ξ::AbstractVector{<:Real},
    X::AbstractMatrix{<:Real}; β::Real=1.0,
    multiplicities::Union{Nothing,AbstractVector{<:Real}}=nothing)

    β > 0 || throw(ArgumentError("β must be positive, got β = $β"))
    d, K = size(X)
    length(ξ) == d || throw(DimensionMismatch(
        "state has length $(length(ξ)) but X has $d rows"))

    r = multiplicities === nothing ? ones(Float64, K) : Float64.(multiplicities)
    length(r) == K || throw(DimensionMismatch(
        "multiplicities has length $(length(r)) but X has $K memories"))
    all(isfinite, r) || throw(ArgumentError("multiplicities must be finite"))
    all(>=(0.0), r) || throw(ArgumentError("multiplicities must be nonnegative"))
    any(>(0.0), r) || throw(ArgumentError("at least one multiplicity must be positive"))

    logits = Float64(β) .* (Float64.(X)' * Float64.(ξ))
    for k in 1:K
        logits[k] += r[k] == 0.0 ? -Inf : log(r[k])
    end
    max_logit = maximum(logits)
    responsibilities = exp.(logits .- max_logit)
    return responsibilities ./ sum(responsibilities)
end


"""
    hopfield_score(ξ, X; β=1.0, multiplicities=nothing)

Evaluate the exact score of the modern Hopfield Boltzmann density:

```math
\\nabla_\\xi\\log p_\\beta(\\xi)
=\\beta\\left[Xa(\\xi)-\\xi\\right].
```
"""
function hopfield_score(ξ::AbstractVector{<:Real}, X::AbstractMatrix{<:Real};
    β::Real=1.0, multiplicities::Union{Nothing,AbstractVector{<:Real}}=nothing)

    a = hopfield_responsibilities(ξ, X; β=β, multiplicities=multiplicities)
    return Float64(β) .* (Float64.(X) * a .- Float64.(ξ))
end


"""
    hopfield_logpdf(ξ, X; β=1.0, multiplicities=nothing) -> Float64

Evaluate the normalized log density of the exact Gaussian mixture induced by
the modern Hopfield energy. Each component has mean `X[:, k]`, covariance
`β⁻¹I`, and weight returned by [`hopfield_component_weights`](@ref).
"""
function hopfield_logpdf(ξ::AbstractVector{<:Real}, X::AbstractMatrix{<:Real};
    β::Real=1.0, multiplicities::Union{Nothing,AbstractVector{<:Real}}=nothing)

    β > 0 || throw(ArgumentError("β must be positive, got β = $β"))
    d, K = size(X)
    length(ξ) == d || throw(DimensionMismatch(
        "state has length $(length(ξ)) but X has $d rows"))

    weights = hopfield_component_weights(X; β=β, multiplicities=multiplicities)
    log_terms = Vector{Float64}(undef, K)
    log_gaussian_constant = 0.5 * d * (log(Float64(β)) - log(2π))
    ξ_float = Float64.(ξ)
    for k in 1:K
        difference = ξ_float .- Float64.(@view X[:, k])
        log_terms[k] = log(weights[k]) + log_gaussian_constant -
            0.5 * Float64(β) * dot(difference, difference)
    end
    max_log_term = maximum(log_terms)
    return max_log_term + log(sum(exp.(log_terms .- max_log_term)))
end


"""
    exact_hopfield_sample(X, S; β=1.0, multiplicities=nothing, seed=nothing)

Draw `S` independent samples from the exact modern Hopfield Boltzmann law.
Returns a named tuple with a `d × S` sample matrix, sampled component indices,
and normalized component weights.
"""
function exact_hopfield_sample(X::AbstractMatrix{<:Real}, S::Int;
    β::Real=1.0, multiplicities::Union{Nothing,AbstractVector{<:Real}}=nothing,
    seed::Union{Int,Nothing}=nothing)

    S > 0 || throw(ArgumentError("S must be positive, got S = $S"))
    β > 0 || throw(ArgumentError("β must be positive, got β = $β"))

    X_float = Float64.(X)
    d, K = size(X_float)
    K > 0 || throw(ArgumentError("X must contain at least one memory"))
    weights = hopfield_component_weights(X_float; β=β, multiplicities=multiplicities)
    rng = seed === nothing ? Random.default_rng() : Random.MersenneTwister(seed)
    component_distribution = Distributions.Categorical(weights)
    components = rand(rng, component_distribution, S)
    samples = Matrix{Float64}(undef, d, S)
    noise_scale = inv(sqrt(Float64(β)))
    for s in 1:S
        samples[:, s] .= @view(X_float[:, components[s]]) .+
            noise_scale .* randn(rng, d)
    end
    return (samples=samples, components=components, weights=weights)
end


"""
    sample(X, ξ₀, T; β=1.0, α=0.1, seed=nothing) -> (t, Ξ)

Run the Stochastic Attention Sampler (Algorithm 1) on a memory matrix `X`.

The sampler implements the Unadjusted Langevin Algorithm (ULA) on the modern 
Hopfield energy ``E(\\boldsymbol{\\xi}) = \\tfrac{1}{2}\\|\\boldsymbol{\\xi}\\|^2 
- \\tfrac{1}{\\beta}\\log\\sum_k \\exp(\\beta\\,\\mathbf{m}_k^\\top \\boldsymbol{\\xi})``, 
yielding the update

```math
\\boldsymbol{\\xi}_{t+1}
= (1-\\alpha)\\,\\boldsymbol{\\xi}_t
  + \\alpha\\,\\mathbf{X}\\,\\operatorname{softmax}(\\beta\\,\\mathbf{X}^\\top\\boldsymbol{\\xi}_t)
  + \\sqrt{2\\alpha/\\beta}\\;\\boldsymbol{\\epsilon}_t,
\\qquad \\boldsymbol{\\epsilon}_t \\sim \\mathcal{N}(\\mathbf{0}, \\mathbf{I}).
```

At each step the algorithm performs three operations:
1. A contraction toward the origin: ``(1-\\alpha)\\boldsymbol{\\xi}_t``
2. A softmax-weighted pull toward stored memories: ``\\alpha\\,\\mathbf{X}\\,\\mathbf{a}_t``
3. An isotropic Gaussian perturbation scaled by ``\\sqrt{2\\alpha/\\beta}``

# Arguments (positional)
- `X::Matrix{Float64}`: Memory matrix of size `d × K`, where each column is a 
  stored pattern of dimension `d`.
- `ξ₀::Vector{Float64}`: Initial state vector of length `d`.
- `T::Int`: Number of iterations to run (the chain produces states 
  ``\\boldsymbol{\\xi}_0, \\boldsymbol{\\xi}_1, \\dots, \\boldsymbol{\\xi}_T``).

# Keyword Arguments
- `β::Float64=1.0`: Inverse temperature. Controls the sharpness of the Boltzmann 
  distribution: large `β` concentrates mass on energy minima (retrieval), small `β` 
  flattens the distribution (diffuse generation).
- `α::Float64=0.1`: Step size (learning rate) in `(0, 1)`. Smaller values reduce 
  ULA discretization bias at the cost of slower mixing.
- `seed::Union{Int, Nothing}=nothing`: Optional random seed for the noise sequence. 
  If `nothing`, the global RNG state is used unchanged.

# Returns
A named tuple `(t, Ξ)` where:
- `t::Vector{Int}`: Time indices `[0, 1, …, T]` of length `T + 1`.
- `Ξ::Matrix{Float64}`: State trajectory of size `(T + 1) × d`. Row `i` is the 
  state ``\\boldsymbol{\\xi}_{i-1}`` at time `t[i]`. Columns correspond to the `d` 
  dimensions of the state space.

# Examples
```julia
# 16 patterns in 64 dimensions, 1000 steps at inverse temperature 5
X = randn(64, 16); X ./= sqrt.(sum(X.^2; dims=1))  # unit-norm columns
ξ₀ = randn(64)
result = sample(X, ξ₀, 1000; β=5.0, α=0.05)
result.t   # [0, 1, …, 1000]
result.Ξ   # 1001 × 64 matrix of states

# Reproducible run
result = sample(X, ξ₀, 500; β=2.0, α=0.1, seed=42)
```
"""
function sample(X::Matrix{Float64}, ξ₀::Vector{Float64}, T::Int;
    β::Float64 = 1.0, α::Float64 = 0.1,
    seed::Union{Int, Nothing} = nothing)

    # --- input validation ---
    d, K = size(X)
    length(ξ₀) == d || throw(DimensionMismatch(
        "Initial state has length $(length(ξ₀)) but X has $d rows"))
    T > 0   || throw(ArgumentError("T must be positive, got T = $T"))
    β > 0   || throw(ArgumentError("β must be positive, got β = $β"))
    0 < α < 1 || throw(ArgumentError("α must be in (0,1), got α = $α"))

    # --- seed the RNG if requested ---
    if seed !== nothing
        Random.seed!(seed)
    end

    # --- pre-allocate output ---
    # Ξ is (T+1) × d: row i holds ξ at time t = i-1
    Ξ = Matrix{Float64}(undef, T + 1, d)
    Ξ[1, :] .= ξ₀  # store initial state

    # --- pre-compute constants ---
    noise_scale = sqrt(2.0 * α / β)  # standard deviation of the Gaussian perturbation

    # --- run the Langevin chain ---
    ξ = copy(ξ₀)  # current state (working vector)
    for t in 1:T

        # Step 1: compute attention weights  a = softmax(β X⊤ ξ)
        logits = β .* (X' * ξ)          # K-vector of logits
        w = softmax(logits)              # NNlib.softmax (numerically stable)

        # Step 2: draw isotropic Gaussian noise  ε ~ N(0, I_d)
        ε = randn(d)

        # Step 3: Langevin update
        #   ξ_{t+1} = (1 - α) ξ_t + α X a_t + √(2α/β) ε_t
        ξ .= (1.0 - α) .* ξ .+ α .* (X * w) .+ noise_scale .* ε

        # store the new state
        Ξ[t + 1, :] .= ξ
    end

    # --- build time index ---
    t_vec = collect(0:T)

    return (t = t_vec, Ξ = Ξ)
end


"""
    mala_sample(X, ξ₀, T; β=1.0, α=0.1, seed=nothing) -> (t, Ξ, accept_rate)

Run the Metropolis-Adjusted Langevin Algorithm (MALA) on the modern Hopfield energy.

MALA proposes each new state via the same Langevin step as Algorithm 1 (ULA), but
then applies a Metropolis–Hastings accept/reject correction to eliminate the
discretization bias inherent in ULA. This produces exact (asymptotically unbiased)
samples from the Boltzmann distribution

```math
p_\\beta(\\boldsymbol{\\xi}) \\;\\propto\\; \\exp\\!\\bigl(-\\beta\\, E(\\boldsymbol{\\xi})\\bigr),
\\qquad
E(\\boldsymbol{\\xi}) = \\tfrac{1}{2}\\|\\boldsymbol{\\xi}\\|^2
  - \\tfrac{1}{\\beta}\\log\\sum_k \\exp(\\beta\\,\\mathbf{m}_k^\\top \\boldsymbol{\\xi}).
```

**Proposal step** (identical to ULA):
```math
\\boldsymbol{\\xi}^*
= (1-\\alpha)\\,\\boldsymbol{\\xi}_t
  + \\alpha\\,\\mathbf{X}\\,\\operatorname{softmax}(\\beta\\,\\mathbf{X}^\\top\\boldsymbol{\\xi}_t)
  + \\sqrt{2\\alpha/\\beta}\\;\\boldsymbol{\\epsilon}_t,
\\qquad \\boldsymbol{\\epsilon}_t \\sim \\mathcal{N}(\\mathbf{0}, \\mathbf{I}).
```

**Acceptance criterion**: the proposal ``\\boldsymbol{\\xi}^*`` is accepted with probability
```math
\\min\\!\\Bigl(1,\\;
  \\frac{p_\\beta(\\boldsymbol{\\xi}^*)\\; q(\\boldsymbol{\\xi}_t \\mid \\boldsymbol{\\xi}^*)}
       {p_\\beta(\\boldsymbol{\\xi}_t)\\; q(\\boldsymbol{\\xi}^* \\mid \\boldsymbol{\\xi}_t)}
\\Bigr),
```
where ``q(\\cdot \\mid \\boldsymbol{\\xi})`` is the Gaussian proposal density centered at the
Langevin mean ``\\mu(\\boldsymbol{\\xi}) = (1-\\alpha)\\boldsymbol{\\xi} + \\alpha\\,\\mathbf{X}\\,\\operatorname{softmax}(\\beta\\,\\mathbf{X}^\\top\\boldsymbol{\\xi})``.
If rejected, the chain stays at ``\\boldsymbol{\\xi}_t``.

# Arguments (positional)
- `X::Matrix{Float64}`: Memory matrix of size `d × K`, where each column is a 
  stored pattern of dimension `d`.
- `ξ₀::Vector{Float64}`: Initial state vector of length `d`.
- `T::Int`: Number of iterations to run.

# Keyword Arguments
- `β::Float64=1.0`: Inverse temperature.
- `α::Float64=0.1`: Step size (learning rate) in `(0, 1)`.
- `seed::Union{Int, Nothing}=nothing`: Optional random seed.

# Returns
A named tuple `(t, Ξ, accept_rate)` where:
- `t::Vector{Int}`: Time indices `[0, 1, …, T]` of length `T + 1`.
- `Ξ::Matrix{Float64}`: State trajectory of size `(T + 1) × d`.
- `accept_rate::Float64`: Fraction of proposals that were accepted, in `[0, 1]`.

# Examples
```julia
X = randn(64, 16); X ./= sqrt.(sum(X.^2; dims=1))
ξ₀ = randn(64)
result = mala_sample(X, ξ₀, 1000; β=5.0, α=0.05)
result.accept_rate  # e.g. 0.73
```
"""
function mala_sample(X::Matrix{Float64}, ξ₀::Vector{Float64}, T::Int;
    β::Float64 = 1.0, α::Float64 = 0.1,
    seed::Union{Int, Nothing} = nothing)

    # --- input validation ---
    d, K = size(X)
    length(ξ₀) == d || throw(DimensionMismatch(
        "Initial state has length $(length(ξ₀)) but X has $d rows"))
    T > 0   || throw(ArgumentError("T must be positive, got T = $T"))
    β > 0   || throw(ArgumentError("β must be positive, got β = $β"))
    0 < α < 1 || throw(ArgumentError("α must be in (0,1), got α = $α"))

    # --- seed the RNG if requested ---
    if seed !== nothing
        Random.seed!(seed)
    end

    # --- pre-allocate output ---
    Ξ = Matrix{Float64}(undef, T + 1, d)
    Ξ[1, :] .= ξ₀

    # --- pre-compute constants ---
    noise_scale = sqrt(2.0 * α / β)       # std dev of the Gaussian perturbation
    noise_var   = 2.0 * α / β             # variance σ² = 2α/β
    log_norm_const = -0.5 * d * log(2π * noise_var)  # constant part of log q(·|·)
    inv_2var = 1.0 / (2.0 * noise_var)    # 1/(2σ²) for the Gaussian exponent

    n_accept = 0  # acceptance counter

    # --- helper: compute Langevin proposal mean μ(ξ) ---
    #   μ(ξ) = (1 - α) ξ + α X softmax(β X⊤ ξ)
    function langevin_mean(ξ::Vector{Float64})
        logits = β .* (X' * ξ)
        w = softmax(logits)
        return (1.0 - α) .* ξ .+ α .* (X * w)
    end

    # --- helper: compute the negative log-unnormalized target  β E(ξ) ---
    #   β E(ξ) = β/2 ‖ξ‖² − log Σ_k exp(β m_k⊤ ξ)
    function neg_log_target(ξ::Vector{Float64})
        logits = β .* (X' * ξ)
        logits_max = maximum(logits)
        lse = logits_max + log(sum(exp.(logits .- logits_max)))
        return 0.5 * β * dot(ξ, ξ) - lse
    end

    # --- helper: log proposal density  log q(y | x) = −‖y − μ(x)‖²/(2σ²) + const ---
    function log_proposal(y::Vector{Float64}, μx::Vector{Float64})
        diff = y .- μx
        return log_norm_const - inv_2var * dot(diff, diff)
    end

    # --- run the MALA chain ---
    ξ = copy(ξ₀)
    μ_curr = langevin_mean(ξ)           # cache the current proposal mean
    nlt_curr = neg_log_target(ξ)        # cache −log p(ξ_t) up to const

    for t in 1:T

        # Step 1: propose  ξ* = μ(ξ_t) + σ ε,  ε ~ N(0, I)
        ε = randn(d)
        ξ_prop = μ_curr .+ noise_scale .* ε

        # Step 2: compute acceptance probability
        #   log α = [ -β E(ξ*) + log q(ξ_t | ξ*) ] - [ -β E(ξ_t) + log q(ξ* | ξ_t) ]
        nlt_prop = neg_log_target(ξ_prop)
        μ_prop   = langevin_mean(ξ_prop)

        log_accept = (-nlt_prop + log_proposal(ξ, μ_prop)) -
                     (-nlt_curr + log_proposal(ξ_prop, μ_curr))

        # Step 3: accept or reject
        if log(rand()) < log_accept
            # accept the proposal
            ξ .= ξ_prop
            μ_curr  = μ_prop
            nlt_curr = nlt_prop
            n_accept += 1
        end
        # if rejected, ξ, μ_curr, nlt_curr remain unchanged

        # store the current state (accepted or rejected)
        Ξ[t + 1, :] .= ξ
    end

    # --- build time index ---
    t_vec = collect(0:T)
    accept_rate = n_accept / T

    return (t = t_vec, Ξ = Ξ, accept_rate = accept_rate)
end
