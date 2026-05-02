#!/usr/bin/env julia
# ──────────────────────────────────────────────────────────────────────────────
# Render the 4x4 grid of *stored* MNIST digit-3 patterns used as the SA memory.
# The output PNG sits in paper-neurips/figs/ next to the other 4x4 grids and is
# included as the leftmost ("Stored") panel in Figure 2 of the paper.
# ──────────────────────────────────────────────────────────────────────────────

@info "Loading environment …"
include(joinpath(@__DIR__, "Include-MNIST.jl"))
@info "Environment loaded."

using FileIO, Images

# ── helpers (matching the MNIST grid renderers used elsewhere) ──────────────
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

# ── Load the same K=100 digit-3 memories used everywhere else ──────────────
const DIGIT = 3
const K = 100
const D = 784
const ϵ = 1e-12

@info "Loading MNIST digit $DIGIT …"
digits = MyMNISTHandwrittenDigitImageDataset(number_of_examples = K)

stored_samples = Vector{Vector{Float64}}()
for i in 1:K
    img = digits[DIGIT][:, :, i]
    x = reshape(transpose(img) |> Matrix, D) |> vec
    x ./= (norm(x) + ϵ)
    push!(stored_samples, x)
end

# ── Render to a 4x4 grid and save next to the existing panels ──────────────
canvas = build_grid(stored_samples; nrows = 4, ncols = 4, gap = 2)
out_path = joinpath(@__DIR__, "..", "..", "paper-neurips", "figs", "Fig_mnist_grid_stored.png")
out_path = abspath(out_path)
save(out_path, Gray.(clamp.(canvas, 0.0, 1.0)))
@info "Wrote $out_path"
