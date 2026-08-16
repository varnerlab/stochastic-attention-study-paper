#!/usr/bin/env julia

# Re-render the production figure from the values reported in Supplementary
# Table S11. The two energy differences reported only as <0.001 are left
# unplotted rather than assigned invented point estimates.

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using Plots
using Colors

include(joinpath(@__DIR__, "plot_stepsize_sweep.jl"))

αs = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]
accept_rates = [1.0, 1.0, 0.992, 0.978, 0.911, 0.747, 0.0, 0.0, 0.0]

αs_valid = αs[1:6]
Δnovelty = [0.001, 0.000, 0.001, 0.002, 0.004, 0.007]
Δdiversity = [0.000, 0.002, 0.002, 0.002, 0.004, 0.008]
Δenergy = [NaN, NaN, 0.002, 0.003, 0.006, 0.011]

output_dir = normpath(joinpath(@__DIR__, "..", "figs"))
paths = render_stepsize_sweep_figure(
    αs,
    accept_rates,
    αs_valid,
    Δnovelty,
    Δdiversity,
    Δenergy;
    output_dir,
)

@info "Rendered production step-size figure" paths
