# setup paths -
const _ROOT = @__DIR__
const _PATH_TO_SRC = joinpath(_ROOT, "..","src");
const _PATH_TO_DATA = joinpath(_ROOT, "..", "data");
const _PATH_TO_FIG = joinpath(_ROOT, "..", "figs");

# activate the project environment (one level up, where Project.toml lives)
using Pkg
Pkg.activate(joinpath(_ROOT, ".."))
if (isfile(joinpath(_ROOT, "..", "Manifest.toml")) == false)
    Pkg.add(path="https://github.com/varnerlab/VLDataScienceMachineLearningPackage.jl.git")
    Pkg.resolve(); Pkg.instantiate(); Pkg.update();
end

# load the required packages -
using VLDataScienceMachineLearningPackage
using Images
using ImageInTerminal
using ImageIO
using OneHotArrays

using Distributions
using Plots
using StatsPlots
using Colors
using LinearAlgebra
using Statistics
using DataFrames
using PrettyTables
using Random
using CSV
using FileIO
using JLD2
using Dates
using NNlib
using CategoricalArrays
using StatsBase
using MultivariateStats

# set the random seed for reproducibility
Random.seed!(1234); # set the random seed for reproducibility

# include the source code for the project -
include(joinpath(_PATH_TO_SRC, "Data.jl"))
include(joinpath(_PATH_TO_SRC, "Compute.jl"))
include(joinpath(_PATH_TO_SRC, "Utilities.jl"))

# publication figures are ALWAYS PDF, never PNG — provides save_pub(path, img)
include(joinpath(_ROOT, "..", "save_pub.jl"))