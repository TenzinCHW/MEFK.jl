push!(LOAD_PATH, "@stdlib")
using Pkg
Pkg.activate(@__DIR__)

include("basic.jl")
#include("basicfull.jl")
include("basicmat.jl")
#include("gpuarr.jl")
#include("gpufull.jl")
include("checkenergy.jl")
include("checkenergy2.jl")

