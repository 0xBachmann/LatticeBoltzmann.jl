using CellArraysIndexing, StaticArrays
using ImplicitGlobalGrid
import MPI

using ParallelStencil

const USE_GPU = false

@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 3, inbounds=true)
else
    @init_parallel_stencil(Threads, Float64, 3, inbounds=true)
end

const method = :D3Q19
const dimension = 3

include("../src/LatticeBoltzmann3D.jl")

thermal_convection_lbm_3D()