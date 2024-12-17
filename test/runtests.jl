using Test

using ImplicitGlobalGrid
import MPI

using LinearAlgebra
using Plots
using ParallelStencil
using ProgressBars


const USE_GPU = false

@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 3, inbounds=false)
else
    @init_parallel_stencil(Threads, Float64, 3, inbounds=false)
end

const method = :D3Q19

include("../src/LatticeBoltzmann3D.jl")

"""
    save_array(Aname, A)

Write an array `A` in binary format to disc. Resulting file is called `Aname` with suffix `".bin"`.  
"""
function save_array(Aname, A)
    fname = string(Aname, ".bin")
    out = open(fname, "w")
    write(out, A)
    close(out)
end

function lb()
    Nx = 40
    Ny = 70
    Nz = 3

    me, dims, nprocs, coords, comm = init_global_grid(Nx, Ny, Nz, periodz=1, periodx=1, periody=1)

    lx = 40
    ly = 70
    lz = 3

    dx, dy, dz = lx / nx_g(), ly / ny_g(), lz / nz_g()

    density_pop = @zeros(Nx + 2, Ny + 2, Nz + 2, Q)
    density_buf = @zeros(Nx + 2, Ny + 2, Nz + 2, Q)
    
    temperature_pop = @zeros(Nx + 2, Ny + 2, Nz + 2, Q)
    temperature_buf = @zeros(Nx + 2, Ny + 2, Nz + 2, Q)

    D = 1e-2
    viscosity = 5e-2

    _τ_temperature = 1. / (D * _cs2 + 0.5)
    _τ_density = 1. / (viscosity * _cs2 + 0.5)

    nt = 1000
    timesteps = 1:nt

    R = lx / 4
    U_init = @zeros(3)
    U_init[2] = 0.2

    velocity = @zeros(Nx, Ny, Nz, 3)
    density = @zeros(Nx, Ny, Nz)
    temperature = @zeros(Nx, Ny, Nz)

    @parallel (1:Nx, 1:Ny, 1:Nz) init!(velocity, density, temperature, U_init, lx, ly, R)
    
    @parallel (2:Nx+1, 2:Ny+1, 2:Nz+1) init_pop!(density_pop, velocity, density)
    @parallel (2:Nx+1, 2:Ny+1, 2:Nz+1) init_pop!(temperature_pop, velocity, temperature)

    for _ in timesteps

        @parallel (1:Nx, 1:Ny, 1:Nz) update_moments!(velocity, density, temperature, density_pop, temperature_pop)
        @parallel (1:Nx, 1:Ny, 1:Nz) apply_external_force!(velocity, lx, ly, R)

        @parallel (2:Nx+1, 2:Ny+1, 2:Nz+1) collision!(density_pop, velocity, density, _τ_density)
        @parallel (2:Nx+1, 2:Ny+1, 2:Nz+1) collision!(temperature_pop, velocity, temperature, _τ_temperature)


        lb_update_halo!(density_pop, comm)
        lb_update_halo!(temperature_pop, comm)

        @parallel (2:Nx+1, 2:Ny+1, 2:Nz+1) streaming!(density_pop, density_buf)
        @parallel (2:Nx+1, 2:Ny+1, 2:Nz+1) streaming!(temperature_pop, temperature_buf)

        density_pop, density_buf = density_buf, density_pop
        temperature_pop, temperature_buf = temperature_buf, temperature_pop 
    end
    finalize_global_grid()
    return density, temperature
end

function load_array(Aname, A)
    fname = string(Aname, ".bin")
    fid=open(fname, "r"); read!(fid, A); close(fid)
end

@testset "testing 3D thermal lbm" begin
    Nx = 40
    Ny = 70
    Nz = 3
    dens, temp = lb()
    @test all(isfinite, dens)
    @test all(isfinite, temp)
    dens_ref = zeros(Float64, Nx, Ny, Nz)
    temp_ref = zeros(Float64, Nx, Ny, Nz)
    load_array("out_test_density", dens_ref)
    load_array("out_test_temperature", temp_ref)
    @test all(dens .≈ dens_ref)
    @test all(temp .≈ temp_ref)
end