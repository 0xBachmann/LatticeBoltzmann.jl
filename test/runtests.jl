using Test

using ImplicitGlobalGrid
import MPI

using LinearAlgebra
using Plots
using ParallelStencil

@init_parallel_stencil(CUDA, Float64, 3, inbounds=false)

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

    density_pop = @zeros(Nx + 2, Ny + 2, Nz + 2, celldims=Q)
    density_buf = @zeros(Nx + 2, Ny + 2, Nz + 2, celldims=Q)
    
    temperature_pop = @zeros(Nx + 2, Ny + 2, Nz + 2, celldims=Q)
    temperature_buf = @zeros(Nx + 2, Ny + 2, Nz + 2, celldims=Q)

    D = 1e-2
    viscosity = 5e-2

    _τ_temperature = 1. / (D * _cs2 + 0.5)
    _τ_density = 1. / (viscosity * _cs2 + 0.5)

    nt = 1000
    timesteps = 0:nt

    R = lx / 4
    U_init = @SVector [0., 0.2, 0.]

    velocity = @zeros(Nx, Ny, Nz, celldims=3)
    density = @ones(Nx, Ny, Nz)
    boundary = Data.Array([((x_g(ix, dx, density) - lx / 2)^2 + (y_g(iy, dy, density) - ly / 3) ^2) < R^2 ? 1. : 0. for ix = 1:Nx, iy = 1:Ny, iz = 1:Nz])
    temperature = @zeros(Nx, Ny, Nz)
    
    @parallel (1:Nx, 1:Ny, 1:Nz) init!(velocity, temperature, boundary, U_init)
    
    @parallel (2:Nx+1, 2:Ny+1, 2:Nz+1) init_pop!(density_pop, velocity, density)
    @parallel (2:Nx+1, 2:Ny+1, 2:Nz+1) init_pop!(temperature_pop, velocity, temperature)

    visdir = "visdir"
    st = ceil(Int, Nx / 10)
    ENV["GKSwstype"]="nul"
    if (me==0) if isdir("$visdir")==false mkdir("$visdir") end; loadpath="$visdir/"; anim=Animation(loadpath,String[]); println("Animation directory: $(anim.dir)") end
    Nx_v, Ny_v, Nz_v = (Nx) * dims[1], (Ny) * dims[2], (Nz) * dims[3]
    (2 * Nx_v * Ny_v * Nz_v * sizeof(Data.Number) > 0.8 * Sys.free_memory()) && error("Not enough memory for visualization.")
    density_v = zeros(Nx_v, Ny_v, Nz_v) # global array for visu
    temperature_v = zeros(Nx_v, Ny_v, Nz_v) # global array for visu
    xi_g, yi_g = LinRange(0, lx, Nx_v), LinRange(0, ly, Ny_v) # inner points only
    iframe = 0
    Xc, Yc = [x for x in xi_g, _ in yi_g], [y for _ in xi_g, y in yi_g]
    Xp, Yp = Xc[1:st:end, 1:st:end], Yc[1:st:end, 1:st:end]

    for i in timesteps
        if me == 0
            dens = heatmap(xi_g, yi_g, Array(density[:, :, Int(ceil((Nz-2)/2))])'; xlims=(xi_g[1], xi_g[end]), ylims=(yi_g[1], yi_g[end]), aspect_ratio=1, c=:turbo, clim=(0,1), title="density")
            # dens = quiver!(Xp[:], Yp[:]; quiver=(velx_p_g[:], vely_p_g[:]), lw=0.5, c=:black)

            temp = heatmap(xi_g, yi_g, Array(temperature[:, :, Int(ceil((Nz-2)/2))])'; xlims=(xi_g[1], xi_g[end]), ylims=(yi_g[1], yi_g[end]), aspect_ratio=1, c=:turbo, clim=(0,1), title="temperature")
            # temp = quiver!(Xp[:], Yp[:]; quiver=(velx_p_g[:], vely_p_g[:]), lw=0.5, c=:black)

            p = plot(dens, temp)
            png(p, "$visdir/$(lpad(iframe += 1, 4, "0")).png")
            save_array("$visdir/out_dens_$(lpad(iframe, 4, "0"))", convert.(Float32, density_v))
            save_array("$visdir/out_temp_$(lpad(iframe, 4, "0"))", convert.(Float32, temperature_v))
        end
        @parallel (1:Nx, 1:Ny, 1:Nz) update_moments!(velocity, density, temperature, density_pop, temperature_pop)
        @parallel (1:Nx, 1:Ny, 1:Nz) apply_external_force!(velocity, boundary, lx, ly, R)

        @parallel (2:Nx+1, 2:Ny+1, 2:Nz+1) collision!(density_pop, velocity, density, _τ_density)
        @parallel (2:Nx+1, 2:Ny+1, 2:Nz+1) collision!(temperature_pop, velocity, temperature, _τ_temperature)

        @parallel (1:Nx+2, 1:Ny+2) periodic_boundary_z!(density_pop)
        @parallel (1:Nx+2, 1:Ny+2) periodic_boundary_z!(temperature_pop)
        @parallel (1:Nx+2, 1:Nz+2) periodic_boundary_y!(density_pop)
        @parallel (1:Nx+2, 1:Nz+2) periodic_boundary_y!(temperature_pop)
        @parallel (1:Ny+2, 1:Nz+2) periodic_boundary_x!(density_pop)
        @parallel (1:Ny+2, 1:Nz+2) periodic_boundary_x!(temperature_pop)

        @parallel (2:Nx+1, 2:Ny+1, 2:Nz+1) streaming!(density_pop, density_buf)
        @parallel (2:Nx+1, 2:Ny+1, 2:Nz+1) streaming!(temperature_pop, temperature_buf)

        # pointer swap
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