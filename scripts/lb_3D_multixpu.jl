using ImplicitGlobalGrid
import MPI

using LinearAlgebra
using Printf
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

    do_vis = true
    nvis = 10
    visdir = "visdir"
    st = ceil(Int, Nx / 10)
    
    @parallel (1:Nx, 1:Ny, 1:Nz) init!(velocity, density, temperature, U_init, lx, ly, R)
    
    @parallel (2:Nx+1, 2:Ny+1, 2:Nz+1) init_pop!(density_pop, velocity, density)
    @parallel (2:Nx+1, 2:Ny+1, 2:Nz+1) init_pop!(temperature_pop, velocity, temperature)

    # @parallel (1:Nx, 1:Ny) periodic_boundary_update!(:z, density_pop, density_buf)
    # @parallel (1:Nx, 1:Ny) periodic_boundary_update!(:z, temperature_pop, temperature_buf)
    # @parallel (1:Nx, 1:Nz) periodic_boundary_update!(:y, density_pop, density_buf)
    # @parallel (1:Nx, 1:Nz) periodic_boundary_update!(:y, temperature_pop, temperature_buf)
    # @parallel (1:Ny, 1:Nz) periodic_boundary_update!(:x, density_pop, density_buf)
    # @parallel (1:Ny, 1:Nz) periodic_boundary_update!(:x, temperature_pop, temperature_buf)
    # @parallel (2:Ny-1, 2:Nz-1) bounce_back_boundary!(:x, density_pop, density_buf)
    # @parallel (2:Ny-1, 2:Nz-1) bounce_back_boundary!(:x, temperature_pop, temperature_buf)
    

    if do_vis
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
    end

    for i in (me == 0 ? ProgressBar(timesteps) : timesteps)

        @parallel (1:Nx, 1:Ny, 1:Nz) update_moments!(velocity, density, temperature, density_pop, temperature_pop)
        @parallel (1:Nx, 1:Ny, 1:Nz) apply_external_force!(velocity, lx, ly, R)

        @parallel (2:Nx+1, 2:Ny+1, 2:Nz+1) collision!(density_pop, velocity, density, _τ_density)
        @parallel (2:Nx+1, 2:Ny+1, 2:Nz+1) collision!(temperature_pop, velocity, temperature, _τ_temperature)


        lb_update_halo!(density_pop, comm)
        lb_update_halo!(temperature_pop, comm)

        @parallel (2:Nx+1, 2:Ny+1, 2:Nz+1) streaming!(density_pop, density_buf)
        @parallel (2:Nx+1, 2:Ny+1, 2:Nz+1) streaming!(temperature_pop, temperature_buf)

        # @parallel (1:Nx, 1:Nz) periodic_boundary_update!(:y, density_pop, density_buf)
        # @parallel (1:Nx, 1:Nz) periodic_boundary_update!(:y, temperature_pop, temperature_buf)
        # @parallel (2:Nx-1, 2:Nz-1) inlet_boundary_conditions!(:y, density_pop, density_buf, U_init, density)
        # @parallel (2:Nx-1, 2:Nz-1) inlet_boundary_conditions!(:y, temperature_pop, temperature_buf, U_init, temperature)

        # @parallel (2:Ny-1, 2:Nz-1) bounce_back_boundary!(:x, density_pop, density_buf)
        # @parallel (2:Ny-1, 2:Nz-1) bounce_back_boundary!(:x, temperature_pop, temperature_buf)
        # bounce_back_boundary(:z, density_pop, density_buf)
        # bounce_back_boundary(:z, temperature_pop, temperature_buf)
        # dirichlet_boundary(:ylower, density_buf, U_init, density)
        # dirichlet_boundary(:ylower, temperature_buf, U_init, temperature)

        # @parallel (1:Nx, 1:Ny) periodic_boundary_update!(:z, density_pop, density_buf) # needed if not multiple threads in z
        # @parallel (1:Nx, 1:Ny) periodic_boundary_update!(:z, temperature_pop, temperature_buf) # needed if not multiple threads in z
        # @parallel (1:Nx, 1:Nz) periodic_boundary_update!(:y, density_pop, density_buf)
        # @parallel (1:Nx, 1:Nz) periodic_boundary_update!(:y, temperature_pop, temperature_buf)
        # @parallel (1:Ny, 1:Nz) periodic_boundary_update!(:x, density_pop, density_buf)
        # @parallel (1:Ny, 1:Nz) periodic_boundary_update!(:x, temperature_pop, temperature_buf)

        density_pop, density_buf = density_buf, density_pop
        temperature_pop, temperature_buf = temperature_buf, temperature_pop 



        if do_vis && (i % nvis == 0)
            gather!(density, density_v)
            gather!(temperature, temperature_v)
            vel_c = copy(velocity[:, :, Int(ceil((Nz-2)/2)), 1:2])
            for i in axes(vel_c, 1)
                for j in axes(vel_c, 2)
                    vel_c[i, j, :] /= norm(vel_c[i, j, :])
                end
            end

            velx_p = vel_c[1:st:end, 1:st:end, 1]
            vely_p = vel_c[1:st:end, 1:st:end, 2]
            velx_p_g = @zeros(size(vel_c[1:st:end, 1:st:end, 1], 1) * dims[1], size(vel_c[1:st:end, 1:st:end, 1], 2) * dims[2])
            vely_p_g = @zeros(size(vel_c[1:st:end, 1:st:end, 2], 1) * dims[1], size(vel_c[1:st:end, 1:st:end, 2], 2) * dims[2])
            gather!(velx_p, velx_p_g)
            gather!(vely_p, vely_p_g)

            if me == 0
                dens = heatmap(xi_g, yi_g, density_v[:, :, Int(ceil((Nz-2)/2))]'; xlims=(xi_g[1], xi_g[end]), ylims=(yi_g[1], yi_g[end]), aspect_ratio=1, c=:turbo, clim=(0,1), title="density")
                # dens = quiver!(Xp[:], Yp[:]; quiver=(velx_p_g[:], vely_p_g[:]), lw=0.5, c=:black)

                temp = heatmap(xi_g, yi_g, temperature_v[:, :, Int(ceil((Nz-2)/2))]'; xlims=(xi_g[1], xi_g[end]), ylims=(yi_g[1], yi_g[end]), aspect_ratio=1, c=:turbo, clim=(0,1), title="temperature")
                # temp = quiver!(Xp[:], Yp[:]; quiver=(velx_p_g[:], vely_p_g[:]), lw=0.5, c=:black)

                p = plot(dens, temp)
                png(p, "$visdir/$(lpad(iframe += 1, 4, "0")).png")
                save_array("$visdir/out_dens_$(lpad(iframe, 4, "0"))", convert.(Float32, density_v))
                save_array("$visdir/out_temp_$(lpad(iframe, 4, "0"))", convert.(Float32, temperature_v))
            end
        end
    end
    if do_vis && me == 0
        run(`ffmpeg -i $visdir/%4d.png ../docs/3D_MULTI_XPU.mp4 -y`)
    end
    finalize_global_grid()
end

lb()