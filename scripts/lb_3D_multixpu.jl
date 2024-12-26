using CellArraysIndexing, StaticArrays
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
const dimension = 3

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
    nx_pop = 40
    ny_pop = 20
    nz_pop = 3

    nx_values = nx_pop - 2
    ny_values = ny_pop - 2
    nz_values = nz_pop - 2

    me, dims, nprocs, coords, comm = init_global_grid(nx_pop, ny_pop, nz_pop, periodx=0, periody=0, periodz=1)

    lx = 20
    ly = 10
    lz = 3

    dx, dy, dz = lx / nx_g(), ly / ny_g(), lz / nz_g()

    density_pop = @zeros(nx_pop, ny_pop, nz_pop, celldims=Q)
    density_pop_buf = @zeros(nx_pop, ny_pop, nz_pop, celldims=Q)
    
    temperature_pop = @zeros(nx_pop, ny_pop, nz_pop, celldims=Q)
    temperature_pop_buf = @zeros(nx_pop, ny_pop, nz_pop, celldims=Q)

    D = 6e-3
    viscosity = 5e-2

    _τ_temperature = 1. / (D * _cs2 + 0.5)
    _τ_density = 1. / (viscosity * _cs2 + 0.5)
    @show 1/_τ_density, 1/_τ_temperature
    Δt = 1. # lattice units

    α = 0.0003
    ρ_0 = 1.
    gravity = @SVector [0., -1., 0] # g = g⋆ * Δx/Δt^2

    nt = 10000
    timesteps = 0:nt


    R = lx / 4
    U_init = @SVector [0., 0.2, 0.]
    ΔT = 1.

    # Ra = α * norm(gravity) * ΔT * ly^3 / (viscosity * k)
    @show(α * norm(gravity) * ΔT * ly^3 / (viscosity * 10)) # eq 8.43 <-- right now these are not unit values?


    velocity = @zeros(nx_values, ny_values, nz_values, celldims=dimension)
    forces = @zeros(nx_values, ny_values, nz_values, celldims=dimension)
    density = @ones(nx_values, ny_values, nz_values)
    boundary = @zeros(nx_values, ny_values, nz_values) # Data.Array([((x_g(ix, dx, density) - lx / 2)^2 + (y_g(iy, dy, density) - ly / 3) ^2) < R^2 ? 1. : 0. for ix = 1:nx_values, iy = 1:ny_values, iz = 1:nz_values])
    temperature = Data.Array([ΔT * exp(-(x_g(ix, dx, density) - lx / 2)^2
                                        -(y_g(iy, dy, density) - ly / 2)^2
                                        # -(z_g(iz, dz, density) - lz / 2)^2
                                        ) for ix = 1:nx_values, iy = 1:ny_values, iz = 1:nz_values])


    do_vis = true
    nvis = 1
    visdir = "visdir"
    st = ceil(Int, nx_values / 20)

    inner_range_pop = (2:nx_pop-1, 2:ny_pop-1, 2:nz_pop-1)
    range_values = (1:nx_values, 1:ny_values, 1:nz_values)
    x_boundary_range = 2:nx_pop-1
    y_boundary_range = 2:ny_pop-1
    z_boundary_range = 2:nz_pop-1
    
    left_boundary_x = coords[1] == 0
    right_boundary_x = coords[1] == dims[1]-1
    left_boundary_y = coords[2] == 0
    right_boundary_y = coords[2] == dims[2]-1
    left_boundary_z = coords[3] == 0
    right_boundary_z = coords[3] == dims[3]-1
    if left_boundary_y || right_boundary_y
        @parallel range_values init!(left_boundary_y, right_boundary_y, velocity, temperature, boundary, U_init, ΔT)
    end
    
    @parallel inner_range_pop init_density_pop!(density_pop, velocity, density)
    @parallel inner_range_pop init_temperature_pop!(temperature_pop, velocity, temperature)

    # # @parallel (1:Nx, 1:Ny) periodic_boundary_update!(:z, density_pop, density_buf)
    # # @parallel (1:Nx, 1:Ny) periodic_boundary_update!(:z, temperature_pop, temperature_buf)
    # # @parallel (1:Nx, 1:Nz) periodic_boundary_update!(:y, density_pop, density_buf)
    # # @parallel (1:Nx, 1:Nz) periodic_boundary_update!(:y, temperature_pop, temperature_buf)
    # # @parallel (1:Ny, 1:Nz) periodic_boundary_update!(:x, density_pop, density_buf)
    # # @parallel (1:Ny, 1:Nz) periodic_boundary_update!(:x, temperature_pop, temperature_buf)
    # # @parallel (2:Ny-1, 2:Nz-1) bounce_back_boundary!(:x, density_pop, density_buf)
    # # @parallel (2:Ny-1, 2:Nz-1) bounce_back_boundary!(:x, temperature_pop, temperature_buf)
    

    if do_vis
        ENV["GKSwstype"]="nul"
        if (me==0) if isdir("$visdir")==false mkdir("$visdir") end; loadpath="$visdir/"; anim=Animation(loadpath,String[]); println("Animation directory: $(anim.dir)") end
        nx_v, ny_v, nz_v = (nx_values) * dims[1], (ny_values) * dims[2], (nz_values) * dims[3]
        (2 * nx_v * ny_v * nz_v * sizeof(Data.Number) > 0.8 * Sys.free_memory()) && error("Not enough memory for visualization.")
        density_v = zeros(nx_v, ny_v, nz_v) # global array for visu
        temperature_v = zeros(nx_v, ny_v, nz_v) # global array for visu
        xi_g, yi_g = LinRange(0, lx, nx_v), LinRange(0, ly, ny_v) # inner points only
        iframe = 0
        Xc, Yc = [x for x in xi_g, _ in yi_g], [y for _ in xi_g, y in yi_g]
        Xp, Yp = Xc[1:st:end, 1:st:end], Yc[1:st:end, 1:st:end]
    end

    for i in (me == 0 ? ProgressBar(timesteps) : timesteps)
        if do_vis && (i % nvis == 0)
            gather!(density, density_v)
            gather!(temperature, temperature_v)
            # vel_c = copy(velocity[:, :, Int(ceil((Nz-2)/2))])
            # for i in axes(vel_c, 1)
            #     for j in axes(vel_c, 2)
            #         vel_c[i, j] /= norm(vel_c[i, j])
            #     end
            # end

            # velx_p = Data.Array([@index vel_c[1, i, j] for i in 1:st:Nx for j in 1:st:Ny])
            # vely_p = Data.Array([@index vel_c[2, i, j] for i in 1:st:Nx for j in 1:st:Ny])
            # velx_p_g = @zeros(size(vel_c[1:st:end, 1:st:end, 1], 1) * dims[1], size(vel_c[1:st:end, 1:st:end, 1], 2) * dims[2])
            # vely_p_g = @zeros(size(vel_c[1:st:end, 1:st:end, 2], 1) * dims[1], size(vel_c[1:st:end, 1:st:end, 2], 2) * dims[2])
            # gather!(velx_p, velx_p_g)
            # gather!(vely_p, vely_p_g)

            if me == 0
                dens = heatmap(xi_g, yi_g, Array(density_v[:, :, Int(ceil(nz_values/2))])'; xlims=(xi_g[1], xi_g[end]), ylims=(yi_g[1], yi_g[end]), aspect_ratio=1, c=:turbo, clim=(0,1), title="density")
                # dens = quiver!(Xp[:], Yp[:]; quiver=(velx_p[:], vely_p[:]), lw=0.5, c=:black)

                temp = heatmap(xi_g, yi_g, Array(temperature_v[:, :, Int(ceil(nz_values/2))])'; xlims=(xi_g[1], xi_g[end]), ylims=(yi_g[1], yi_g[end]), aspect_ratio=1, c=:turbo, clim=(-ΔT/2,ΔT/2), title="temperature")
                # temp = quiver!(Xp[:], Yp[:]; quiver=(velx_p[:], vely_p[:]), lw=0.5, c=:black)

                p = plot(dens, temp, layout=(2, 1))
                png(p, "$visdir/$(lpad(iframe += 1, 4, "0")).png")
                save_array("$visdir/out_dens_$(lpad(iframe, 4, "0"))", convert.(Float32, Array(density_v)))
                save_array("$visdir/out_temp_$(lpad(iframe, 4, "0"))", convert.(Float32, Array(temperature_v)))
            end
        end

        # collision_temperature
        # boundary condition temperature
        # streaming temperature
        # update temperature

        # compute buoyancy force
        # collision density, taking force into account
        # boundary condition density
        # streaming density
        # update velocity and density

        @parallel range_values compute_force!(forces, temperature, gravity, α, ρ_0)
        @parallel range_values update_moments!(velocity, density, temperature, density_pop, temperature_pop, forces)
        @parallel range_values apply_external_force!(velocity, boundary)

        @parallel inner_range_pop collision_density!(density_pop, velocity, density, forces, _τ_density)
        @parallel inner_range_pop collision_temperature!(temperature_pop, velocity, temperature, _τ_temperature)


        # @parallel (1:nx_pop, 1:ny_pop) periodic_boundary_z!(density_pop)
        # @parallel (1:nx_pop, 1:ny_pop) periodic_boundary_z!(temperature_pop)
        # @parallel (1:nx_pop, 1:nz_pop) periodic_boundary_y!(density_pop)
        # @parallel (1:nx_pop, 1:nz_pop) periodic_boundary_y!(temperature_pop)
        # @parallel (1:ny_pop, 1:nz_pop) periodic_boundary_x!(density_pop)
        # @parallel (1:ny_pop, 1:nz_pop) periodic_boundary_x!(temperature_pop)

        

        # @parallel (y_boundary_range, z_boundary_range) bounce_back_x!(density_pop)
        # @parallel (y_boundary_range, z_boundary_range) bounce_back_x!(temperature_pop)
        # @parallel (x_boundary_range, z_boundary_range) bounce_back_y!(density_pop)
        # # @parallel (x_boundary_range, z_boundary_range) bounce_back_y!(temperature_pop)
        # @parallel (x_boundary_range, z_boundary_range) anti_bounce_back_temperature_y!(temperature_pop, velocity, temperature, ΔT/2, -ΔT/2)

        update_halo!(density_pop, temperature_pop)

        @parallel inner_range_pop streaming!(density_pop, density_pop_buf)
        @parallel inner_range_pop streaming!(temperature_pop, temperature_pop_buf)

        @parallel (y_boundary_range, z_boundary_range) bounce_back_x!(left_boundary_x, right_boundary_x, density_pop, density_pop_buf)
        @parallel (y_boundary_range, z_boundary_range) bounce_back_x!(left_boundary_x, right_boundary_x, temperature_pop, temperature_pop_buf)
        @parallel (x_boundary_range, z_boundary_range) bounce_back_y!(left_boundary_y, right_boundary_y, density_pop, density_pop_buf)
        # @parallel (x_boundary_range, z_boundary_range) bounce_back_y!(left_boundary_y, right_boundary_y, temperature_pop, temperature_pop_buf)
        @parallel (x_boundary_range, z_boundary_range) anti_bounce_back_temperature_y!(left_boundary_y, right_boundary_y, temperature_pop, temperature_pop_buf, velocity, temperature, ΔT/2, -ΔT/2)


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

        density_pop, density_pop_buf = density_pop_buf, density_pop
        temperature_pop, temperature_pop_buf = temperature_pop_buf, temperature_pop 
        
    end
    if do_vis && me == 0
        run(`ffmpeg -i $visdir/%4d.png ../docs/3D_MULTI_XPU.mp4 -y`)
    end
    finalize_global_grid()
end

lb()