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
    @init_parallel_stencil(CUDA, Float64, 3, inbounds=true)
else
    @init_parallel_stencil(Threads, Float64, 3, inbounds=true)
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
    ny_pop = Int(nx_pop/2)
    nz_pop = nx_pop

    nx_values = nx_pop - 2
    ny_values = ny_pop - 2
    nz_values = nz_pop - 2

    me, dims, nprocs, coords, comm = init_global_grid(nx_pop, ny_pop, nz_pop, periodx=0, periody=0, periodz=0)

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
    αρ_0gravity = α * ρ_0 * gravity

    nt = 1000
    startup = Int(nt/10)
    t_tic = 0.
    timesteps = 0:nt


    R = lx / 4
    U_init = @SVector [0., 0.2, 0.]
    ΔT = 1.

    # Ra = α * norm(gravity) * ΔT * ly^3 / (viscosity * k)
    @show(α * norm(gravity) * ΔT * ly^3 / (viscosity * 10)) # eq 8.43 <-- right now these are not unit values?


    velocity = @zeros(nx_values, ny_values, nz_values, celldims=dimension)
    # forces = @zeros(nx_values, ny_values, nz_values, celldims=dimension)
    density = @ones(nx_values, ny_values, nz_values)
    boundary = @zeros(nx_values, ny_values, nz_values) # Data.Array([((x_g(ix, dx, density) - lx / 2)^2 + (y_g(iy, dy, density) - ly / 3) ^2) < R^2 ? 1. : 0. for ix = 1:nx_values, iy = 1:ny_values, iz = 1:nz_values])
    temperature = Data.Array([ΔT * exp(-(x_g(ix, dx, density) - lx / 2)^2
                                        -(y_g(iy, dy, density) - ly / 2)^2
                                        -(z_g(iz, dz, density) - lz / 2)^2
                                        ) for ix = 1:nx_values, iy = 1:ny_values, iz = 1:nz_values])


    do_vis = true
    nvis = 10
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

    if do_vis
        ENV["GKSwstype"]="nul"
        if (me==0) if isdir("$visdir")==false mkdir("$visdir") end; loadpath="$visdir/"; anim=Animation(loadpath,String[]); println("Animation directory: $(anim.dir)") end
        nx_v, ny_v, nz_v = (nx_values) * dims[1], (ny_values) * dims[2], (nz_values) * dims[3]
        (2 * nx_v * ny_v * nz_v * sizeof(Data.Number) > 0.8 * Sys.free_memory()) && error("Not enough memory for visualization.")
        density_v = @zeros(nx_v, ny_v, nz_v) # global array for visu
        temperature_v = @zeros(nx_v, ny_v, nz_v) # global array for visu
        xi_g, yi_g = LinRange(0, lx, nx_v), LinRange(0, ly, ny_v) # inner points only
        iframe = 0
        Xc, Yc = [x for x in xi_g, _ in yi_g], [y for _ in xi_g, y in yi_g]
        Xp, Yp = Xc[1:st:end, 1:st:end], Yc[1:st:end, 1:st:end]
    end

    for i in (me == 0 ? ProgressBar(timesteps) : timesteps)
        if i == startup
            t_tic = Base.time()
        end
        if do_vis && (i % nvis == 0)
            # gather!(density, density_v)
            # gather!(temperature, temperature_v)
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
                dens = heatmap(xi_g, yi_g, Array(density[:, :, Int(ceil(nz_values/2))])'; xlims=(xi_g[1], xi_g[end]), ylims=(yi_g[1], yi_g[end]), aspect_ratio=1, c=:turbo, clim=(0,1), title="density")
                # dens = quiver!(Xp[:], Yp[:]; quiver=(velx_p[:], vely_p[:]), lw=0.5, c=:black)

                temp = heatmap(xi_g, yi_g, Array(temperature[:, :, Int(ceil(nz_values/2))])'; xlims=(xi_g[1], xi_g[end]), ylims=(yi_g[1], yi_g[end]), aspect_ratio=1, c=:turbo, clim=(-ΔT/2,ΔT/2), title="temperature")
                # temp = quiver!(Xp[:], Yp[:]; quiver=(velx_p[:], vely_p[:]), lw=0.5, c=:black)

                p = plot(dens, temp, layout=(2, 1))
                png(p, "$visdir/$(lpad(iframe += 1, 4, "0")).png")
                save_array("$visdir/out_dens_$(lpad(iframe, 4, "0"))", convert.(Float32, Array(density)))
                save_array("$visdir/out_temp_$(lpad(iframe, 4, "0"))", convert.(Float32, Array(temperature)))
            end
        end

        @parallel range_values update_moments!(velocity, density, temperature, density_pop, temperature_pop, αρ_0gravity)

        @hide_communication (8, 8, 4) computation_calls=1 begin
            # @parallel collision_density!(density_pop, velocity, density, forces, _τ_density)
            # @parallel collision_temperature!(temperature_pop, velocity, temperature, _τ_temperature)
            @parallel collision(density_pop, temperature_pop, velocity, density, temperature, αρ_0gravity, _τ_density, _τ_temperature)
            update_halo!(density_pop, temperature_pop)
        end

        # @parallel inner_range_pop streaming!(density_pop, density_pop_buf)
        # @parallel inner_range_pop streaming!(temperature_pop, temperature_pop_buf)
        @parallel inner_range_pop streaming!(density_pop, density_pop_buf, temperature_pop, temperature_pop_buf)


        if left_boundary_x 
            @parallel (y_boundary_range, z_boundary_range) bounce_back_x_left!(density_pop, density_pop_buf)
            @parallel (y_boundary_range, z_boundary_range) bounce_back_x_left!(temperature_pop, temperature_pop_buf)
            @parallel (y_boundary_range, z_boundary_range) bounce_back_x_left!(temperature_pop, temperature_pop_buf)
        end

        if right_boundary_x
            @parallel (y_boundary_range, z_boundary_range) bounce_back_x_right!(density_pop, density_pop_buf)
            @parallel (y_boundary_range, z_boundary_range) bounce_back_x_right!(density_pop, density_pop_buf)
            @parallel (y_boundary_range, z_boundary_range) bounce_back_x_right!(temperature_pop, temperature_pop_buf)
        end

        if left_boundary_z
            @parallel (x_boundary_range, y_boundary_range) bounce_back_z_left!(density_pop, density_pop_buf)
            @parallel (x_boundary_range, y_boundary_range) bounce_back_z_left!(temperature_pop, temperature_pop_buf)
        end
         
        if right_boundary_z
            @parallel (x_boundary_range, y_boundary_range) bounce_back_z_right!(density_pop, density_pop_buf)
            @parallel (x_boundary_range, y_boundary_range) bounce_back_z_right!(temperature_pop, temperature_pop_buf)
        end

        if left_boundary_y
            @parallel (x_boundary_range, z_boundary_range) bounce_back_y_left!(density_pop, density_pop_buf)
            # @parallel (x_boundary_range, z_boundary_range) bounce_back_y!(temperature_pop, temperature_pop_buf)
            @parallel (x_boundary_range, z_boundary_range) anti_bounce_back_temperature_y_left!(temperature_pop_buf, velocity, temperature, -ΔT/2)
    
        end
         
        if right_boundary_y
            @parallel (x_boundary_range, z_boundary_range) bounce_back_y_right!(density_pop, density_pop_buf)
            # @parallel (x_boundary_range, z_boundary_range) bounce_back_y!(temperature_pop, temperature_pop_buf)
            @parallel (x_boundary_range, z_boundary_range) anti_bounce_back_temperature_y_right!(temperature_pop_buf, velocity, temperature, ΔT/2)
        end



        density_pop, density_pop_buf = density_pop_buf, density_pop
        temperature_pop, temperature_pop_buf = temperature_pop_buf, temperature_pop 
        
    end
    if do_vis && me == 0
        run(`ffmpeg -i $visdir/%4d.png ../docs/3D_MULTI_XPU.mp4 -y`)
    end

    t_toc = Base.time() - t_tic
    A_eff = 2 * (
        4 * 19 + # pop and buff for density and temperature
        2 * 3 + # velocity and forces
        2 # density and temperature
    )/1e9*nx_pop*ny_pop*nz_pop*sizeof(Float64)  # Effective main memory access per iteration [GB]
    niter = length(timesteps) - startup
    t_it = t_toc / niter
    T_eff = A_eff/t_it 
    if me == 0
        println("Time = $t_toc sec, T_eff = $T_eff GB/s (niter = $niter)")
    end

    finalize_global_grid()
end

lb()