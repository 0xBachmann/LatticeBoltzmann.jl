using Test
using ParallelStencil

@init_parallel_stencil(Threads, Float64, 3, inbounds=false)

const method = :D3Q19

include("../src/LatticeBoltzmann3D.jl")

function thermal_convection_testing()
    # numerics
    nx_pop = 40
    ny_pop = Int(nx_pop/2)
    nz_pop = nx_pop

    nx_values = nx_pop - 2
    ny_values = ny_pop - 2
    nz_values = nz_pop - 2

    init_global_grid(nx_pop, ny_pop, nz_pop)
    # physics
    lx = 20
    ly = lx * ny_pop / nx_pop
    lz = lx * ny_pop / nx_pop
    dx, dy, dz = lx / nx_values, ly / ny_values, lz / nz_values    

    α = 0.0003 #6.9e-5
    ρ_0 = 1.
    gravity = @SVector [0., -1., 0.]
    αρ_0gravity = α * ρ_0 * gravity
    ΔT = 1. #200.

    ν = 5e-2 # viscosity close to 0 -> near incompressible limit
    # Ra = α * norm(gravity) * ΔT * ly^3 / (viscosity * k)
    κ = α * norm(gravity) * ΔT * ly^3 / (ν * 1000.)

    dt = dx / 10 # to regulate lattice velocity

    # lattice units
    ρ_lattice = 1.
    dx_lattice = 1.
    dt_lattice = 1.

    gravity_lattice = gravity .* dt^2 ./ dx
    α_lattice = α * ΔT
    αρ_0gravity_lattice = α_lattice * ρ_0 / ρ_lattice * gravity_lattice

    ΔT_lattice = 1.
    H_lattice = ly / dx
    κ_lattice = κ * dt / dx^2
    ν_lattice = ν * dt / dx^2

    _τ_temperature_lattice = 1. / (κ_lattice * _cs2 + 0.5)
    _τ_density_lattice = 1. / (ν_lattice * _cs2 + 0.5)

    # timing
    timesteps = 0:1000
    progress = Progress(length(timesteps))
    
    # init
    density_pop = @zeros(nx_pop, ny_pop, nz_pop, celldims=Q)
    density_pop_buf = @zeros(nx_pop, ny_pop, nz_pop, celldims=Q)
    
    temperature_pop = @zeros(nx_pop, ny_pop, nz_pop, celldims=Q)
    temperature_pop_buf = @zeros(nx_pop, ny_pop, nz_pop, celldims=Q)
    
    forces = @zeros(nx_values, ny_values, nz_values, celldims=dimension)
    velocity = @zeros(nx_values, ny_values, nz_values, celldims=dimension)
    density = @ones(nx_values, ny_values, nz_values)
    temperature = Data.Array([ΔT_lattice * exp(
                                            -(x_g(ix, dx, density) - lx / 2)^2
                                            -(y_g(iy, dy, density) - ly / 2)^2
                                            -(z_g(iz, dz, density) - lz / 2)^2
                                            ) for ix = 1:nx_values, iy = 1:ny_values, iz = 1:nz_values])
    
    # boundary and ranges
    inner_range_pop = (2:nx_pop-1, 2:ny_pop-1, 2:nz_pop-1)
    range_values = (1:nx_values, 1:ny_values, 1:nz_values)
    x_boundary_range = 2:nx_pop-1
    y_boundary_range = 2:ny_pop-1
    z_boundary_range = 2:nz_pop-1
    
    @parallel range_values init!(true, true, temperature, ΔT_lattice)
    
    # initialize populations from equilibrium distribution
    @parallel inner_range_pop init_density_pop!(density_pop, velocity, density)
    @parallel inner_range_pop init_temperature_pop!(temperature_pop, velocity, temperature)
    

    # time loop
    for _ in timesteps
        # compute forces
        @parallel range_values compute_force!(forces, temperature, gravity, α, ρ_0)

        # compute moments (density, velocity and temperature)
        @parallel range_values update_moments!(velocity, density, temperature, density_pop, temperature_pop, forces)

        # collsion
        @parallel collision!(density_pop, temperature_pop, velocity, density, temperature, forces, _τ_density_lattice, _τ_temperature_lattice)

        # streaming
        @parallel inner_range_pop streaming!(density_pop, density_pop_buf, temperature_pop, temperature_pop_buf)

        # boundary conditions
        @parallel (y_boundary_range, z_boundary_range) bounce_back_x_left!(density_pop, density_pop_buf)
        @parallel (y_boundary_range, z_boundary_range) bounce_back_x_left!(temperature_pop, temperature_pop_buf)

        @parallel (y_boundary_range, z_boundary_range) bounce_back_x_right!(density_pop, density_pop_buf)
        @parallel (y_boundary_range, z_boundary_range) bounce_back_x_right!(temperature_pop, temperature_pop_buf)

        @parallel (x_boundary_range, y_boundary_range) bounce_back_z_left!(density_pop, density_pop_buf)
        @parallel (x_boundary_range, y_boundary_range) bounce_back_z_left!(temperature_pop, temperature_pop_buf)

        @parallel (x_boundary_range, y_boundary_range) bounce_back_z_right!(density_pop, density_pop_buf)
        @parallel (x_boundary_range, y_boundary_range) bounce_back_z_right!(temperature_pop, temperature_pop_buf)

        @parallel (x_boundary_range, z_boundary_range) bounce_back_y_left!(density_pop, density_pop_buf)
        @parallel (x_boundary_range, z_boundary_range) anti_bounce_back_temperature_y_left!(temperature_pop_buf, velocity, temperature, ΔT_lattice/2)

        @parallel (x_boundary_range, z_boundary_range) bounce_back_y_right!(density_pop, density_pop_buf)
        @parallel (x_boundary_range, z_boundary_range) anti_bounce_back_temperature_y_right!(temperature_pop_buf, velocity, temperature, -ΔT_lattice/2)

        # pointer swap
        density_pop, density_pop_buf = density_pop_buf, density_pop
        temperature_pop, temperature_pop_buf = temperature_pop_buf, temperature_pop 
        
        # progress bar
        next!(progress)
    end
    finalize_global_grid()
    return density, temperature
end

function load_array!(Aname, A)
    fname = string(Aname, ".bin")
    fid=open(fname, "r"); read!(fid, A); close(fid)
end

@testset "testing 3D thermal lbm" begin
    Nx = 38
    Ny = 18
    Nz = 38
    dens, temp = thermal_convection_testing()
    @test all(isfinite, dens)
    @test all(isfinite, temp)
    dens_ref = zeros(Float64, Nx, Ny, Nz)
    temp_ref = zeros(Float64, Nx, Ny, Nz)
    load_array!("out_test_density", dens_ref)
    load_array!("out_test_temperature", temp_ref)
    @show(dens - dens_ref)
    @show(temp - temp_ref)
    @test all(dens .≈ dens_ref)
    @test all(temp .≈ temp_ref)
end