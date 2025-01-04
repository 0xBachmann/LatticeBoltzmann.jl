using CellArraysIndexing, StaticArrays
using ParallelStencil
using ImplicitGlobalGrid
import MPI

using Plots
using ParallelStencil
using LinearAlgebra
using ProgressMeter


@static if method == :D3Q15
    const Q = 15
    const directions = SA[
        SA[0, 0, 0], 
        SA[1, 0, 0], SA[-1, 0, 0], SA[0, 1, 0], SA[0, -1, 0], SA[0, 0, 1], SA[0, 0, -1], 
        SA[1, 1, 1], SA[-1, -1, -1], SA[1, 1, -1], SA[-1, -1, 1], SA[1, -1, 1], SA[-1, 1, -1], SA[-1, 1, 1], SA[1, -1, -1]
        ]
    const weights = SA[
        2/9, 
        1/9, 1/9, 1/9, 1/9, 1/9, 1/9, 
        1/72, 1/72, 1/72, 1/72, 1/72, 1/72, 1/72, 1/72
        ]
    const dimension = 3
elseif method == :D3Q19
    const Q = 19
    const directions = SA[
        SA[0, 0, 0], 
        SA[1, 0, 0], SA[-1, 0, 0], SA[0, 1, 0], SA[0, -1, 0], SA[0, 0, 1], SA[0, 0, -1], 
        SA[1, 1, 0], SA[-1, -1, 0], SA[1, -1, 0], SA[-1, 1, 0], SA[1, 0, 1], SA[-1, 0, -1], SA[-1, 0, 1], SA[1, 0, -1], SA[0, 1, 1], SA[0, -1, -1], SA[0, 1, -1], SA[0, -1, 1]
        ]
    const weights = SA[
        1/3, 1/18, 1/18, 1/18, 1/18, 1/18, 1/18, 1/36, 1/36, 1/36, 1/36, 1/36, 1/36, 1/36, 1/36, 1/36, 1/36, 1/36, 1/36
        ]
        const dimension = 3
elseif method == :D3Q27
    const Q = 27
    const directions = SA[
        SA[0, 0, 0],
        SA[1, 0, 0], SA[-1, 0, 0], SA[0, 1, 0], SA[0, -1, 0], SA[0, 0, 1], SA[0, 0, -1], 
        SA[1, 1, 0], SA[-1, -1, 0], SA[1, -1, 0], SA[-1, 1, 0], SA[1, 0, 1], SA[-1, 0, -1], SA[-1, 0, 1], SA[1, 0, -1], SA[0, 1, 1], SA[0, -1, -1], SA[0, 1, -1], SA[0, -1, 1],
        SA[1, 1, 1], SA[-1, -1, -1], SA[1, 1, -1], SA[-1, -1, 1], SA[1, -1, 1], SA[-1, 1, -1], SA[-1, 1, 1], SA[1, -1, -1]
    ]
    const weights = SA[
        8/27,
        2/27, 2/27, 2/27, 2/27, 2/27, 2/27,
        1/54, 1/54, 1/54, 1/54, 1/54, 1/54, 1/54, 1/54, 1/54, 1/54, 1/54, 1/54,
        1/216, 1/216, 1/216, 1/216, 1/216, 1/216, 1/216, 1/216
    ]
    const dimension = 3
else
    @assert false "method not defined"
end
@assert Q == size(directions, 1) "Q=$Q, size of directions=$(size(directions, 1))"
@assert Q == size(weights, 1) "Q=$Q, size of weights=$(size(weights, 1))"
    
# Speed of sound (in lattice units)
const _cs2 = 3. # cs^2 = 1./3. * (dx**2/dt**2)
const _cs4 = 9.

"""
    @parallel collision!(density_pop, temperature_pop, velocity, density, temperature, gravity, _τ_density, _τ_temperature)
    
Applies collision operator to the density and temperature populations.

# Arguments
- `density_pop::CellArray`: Density population distribution function.
- `temperature_pop::CellArray`: Temperature population distribution function.
- `velocity::CellArray`: Local velocity at each point in the domain.
- `density::Array`: Local density at each point in the domain.
- `temperature::Array`: Local temperature at each point in the domain.
- `forces::CellArray`: Local external forces at each point in the domain.
- `_τ_density::Float64`: Inverse relaxation time for density.
- `_τ_temperature::Float64`: Inverse relaxation time for temperature.
"""
@parallel_indices (i, j, k) function collision!(density_pop, temperature_pop, velocity, density, temperature, forces, _τ_density, _τ_temperature)
    if (1 < i && i < size(density_pop, 1)) && (1 < j && j < size(density_pop, 2)) && (1 < k && k < size(density_pop, 3))
        v = velocity[i-1, j-1, k-1]
        for q in 1:Q
            @index density_pop[q, i, j, k] = (1. - _τ_density) * @index(density_pop[q, i, j, k]) + _τ_density * f_eq(q, v, density[i-1, j-1, k-1]) + source_term(q, v, forces[i-1, j-1, k-1], _τ_density)
            @index temperature_pop[q, i, j, k] = (1. - _τ_temperature) * @index(temperature_pop[q, i, j, k]) + _τ_temperature * temp_eq(q, v, temperature[i-1, j-1, k-1])
        end
    end
    return
end

"""
    @parallel (1:ny, 1:nz) periodic_boundary_x!(pop)

Enforces periodic boundary conditions along the x-direction.

# Arguments
- `pop::{Array, CellArray}`: Population distribution function to be updated.
"""
@parallel_indices (i, j) function periodic_boundary_x!(pop)
    pop[1, i, j] = pop[end-1, i, j]
    pop[end, i, j] = pop[2, i, j]
    return
end

"""
    @parallel (1:nx, 1:nz) periodic_boundary_y!(pop)

Enforces periodic boundary conditions along the y-direction.

# Arguments
- `pop::{Array, CellArray}`: Population distribution function to be updated.
"""
@parallel_indices (i, k) function periodic_boundary_y!(pop)
    pop[i, 1, k] = pop[i, end-1, k]
    pop[i, end, k] = pop[i, 2, k]
    return
end

"""
    @parallel (1:nx, 1:ny) periodic_boundary_z!(pop)

Enforces periodic boundary conditions along the z-direction.

# Arguments
- `pop::{Array, CellArray}`: Population distribution function to be updated.
"""
@parallel_indices (i, j) function periodic_boundary_z!(pop)
    pop[i, j, 1] = pop[i, j, end-1]
    pop[i, j, end] = pop[i, j, 2]
    return
end


"""
    @parallel (2:ny-1, 2:nz-1) anti_bounce_back_temperature_x_right!(pop_buf, velocity, values, dirichlet_value) 

Enforces Dirichlet temperature boundary conditions via the anti-bounce-back method at the right boundary along the x-direction.

# Arguments
- `pop_buf::CellArray`: Buffer for the updated Temperature population distribution.
- `velocity::CellArray`: Local velocity at each point.
- `values::Array`: Local values of a field (i.e. temperature).
- `dirichlet_value::Float64`: Dirichlet boundary condition value for temperature.
"""
@parallel_indices (j, k) function anti_bounce_back_temperature_x_right!(pop_buf, velocity, values, dirichlet_value) 
    Nx = size(pop_buf, 1)
    for q in 1:Q
        if directions[q][1] == 1
            flipped_dir = (q%2 == 0) ? q+1 : q-1
            value = -temp_eq(q, velocity[Nx - 2, j - 1, k - 1], values[Nx - 2, j - 1, k - 1]) + 2 * weights[q] * dirichlet_value
            @index pop_buf[flipped_dir, Nx - 1, j, k] = value
        end
    end 
    return
end

"""
    @parallel (2:ny-1, 2:nz-1) anti_bounce_back_temperature_x_left!(pop_buf, velocity, values, dirichlet_value) 

Enforces Dirichlet temperature boundary conditions via the anti-bounce-back method at the left boundary along the x-direction.

# Arguments
- `pop_buf::CellArray`: Buffer for the updated Temperature population distribution.
- `velocity::CellArray`: Local velocity at each point.
- `values::Array`: Local values of a field (i.e. temperature).
- `dirichlet_value::Float64`: Dirichlet boundary condition value for temperature.
"""
@parallel_indices (j, k) function anti_bounce_back_temperature_x_left!(pop_buf, velocity, values, dirichlet_value) 
    for q in 1:Q
        if directions[q][1] == -1
            flipped_dir = (q%2 == 0) ? q+1 : q-1
            value = -temp_eq(q, velocity[1, j - 1, k - 1], values[1, j - 1, k - 1]) + 2 * weights[q] * dirichlet_value
            @index pop_buf[flipped_dir, 2, j, k] = value
        end
    end
    return
end

"""
    @parallel (2:nx-1, 2:nz-1) anti_bounce_back_temperature_y_right!(pop_buf, velocity, values, dirichlet_value) 

Enforces Dirichlet temperature boundary conditions via the anti-bounce-back method at the right boundary along the y-direction.

# Arguments
- `pop_buf::CellArray`: Buffer for the updated Temperature population distribution.
- `velocity::CellArray`: Local velocity at each point.
- `values::Array`: Local values of a field (i.e. temperature).
- `dirichlet_value::Float64`: Dirichlet boundary condition value for temperature.
"""
@parallel_indices (i, k) function anti_bounce_back_temperature_y_right!(pop_buf, velocity, values, dirichlet_value) 
    Ny = size(pop_buf, 2)
    for q in 1:Q
        if directions[q][2] == 1
            flipped_dir = (q%2 == 0) ? q+1 : q-1
            value = -temp_eq(q, velocity[i - 1, Ny - 2, k - 1], values[i - 1, Ny - 2, k - 1]) + 2 * weights[q] * dirichlet_value
            @index pop_buf[flipped_dir, i, Ny - 1, k] = value
        end
    end
    return 
end

"""
    @parallel (2:nx-1, 2:nz-1) anti_bounce_back_temperature_y_left!(pop_buf, velocity, values, dirichlet_value) 

Enforces Dirichlet temperature boundary conditions via the anti-bounce-back method at the left boundary along the y-direction.

# Arguments
- `pop_buf::CellArray`: Buffer for the updated Temperature population distribution.
- `velocity::CellArray`: Local velocity at each point.
- `values::Array`: Local values of a field (i.e. temperature).
- `dirichlet_value::Float64`: Dirichlet boundary condition value for temperature.
"""
@parallel_indices (i, k) function anti_bounce_back_temperature_y_left!(pop_buf, velocity, values, dirichlet_value) 
    for q in 1:Q
        if directions[q][2] == -1
            flipped_dir = (q%2 == 0) ? q+1 : q-1
            value = -temp_eq(q, velocity[i - 1, 1, k - 1], values[i - 1, 1, k - 1]) + 2 * weights[q] * dirichlet_value
            @index pop_buf[flipped_dir, i, 2, k] = value
        end
    end
    return
end

"""
    @parallel (2:nx-1, 2:ny-1) anti_bounce_back_temperature_z_right!(pop_buf, velocity, values, dirichlet_value) 

Enforces Dirichlet temperature boundary conditions via the anti-bounce-back method at the right boundary along the z-direction.

# Arguments
- `pop_buf::CellArray`: Buffer for the updated Temperature population distribution.
- `velocity::CellArray`: Local velocity at each point.
- `values::Array`: Local values of a field (i.e. temperature).
- `dirichlet_value::Float64`: Dirichlet boundary condition value for temperature.
"""
@parallel_indices (i, j) function anti_bounce_back_temperature_z_right!(pop_buf, velocity, values, dirichlet_value) 
    Nz = size(pop_buf, 3)
    for q in 1:Q
        if directions[q][3] == 1
            flipped_dir = (q%2 == 0) ? q+1 : q-1
            value = -temp_eq(q, velocity[i - 1, j - 1, Nz - 2], values[i - 1, j - 1, Nz - 2]) + 2 * weights[q] * dirichlet_value
            @index pop_buf[flipped_dir, i, j, Nz - 1] = value
        end
    end
    return
end

"""
    @parallel (2:nx-1, 2:ny-1) anti_bounce_back_temperature_z_left!(pop_buf, velocity, values, dirichlet_value) 

Enforces Dirichlet temperature boundary conditions via the anti-bounce-back method at the left boundary along the z-direction.

# Arguments
- `pop_buf::CellArray`: Buffer for the updated Temperature population distribution.
- `velocity::CellArray`: Local velocity at each point.
- `values::Array`: Local values of a field (i.e. temperature).
- `dirichlet_value::Float64`: Dirichlet boundary condition value for temperature.
"""
@parallel_indices (i, j) function anti_bounce_back_temperature_z_left!(pop_buf, velocity, values, dirichlet_value) 
    for q in 1:Q
        if directions[q][3] == -1
            flipped_dir = (q%2 == 0) ? q+1 : q-1
            value = -temp_eq(q, velocity[i - 1, j - 1, 1], values[i - 1, j - 1, 1]) + 2 * weights[q] * dirichlet_value
            @index pop_buf[flipped_dir, i, j, 2] = value
        end
    end
    return
end

"""
    @parallel (2:ny-1, 2:nz-1) bounce_back_x_right!(pop, pop_buf)

Enforces No-Slip boundary conditions via the bounce-back method at the right boundary along the x-direction.

# Arguments
- `pop::CellArray`: Population distribution function.
- `pop_buf::CellArray`: Buffer for the updated population distribution.
"""
@parallel_indices (j, k) function bounce_back_x_right!(pop, pop_buf)
    Nx = size(pop, 1)
    for q in 1:Q
        if directions[q][1] == 1
            flipped_dir = (q%2 == 0) ? q+1 : q-1
            flipped_value = @index pop[q, Nx-1, j, k]
            @index pop_buf[flipped_dir, Nx-1, j, k] = flipped_value
        end
    end
    return
end

"""
    @parallel (2:ny-1, 2:nz-1) bounce_back_x_left!(pop, pop_buf)

Enforces No-Slip boundary conditions via the bounce-back method at the left boundary along the x-direction.

# Arguments
- `pop::CellArray`: Population distribution function.
- `pop_buf::CellArray`: Buffer for the updated population distribution.
"""
@parallel_indices (j, k) function bounce_back_x_left!(pop, pop_buf)
    for q in 1:Q
        if directions[q][1] == -1
            flipped_dir = (q%2 == 0) ? q+1 : q-1
            flipped_value = @index pop[q, 2, j, k]
            @index pop_buf[flipped_dir, 2, j, k] = flipped_value
        end
    end   
    return
end

"""
    @parallel (2:nx-1, 2:nz-1) bounce_back_y_right!(pop, pop_buf)

Enforces No-Slip boundary conditions via the bounce-back method at the right boundary along the y-direction.

# Arguments
- `pop::CellArray`: Population distribution function.
- `pop_buf::CellArray`: Buffer for the updated population distribution.
"""
@parallel_indices (i, k) function bounce_back_y_right!(pop, pop_buf)
    Ny = size(pop, 2)
    for q in 1:Q
        if directions[q][2] == 1
            flipped_dir = (q%2 == 0) ? q+1 : q-1
            flipped_value = @index pop[q, i, Ny - 1, k]
            @index pop_buf[flipped_dir, i, Ny - 1, k] = flipped_value
        end
    end
    return
end

"""
    @parallel (2:nx-1, 2:nz-1) bounce_back_y_left!(pop, pop_buf)

Enforces No-Slip boundary conditions via the bounce-back method at the left boundary along the y-direction.

# Arguments
- `pop::CellArray`: Population distribution function.
- `pop_buf::CellArray`: Buffer for the updated population distribution.
"""
@parallel_indices (i, k) function bounce_back_y_left!(pop, pop_buf)
    for q in 1:Q
        if directions[q][2] == -1
            flipped_dir = (q%2 == 0) ? q+1 : q-1
            flipped_value = @index pop[q, i, 2, k]
            @index pop_buf[flipped_dir, i, 2, k] = flipped_value
        end
    end   
    return
end

"""
    @parallel (2:nx-1, 2:ny-1) bounce_back_z_right!(pop, pop_buf)

Enforces No-Slip boundary conditions via the bounce-back method at the right boundary along the z-direction.

# Arguments
- `pop::CellArray`: Population distribution function.
- `pop_buf::CellArray`: Buffer for the updated population distribution.
"""
@parallel_indices (i, j) function bounce_back_z_right!(pop, pop_buf)
    Nz = size(pop, 3)
    for q in 1:Q
        if directions[q][3] == 1
            flipped_dir = (q%2 == 0) ? q+1 : q-1
            flipped_value = @index pop[q, i, j, Nz - 1]
            @index pop_buf[flipped_dir, i, j, Nz - 1] = flipped_value
        end
    end
    return
end

"""
    @parallel (2:nx-1, 2:ny-1) bounce_back_z_left!(pop, pop_buf)

Enforces No-Slip boundary conditions via the bounce-back method at the left boundary along the z-direction.

# Arguments
- `pop::CellArray`: Population distribution function.
- `pop_buf::CellArray`: Buffer for the updated population distribution.
"""
@parallel_indices (i, j) function bounce_back_z_left!(pop, pop_buf)
    for q in 1:Q
        if directions[q][3] == -1
            flipped_dir = (q%2 == 0) ? q+1 : q-1
            flipped_value = @index pop[q, i, j, 2]
            @index pop_buf[flipped_dir, i, j, 2] = flipped_value
        end
    end   
    return
end

"""
    @parallel (2:nx-1, 2:ny-1, 2_nz-1) streaming!(density_pop, density_pop_buf, temperature_pop, temperature_pop_buf)

Performs streaming for density and temperature populations.

# Arguments
- `density_pop::CellArray`: Density population distribution function.
- `density_pop_buf::CellArray`: Buffer for the updated density population.
- `temperature_pop::CellArray`: Temperature population distribution function.
- `temperature_pop_buf::CellArray`: Buffer for the updated temperature population.
"""
@parallel_indices (i, j, k) function streaming!(density_pop, density_pop_buf, temperature_pop, temperature_pop_buf)
    for q in 1:Q
        @index density_pop_buf[q, i, j, k] = @index density_pop[q, i - directions[q][1], j - directions[q][2], k - directions[q][3]]
        @index temperature_pop_buf[q, i, j, k] = @index temperature_pop[q, i - directions[q][1], j - directions[q][2], k - directions[q][3]]
    end
    return
end

@parallel_indices (i, j, k) function init!(left_boundary, right_boundary, temperature, ΔT)    
    if left_boundary && j == 1
        temperature[i, j, k] = ΔT / 2
    elseif right_boundary && j == size(temperature, 2)
        temperature[i, j, k] = -ΔT / 2
    end
    return
end

"""
    @parallel (1:nx, 1:ny, 1:nz) update_moments!(velocity, density, temperature, density_pop, temperature_pop, gravity)
Updates the macroscopic moments (velocity, density, temperature).

# Arguments
- `velocity::CellArray`: Velocity field to be updated.
- `density::Array`: Density field to be updated.
- `temperature::Array`: Temperature field to be updated.
- `density_pop::CellArray`: Density population distribution function.
- `temperature_pop::CellArray`: Temperature population distribution function.
- `forces::CellArray`: External forces (gravity).
"""
@parallel_indices (i, j, k) function update_moments!(velocity, density, temperature, density_pop, temperature_pop, forces)
    cell_density = 0.
    cell_velocity = @SVector zeros(3)
    cell_temperature = 0.
    for q in 1:Q
        cell_density += @index density_pop[q, i + 1, j + 1, k + 1]
        cell_temperature += @index temperature_pop[q, i + 1, j + 1, k + 1]
        cell_velocity += directions[q] * @index density_pop[q, i + 1, j + 1, k + 1]
    end

    cell_velocity += forces[i, j, k] / 2
    velocity[i, j, k] = cell_velocity / cell_density
    density[i, j, k] = cell_density
    temperature[i, j, k] = cell_temperature
    return
end

"""
    f_eq(q, velocity, density)
Computes the equilibrium distribution function for density.

# Arguments
- `q::Int`: Index of the discrete velocity direction.
- `velocity::SVector`: Local velocity at the given point.
- `density::Float`: Local density at the given point.

# Returns
- `Float`: The equilibrium distribution function for density.

The function is based on the discrete velocity directions and weights, incorporating 
macroscopic variables density and velocity. It includes second-order terms to
enhance accuracy.
"""
@inline function f_eq(q, velocity, density)
    uc = dot(velocity, directions[q])
    uc2 = uc^2
    u2 = norm(velocity) ^ 2
    return weights[q] * density * (1. + uc * _cs2 + 0.5 * uc2 * _cs4 - 0.5 * u2 * _cs2)
end

"""
    source_term(q, velocity, force, _τ) 
Computes the source term originating from external forces.

# Arguments
- `q::Int`: Index of the discrete velocity direction.
- `velocity::SVector`: Local velocity at the given point.
- `force::SVector`: External force vector acting on the fluid.
- `_τ::Float`: Relaxation time for the collision operator.

# Returns
- `Float`: The source term for the given velocity direction.

The source term accounts for the influence of external forces on the population distribution function (e.g. from buoyancy).
"""
@inline function source_term(q, velocity, force, _τ)
    cf = dot(directions[q], force)
    return (1. - _τ / 2) * weights[q] * (cf * _cs2 + cf * dot(directions[q], velocity) * _cs4 - dot(velocity, force) * _cs2)
end

"""
    temp_eq(q, velocity, temperature)
Computes the equilibrium distribution function for temperature.

# Arguments
- `q::Int`: Index of the discrete velocity direction.
- `velocity::SVector`: Local velocity at the given point.
- `temperature::Float`: Local temperature at the given point.

# Returns
- `Float`: The equilibrium distribution function for temperature.

This function is similar to the equilibrium distribution for density but applies to the 
temperature field, with adjustments for temperature-dependent variables.
"""
@inline function temp_eq(q, velocity, temperature)
    uc = dot(velocity, directions[q])
    return weights[q] * temperature * (1. + uc * _cs2)
end

"""
    @parallel_indices (i, j, k) init_density_pop!(density_pop, velocity, density)

Initializes the density population distribution function (`density_pop`) based on the equilibrium distribution function `f_eq`.

# Arguments
- `density_pop::CellArray`: The density population distribution function to be updated.
- `velocity::CellArray`: Velocity field used in the computation of the equilibrium distribution.
- `density::Array`: Density field required for the equilibrium calculation.
"""
@parallel_indices (i, j, k) function init_density_pop!(density_pop, velocity, density)
    for q in 1:Q
        @index density_pop[q, i, j, k] = f_eq(q, velocity[i-1, j-1, k-1], density[i-1, j-1, k-1])
    end
    return
end

"""
    @parallel_indices (i, j, k) init_temperature_pop!(temperature_pop, velocity, density)

Initializes the temperature population distribution function (`temperature_pop`) based on the equilibrium distribution function `temp_eq`.

# Arguments
- `temperature_pop::CellArray`: The temperature population distribution function to be updated.
- `velocity::CellArray`: Velocity field used in the computation of the equilibrium distribution.
- `temperature::Array`: Temperature field required for the equilibrium calculation.
"""
@parallel_indices (i, j, k) function init_temperature_pop!(temperature_pop, velocity, temperature)
    for q in 1:Q
        @index temperature_pop[q, i, j, k] = temp_eq(q, velocity[i-1, j-1, k-1], temperature[i-1, j-1, k-1])
    end
    return
end

"""
    @parallel_indices (i, j, k) compute_force!(forces, temperature, gravity, α, ρ_0)

Computes the external forces acting on the system, accounting for temperature variations and gravitational acceleration.

# Arguments
- `forces::CellArray`: Array to store the computed force values at each grid point.
- `temperature::Array`: Temperature field used to compute the force.
- `gravity::SVector`: Gravitational acceleration vector.
- `α::Float64`: Thermal expansion coefficient determining the force magnitude based on temperature variations.
- `ρ_0::Float64`: Reference density for the force calculation.
"""
@parallel_indices (i, j, k) function compute_force!(forces, temperature, gravity, α, ρ_0)
    forces[i, j, k] = -α * ρ_0 * temperature[i, j, k] .* gravity
    return
end

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

"""
    thermal_convection_lbm_3D(;N=40, nt=10000, Ra=1000., do_vis=true, vis_file="vis.gif", manage_MPI=true)
Simulates 3D thermal convection using the Lattice Boltzmann Method (LBM).

# Keyword Arguments
- `N::Int`: Number of grid points in the largest dimension (default: 40).
- `nt::Int`: Number of timesteps for the simulation (default: 1000).
- `Ra::Float64`: Rayleigh number, determining the buoyancy-driven flow (default: 100).
- `do_vis::Bool`: Flag to enable or disable visualization during simulation (default: true).
- `vis_file::String`: Filename for the resulting animation if `do_vis` is set (default: "vis.gif").
- `manage_MPI::Bool`: Flag to enable or disable IGG's managing of MPI, allows multiple calls withour re-initializing MPI (default: true).

# Description
This function performs a 3D thermal convection simulation using the LBM. The simulation 
is characterized by a rectangular domain and includes computation of density, temperature, 
and velocity fields. 

If `do_vis` is enabled, it generates visualizations of the temperature and density fields at 
specified intervals and creates an animation.

# Example
```julia
thermal_convection_lbm_3D(N=50, nt=2000, Ra=500, do_vis=true)
```

This example simulates a domain with a grid size of 50x25x50 for 2000 timesteps, using a Rayleigh 
number of 500, with visualization enabled.
"""    
function thermal_convection_lbm_3D(; N=40, nt=10000, Ra=1000., do_vis=true, vis_file="vis.gif", manage_MPI=true)
    # numerics
    nx_pop = N
    ny_pop = Int(nx_pop/2)
    nz_pop = nx_pop

    nx_values = nx_pop - 2
    ny_values = ny_pop - 2
    nz_values = nz_pop - 2

    # initialize IGG
    me, dims, nprocs, coords, comm = init_global_grid(nx_pop, ny_pop, nz_pop, periodx=0, periody=0, periodz=0, init_MPI=manage_MPI)
    b_width = (8, 8, 8)

    # physics
    lx = 20
    ly = lx * ny_pop / nx_pop
    lz = lx * ny_pop / nx_pop
    dx, dy, dz = lx / nx_g(), ly / ny_g(), lz / nz_g()    

    α = 0.0003 #6.9e-5
    ρ_0 = 1.
    gravity = @SVector [0., -1., 0.]
    αρ_0gravity = α * ρ_0 * gravity
    ΔT = 1. #200.

    ν = 5e-2 # viscosity close to 0 -> near incompressible limit
    # Ra = α * norm(gravity) * ΔT * ly^3 / (viscosity * k)
    κ = α * norm(gravity) * ΔT * ly^3 / (ν * Ra)

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
    startup = Int(nt/10)
    t_tic = 0.
    timesteps = 0:nt
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
    
    # test whether local gird is touching global boundary
    _, periods, _ = MPI.Cart_get(comm)
    left_boundary_x = coords[1] == 0 && !periods[0]
    right_boundary_x = (coords[1] == dims[1]-1) && !periods[1]
    left_boundary_y = coords[2] == 0 && !periods[0]
    right_boundary_y = (coords[2] == dims[2]-1) && !periods[2]
    left_boundary_z = coords[3] == 0 && !periods[0]
    right_boundary_z = (coords[3] == dims[3]-1) && !periods[3]
    if left_boundary_y || right_boundary_y
        @parallel range_values init!(left_boundary_y, right_boundary_y, temperature, ΔT_lattice)
    end
    
    # initialize populations from equilibrium distribution
    @parallel inner_range_pop init_density_pop!(density_pop, velocity, density)
    @parallel inner_range_pop init_temperature_pop!(temperature_pop, velocity, temperature)
    
    # visualization
    nvis = 10
    visdir = "visdir"

    if do_vis
        ENV["GKSwstype"]="nul"
        if (me==0) if isdir("$visdir")==false mkdir("$visdir") end; loadpath="$visdir/"; anim=Animation(loadpath,String[]); println("Animation directory: $(anim.dir)") end
        nx_v, ny_v, nz_v = (nx_values) * dims[1], (ny_values) * dims[2], (nz_values) * dims[3]
        if (2 * nx_v * ny_v * nz_v * sizeof(Data.Number) > 0.8 * Sys.free_memory()) error("Not enough memory for visualization.") end
        density_c = zeros(nx_values, ny_values, nz_values) # local cpu array
        temperature_c = zeros(nx_values, ny_values, nz_values) # local cpu array
        density_v = zeros(nx_v, ny_v, nz_v) # global array for visu
        temperature_v = zeros(nx_v, ny_v, nz_v) # global array for visu
        xi_g, yi_g = LinRange(0, lx, nx_v), LinRange(0, ly, ny_v) # inner points only
        iframe = 0
    end

    # time loop
    for i in timesteps
        # timing
        if i == startup
            t_tic = Base.time()
        end
        # visualization
        if do_vis && (i % nvis == 0)
            density_c .= Array(density) .* ρ_0; gather!(density_c, density_v)
            temperature_c .= Array(temperature) .* ΔT; gather!(temperature_c, temperature_v)
            if me == 0
                dens = heatmap(xi_g, yi_g, density_v[:, :, Int(ceil(nz_v/2))]'; xlims=(xi_g[1], xi_g[end]), ylims=(yi_g[1], yi_g[end]), aspect_ratio=1, c=:turbo, clim=(0,1.1*ρ_0), title="density")
                temp = heatmap(xi_g, yi_g, temperature_v[:, :, Int(ceil(nz_v/2))]'; xlims=(xi_g[1], xi_g[end]), ylims=(yi_g[1], yi_g[end]), aspect_ratio=1, c=:turbo, clim=(-ΔT/2,ΔT/2), title="temperature")

                p = plot(dens, temp, layout=(2, 1))
                png(p, "$visdir/$(lpad(iframe += 1, 4, "0")).png")
                save_array("$visdir/out_dens_$(lpad(iframe, 4, "0"))", convert.(Float32, density_v))
                save_array("$visdir/out_temp_$(lpad(iframe, 4, "0"))", convert.(Float32, temperature_v))
            end
        end

        # compute forces
        @parallel range_values compute_force!(forces, temperature, gravity, α, ρ_0)

        # compute moments (density, velocity and temperature)
        @parallel range_values update_moments!(velocity, density, temperature, density_pop, temperature_pop, forces)

        # collsion and exchange halo
        @hide_communication b_width begin
            @parallel collision!(density_pop, temperature_pop, velocity, density, temperature, forces, _τ_density_lattice, _τ_temperature_lattice)
            update_halo!(density_pop, temperature_pop)
        end

        # streaming
        @parallel inner_range_pop streaming!(density_pop, density_pop_buf, temperature_pop, temperature_pop_buf)

        # boundary conditions
        begin
            if left_boundary_x 
                @parallel (y_boundary_range, z_boundary_range) bounce_back_x_left!(density_pop, density_pop_buf)
                @parallel (y_boundary_range, z_boundary_range) bounce_back_x_left!(temperature_pop, temperature_pop_buf)
            end

            if right_boundary_x
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
                @parallel (x_boundary_range, z_boundary_range) anti_bounce_back_temperature_y_left!(temperature_pop_buf, velocity, temperature, ΔT_lattice/2)
        
            end
            
            if right_boundary_y
                @parallel (x_boundary_range, z_boundary_range) bounce_back_y_right!(density_pop, density_pop_buf)
                @parallel (x_boundary_range, z_boundary_range) anti_bounce_back_temperature_y_right!(temperature_pop_buf, velocity, temperature, -ΔT_lattice/2)
            end
        end

        # pointer swap
        density_pop, density_pop_buf = density_pop_buf, density_pop
        temperature_pop, temperature_pop_buf = temperature_pop_buf, temperature_pop 
        
        # progress bar
        if (me == 0) next!(progress) end
    end

    # visualization
    if do_vis && me == 0
        run(`ffmpeg -i $visdir/%4d.png ../docs/plots/"$vis_file" -y`)
    end

    # effective memory throughput
    t_toc = Base.time() - t_tic
    A_eff = 2 * (
        4 * Q + # pop and buff for density and temperature
        2 * 3 + # velocity and forces
        2 * 1 # density and temperature
    ) / 1e9*nx_pop*ny_pop*nz_pop*sizeof(Float64) # Effective main memory access per iteration [GB]
    niter = length(timesteps) - startup
    t_it = t_toc / niter
    T_eff = A_eff/t_it 
    if me == 0
        println("Time = $t_toc sec, T_eff = $T_eff GB/s (niter = $niter)")
    end

    # finalize IGG
    finalize_global_grid(finalize_MPI=manage_MPI)
end