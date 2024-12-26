using CellArraysIndexing, StaticArrays
using ParallelStencil
using ImplicitGlobalGrid
import MPI

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
else
    @assert false "method not defined"
end
@assert Q == size(directions, 1) "Q=$Q, size of directions=$(size(directions, 1))"
@assert Q == size(weights, 1) "Q=$Q, size of weights=$(size(weights, 1))"
    
# Speed of sound (in lattice units)
const _cs2 = 3. # cs^2 = 1./3. * (dx**2/dt**2)
const _cs4 = 9.

@parallel_indices (i, j, k) function collision_density!(pop, velocity, density, forces, _τ)
    v = velocity[i-1, j-1, k-1]
    for q in 1:Q
        @index pop[q, i, j, k] = (1. - _τ) * @index(pop[q, i, j, k]) + _τ * f_eq(q, v, density[i-1, j-1, k-1]) + source_term(q, v, forces[i-1, j-1, k-1], _τ)
    end
    return
end

@parallel_indices (i, j, k) function collision_temperature!(pop, velocity, temperature, _τ)
    v = velocity[i-1, j-1, k-1]
    for q in 1:Q
        @index pop[q, i, j, k] = (1. - _τ) * @index(pop[q, i, j, k]) + _τ * temp_eq(q, v, temperature[i-1, j-1, k-1])
    end
    return
end

@parallel_indices (i, j) function periodic_boundary_x!(pop)
    Nx = size(pop, 1)
    pop[1, i, j] = pop[Nx-1, i, j]
    pop[Nx, i, j] = pop[2, i, j]
    return
end
@parallel_indices (i, k) function periodic_boundary_y!(pop)
    Ny = size(pop, 2)
    pop[i, 1, k] = pop[i, Ny-1, k]
    pop[i, Ny, k] = pop[i, 2, k]
    return
end
@parallel_indices (i, j) function periodic_boundary_z!(pop)
    Nz = size(pop, 3)
    pop[i, j, 1] = pop[i, j, Nz-1]
    pop[i, j, Nz] = pop[i, j, 2]
    return
end

@parallel_indices (j, k) function anti_bounce_back_temperature_x_right!(pop, pop_buf, velocity, values, dirichlet_value) 
    Nx = size(pop, 1)
    for q in 1:Q
        if directions[q][1] == 1
            flipped_dir = (q%2 == 0) ? q+1 : q-1
            value = -temp_eq(q, velocity[Nx - 2, j - 1, k - 1], values[Nx - 2, j - 1, k - 1]) + 2 * weights[q] * dirichlet_value
            @index pop_buf[flipped_dir, Nx - 1, j, k] = value
        end
    end 
    return
end

@parallel_indices (j, k) function anti_bounce_back_temperature_x_left!(pop, pop_buf, velocity, values, dirichlet_value) 
    for q in 1:Q
        if directions[q][1] == -1
            flipped_dir = (q%2 == 0) ? q+1 : q-1
            value = -temp_eq(q, velocity[1, j - 1, k - 1], values[1, j - 1, k - 1]) + 2 * weights[q] * dirichlet_value
            @index pop_buf[flipped_dir, 2, j, k] = value
        end
    end
    return
end

@parallel_indices (i, k) function anti_bounce_back_temperature_y_right!(pop, pop_buf, velocity, values, dirichlet_value) 
    Ny = size(pop, 2)
    for q in 1:Q
        if directions[q][2] == 1
            flipped_dir = (q%2 == 0) ? q+1 : q-1
            value = -temp_eq(q, velocity[i - 1, Ny - 2, k - 1], values[i - 1, Ny - 2, k - 1]) + 2 * weights[q] * dirichlet_value
            @index pop_buf[flipped_dir, i, Ny - 1, k] = value
        end
    end
    return
end

@parallel_indices (i, k) function anti_bounce_back_temperature_y_left!(pop, pop_buf, velocity, values, dirichlet_value) 
    for q in 1:Q
        if directions[q][2] == -1
            flipped_dir = (q%2 == 0) ? q+1 : q-1
            value = -temp_eq(q, velocity[i - 1, 1, k - 1], values[i - 1, 1, k - 1]) + 2 * weights[q] * dirichlet_value
            @index pop_buf[flipped_dir, i, 2, k] = value
        end
    end
    return
end

@parallel_indices (i, j) function anti_bounce_back_temperature_z_right!(pop, pop_buf, velocity, values, dirichlet_value) 
    Nz = size(pop, 3)
    for q in 1:Q
        if directions[q][3] == 1
            flipped_dir = (q%2 == 0) ? q+1 : q-1
            value = -temp_eq(q, velocity[i - 1, j - 1, Nz - 2], values[i - 1, j - 1, Nz - 2]) + 2 * weights[q] * dirichlet_value
            @index pop_buf[flipped_dir, i, j, Nz - 1] = value
        end
    end
    return
end

@parallel_indices (i, j) function anti_bounce_back_temperature_z_left!(pop, pop_buf, velocity, values, dirichlet_value) 
    for q in 1:Q
        if directions[q][3] == -1
            flipped_dir = (q%2 == 0) ? q+1 : q-1
            value = -temp_eq(q, velocity[i - 1, j - 1, 1], values[i - 1, j - 1, 1]) + 2 * weights[q] * dirichlet_value
            @index pop_buf[flipped_dir, i, j, 2] = value
        end
    end
    return
end

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


@parallel_indices (i, j, k) function streaming!(pop, pop_buf)
    for q in 1:Q
        @index pop_buf[q, i, j, k] = @index pop[q, i - directions[q][1], j - directions[q][2], k - directions[q][3]]
    end
    return
end

@parallel_indices (i, j, k) function init!(left_boundary, right_boundary, velocity, temperature, boundary, U_init, ΔT)    
    # if boundary[i, j, k] == 0.
    #     velocity[i, j, k] = U_init
    # else 
    #     temperature[i, j, k] = 1.
    # end

    if left_boundary && j == 1
        temperature[i, j, k] = ΔT / 2
    elseif right_boundary && j == size(temperature, 2)
        temperature[i, j, k] = -ΔT / 2
    end
    return
end

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

@parallel_indices (i, j, k) function apply_external_force!(velocity, boundary)
    if boundary[i, j, k] != 0.
        velocity[i, j, k] = @SVector zeros(3)
    end
    return
end

@inline function f_eq(q, velocity, density)
    uc = dot(velocity, directions[q])
    uc2 = uc^2
    u2 = norm(velocity) ^ 2
    return weights[q] * density * (1. + uc * _cs2 + 0.5 * uc2 * _cs4 - 0.5 * u2 * _cs2)
end

@inline function source_term(q, velocity, force, _τ)
    cf = dot(directions[q], force)
    return (1. - _τ / 2) * weights[q] * (cf * _cs2 + cf * dot(directions[q], velocity) * _cs4 - dot(velocity, force) * _cs2)
end


@inline function temp_eq(q, velocity, temperature)
    uc = dot(velocity, directions[q])
    uc2 = uc^2
    u2 = norm(velocity) ^ 2
    return weights[q] * temperature * (1. + uc * _cs2 + 0.5 * uc2 * _cs4 - 0.5 * u2 * _cs2)
    # uc = dot(velocity, directions[q])
    # return weights[q] * temperature * (1. + uc * _cs2)
end

@parallel_indices (i, j, k) function init_density_pop!(density_pop, velocity, values)
    for q in 1:Q
        @index density_pop[q, i, j, k] = f_eq(q, velocity[i-1, j-1, k-1], values[i-1, j-1, k-1])
    end
    return
end

@parallel_indices (i, j, k) function init_temperature_pop!(temperature_pop, velocity, values)
    for q in 1:Q
        @index temperature_pop[q, i, j, k] = temp_eq(q, velocity[i-1, j-1, k-1], values[i-1, j-1, k-1])
    end
    return
end

@parallel_indices (i, j, k) function compute_force!(forces, temperature, gravity, α, ρ_0)
    forces[i, j, k] = -α * ρ_0 * temperature[i, j, k] .* gravity
    return
end