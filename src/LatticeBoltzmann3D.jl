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

@views function dirichlet_boundary!(boundary, pop, velocity, values)
    if boundary == :xupper
        x = size(pop, 1)
        for j in axes(pop, 2)[2:end-1]
            for k in axes(pop, 3)[2:end-1]
                for q in 1:Q
                    pop[x, j, k, q] = f_eq(q, velocity, values[x, j, k])
                end      
            end               
        end
    elseif  boundary == :zlower
        z = 1
        for i in axes(pop, 1)[2:end-1]
            for j in axes(pop, 2)[2:end-1]
                for q in 1:Q
                    pop[i, j, z, q] = f_eq(q, velocity, values[i, j, z])
                end
            end
        end
    elseif boundary == :ylower
        y = 1
        for i in axes(pop, 1)[2:end-1]
            for k in axes(pop, 3)[2:end-1]
                for q in 1:Q
                    pop[i, y, k, q] = f_eq(q, velocity, values[i, y, k])
                end      
            end      
        end
    elseif boundary == :yupper
        y = size(pop, 2)
        for i in axes(pop, 1)[2:end-1]
            for k in axes(pop, 3)[2:end-1]
                for q in 1:Q
                    pop[i, y, k, q] = f_eq(q, velocity, values[i, y, k])
                end      
            end           
        end
    elseif boundary == :xlower
        x = 1
        for j in axes(pop, 2)[2:end-1]
            for k in axes(pop, 3)[2:end-1]
                for q in 1:Q
                    pop[x, j, k, q] = f_eq(q, velocity, values[x, j, k])
                end      
            end      
        end
    elseif boundary == :zupper
        z = size(popo, 3)
        for i in axes(pop, 1)[2:end-1]
            for j in axes(pop, 2)[2:end-1]
                for q in 1:Q
                    pop[i, j, z] = f_eq(q, velocity, values[i, j, z])
                end
            end
        end
    end
end

@parallel_indices (i, j) function periodic_boundary_x!(pop)
    Nx = size(pop, 1)
        # for q in 1:Q
        #     yidx = mod(i - directions[q][2] - 1, Ny) + 1
        #     zidx = mod(j - directions[q][3] - 1, Nz) + 1
            # if directions[q][1] == 1
                pop[1, i, j] = pop[Nx-1, i, j]
            # elseif directions[q][1] == -1
                pop[Nx, i, j] = pop[2, i, j]
            # end
        # end   
    return
end
@parallel_indices (i, k) function periodic_boundary_y!(pop)
    Ny = size(pop, 2)
        # for q in 1:Q
        #     xidx = mod(i - directions[q][1] - 1, Nx) + 1
        #     zidx = mod(j - directions[q][3] - 1, Nz) + 1
            # if directions[q][2] == 1
                pop[i, 1, k] = pop[i, Ny-1, k]
            # elseif directions[q][2] == -1
                pop[i, Ny, k] = pop[i, 2, k]
            # end
        # end 
    return
end
@parallel_indices (i, j) function periodic_boundary_z!(pop)
    Nz = size(pop, 3)

        # for q in 1:Q
        #     xidx = mod(i - directions[q][1] - 1, Nx) + 1
        #     yidx = mod(j - directions[q][2] - 1, Ny) + 1
            # if directions[q][3] == 1
                pop[i, j, 1] = pop[i, j, Nz-1]
            # elseif directions[q][3] == -1
                pop[i, j, Nz] = pop[i, j, 2]
            # end
        # end
    return
end


@parallel_indices (i, j) function inlet_boundary_conditions!(dimension, pop, pop_buf, U_inlet, values)
    if dimension == :x
        lx = 2
        rx = size(pop, 1) - 1
        for q in 1:Q
            if directions[q][1] == 1
                pop_buf[rx, i, j, q] = pop[rx, i, j, (q%2 == 0) ? q+1 : q-1] - 2 * _cs2 * weights[q] * values[rx - 1, i - 1, j - 1] * dot(U_inlet, directions[q])
            elseif directions[q][1] == -1
                pop_buf[lx, i, j, q] = pop[lx, i, j, (q%2 == 0) ? q+1 : q-1] - 2 * _cs2 * weights[q] * values[lx - 1, i - 1, j - 1] * dot(U_inlet, directions[q])
            end
        end   
    elseif dimension == :y
        ly = 2
        ry = size(pop, 2) - 1
        n = @zeros(3)
        n[2] = -1
        for q in 1:Q   
            ci = -directions[q]
            ti = ci - dot(ci, n) * n 
            corr = 0
            for qq in 1:Q
                cj = directions[qq]
                corr += pop[i, ly, j, qq] * dot(ti, cj) * (1 - abs(dot(cj, n)))
            end
            corr /= 2
            # if directions[q][2] == 1
            #     pop_buf[i, ry, j, q] = pop[i, ry, j, (q%2 == 0) ? q+1 : q-1] - 2 * _cs2 * weights[q] * values[i - 1, ry - 1, j - 1] * dot(U_inlet, directions[q])
            # else
            if directions[q][2] == -1
                pop_buf[i, ly, j, q] = pop[i, ly, j, (q%2 == 0) ? q+1 : q-1] - values[i-1, ry-1, j-1] * (dot(ci, U_inlet) / 2 + dot(ti, U_inlet)) / 6 + corr 
            end    
        end
    elseif dimension == :z
        lz = 2
        rz = size(pop, 3) - 1
        for q in 1:Q
            if directions[q][3] == 1 
                pop_buf[i, j, rz, q] = pop[i, j, rz, (q%2 == 0) ? q+1 : q-1] - 2 * _cs2 * weights[q] * values[i - 1, j - 1, rz - 1] * dot(U_inlet, directions[q])
            elseif directions[q][3] == -1
                pop_buf[i, j, lz, q] = pop[i, j, lz, (q%2 == 0) ? q+1 : q-1] - 2 * _cs2 * weights[q] * values[i - 1, j - 1, lz - 1] * dot(U_inlet, directions[q])
            end
        end
    end
    return
end

@parallel_indices (j, k) function anti_bounce_back_temperature_x!(pop, velocity, values, dirichlet_value_l, dirichlet_value_r) 
    Nx = size(pop, 1)
    for q in 1:Q
        yidx = j + directions[q][2]
        zidx = k + directions[q][3]
        if directions[q][1] == 1
            flipped_dir = (q%2 == 0) ? q+1 : q-1
            value = -temp_eq(q, velocity[Nx - 2, j - 1, k - 1], values[Nx - 2, j - 1, k - 1]) + 2 * weights[q] * dirichlet_value_r
            @index pop[flipped_dir, Nx, yidx, zidx] = value
        end
        if directions[q][1] == -1
            flipped_dir = (q%2 == 0) ? q+1 : q-1
            value = -temp_eq(q, velocity[1, j - 1, k - 1], values[1, j - 1, k - 1]) + 2 * weights[q] * dirichlet_value_l
            @index pop[flipped_dir, 1, yidx, zidx] = value
        end
    end
    return
end

@parallel_indices (i, k) function anti_bounce_back_temperature_y!(pop, velocity, values, dirichlet_value_l, dirichlet_value_r) 
    Ny = size(pop, 2)
    for q in 1:Q
        xidx = i + directions[q][1]
        zidx = k + directions[q][3]
        if directions[q][2] == 1
            flipped_dir = (q%2 == 0) ? q+1 : q-1
            value = -temp_eq(q, velocity[i - 1, Ny - 2, k - 1], values[i - 1, Ny - 2, k - 1]) + 2 * weights[q] * dirichlet_value_r
            @index pop[flipped_dir, xidx, Ny, zidx] = value
        end
        if directions[q][2] == -1
            flipped_dir = (q%2 == 0) ? q+1 : q-1
            value = -temp_eq(q, velocity[i - 1, 1, k - 1], values[i - 1, 1, k - 1]) + 2 * weights[q] * dirichlet_value_l
            @index pop[flipped_dir, xidx, 1, zidx] = value
        end
    end
    return
end

@parallel_indices (i, j) function anti_bounce_back_temperature_z!(pop, velocity, values, dirichlet_value_l, dirichlet_value_r) 
    Nx = size(pop, 1)
    for q in 1:Q
        xidx = i + directions[q][1]
        yidx = j + directions[q][2]
        if directions[q][3] == 1
            flipped_dir = (q%2 == 0) ? q+1 : q-1
            value = -temp_eq(q, velocity[i - 1, j - 1, Nz - 2], values[i - 1, j - 1, Nz - 2]) + 2 * weights[q] * dirichlet_value_r
            @index pop[flipped_dir, xidx, yidx, Nz] = value
        end
        if directions[q][3] == -1
            flipped_dir = (q%2 == 0) ? q+1 : q-1
            value = -temp_eq(q, velocity[i - 1, j - 1, 1], values[i - 1, j - 1, 1]) + 2 * weights[q] * dirichlet_value_l
            @index pop[flipped_dir, xidx, yidx, 1] = value
        end
    end
    return
end

@parallel_indices (j, k) function bounce_back_x!(pop)
    Nx = size(pop, 1)
    for q in 1:Q
        yidx = j + directions[q][2]
        zidx = k + directions[q][3]
        if directions[q][1] == 1
            flipped_dir = (q%2 == 0) ? q+1 : q-1
            flipped_value = @index pop[q, Nx - 1, j, k]
            @index pop[flipped_dir, Nx, yidx, zidx] = flipped_value
        end
        if directions[q][1] == -1
            flipped_dir = (q%2 == 0) ? q+1 : q-1
            flipped_value = @index pop[q, 2, j, k]
            @index pop[flipped_dir, 1, yidx, zidx] = flipped_value
        end
    end   
    return
end

@parallel_indices (i, k) function bounce_back_y!(pop)
    Ny = size(pop, 2)
    for q in 1:Q
        xidx = i + directions[q][1]
        zidx = k + directions[q][3]
        if directions[q][2] == 1
            flipped_dir = (q%2 == 0) ? q+1 : q-1
            flipped_value = @index pop[q, i, Ny - 1, k]
            @index pop[flipped_dir, xidx, Ny, zidx] = flipped_value
        end
        if directions[q][2] == -1
            flipped_dir = (q%2 == 0) ? q+1 : q-1
            flipped_value = @index pop[q, i, 2, k]
            @index pop[flipped_dir, xidx, 1, zidx] = flipped_value
        end
    end   
    return
end

@parallel_indices (i, j) function bounce_back_z!(pop)
    Nz = size(pop, 3)
    for q in 1:Q
        xidx = i + directions[q][1]
        yidx = j + directions[q][2]
        if directions[q][3] == 1
            flipped_dir = (q%2 == 0) ? q+1 : q-1
            flipped_value = @index pop[q, i, j, Nz - 1]
            @index pop[flipped_dir, xidx, yidx, Nz] = flipped_value
        end
        if directions[q][3] == -1
            flipped_dir = (q%2 == 0) ? q+1 : q-1
            flipped_value = @index pop[q, i, j, 2]
            @index pop[flipped_dir, xidx, yidx, 1] = flipped_value
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

@parallel_indices (i, j, k) function init!(velocity, temperature, boundary, U_init, ΔT)    
    if boundary[i, j, k] == 0.
        velocity[i, j, k] = U_init
    else 
        temperature[i, j, k] = 1.
    end
    # if j == 1
    #     temperature[i, j, k] = ΔT / 2
    # elseif j == size(temperature, 2)
    #     temperature[i, j, k] = -ΔT / 2
    # end
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

@parallel_indices (i, j, k) function apply_external_force!(velocity, boundary, lx, ly, R)
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

@views function lb_update_halo!(pop, comm)
    MPI.Barrier(comm)
    Nx, Ny, Nz = size(pop)
    me = MPI.Comm_rank(comm)
    dims, periods, coords = MPI.Cart_get(comm)
    

    reqs = Vector{MPI.Request}()
    recvbuffs = Vector{MPI.Buffer}()

    function valid_neighbour(neighbour_coords)
        return !(any((neighbour_coords .< 0) .& (periods .== 0)) || any((neighbour_coords .>= dims) .& (periods .== 0)))
    end

    function boundary_dims(dir, role)
        if role == :sender
            if dir[1] == 1
                xidx = Nx - 1
            elseif dir[1] == -1
                xidx = 2
            else
                xidx = 2:Nx - 1
            end
            if dir[2] == 1
                yidx = Ny - 1
            elseif dir[2] == -1
                yidx = 2
            else
                yidx = 2:Ny - 1
            end
            if dir[3] == 1
                zidx = Nz - 1
            elseif dir[3] == -1
                zidx = 2
            else
                zidx = 2:Nz - 1
            end
        elseif role == :receiver
            if dir[1] == 1
                xidx = 1
            elseif dir[1] == -1
                xidx = Nx
            else
                xidx = 2:Nx - 1
            end
            if dir[2] == 1
                yidx = 1
            elseif dir[2] == -1
                yidx = Ny
            else
                yidx = 2:Ny - 1
            end
            if dir[3] == 1
                zidx = 1
            elseif dir[3] == -1
                zidx = Nz
            else
                zidx = 2:Nz - 1
            end
        else
            @assert(false)
        end
        return xidx, yidx, zidx

    end

    for q in 2:Q # no need to exchange with self
        dir = directions[q]
        if valid_neighbour(coords + dir)
            neighbour = MPI.Cart_rank(comm, coords + dir)
            xidx, yidx, zidx = boundary_dims(dir, :sender)
            sendbuff = pop[xidx, yidx, zidx, :]

            sreq = MPI.Isend(sendbuff, comm, dest=neighbour, tag=0)
            append!(reqs, [sreq])
        end
        if valid_neighbour(coords - dir)
            neighbour = MPI.Cart_rank(comm, coords - dir)

            xidx, yidx, zidx = boundary_dims(dir, :receiver)
            append!(recvbuffs, [MPI.Buffer(Array{Float64}(undef, size(pop[xidx, yidx, zidx, :])))]) # Array{Float64}(undef, length(xidx), length(yidx), length(zidx), Q)

            rreq = MPI.Irecv!(recvbuffs[end], comm, source=neighbour, tag=0)
            append!(reqs, [rreq])
        end
    end

    MPI.Waitall(MPI.RequestSet(reqs))

    req_idx = 1
    for q in 2:Q
        dir = directions[q]
        if valid_neighbour(coords - dir)
            xidx, yidx, zidx = boundary_dims(dir, :receiver)

            # TODO make parallel copy
            pop[xidx, yidx, zidx, :] .= recvbuffs[req_idx].data

            req_idx += 1
        end
    end
    MPI.Barrier(comm)
    return
end

@parallel_indices (i, j, k) function compute_force!(forces, temperature, gravity, α, ρ_0)
    forces[i, j, k] = -α * ρ_0 * temperature[i, j, k] .* gravity
    return
end