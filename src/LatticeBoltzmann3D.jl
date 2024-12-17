using ParallelStencil
using ImplicitGlobalGrid
import MPI

@static if method == :D3Q15
    const Q = 15
    const directions = [
        [0, 0, 0], 
        [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1], 
        [1, 1, 1], [-1, -1, -1], [1, 1, -1], [-1, -1, 1], [1, -1, 1], [-1, 1, -1], [-1, 1, 1], [1, -1, -1]
        ]
    const weights = [
        2/9, 
        1/9, 1/9, 1/9, 1/9, 1/9, 1/9, 
        1/72, 1/72, 1/72, 1/72, 1/72, 1/72, 1/72, 1/72
        ]
elseif method == :D3Q19
    const Q = 19
    const directions = [
        [0, 0, 0], 
        [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1], 
        [1, 1, 0], [-1, -1, 0], [1, -1, 0], [-1, 1, 0], [1, 0, 1], [-1, 0, -1], [-1, 0, 1], [1, 0, -1], [0, 1, 1], [0, -1, -1], [0, 1, -1], [0, -1, 1]
        ]
    const weights = [
        1/3, 1/18, 1/18, 1/18, 1/18, 1/18, 1/18, 1/36, 1/36, 1/36, 1/36, 1/36, 1/36, 1/36, 1/36, 1/36, 1/36, 1/36, 1/36
        ]
elseif method == :D3Q27
    const Q = 27
    const directions = [
        [0, 0, 0],
        [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1], 
        [1, 1, 0], [-1, -1, 0], [1, -1, 0], [-1, 1, 0], [1, 0, 1], [-1, 0, -1], [-1, 0, 1], [1, 0, -1], [0, 1, 1], [0, -1, -1], [0, 1, -1], [0, -1, 1],
        [1, 1, 1], [-1, -1, -1], [1, 1, -1], [-1, -1, 1], [1, -1, 1], [-1, 1, -1], [-1, 1, 1], [1, -1, -1]
    ]
    const weights = [
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

@parallel_indices (i, j, k) function collision!(pop, velocity, values, _τ)
    v = velocity[i-1, j-1, k-1, :]
    for q in 1:Q
        pop[i, j, k, q] = (1. - _τ) * pop[i, j, k, q] + _τ * f_eq(q, v, values[i-1, j-1, k-1])
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
                    pop[i, j, z] = f_eq(q, velocity, values[i, j, z])
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

@parallel_indices (i, j) function periodic_boundary_update!(dimension, pop, pop_buf)
    Nx = size(pop, 1)
    Ny = size(pop, 2)
    Nz = size(pop, 3)
    if dimension == :x
        for q in 1:Q
            yidx = mod(i - directions[q][2] - 1, Ny) + 1
            zidx = mod(j - directions[q][3] - 1, Nz) + 1
            # if directions[q][1] == 1
                pop_buf[1, i, j, q] = pop[Nx-1, i, j, q]
            # elseif directions[q][1] == -1
                pop_buf[Nx, i, j, q] = pop[2, i, j, q]
            # end
        end   
    elseif dimension == :y
        for q in 1:Q
            xidx = mod(i - directions[q][1] - 1, Nx) + 1
            zidx = mod(j - directions[q][3] - 1, Nz) + 1
            # if directions[q][2] == 1
                pop_buf[i, 1, j, q] = pop[i, Ny-1, j, q]
            # elseif directions[q][2] == -1
                pop_buf[i, Ny, j, q] = pop[i, 2, j, q]
            # end
        end    
    elseif dimension == :z
        for q in 1:Q
            xidx = mod(i - directions[q][1] - 1, Nx) + 1
            yidx = mod(j - directions[q][2] - 1, Ny) + 1
            # if directions[q][3] == 1
                pop_buf[i, j, 1, q] = pop[i, j, Nz-1, q]
            # elseif directions[q][3] == -1
                pop_buf[i, j, Nz, q] = pop[i, j, 2, q]
            # end
        end
    end
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

@parallel_indices (i, j) function bounce_back_boundary!(dimension, pop, pop_buf)
    if dimension == :x
        Nx = size(pop, 1)
        for q in 1:Q
            yidx = i + directions[q][2]
            zidx = j + directions[q][3]
            if directions[q][1] == 1
                pop_buf[1, i, j, q] = pop[2, yidx, zidx, (q%2 == 0) ? q+1 : q-1]
            end
            if directions[q][1] == -1
                pop_buf[Nx, i, j, q] = pop[Nx - 1, yidx, zidx, (q%2 == 0) ? q+1 : q-1]
            end
        end   
    elseif dimension == :y
        Ny = size(pop, 2)
        for q in 1:Q    
            xidx = i + directions[q][1]
            zidx = j + directions[q][3]
            if directions[q][2] == 1
                pop_buf[i, 1, j, q] = pop[xidx, 2, zidx, (q%2 == 0) ? q+1 : q-1]
            end
            if directions[q][2] == -1
                pop_buf[i, Ny, j, q] = pop[xidx, Ny - 1, zidx, (q%2 == 0) ? q+1 : q-1]
            end    
        end
    elseif dimension == :z
        Nz = size(pop, 3)
        for q in 1:Q
            xidx = i + directions[q][1]
            yidx = j + directions[q][2]
            if directions[q][3] == 1 
                pop_buf[i, j, 1, q] = pop[xidx, yidx, 2, (q%2 == 0) ? q+1 : q-1]
            elseif directions[q][3] == -1
                pop_buf[i, j, Nz, q] = pop[xidx, yidx, Nz - 1, (q%2 == 0) ? q+1 : q-1]
            end
        end
    end
    return
end

@parallel_indices (i, j, k) function streaming!(pop, pop_buf)
    for q in 1:Q
        pop_buf[i, j, k, q] = pop[i - directions[q][1], j - directions[q][2], k - directions[q][3], q]
    end
    return
end

@parallel_indices (i, j, k) function init!(velocity, density, temperature, U_init, lx, ly, R)
    density[i, j, k] = 1

    dx = lx / nx_g()
    dy = ly / ny_g()
    x = x_g(i, dx, velocity)
    y = y_g(j, dy, velocity)
    
    if ((x - lx / 2)^2 + (y - ly / 3) ^2) < R^2
        velocity[i, j, k, :] = @zeros(3)
        temperature[i, j, k] = 1
    else 
        velocity[i, j, k, :] = U_init
        temperature[i, j, k] = 0
    end
    # temperature[i, j, k] = MPI.Comm_rank(MPI.COMM_WORLD)
    return
end

@parallel_indices (i, j, k) function update_moments!(velocity, density, temperature, density_pop, temperature_pop)
    cell_density = 0
    cell_velocity = @zeros(3)
    cell_temperature = 0
    for q in 1:Q
        cell_density += density_pop[i + 1, j + 1, k + 1, q]
        cell_temperature += temperature_pop[i + 1, j + 1, k + 1, q]
        cell_velocity .+= directions[q] * density_pop[i + 1, j + 1, k + 1, q]
    end

    cell_velocity /= cell_density
    velocity[i, j, k, :] = cell_velocity
    density[i, j, k] = cell_density
    temperature[i, j, k] = cell_temperature
    return
end

@parallel_indices (i, j, k) function apply_external_force!(velocity, lx, ly, R)

    dx = lx / nx_g()
    dy = ly / ny_g()
    x = x_g(i, dx, velocity)
    y = y_g(j, dy, velocity)

    if ((x - lx / 2)^2 + (y - ly / 3) ^2) < R^2
        velocity[i, j, k, :] = @zeros(3)
    end
    return
end

@views function f_eq(q, velocity, density)
    uc = dot(velocity, directions[q])
    uc2 = uc^2
    u2 = norm(velocity) ^ 2
    return weights[q] * density * (1. + uc * _cs2 + 0.5 * uc2 * _cs4 - 0.5 * u2 * _cs2)
end

@parallel_indices (i, j, k) function init_pop!(pop, velocity, values)
    for q in 1:Q
        pop[i, j, k, q] = f_eq(q, velocity[i-1, j-1, k-1, :], values[i-1, j-1, k-1])
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