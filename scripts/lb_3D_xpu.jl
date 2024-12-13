using LinearAlgebra
using Plots
using ParallelStencil

const USE_GPU = false

@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 3, inbounds=false)
else
    @init_parallel_stencil(Threads, Float64, 3, inbounds=false)
end


const method = :D3Q19

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
    if dimension == :x
        Nx = size(pop, 1)
        for q in 1:Q
            pop_buf[1, i, j, q] = pop[Nx-1, i, j, q]
            pop_buf[Nx, i, j, q] = pop[2, i, j, q]
        end   
    elseif dimension == :y
        Ny = size(pop, 2)
        for q in 1:Q
            pop_buf[i, 1, j, q] = pop[i, Ny-1, j, q]
            pop_buf[i, Ny, j, q] = pop[i, 2, j, q]
        end    
    elseif dimension == :z
        Nz = size(pop, 3)
        for q in 1:Q
            pop_buf[i, j, 1, q] = pop[i, j, Nz-1, q]
            pop_buf[i, j, Nz, q] = pop[i, j, 2, q]
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
        for q in 1:Q    
            # if directions[q][2] == 1
            #     pop_buf[i, ry, j, q] = pop[i, ry, j, (q%2 == 0) ? q+1 : q-1] - 2 * _cs2 * weights[q] * values[i - 1, ry - 1, j - 1] * dot(U_inlet, directions[q])
            # else
            if directions[q][2] == -1
                pop_buf[i, ly, j, q] = pop[i, ly, j, (q%2 == 0) ? q+1 : q-1] - 2 * _cs2 * weights[q] * values[i - 1, ly - 1, j - 1] * dot(U_inlet, directions[q])
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
        lx = 2
        rx = size(pop, 1) - 1
        for q in 1:Q
            if directions[q][1] == 1
                pop_buf[rx, i, j, q] = pop[rx, i, j, (q%2 == 0) ? q+1 : q-1]
            end
            if directions[q][1] == -1
                pop_buf[lx, i, j, q] = pop[lx, i, j, (q%2 == 0) ? q+1 : q-1]
            end
        end   
    elseif dimension == :y
        ly = 2
        ry = size(pop, 2) - 1
        for q in 1:Q    
            if directions[q][2] == 1
                pop_buf[i, ry, j, q] = pop[i, ry, j, (q%2 == 0) ? q+1 : q-1]
            end
            if directions[q][2] == -1
                pop_buf[i, ly, j, q] = pop[i, ly, j, (q%2 == 0) ? q+1 : q-1]
            end    
        end
    elseif dimension == :z
        lz = 2
        rz = size(pop, 3) - 1
        for q in 1:Q
            if directions[q][3] == 1 
                pop_buf[i, j, rz, q] = pop[i, j, rz, (q%2 == 0) ? q+1 : q-1]
            elseif directions[q][3] == -1
                pop_buf[i, j, lz, q] = pop[i, j, lz, (q%2 == 0) ? q+1 : q-1]
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

@parallel_indices (i, j, k) function init!(velocity, density, temperature, U_init, R)
    density[i, j, k] = 1

    Nx = size(velocity, 1)
    Ny = size(velocity, 2)

    if ((i - Nx/2)^2 + (j - Ny/3)^2) < R^2
        velocity[i, j, k, :] = @zeros(3)
        temperature[i, j, k] = 1
    else 
        velocity[i, j, k, :] = U_init
        temperature[i, j, k] = 0
    end
    return
end

@parallel_indices (i, j, k) function update_moments!(velocity, density, temperature, density_pop, temperature_pop)
    cell_density = 0
    cell_velocity = @zeros(3)
    cell_temperature = 0
    for q in 1:Q
        cell_density += density_pop[i + 1, j + 1, k + 1, q]
        cell_temperature += temperature_pop[i + 1, j + 1, k + 1, q]
        cell_velocity += directions[q] * density_pop[i + 1, j + 1, k + 1, q]
    end

    cell_velocity /= cell_density
    velocity[i, j, k, :] = cell_velocity
    density[i, j, k] = cell_density
    temperature[i, j, k] = cell_temperature
    return
end

@parallel_indices (i, j, k) function apply_external_force!(velocity, R)

    Nx = size(velocity, 1)
    Ny = size(velocity, 2)
    
    if ((i - Nx/2)*(i - Nx/2) + (j - Ny/3)*(j - Ny/3)) < R^2
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

function lb()
    Nx = 40
    Ny = 70
    Nz = 1

    lx = 40
    ly = 40

    xc, yc = LinRange(0, lx, Nx), LinRange(0, ly, Ny)


    density_pop = @zeros(Nx + 2, Ny + 2, Nz + 2, Q)
    density_buf = @zeros(Nx + 2, Ny + 2, Nz + 2, Q)
    
    temperature_pop = @zeros(Nx + 2, Ny + 2, Nz + 2, Q)
    temperature_buf = @zeros(Nx + 2, Ny + 2, Nz + 2, Q)

    velocity = @zeros(Nx, Ny, Nz, 3)
    density = @zeros(Nx, Ny, Nz)
    temperature = @zeros(Nx, Ny, Nz)

    D = 1e-2
    viscosity = 5e-2

    _τ_temperature = 1. / (D * _cs2 + 0.5)
    _τ_density = 1. / (viscosity * _cs2 + 0.5)

    nt = 1000

    R = Nx / 5
    U_init = @zeros(3)
    U_init[2] = 0.2

    do_vis = true
    nvis = 10
    anim = Animation()
    st = ceil(Int, Nx / 15)
    Xc, Yc = [x for x in xc, _ in yc], [y for _ in xc, y in yc]
    Xp, Yp = Xc[1:st:end, 1:st:end], Yc[1:st:end, 1:st:end]

    @parallel (1:Nx, 1:Ny, 1:Nz) init!(velocity, density, temperature, U_init, R)
    
    @parallel (2:Nx+1, 2:Ny+1, 2:Nz+1) init_pop!(density_pop, velocity, density)
    @parallel (2:Nx+1, 2:Ny+1, 2:Nz+1) init_pop!(temperature_pop, velocity, temperature)

    @parallel (1:Ny+2, 1:Nz+2) periodic_boundary_update!(:x, density_pop, density_buf)
    @parallel (1:Ny+2, 1:Nz+2) periodic_boundary_update!(:x, temperature_pop, temperature_buf)
    @parallel (1:Nx+2, 1:Nz+2) periodic_boundary_update!(:y, density_pop, density_buf)
    @parallel (1:Nx+2, 1:Nz+2) periodic_boundary_update!(:y, temperature_pop, temperature_buf)

    for i in 1:nt
        @parallel (1:Nx, 1:Ny, 1:Nz) apply_external_force!(velocity, R)

        @parallel (2:Nx+1, 2:Ny+1, 2:Nz+1) collision!(density_pop, velocity, density, _τ_density)
        @parallel (2:Nx+1, 2:Ny+1, 2:Nz+1) collision!(temperature_pop, velocity, temperature, _τ_temperature)

        @parallel (2:Nx+1, 2:Ny+1, 2:Nz+1) streaming!(density_pop, density_buf)
        @parallel (2:Nx+1, 2:Ny+1, 2:Nz+1) streaming!(temperature_pop, temperature_buf)

        # @parallel (1:Nx+2, 1:Nz+2) periodic_boundary_update!(:y, density_pop, density_buf)
        # @parallel (1:Nx+2, 1:Nz+2) periodic_boundary_update!(:y, temperature_pop, temperature_buf)
        # @parallel (2:Nx+1, 2:Nz+1) inlet_boundary_conditions!(:y, density_pop, density_buf, U_init, density)
        # @parallel (2:Nx+1, 2:Nz+1) inlet_boundary_conditions!(:y, temperature_pop, temperature_buf, U_init, temperature)
        @parallel (1:Nx+2, 1:Ny+2) periodic_boundary_update!(:z, density_pop, density_buf)
        @parallel (1:Nx+2, 1:Ny+2) periodic_boundary_update!(:z, temperature_pop, temperature_buf)

        @parallel (2:Ny+1, 2:Nz+1) bounce_back_boundary!(:x, density_pop, density_buf)
        @parallel (2:Ny+1, 2:Nz+1) bounce_back_boundary!(:x, temperature_pop, temperature_buf)
        # bounce_back_boundary(:z, density_pop, density_buf)
        # bounce_back_boundary(:z, temperature_pop, temperature_buf)
        # dirichlet_boundary(:ylower, density_buf, U_init, density)
        # dirichlet_boundary(:ylower, temperature_buf, U_init, temperature)

        density_pop, density_buf = density_buf, density_pop
        temperature_pop, temperature_buf = temperature_buf, temperature_pop 

        @parallel (1:Nx, 1:Ny, 1:Nz) update_moments!(velocity, density, temperature, density_pop, temperature_pop)

        if do_vis && (i % nvis == 0)
            vel_c = copy(velocity[:, :, 1, 1:2])
            for i in axes(vel_c, 1)
                for j in axes(vel_c, 2)
                    vel_c[i, j, :] /= norm(vel_c[i, j, :])
                end
            end

            velx_p = vel_c[1:st:end, 1:st:end, 1]
            vely_p = vel_c[1:st:end, 1:st:end, 2]

            heatmap(xc, yc, density[:, :, 1]', xlims=(xc[1], xc[end]), ylims=(yc[1], yc[end]), title="density", c=:turbo, clim=(0,1))
            dens = quiver!(Xp[:], Yp[:]; quiver=(velx_p[:], vely_p[:]), lw=0.5, c=:black)
            
            heatmap(xc, yc, temperature[:, :, 1]', xlims=(xc[1], xc[end]), ylims=(yc[1], yc[end]), title="temperature", c=:turbo, clim=(0,1))
            temp = quiver!(Xp[:], Yp[:]; quiver=(velx_p[:], vely_p[:]), lw=0.5, c=:black)
            
            plot(dens, temp)
            frame(anim)
        end
    end
    if do_vis
        gif(anim, "../docs/3D_XPU_LB.gif")
    end
end

lb()