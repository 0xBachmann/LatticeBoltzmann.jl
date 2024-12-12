using LinearAlgebra
using Plots
using ParallelStencil

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

@views function collision!(pop, velocity, values, _τ)
    for i in axes(pop, 1)[2:end]
        for j in axes(pop, 2)[2:end]
            for k in axes(pop, 3)[2:end]
                v = velocity[i, j, k, :]
                for q in 1:Q
                    pop[i, j, k, q] = (1. - _τ) * pop[i, j, k, q] + _τ * f_eq(q, v, values[i, j, k])
                end
            end
        end
    end
end

@views function dirichlet_boundary(boundary, pop, velocity, values)
    if boundary == :xupper
        x = size(pop, 1)
        for j in axes(pop, 2)[2:end-1]
            for k in axes(pop, 3)[2:end-1]
                for q in 1:Q
                    @inbounds pop[x, j, k, q] = f_eq(q, velocity, values[x, j, k])
                end      
            end               
        end
    elseif  boundary == :zlower
        z = 1
        for i in axes(pop, 1)[2:end-1]
            for j in axes(pop, 2)[2:end-1]
                for q in 1:Q
                    @inbounds pop[i, j, z] = f_eq(q, velocity, values[i, j, z])
                end
            end
        end
    elseif boundary == :ylower
        y = 1
        for i in axes(pop, 1)[2:end-1]
            for k in axes(pop, 3)[2:end-1]
                for q in 1:Q
                    @inbounds pop[i, y, k, q] = f_eq(q, velocity, values[i, y, k])
                end      
            end      
        end
    elseif boundary == :yupper
        y = size(pop, 2)
        for i in axes(pop, 1)[2:end-1]
            for k in axes(pop, 3)[2:end-1]
                for q in 1:Q
                    @inbounds pop[i, y, k, q] = f_eq(q, velocity, values[i, y, k])
                end      
            end           
        end
    elseif boundary == :xlower
        x = 1
        for j in axes(pop, 2)[2:end-1]
            for k in axes(pop, 3)[2:end-1]
                for q in 1:Q
                    @inbounds pop[x, j, k, q] = f_eq(q, velocity, values[x, j, k])
                end      
            end      
        end
    elseif boundary == :zupper
        z = size(popo, 3)
        for i in axes(pop, 1)[2:end-1]
            for j in axes(pop, 2)[2:end-1]
                for q in 1:Q
                    @inbounds pop[i, j, z] = f_eq(q, velocity, values[i, j, z])
                end
            end
        end
    end
end

@views function periodic_boundary_conditions(dimension, pop, pop_buf)
    if dimension == :x
        Nx = size(pop, 1)
        for j in axes(pop, 2)[2:end-1]
            for k in axes(pop, 3)[2:end-1]
                for q in 1:Q
                    for i in [1, Nx]
                        xidx = i - directions[q][1]
                        if xidx == 0
                            xidx = Nx
                        elseif xidx == Nx + 1
                            xidx = 1
                        end
                        pop_buf[i, j, k, q] = pop[xidx, j - directions[q][2], k - directions[q][3], q]
                    end
                end   
            end         
        end
    elseif dimension == :y
        Ny = size(pop, 2)
        for i in axes(pop, 1)[2:end-1]
            for k in axes(pop, 3)[2:end-1]
                for q in 1:Q
                    for j in [1, Ny]
                        yidx = i - directions[q][2]
                        if yidx == 0
                            yidx = Nx
                        elseif yidx == Nx + 1
                            yidx = 1
                        end
                        pop_buf[i, j, k, q] = pop[i - directions[q][1], yidx, k - directions[q][3], q]
                    end
                end    
            end        
        end
    elseif dimension == :z
        Nz = size(pop, 3)
        for i in axes(pop, 1)[2:end-1]
            for j in axes(pop, 2)[2:end-1]
                for q in 1:Q
                    for j in [1, Ny]
                        zidx = i - directions[q][3]
                        if zidx == 0
                            zidx = Nx
                        elseif zidx == Nx + 1
                            zidx = 1
                        end
                        pop_buf[i, j, k, q] = pop[i - directions[q][1], j - directions[q][2], zidx, q]
                    end
                end
            end
        end
    end
end

@views function bounce_back_boundary(dimension, pop, pop_buf)
    if dimension == :x
        lx = 1
        rx = size(pop, 1)
        for j in axes(pop, 2)[2:end-1]
            for k in axes(pop, 3)[2:end-1]
                for q in 1:Q
                    if directions[q][1] == 1
                        pop_buf[rx, j, k, q] = pop[rx, j, k, (q%2 == 0) ? q+1 : q-1]
                    end
                    if directions[q][1] == -1
                        pop_buf[lx, j, k, q] = pop[lx, j, k, (q%2 == 0) ? q+1 : q-1]
                    end
                end   
            end         
        end
    elseif dimension == :y
        ly = 1
        ry = size(pop, 2)
        for i in axes(pop, 1)[2:end-1]
            for k in axes(pop, 3)[2:end-1]
                for q in 1:Q    
                    if directions[q][2] == 1
                        pop_buf[i, ry, k, q] = pop[i, ry, k, (q%2 == 0) ? q+1 : q-1]
                    end
                    if directions[q][2] == -1
                        pop_buf[i, ly, k, q] = pop[i, ly, k, (q%2 == 0) ? q+1 : q-1]
                    end
                end    
            end        
        end
    elseif dimension == :z
        lz = 1
        rz = size(pop, 3)
        for i in axes(pop, 1)[2:end-1]
            for j in axes(pop, 2)[2:end-1]
                for q in 1:Q
                    if directions[q][3] == 1 
                        pop_buf[i, j, rz, q] = pop[i, j, rz, (q%2 == 0) ? q+1 : q-1]
                    elseif directions[q][3] == -1
                        pop_buf[i, j, lz, q] = pop[i, j, lz, (q%2 == 0) ? q+1 : q-1]
                    end
                end
            end
        end
    end
end

@views function streaming!(pop, pop_buf)
    for i in axes(pop, 1)[2:end-1]
        for j in axes(pop, 2)[2:end-1]
            for k in axes(pop, 3)[2:end-1]
                for q in 1:Q
                    pop_buf[i, j, k, q] = pop[i - directions[q][1], j - directions[q][2], k - directions[q][3], q]
                end
            end
        end
    end
end

@views function init!(velocity, density, temperature, U_init, R)
    density .= 1
    temperature .= 0

    Nx = size(velocity, 1)
    Ny = size(velocity, 2)
    Nz = size(velocity, 3)

    for i in axes(velocity, 1)
        for j in axes(velocity, 2)
            for k in axes(velocity, 3)
                if ((i - Nx/2)^2 + (j - Ny/3)^2) < R^2
                    velocity[i, j, k, :] = zeros(3)
                    temperature[i, j, k] = 1
                else 
                    velocity[i, j, k, :] = U_init
                end
            end
        end
    end
end

@views function update_moments!(velocity, density, temperature, density_pop, temperature_pop)
    for i in axes(velocity, 1)
        for j in axes(velocity, 2)
            for k in axes(velocity, 3)
                cell_density = 0
                cell_velocity = zeros(3)
                cell_temperature = 0
                for q in 1:Q
                    cell_density += density_pop[i, j, k, q]
                    cell_temperature += temperature_pop[i, j, k, q]
                    cell_velocity += directions[q] * density_pop[i, j, k, q]
                end

                cell_velocity /= cell_density
                velocity[i, j, k, :] = cell_velocity
                density[i, j, k] = cell_density
                temperature[i, j, k] = cell_temperature
            end
        end
    end
end

@views function apply_external_force!(velocity, R)

    Nx = size(velocity, 1)
    Ny = size(velocity, 2)
    
    for i in axes(velocity, 1)
        for j in axes(velocity, 2)
            for k in axes(velocity, 3)
                if ((i - Nx/2)*(i - Nx/2) + (j - Ny/3)*(j - Ny/3)) < R^2
                    velocity[i, j, k, :] = zeros(3)
                end
            end
        end
    end
end

@views function f_eq(q, velocity, density)
    uc = dot(velocity, directions[q])
    uc2 = uc^2
    u2 = norm(velocity) ^ 2
    return weights[q] * density * (1. + uc * _cs2 + 0.5 * uc2 * _cs4 - 0.5 * u2 * _cs2)
end

@views function init_pop!(pop, velocity, values)
    for i in axes(pop, 1)
        for j in axes(pop, 2)
            for k in axes(pop, 3)
                for q in 1:Q
                    @inbounds pop[i, j, k, q] = f_eq(q, velocity[i, j, k, :], values[i, j, k])
                end
            end
        end
    end
end

function lb()
    Nx = 40
    Ny = 70
    Nz = 1

    lx = 40
    ly = 40

    xc, yc = LinRange(0, lx, Nx + 2), LinRange(0, ly, Ny + 2)


    density_pop = zeros(Nx + 2, Ny + 2, Nz + 2, Q)
    density_buf = copy(density_pop)
    
    temperature_pop = zeros(Nx + 2, Ny + 2, Nz + 2, Q)
    temperature_buf = copy(temperature_pop)

    velocity = zeros(Nx + 2, Ny + 2, Nz + 2, 3)
    density = zeros(Nx + 2, Ny + 2, Nz + 2)
    temperature = zeros(Nx + 2, Ny + 2, Nz + 2)

    D = 1e-2
    viscosity = 5e-2

    _τ_temperature = 1. / (D * _cs2 + 0.5)
    _τ_density = 1. / (viscosity * _cs2 + 0.5)

    nt = 1000

    R = Nx / 5
    U_init = zeros(3)
    U_init[2] = 0.2

    do_vis = true
    nvis = 5
    anim = Animation()
    st = ceil(Int, Nx / 20)
    Xc, Yc = [x for x in xc[2:end-1], _ in yc[2:end-1]], [y for _ in xc[2:end-1], y in yc[2:end-1]]
    Xp, Yp = Xc[1:st:end, 1:st:end], Yc[1:st:end, 1:st:end]

    init!(velocity, density, temperature, U_init, R)

    init_pop!(density_pop, velocity, density)
    init_pop!(temperature_pop, velocity, temperature)

    for i in 1:nt
        apply_external_force!(velocity, R)

        collision!(density_pop, velocity, density, _τ_density)
        collision!(temperature_pop, velocity, temperature, _τ_temperature)

        streaming!(density_pop, density_buf)
        streaming!(temperature_pop, temperature_buf)

        bounce_back_boundary(:x, density_pop, density_buf)
        bounce_back_boundary(:x, temperature_pop, temperature_buf)
        # bounce_back_boundary(:z, density_pop, density_buf)
        # bounce_back_boundary(:z, temperature_pop, temperature_buf)
        # dirichlet_boundary(:ylower, density_buf, U_init, density)
        # dirichlet_boundary(:ylower, temperature_buf, U_init, temperature)

        density_pop, density_buf = density_buf, density_pop
        temperature_pop, temperature_buf = temperature_buf, temperature_pop 

        update_moments!(velocity, density, temperature, density_pop, temperature_pop)

        if do_vis && (i % nvis == 0)
            vel_c = copy(velocity[:, :, 1, :])
            for i in axes(vel_c, 1)
                for j in axes(vel_c, 2)
                    vel_c[i, j, :] /= norm(vel_c[i, j, :])
                end
            end

            velx_p = vel_c[1:st:end, 1:st:end, 1]
            vely_p = vel_c[1:st:end, 1:st:end, 2]

            heatmap(xc, yc, density[2:end-1, 2:end-1, 1]', xlims=(xc[1], xc[end]), ylims=(yc[1], yc[end]), title="density", c=:turbo)
            dens = quiver!(Xp[:], Yp[:]; quiver=(velx_p[:], vely_p[:]), lw=0.5, c=:black)
            
            heatmap(xc, yc, temperature[2:end-1, 2:end-1, 1]', xlims=(xc[1], xc[end]), ylims=(yc[1], yc[end]), title="temperature", c=:turbo)
            temp = quiver!(Xp[:], Yp[:]; quiver=(velx_p[:], vely_p[:]), lw=0.5, c=:black)
            
            plot(dens, temp)
            frame(anim)
        end
    end
    if do_vis
        gif(anim, "../docs/3D_LB.gif")
    end
end

lb()