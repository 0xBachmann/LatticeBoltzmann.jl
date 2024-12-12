using LinearAlgebra
using Plots
using ParallelStencil

const method = :D2Q9

@static if method == :D2Q5
    const Q = 5
    const directions = [
        [0, 0], 
        [1, 0], [-1, 0], [0, 1], [0, -1]
        ]
    const weights = [
        1/3, 
        1/6, 1/6, 1/6, 1/6
        ]
elseif method == :D2Q9
    const Q = 9
    const directions = [
        [0, 0], 
        [1, 0], [-1, 0], [0, 1], [0, -1], 
        [1, 1], [-1, -1], [1, -1], [-1, 1]
        ]
    const weights = [
        4/9, 
        1/9, 1/9, 1/9, 1/9, 
        1/36, 1/36, 1/36, 1/36
        ]
end
    
# Speed of sound (in lattice units)
const _cs2 = 3. # cs^2 = 1./3. * (dx**2/dt**2)
const _cs4 = 9.

@views function collision!(pop, velocity, values, _τ)
    for i in axes(pop, 1)
        for j in axes(pop, 2)
            v = velocity[i, j, :]
            for q in axes(pop, 3)
                pop[i, j, q] = (1. - _τ) * pop[i, j, q] + _τ * f_eq(q, v, values[i, j])
            end
        end
    end
end

@views function dirichlet_boundary(boundary, pop, velocity, values)
    if boundary == :ylower
        y = 1
        for i in axes(pop, 1)
            for q in axes(pop, 3)
                @inbounds pop[i, y, q] = f_eq(q, velocity, values[i, y])
            end            
        end
    elseif boundary == :yupper
        y = size(pop, 2)
        for i in axes(pop, 1)
            for q in axes(pop, 3)
                @inbounds pop[i, y, q] = f_eq(q, velocity, values[i, y])
            end            
        end
    elseif boundary == :xlower
        x = 1
        for j in axes(pop, 2)
            for q in axes(pop, 3)
                @inbounds pop[x, j, q] = f_eq(q, velocity, values[x, j])
            end            
        end
    elseif boundary == :xupper
        x = size(pop, 1)
        for j in axes(pop, 2)
            for q in axes(pop, 3)
                @inbounds pop[x, j, q] = f_eq(q, velocity, values[x, j])
            end            
        end
    end
end

@views function bounce_back_boundary(dimension, pop, pop_buf)
    if dimension == :y
        ly = 1
        ry = size(pop, 2)
        for i in axes(pop, 1)
            for q in axes(pop, 3)
                if directions[q][2] == 1
                    pop_buf[i, ry, q] = pop[i, ry, (q%2 == 0) ? q+1 : q-1]
                end
                if directions[q][2] == -1
                    pop_buf[i, ly, q] = pop[i, ly, (q%2 == 0) ? q+1 : q-1]
                end
            end            
        end
    elseif dimension == :x
        lx = 1
        rx = size(pop, 1)
        for j in axes(pop, 2)
            for q in axes(pop, 3)
                if directions[q][1] == 1
                    pop_buf[rx, j, q] = pop[rx, j, (q%2 == 0) ? q+1 : q-1]
                end
                if directions[q][1] == -1
                    pop_buf[lx, j, q] = pop[lx, j, (q%2 == 0) ? q+1 : q-1]
                end
            end            
        end
    end
end

@views function streaming!(pop, pop_buf)
    for i in axes(pop, 1)
        for j in axes(pop, 2)
            for q in axes(pop, 3)
                xidx = i - directions[q][1]
                if xidx == 0
                    xidx = size(pop, 1)
                elseif xidx == size(pop, 1) + 1
                    xidx = 1
                end
                yidx = j - directions[q][2]
                if yidx == 0
                    yidx = size(pop, 1)
                elseif yidx == size(pop, 2) + 1
                    yidx = 1
                end
                pop_buf[i, j, q] = pop[xidx, yidx, q]
            end
        end
    end
end

@views function init!(velocity, density, temperature, U_init, R)
    density .= 1
    temperature .= 0

    Nx = size(velocity, 1)
    Ny = size(velocity, 2)

    for i in axes(velocity, 1)
        for j in axes(velocity, 2)
            if ((i - Nx/2)*(i - Nx/2) + (j - Ny/3)*(j - Ny/3)) < R^2
                velocity[i, j, :] = zeros(2)
                temperature[i, j] = 1
            else 
                velocity[i, j, :] = U_init
            end
        end
    end
end

@views function update_moments!(velocity, density, temperature, density_pop, temperature_pop)
    for i in axes(velocity, 1)
        for j in axes(velocity, 2)
            cell_density = 0
            cell_velocity = zeros(2)
            cell_temperature = 0
            for q in axes(density_pop, 3)
                cell_density += density_pop[i, j, q]
                cell_temperature += temperature_pop[i, j, q]
                cell_velocity += directions[q] * density_pop[i, j, q]
            end

            cell_velocity /= cell_density
            velocity[i, j, :] = cell_velocity
            density[i, j] = cell_density
            temperature[i, j] = cell_temperature
        end
    end

end

@views function apply_external_force!(velocity, R)

    Nx = size(velocity, 1)
    Ny = size(velocity, 2)
    
    for i in axes(velocity, 1)
        for j in axes(velocity, 2)
            if ((i - Nx/2)*(i - Nx/2) + (j - Ny/3)*(j - Ny/3)) < R^2
                velocity[i, j, :] = zeros(2)
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
            for q in axes(pop, 3)
                @inbounds pop[i, j, q] = f_eq(q, velocity[i, j, :], values[i, j])
            end
        end
    end
end

function lb()
    Nx = 40
    Ny = 70

    lx = 40
    ly = 40

    xc, yc = LinRange(0, lx, Nx), LinRange(0, ly, Ny)


    density_pop = zeros(Nx, Ny, Q)
    density_buf = copy(density_pop)
    
    temperature_pop = zeros(Nx, Ny, Q)
    temperature_buf = copy(temperature_pop)

    velocity = zeros(Nx, Ny, 2)
    density = zeros(Nx, Ny)
    temperature = zeros(Nx, Ny)

    D = 1e-2
    viscosity = 5e-2

    _τ_temperature = 1. / (D * _cs2 + 0.5)
    _τ_density = 1. / (viscosity * _cs2 + 0.5)

    nt = 1000

    R = Nx / 5
    U_init = zeros(2)
    U_init[2] = 0.2

    do_vis = true
    nvis = 5
    anim = Animation()
    st = ceil(Int, Nx / 25)
    Xc, Yc = [x for x in xc, y in yc], [y for x in xc, y in yc]
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
        # dirichlet_boundary(:xlower, density_buf, U_init, density)
        # dirichlet_boundary(:xlower, temperature_buf, U_init, temperature)

        density_pop, density_buf = density_buf, density_pop
        temperature_pop, temperature_buf = temperature_buf, temperature_pop 

        update_moments!(velocity, density, temperature, density_pop, temperature_pop)

        if do_vis && (i % nvis == 0)
            vel_c = copy(velocity)
            for i in axes(vel_c, 1)
                for j in axes(vel_c, 2)
                    vel_c[i, j, :] /= norm(vel_c[i, j, :])
                end
            end

            velx_p = vel_c[1:st:end, 1:st:end, 1]
            vely_p = vel_c[1:st:end, 1:st:end, 2]

            heatmap(xc, yc, density', xlims=(xc[1], xc[end]), ylims=(yc[1], yc[end]), title="density", c=:turbo)
            dens = quiver!(Xp[:], Yp[:]; quiver=(velx_p[:], vely_p[:]), lw=0.5, c=:black)
            
            heatmap(xc, yc, temperature', xlims=(xc[1], xc[end]), ylims=(yc[1], yc[end]), title="temperature", c=:turbo)
            temp = quiver!(Xp[:], Yp[:]; quiver=(velx_p[:], vely_p[:]), lw=0.5, c=:black)
            
            plot(dens, temp)
            frame(anim)
        end
    end
    if do_vis
        gif(anim, "../docs/2D_LB.gif")
    end
end

lb()