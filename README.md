# LatticeBoltzmann.jl

[![CI action](https://github.com/0xBachmann/LatticeBoltzmann.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/0xBachmann/LatticeBoltzmann.jl/actions/workflows/CI.yml)

This project implements a highly parallelized version of the Lattice Boltzmann Method (LBM), designed to simulate fluid dynamics and thermal convection in three-dimensional domains. Leveraging modern multi-core CPUs and GPUs to achieve exceptional performance and scalability is crucial for good performance. Support for both (multi) CPU and GPU is implemented.

## Lattice Boltzmann Method

Unlike traditional Computational Fluid Dynamics (CFD) methods, which numerically solve the conservation equations for macroscopic properties such as mass, momentum, and energy, the Lattice Boltzmann Method (LBM) models fluids as collections of fictitious particles. These particles experience successive propagation and collision processes on a discrete lattice.

LBM orignates from the Boltzmann equations and is capable of solving different PDE's such as Navier-Stokes equations or advection-diffusion. LBM operates by evolving the distribution function $f_i(\boldsymbol{x}, t)$, which represents the probability of particles moving in discrete directions $\boldsymbol{e}_i$ at position $\boldsymbol{x}$ and time $t$. The method uses the discrete lattice structure where $f_i$ are updated via a two step process: _collision_ and _streaming_.

1. **Collision**
   Redistribution of $f_i$
   $$
   f_i^\star(\boldsymbol{x}, t) = f_i(\boldsymbol{x}, t) + \Omega_i,
   $$
   where $\Omega_i$ is the collision term describing the interaction between particles. The most common collsion term is the Bhatnagar–Gross–Krook (BGK) approximation
   $$
   \Omega_i = \frac{1}{\tau}(f_i^\mathrm{eq} - f_i(\boldsymbol{x}, t))
   $$
   where $\tau$ is the relaxation time controlling viscosity and $f_i^\mathrm{eq}$ is some local equilibrium distribution.
2. **Streaming**
   Propagate $f_i^\star$ along discrete lattice direction $\boldsymbol{e}_i$
   $$
   f_i(\boldsymbol{x} + \boldsymbol{e}_i\Delta t, t + \Delta t) = f_i^\star(\boldsymbol{x}, t).
   $$

### Equilibrium Distribution

The equilibrium distribution $f_i^\mathrm{eq}$ is derived from the Maxwell-Boltzmann distribution and can be approximated via Taylor series as
$$
f_i^\mathrm{eq} = w_i\rho\left(1 + \frac{\boldsymbol{e}_i\cdot\boldsymbol{u}}{c_s^2} + \frac{(\boldsymbol{e}_i\cdot\boldsymbol{u})^2}{2c_s^4} - \frac{\boldsymbol{u}\cdot\boldsymbol{u}}{2c_s^2}\right),
$$
where $\rho$ is the local fluid density, $\boldsymbol{u}$ the local fluid velocity,  $w_i$ some weights and $c_s$ the lattice speed of sound. Both $w_i$ and $c_s$ depend on the choice of lattice, i.e. the directions $\boldsymbol{e}_i$. The macroscopic values density $\rho$ and velocity $\boldsymbol{u}$ can be obtained via
$$
\rho = \sum_i f_i,\qquad \rho\boldsymbol{u}=\sum_i f_i\boldsymbol{e}_i.
$$

### DnQm Lattices

Lattice Boltzmann models can be operated on a number of different lattices, both cubic and triangular, and with or without rest particles in the discrete distribution function. A popular way to classify different lattices is with the _DnQm_ scheme. Here Dn stands for n dimension and Qm for m directions. For example, D2Q9 is a two dimensional grid with 9 directions. The direction and weights are given below.

#### D2Q9 Direction Vectors and Weights

| Index (\(i\)) | Direction Vector (\( \boldsymbol{e}_i \)) | Weight (\(w_i\))   |
|---------------|-------------------------------------|--------------------|
| 0             | (0, 0)                             | \( \frac{4}{9} \)  |
| 1             | (1, 0)                             | \( \frac{1}{9} \)  |
| 2             | (0, 1)                             | \( \frac{1}{9} \)  |
| 3             | (-1, 0)                            | \( \frac{1}{9} \)  |
| 4             | (0, -1)                            | \( \frac{1}{9} \)  |
| 5             | (1, 1)                             | \( \frac{1}{36} \) |
| 6             | (-1, 1)                            | \( \frac{1}{36} \) |
| 7             | (-1, -1)                           | \( \frac{1}{36} \) |
| 8             | (1, -1)                            | \( \frac{1}{36} \) |

Thus each point interacts with all of its nearest neighbours. It is also possible to interact with farther neighbours but this project only supports nearest neighbours (i.e. 8 neighbours in 2D and 26 neighbours in 3D).

### Boundary Conditions

One strength of LBM is its capability to handle complex boundaries. Only in the streaming step boundaries need to be considered and only the directions that are coming out of a wall or outside of the domain are unknown.

#### Neumann Boundary Conditions

The simplest boundary conditions are bounce-back method which simulates no-slip velocity boundary condition, i.e. walls. The working principle of bounce-back boundaries is that populations hitting a rigid wall during propagation are reflected back to where they originally came from. This is captured by the following streaming step in case direction $i$ points into a wall or outside of the boundary
$$
f_{\bar{i}}(\boldsymbol{x}_b, t + \Delta t) = f_i^\star(\boldsymbol{x}_b, t),
$$
where $\boldsymbol{x}_b$ is a node next to a boundary and $\bar{i}$ is the opposite direction of $i$, i.e. $\boldsymbol{e}_{\bar{i}} = -\boldsymbol{e}_i$. Thus the unkonw are taken from within the domain.

#### Dirichlet Boundary Conditions

When we want to enforce a specific value at a boundary, we need to prescribe the values of the incoming directions. An immediate idea would be to just take the equilibrium distribution $f_i^\mathrm{eq}$, however this only achieves limited accuracy. [Zou/He][Zou/He] propose more involved boundary conditions for pressure and velocity boundary conditions, which are very common.
Due to its simplicity however, this project uses anti-bounce-back for dirichlet boundary conditions (see [Krueger et al][Krueger et al]). The anti-bounce-back method is very similar in form to the bounce-back
$$
f_{\bar{i}}(\boldsymbol{x}_b, t + \Delta t) = -f_i^\star(\boldsymbol{x}_b, t) + 2f_i^\mathrm{eq}(\boldsymbol{x}_w, t + \Delta t),
$$
where $\boldsymbol{x}_w$ is the location of the wlal. Dirichlet values enter via the equilibirum distribution. In case the wall is at rest, i.e. has zero velocity, anti-bounce-back simplifies to
$$
f_{\bar{i}}(\boldsymbol{x}_b, t + \Delta t) = -f_i^\star(\boldsymbol{x}_b, t) + 2w_iC_w,
$$
where $C_w$ is the to be imposed Dirichlet value.

### Thermal LBM

To incorporate temperature into the system can be done straightforward by adding a second population $g_i$ for the energy. This population follows the same steps as $f_i$

```math
g_i(\boldsymbol{x} + \boldsymbol{e}_i\Delta t, t + \Delta t) = g_i(\boldsymbol{x}, t) + \Omega_i.
```

Then the local temperature can be recovered by 
$$
T = \sum_i g_i
$$

#### Boussinesq Approximation

Temperature induced density change will lead to a buoyancy force leading to coupling between temperature and density and thus between populations $f_i$ and $g_i$. Using the _Boussinesq Approximation_ the buoyancy force is given by
$$
\boldsymbol{F}_b = -\alpha\rho_0(T-T_0)\boldsymbol{g},
$$
where $\alpha$ is the thermal expansion coefficient of the fluid and $\boldsymbol{g}$ is gravity. To account for this force, we need to make two small adjustments to the LBM procedure.

1. Account for force when computing velocity:
   $$
   \boldsymbol{u} = \frac{1}{\rho}\left(\sum_i f_i\boldsymbol{e}_i + \frac{\boldsymbol{F}\Delta t}{2}\right)
   $$
2. Compute source term $S_i$
   $$
   S_i = \left(1-\frac{\Delta t}{2\tau}\right)w_i\left(\frac{\boldsymbol{F}\cdot\boldsymbol{e}_i}{c_s^2} + \frac{(\boldsymbol{F}\cdot\boldsymbol{e}_i)(\boldsymbol{u}\cdot\boldsymbol{e}_i)}{c_s^4} - \frac{\boldsymbol{F}\cdot\boldsymbol{u}}{c_s^2}\right)
   $$
   and add it to collision of the density population
   $$
   f_i^\star(\boldsymbol{x}, t) = f_i(\boldsymbol{x}, t) + \Omega_i + S_i.
   $$

If viscous heating and compression work are relevant, then an additional source term needs to be added to the temperature population as well (see [Krueger et al]).

## References

\[1\] [Qisu Zou, Xiaoyi He; On pressure and velocity boundary conditions for the lattice Boltzmann BGK model. Physics of Fluids 1 June 1997; 9 (6): 1591–1598.][Zou/He]

\[2\] [Krueger, T., Kusumaatmaja, H., Kuzmin, A., Shardt, O., Silva, G., & Viggen, E. M. (2016). The Lattice Boltzmann Method: Principles and Practice. (Graduate Texts in Physics). Springer.][Krueger et al]

[Zou/He]: https://doi.org/10.1063/1.869307

[Krueger et al]: https://doi.org/10.1007/978-3-319-44649-3
