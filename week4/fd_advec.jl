using LinearAlgebra
using SparseArrays
using Plots

"This routine solves the periodic advection-diffusion equation using Forward Euler
in time and finite differences in space."

m = 100 # number of points

# define spatial grid
x = LinRange(-1,1,m+2)
xint = x[1:end-1]
h = x[2]-x[1]

ϵ = h/2 # diffusion coefficient
dt = .25*h # timestep
T = 8.0 # final time

# initial condition and forcing
u0(x) = sin(pi*x)
f(x,t) = 0.0

# approximation of second derivative
A = (1/h^2) * spdiagm(0=>2*ones(m+1),-1=>-ones(m),1=>-ones(m))
A[1,end] = -1/h^2
A[end,1] = -1/h^2

# approximation of first derivative
Q = (1/h) * spdiagm(1=>ones(m),-1=>-ones(m))
Q[1,end] = -1/h
Q[end,1] = 1/h

u = u0.(xint)
Nsteps = ceil(Int,T/dt)
dt = T / Nsteps

interval = 10
@gif for k = 1:Nsteps
    tk = k*dt
    u .= u + dt * (f.(xint,tk) - (Q + ϵ*A)*u)
    if k % interval==0
        plot(xint,u,linewidth=2,label="Solution",ylims=(-1.0,3.0))
        println("on timestep $k out of $Nsteps.")
    end
end every interval
