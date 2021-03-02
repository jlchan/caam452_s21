using LinearAlgebra
using SparseArrays
using Plots

"This routine solves the advection equation with an inflow boundary condition using
Forward Euler in time and finite differences in space."

m = 100 # number of points

# define spatial grid
x = LinRange(-1,1,m+2)
xint = x[2:end]
h = x[2]-x[1]

α(t) = sin(pi*t)
dt = h # timestep
T = 4.0 # final time

# initial condition and forcing
u0(x) = 0.0
f(x,t) = 0.0
function F(α,xint,t,h)
    fvec = f.(xint,t)
    fvec[1] += α(t)/h
    return fvec
end

# upwind approximation of first derivative
Q = (1/h) * spdiagm(0=>ones(m+1),-1=>-ones(m))

u = u0.(xint)
Nsteps = ceil(Int,T/dt)
dt = T / Nsteps

interval = 1
plot()
@gif for k = 1:Nsteps
    tk = k*dt
    u .= u + dt * (F(α,xint,tk,h) - Q*u)
    if k % interval==0
        plot(xint,u,linewidth=2,label="Solution")
        println("on timestep $k out of $Nsteps.")
    end
end every interval

# plot(xint,u,linewidth=2,label="Solution")
# plot!(x,u0.(x .- T),linewidth=2,label="Exact solution",ylims=(-1.0,2.0))
