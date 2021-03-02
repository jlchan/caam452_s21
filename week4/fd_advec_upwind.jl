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

dt = .5*h # timestep
T = 2.0 # final time

# initial condition and forcing
u0(x) = sin(pi*x)
f(x,t) = 0.0

# upwind approximation of first derivative
Q = (1/h) * spdiagm(0=>ones(m+1),-1=>-ones(m))
Q[1,end] = -1/h

u = u0.(xint)
Nsteps = ceil(Int,T/dt)
dt = T / Nsteps

interval = 10
@gif for k = 1:Nsteps
    tk = k*dt
    u .= u - dt * Q*u
    if k % interval==0
        plot(xint,u,linewidth=2,label="Solution")
        plot!(x,u0.(x .- tk),linewidth=2,label="Exact solution",ylims=(-1.0,2.0))
        # error = @. abs(u - u0(xint - tk))
        # plot(xint,error,linewidth=2,label="Error",ylims=(-1e-13,1e-13))
        println("on timestep $k out of $Nsteps.")
    end
end every interval

plot(xint,u,linewidth=2,label="Solution")
plot!(x,u0.(x .- T),linewidth=2,label="Exact solution",ylims=(-1.0,2.0))
