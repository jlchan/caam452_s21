using LinearAlgebra
using SparseArrays
using Plots

"This routine solves the heat equation using Backwards Euler in time and finite differences
in space. The output is saved as a .gif file (which should play automatically when this file is run)."

m = 100 # number of points
dt = .01 # time-step
T = 2.0

u0(x) = 0.0 # initial condition
f(x) = 5*(Float64((x > -.5) && (x <= 0.0)) - Float64((x < .5) && (x >= 0.0)))
f(x,t) = f(x)*exp(-t)
α(t) = 1.0
β(t) = sin(2*pi*t) # du/dx

# define spatial grid
x = LinRange(-1,1,m+2)
xint = x[2:end-1]
h = x[2]-x[1]
A = (1/h^2) * spdiagm(0=>2*ones(m),-1=>-ones(m-1),1=>-ones(m-1))

function F(xint,t)
    b = f.(xint,t)
    b[1] += α(t)/h^2
    b[end] += β(t)/h^2
    return b
end

u = u0.(xint)
Nsteps = ceil(Int,T/dt)
dt = T / Nsteps

interval = 2
@gif for i = 1:Nsteps
    t = (i+1)*dt
    u .= (I + dt*A)\(u + dt*F(xint,t))
    if i % interval == 0 # % = mod function
        plot(xint,u,linewidth=2,label="Solution",ylims=(-1.0,3.0))
        println("on timestep $i out of $Nsteps.")
    end
end every interval
