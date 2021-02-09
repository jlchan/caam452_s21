using LinearAlgebra
using SparseArrays
using Plots

"This routine solves the heat equation using Forward Euler in time and finite differences
in space. Note that this is unstable unless dt ~ O(h^2)."

m = 100 # number of points
T = .01
dt = .0005

u0(x) = 0.0 # initial condition
f(x) = 5*(Float64((x > -.5) && (x <= 0.0)) - Float64((x < .5) && (x >= 0.0)))
f(x,t) = f(x)*exp(-t)
α(t) = 1.0
β(t) = pi

# define spatial grid
x = LinRange(-1,1,m+2)
xint = x[2:end-1]
h = x[2]-x[1]
A = (1/h^2) * spdiagm(0=>2*ones(m),-1=>-ones(m-1),1=>-ones(m-1))

function F(t)
    b = f.(xint,t) # look for f(x,t)
    b[1] += α(t)/h^2
    b[end] += β(t)/h^2
    return b
end

u = u0.(xint)
Nsteps = ceil(Int,T/dt)
dt = T / Nsteps

interval = 1
@gif for k = 1:Nsteps
    tk = k*dt
    u .= u + dt * (F(tk) - A*u)
    if k % interval==0
        plot(xint,u,linewidth=2,label="Solution",ylims=(-1.0,3.0))
        println("on timestep $k out of $Nsteps.")
    end
end every interval
