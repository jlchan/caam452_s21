using LinearAlgebra
using SparseArrays
using Plots

m = 100 # number of points
FinalTime = 1.0
dt = .01

x = LinRange(-1,1,m+2)
xint = x[2:end-1]

u0(x) = 0.0
f(x) = 10*(Float64((x > -.5) && (x <= 0.0)) - Float64((x < .5) && (x >= 0.0)))
f(x,t) = f(x)*exp(-t)
α,β = 1.0,pi

h = x[2]-x[1]
A = (1/h^2) * spdiagm(0=>2*ones(m),-1=>-ones(m-1),1=>-ones(m-1))
b = f.(xint)
b[1] += α/h^2
b[m] += β/h^2

u = u0.(xint)
Nsteps = ceil(Int,FinalTime/dt)
interval = 1
@gif for i = 1:Nsteps
    t = i*dt
    u .= (I + dt*A)\(@. u + dt*f(xint,t))
    if i%interval==0
        plot(xint,u,linewidth=2,label="Solution",ylims=(-1.0,1.0))
        println("on timestep $i out of $Nsteps.")
    end
end every interval
