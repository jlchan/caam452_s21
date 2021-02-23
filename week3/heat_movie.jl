using LinearAlgebra
using SparseArrays
using Plots

"This routine generates a gif movie showing how the heat equation rapidly dissipates
high frequencies."

# m = number of grid points
function build_FD(m,f,α,β)
    x = LinRange(-1,1,m+2)
    xint = x[2:end-1]
    h = x[2]-x[1]
    A = (1/h^2) * spdiagm(0=>2*ones(m),-1=>-ones(m-1),1=>-ones(m-1))

    function F(t)
        b = f.(xint,t)
        b[1] += α(t)/h^2
        b[m] += β(t)/h^2
        return b
    end
    return A,x,xint,h,F
end

m = 200
T = .025

dt = .005/m

A,x,xint,h,F = build_FD(m,(x,t)->0,t->0,t->0)

u0(x) = sin(pi*x) + .1*sin(2*pi*x) - .33*sin(3*pi*x) + 10*sin(8*pi*x)
u = u0.(xint)
Nsteps = ceil(Int,T/dt)
dt = T/Nsteps
@gif for i = 1:Nsteps
    tprev = (i-1)*dt
    t = i*dt
    u .= (I + .5*dt*A)\((I - .5*dt*A)*u + .5*dt*(F(t)+F(tprev)))
    if i%10==0
        plot(xint,u,mark=:dot,leg=false)
        println("on timestep $i out of $Nsteps")
    end
end every 10
