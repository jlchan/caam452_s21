using LinearAlgebra
using SparseArrays
using Plots

"This routine solves the periodic advection equation using Forward Euler in time
and finite volumes in space."

m = 100 # number of points

# define spatial grid
xv = LinRange(-1,1,m+2)
h = xv[2]-xv[1]
x = xv[1:end-1] .+ h/2 # m+1 cell centers

dt = .99*h # timestep
T = 10.0 # final time

# initial condition and forcing
u0(x) = exp(-25*x^2)

# numerical flux
f(u) = u
f_central(uL,uR) = .5*(f(uL) + f(uR))
f(uL,uR) = .5*(f(uL) + f(uR)) - 0*.5*(uR-uL)

u = u0.(x)
Nsteps = ceil(Int,T/dt)
dt = T / Nsteps

interval = 100
plot()
unorm = zeros(Nsteps)
# @gif
for k = 1:Nsteps
    global u
    for i = 1:length(u)
        if i==1
            fR = f(u[1],u[2])
            fL = f(u[end],u[1])
        elseif i==length(u)
            fR = f(u[end],u[1])
            fL = f(u[end-1],u[end])
        else
            fR = f(u[i],u[i+1])
            fL = f(u[i-1],u[i])
        end
        u[i] -= dt*(fR-fL)/h # forward Euler
    end
    unorm[k] = norm(u)

    if k % interval==0
        # plot(x,u,linewidth=2,legend=false,title="Solution at time $(k*dt)",ylims=(-1.5,1.5))
        println("on timestep $k out of $Nsteps.")
    end
end #every interval
# plot(x,u,linewidth=2,legend=false,title="Solution at final time $(T)",ylims=(-1.5,1.5))
