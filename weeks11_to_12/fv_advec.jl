using LinearAlgebra
using SparseArrays
using Plots

"This routine solves the periodic advection equation using finite volumes."

m = 200 # number of points

# define spatial grid
xv = LinRange(-1,1,m+2)
Δx = xv[2]-xv[1]
x = xv[1:end-1] .+ Δx/2 # m+1 cell centers

a = 1 # advection speed
Δt = .9*Δx / a # timestep
T = 2.0 # final time

# initial condition and forcing
u0(x) = exp(-25*x^2)
u = u0.(x)
f(uL,uR) = uL

# indexing for periodic boundaries
index_left = [m+1; 1:m+1]
index_right = [1:m+1; 1]
fv_flux = zeros(m+2) # storage at interfaces

Nsteps = ceil(Int,T/Δt)
Δt = T / Nsteps
interval = 10
plot()
unorm = zeros(Nsteps)
@gif for k = 1:Nsteps
    for i = 1:m+2 # loop over cell interfaces
        left = index_left[i]
        right = index_left[i]
        fv_flux[i] = f(u[left],u[right])
    end
    for i = 1:m+1
        u[i] = u[i] - a*Δt/Δx * (fv_flux[i+1]-fv_flux[i])
    end

    if k % interval==0
        plot(x,u,linewidth=2,legend=false,title="Solution at time $(k*Δt)",ylims=(-1.5,1.5))
        println("on timestep $k out of $Nsteps.")
    end
end every interval
# plot(x,u,linewiΔth=2,legend=false,title="Solution at final time $(T)",ylims=(-1.5,1.5))
