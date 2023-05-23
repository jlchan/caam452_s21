using LinearAlgebra
using SparseArrays
using Plots

"This routine solves the periodic wave equation using finite volumes."

m = 400 # number of points

# define spatial grid
xv = LinRange(-1,1,m+2)
Δx = xv[2]-xv[1]
x = xv[1:end-1] .+ Δx/2 # m+1 cell centers

Δt = .9*Δx # timestep
T = 2.0 # final time

# initial condition and forcing
p0(x) = exp(-100*(x+.5)^2)
u0(x) = 0.

# finite volume flux function
function f(pL,uL,pR,uR)
    λ, R = eigen([0 1;1 0]);
    vL = R\[pL;uL]
    vR = R\[pR;uR]
    flux = R*(max.(λ,0).*vL + min.(λ,0).*vR)
    return flux[1],flux[2]
end
# slightly more expanded analytical flux
function f(pL,uL,pR,uR)
    # v1L = -pL+uL
    v2L = pL+uL
    v1R = -pR+uR
    # v2R = pR+uR
    v1 = -v1R
    v2 = v2L
    pflux = .5*(-v1+v2)
    uflux = .5*(v1+v2)
    return pflux, uflux
end
# cheapest form
function f(pL,uL,pR,uR)
    pflux = .5*(uL+uR) - .5*(pR-pL)
    uflux = .5*(pL+pR) - .5*(uR-uL)
    return pflux,uflux
end

p = p0.(x)
u = u0.(x)
Nsteps = ceil(Int,T/Δt)
Δt = T / Nsteps

index_left = [m+1; 1:m+1]
index_right = [1:m+1; 1]

interval = 10
plot()
unorm = zeros(Nsteps)
pflux = zeros(m+2)
uflux = zeros(m+2)
@gif for k = 1:Nsteps
    # global u,f
    for i = 1:m+2 # loop over cell interfaces
        left = index_left[i]
        right = index_right[i]
        flux_i = f(p[left],u[left],p[right],u[right])
        pflux[i] = flux_i[1]
        uflux[i] = flux_i[2]
    end
    for i = 1:m+1
        p[i] = p[i] - Δt/Δx * (pflux[i+1]-pflux[i])
        u[i] = u[i] - Δt/Δx * (uflux[i+1]-uflux[i])
    end

    if k % interval==0
        plot(x,p,linewiΔth=2,legend=false,title="Solution at time $(k*Δt)",ylims=(-1.5,1.5))
        println("on timestep $k out of $Nsteps.")
    end
end every interval
# plot(x,u,linewiΔth=2,legend=false,title="Solution at final time $(T)",ylims=(-1.5,1.5))
