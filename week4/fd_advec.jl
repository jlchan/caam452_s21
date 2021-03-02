using LinearAlgebra
using SparseArrays
using Plots

"This routine solves the periodic advection-diffusion equation using Forward Euler
in time and 2nd order finite differences in space."

m = 100 # number of points

# define spatial grid
x = LinRange(-1,1,m+2) # x_0, x_1, ..., x_m, x_{m+1} = x_0
xint = x[1:end-1]
h = x[2]-x[1]

a = 1.0 # advection speed
ϵ = h/2 # diffusion coefficient
dt = h # timestep
T = 2.0 # final time

# initial condition and forcing
u0(x) = sin(pi*x)
f(x,t) = 0.0

# approximation of second derivative: largest eig = O(1/h^2)
A = (1/h^2) * spdiagm(0=>2*ones(m+1),-1=>-ones(m),1=>-ones(m))
A[1,end] = -1/h^2
A[end,1] = -1/h^2

# approximation of first derivative: largest eig = O(1/h)
Q = (1/(2*h)) * spdiagm(1=>ones(m),-1=>-ones(m))
Q[1,end] = -1/(2*h)
Q[end,1] = 1/(2*h)

u = u0.(xint)
Nsteps = ceil(Int,T/dt)
dt = T / Nsteps

interval = 10
@gif for k = 1:Nsteps
    tk = k*dt
    u .= u + dt * (f.(xint,tk) - (a*Q + ϵ*A)*u)
    if k % interval==0
        plot(xint,u,linewidth=2,label="Solution",ylims=(-1.5,1.5))
        println("on timestep $k out of $Nsteps.")
    end
end every interval

maxλ = Float64[]
ϵvec = .0:.0001:.025
for ϵ = ϵvec
    append!(maxλ,maximum(abs.(eigvals(Matrix(I - dt*(Q + ϵ*A))))))
end
scatter(ϵvec,maxλ,leg=false)
plot!(title="Largest eigenvalue magnitude with ϵ")
