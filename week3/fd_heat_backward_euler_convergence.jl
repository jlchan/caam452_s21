using LinearAlgebra
using SparseArrays
using Plots
using ForwardDiff

"This routine runs a convergence test for the heat equation using Backwards Euler
in time and finite differences in space. The exact solution is generated using the
method of manufactured solutions."

uexact(x,t) = log(2+sin(pi*x))*exp(-t)*t
dudx_exact(x,t) = ForwardDiff.derivative(x->uexact(x,t),x)
# du/dt - d^2u/dx^2 = f
f(x,t) = ForwardDiff.derivative(t->uexact(x,t),t)-ForwardDiff.derivative(x->dudx_exact(x,t),x) # -d^2u/dx^2 = f
u0(x) = uexact(x,0)
α(t) = uexact(-1,t)
β(t) = uexact(1,t)

function solve(m, f, α, β, u0, T, dt)
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

    u = u0.(xint)
    Nsteps = ceil(Int,T/dt)
    for i = 1:Nsteps
        t = i*dt
        u .= (I + dt*A)\(u .+ dt*F(t))
    end
    return u,x,xint,h
end

# m = 100
# u,x,xint,h = solve(m, f, α, β, u0, T, dt)
# plot(xint,u,label="FD solution")
# x = LinRange(-1,1,1000)
# plot!(x,uexact.(x,T),label="Exact solution")

mvec = 2 .^ (2:6)
hvec = zeros(length(mvec))
err = zeros(length(mvec))
for (i,m) in enumerate(mvec)

    T = 1.0
    dt = 1/m^2

    u,x,xint,h = solve(m,f,α,β,u0,T,dt)
    hvec[i] = h
    err[i] = maximum(@. abs(uexact(xint,T) - u))
end
plot(hvec,err,marker=:dot,label="Max error")
plot!(hvec,hvec,linestyle=:dash,label="O(h)")
plot!(hvec,hvec.^2,linestyle=:dash,label="O(h^2)")
plot!(xaxis=:log,yaxis=:log,legend=:bottomright)
