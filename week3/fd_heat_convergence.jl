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
f(x,t) = ForwardDiff.derivative(t->uexact(x,t),t) -
         ForwardDiff.derivative(x->dudx_exact(x,t),x)
u0(x) = uexact(x,0)
α(t) = uexact(-1,t)
β(t) = uexact(1,t)

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

function solve_backward_Euler(m, f, α, β, u0, T, dt)

    A,x,xint,h,F = build_FD(m,f,α,β)

    u = u0.(xint)
    Nsteps = ceil(Int,T/dt)
    dt = T/Nsteps
    for i = 1:Nsteps
        t = i*dt
        u .= (I + dt*A)\(u .+ dt*F(t))
    end
    return u,x,xint,h
end

function solve_Crank_Nicolson(m, f, α, β, u0, T, dt)

    A,x,xint,h,F = build_FD(m,f,α,β)

    u = u0.(xint)
    Nsteps = ceil(Int,T/dt)
    dt = T/Nsteps
    for i = 1:Nsteps
        tprev = (i-1)*dt
        t = i*dt
        u .= (I + .5*dt*A)\((I - .5*dt*A)*u .+ .5*dt.*(F(t)+F(tprev)))
    end
    return u,x,xint,h
end

# m = 40
# T = .01
# dt = T
# u,x,xint,h = solve_backward_Euler(m, f, α, β, u0, T, dt)
# plot(xint,u,mark=:dot,label="FD solution")
# x = LinRange(-1,1,1000)
# plot!(x,uexact.(x,T),label="Exact solution")

mvec = 2 .^ (2:8)
hvec = zeros(length(mvec))
hvec = Float64[]
err1 = Float64[]
err2 = Float64[]
for (i,m) in enumerate(mvec)

    T = 1.0
    dt = 2/m

    u,x,xint,h = solve_backward_Euler(m,f,α,β,u0,T,dt)
    push!(err1,maximum(@. abs(uexact(xint,T) - u)))

    u,x,xint,h = solve_Crank_Nicolson(m,f,α,β,u0,T,dt)
    push!(err2,maximum(@. abs(uexact(xint,T) - u)))

    push!(hvec,h)
end
plot(hvec,err1,marker=:dot,label="Max error (backwards Euler)")
plot!(hvec,err2,marker=:dot,label="Max error (Crank-Nicolson)")
plot!(hvec,hvec,linestyle=:dash,label="O(h)")
plot!(hvec,hvec.^2,linestyle=:dash,label="O(h^2)")
plot!(xaxis=:log,yaxis=:log,legend=:bottomright)
