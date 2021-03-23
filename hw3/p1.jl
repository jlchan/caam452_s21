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
    tvec = Float64[]
    energy = Float64[]
    for i = 1:Nsteps
        t = i*dt
        u .= (I + dt*A)\(u .+ dt*F(t))
        push!(tvec,t)
        push!(energy,h*dot(u,u))
    end
    return u,x,xint,h,tvec,energy
end

function solve_Crank_Nicolson(m, f, α, β, u0, T, dt)

    A,x,xint,h,F = build_FD(m,f,α,β)

    u = u0.(xint)
    Nsteps = ceil(Int,T/dt)
    dt = T/Nsteps
    tvec = Float64[]
    energy = Float64[]
    for i = 1:Nsteps
        tprev = (i-1)*dt
        t = i*dt
        u .= (I + .5*dt*A)\((I - .5*dt*A)*u .+ .5*dt.*(F(t)+F(tprev)))
        push!(tvec,t)
        push!(energy,h*dot(u,u))
    end
    return u,x,xint,h,tvec,energy
end

m = 100
T = 5.0
dt = 100/m
f(x,t) = 0.0
α(t) = 0.0
β(t) = 0.0
u0(x) = exp(-10*x^2)
u,x,xint,h,tvec,E1 = solve_backward_Euler(m, f, α, β, u0, T, dt)
u,x,xint,h,tvec,E2 = solve_Crank_Nicolson(m, f, α, β, u0, T, dt)
plot(tvec,E1,yaxis=:log,label="Backwards Euler")
plot!(tvec,E2,yaxis=:log,label="Crank Nicolson")
plot!(leg=:bottomleft)
