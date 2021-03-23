using LinearAlgebra
using SparseArrays
using Plots
using ForwardDiff

uexact(x) = log(2+sin(pi*x))
dudx_exact(x) = ForwardDiff.derivative(uexact,x)
f(x) = -ForwardDiff.derivative(dudx_exact,x) # -d2u/d2x = f
α = uexact(-1)
β = uexact(1) - dudx_exact(1)

function solve(m,f,α,β)
    x = LinRange(-1,1,m+2) # add 2 endpoints
    xint = x[2:end]
    h = x[2]-x[1]
    A = (1/h^2) * spdiagm(0=>2*ones(m+1),-1=>-ones(m),1=>-ones(m))
    A[end,end] = (1-h)/h^2
    b = f.(xint)
    b[1] += α/h^2
    b[end] = .5*f(x[end]) - β/h
    u = A\b
    return u,x,xint,h
end

mvec = 2 .^ (2:9)
hvec = zeros(length(mvec))
err_inf = zeros(length(mvec))
for (i,m) in enumerate(mvec)
    u,x,xint,h = solve(m,f,α,β)
    hvec[i] = h
    err_inf[i] = maximum(@. abs(uexact(xint) - u))
end

plot(hvec,err_inf,marker=:dot,label="Max error")
plot!(hvec,hvec.^2,linestyle=:dash,label="O(h^2)")
plot!(xaxis=:log,yaxis=:log,legend=:bottomright)
