using LinearAlgebra
using SparseArrays
using Plots
using ForwardDiff

uexact(x) = log(2+sin(pi*x))
dudx_exact(x) = ForwardDiff.derivative(uexact,x)
f(x) = dudx_exact(x)-ForwardDiff.derivative(dudx_exact,x) # du/dx - d^2u/dx^2 = f(x)
α = uexact(-1)
β = uexact(1)



function solve(m,f,α,β)
    x = LinRange(-1,1,m+2) # add 2 endpoints
    xint = x[2:end-1]
    h = x[2]-x[1]
    A = (1/h^2) * spdiagm(0=>2*ones(m),-1=>-ones(m-1),1=>-ones(m-1))
    Q = (1/(2*h)) * spdiagm(-1=>-ones(m-1),1=>ones(m-1))
    b = f.(xint)
    b[1] += α/h^2 + α/(2*h)
    b[end] += β/h^2 - β/(2*h)
    u = (Q+A)\b
    return u,x,xint,h
end

function solve(m,ϵ,f,α,β)
    x = LinRange(-1,1,m+2) # add 2 endpoints
    xint = x[2:end-1]
    h = x[2]-x[1]
    A = (1/h^2) * spdiagm(0=>2*ones(m),-1=>-ones(m-1),1=>-ones(m-1))
    Q = (1/(2*h)) * spdiagm(-1=>-ones(m-1),1=>ones(m-1))
    # Q = (1/(h)) * spdiagm(0=>-ones(m-1),1=>ones(m-1))
    b = f.(xint)
    b[1] += ϵ*α/h^2 + α/(2*h)
    b[end] += ϵ*β/h^2 - β/(2*h)
    u = (Q+ϵ*A)\b
    return u,x,xint,h
end

ϵ = .0001
u,x,xint,h = solve(100,ϵ,x->1,0,0)
plot(xint,u,leg=false)

# mvec = 2 .^ (2:9)
# hvec = zeros(length(mvec))
# err_inf = zeros(length(mvec))
# for (i,m) in enumerate(mvec)
#
#     ϵ = 1.0
#     u,x,xint,h = solve(m,ϵ,f,α,β)
#     hvec[i] = h
#     err_inf[i] = maximum(@. abs(uexact(xint) - u))
# end
#
# plot(hvec,err_inf,marker=:dot,label="Max error")
# plot!(hvec,hvec,linestyle=:dash,label="O(h)")
# plot!(hvec,hvec.^2,linestyle=:dash,label="O(h^2)")
# plot!(xaxis=:log,yaxis=:log,legend=:bottomright)
