using LinearAlgebra
using SparseArrays
using Plots
using ForwardDiff

"This code performs a convergence test for two finite difference methods applied to
Laplace's equation with variable diffusivity κ(x)."

# variable diffusivity
κ(x) = 1 + .5*sin(pi*x) # assumed κ(x) >= κ_min > 0
# κ(x) = Float64(1 + (x>0))

# method of manufactured solutions: define the exact solution, then figure out
# what f,α,β should be to make it the solution to your PDE
uexact(x) = log(2+sin(pi*x))
dudx_exact(x) = ForwardDiff.derivative(uexact,x)
f(x) = -ForwardDiff.derivative(x->κ(x)*dudx_exact(x),x) # -d/dx (κ(x)*du/dx) = f
α = uexact(-1)
β = uexact(1)

function solve(m,κ,f,α,β)
    x = LinRange(-1,1,m+2) # points x_0, x_1, ... x_m, x_m+1
    xint = x[2:end-1]
    h = x[2]-x[1]

    xmid = @. x[1:end-1] + h/2 # x_0 + h/2, x_1 + h/2, ..., x_m + h/2
    # (re-indexing using fractional indices) x_1/2, x_3/2, ..., x_i+1/2, ..., x_m+1/2
    k = κ.(xmid)

    #                        κ_{i-1/2}+κ_{i+1/2}      κ_{i+1/2}      κ_{i+1/2}
    A = (1/h^2) * spdiagm(0=>k[1:end-1]+k[2:end],-1=>-k[2:end-1],1=>-k[2:end-1])
    b = f.(xint)
    b[1] += k[1]*α/h^2
    b[end] += k[end]*β/h^2

    u = A\b
    return u,x,xint,h
end

m = 40
u,x,xint,h = solve(m,κ,f,α,β)
plot(xint,u,mark=:dot,markersize=2,label="FD solution")
plot!(x,uexact.(x),label="Exact solution")
plot!(leg=:bottomright)

# mvec = 2 .^ (2:9)
# hvec = zeros(length(mvec))
# err = zeros(length(mvec))
# for (i,m) in enumerate(mvec)
#     u,x,xint,h = solve(m,κ,f,α,β)
#     hvec[i] = h
#     err[i] = maximum(@. abs(uexact(xint) - u))
# end
# plot(hvec,err,marker=:dot,label="Max error")
# plot!(hvec,hvec.^2,linestyle=:dash,label="O(h^2)")
# plot!(xaxis=:log,yaxis=:log,legend=:bottomright)
