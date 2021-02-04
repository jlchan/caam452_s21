using LinearAlgebra
using SparseArrays
using Plots
using ForwardDiff

# variable diffusivity
κ(x) = 1 + .5*sin(1+pi*x)

# method of manufactured solutions: define the exact solution, then figure out
# what f,α,β should be to make it the solution to your PDE
uexact(x) = log(2+sin(pi*x))
dudx_exact(x) = ForwardDiff.derivative(uexact,x)
f(x) = -ForwardDiff.derivative(x->κ(x)*dudx_exact(x),x) # -d^2u/dx^2 = f
α = uexact(-1)
β = uexact(1)

# The O(h^2) accurate method
function solve(m,κ,f,α,β)
    x = LinRange(-1,1,m+2) # add 2 endpoints
    xint = x[2:end-1] # leave x_{m+1} in the list of points!
    h = x[2]-x[1]

    xmid = @. x[1:end-1] + h/2
    k = κ.(xmid)

    A = (1/h^2) * spdiagm(0=>k[1:end-1]+k[2:end],-1=>-k[2:end-1],1=>-k[2:end-1])
    b = f.(xint)
    b[1] += k[1]*α/h^2
    b[end] += k[end]*β/h^2

    u = A\b
    return u,x,xint,h
end

m = 20
u,x,xint,h = solve(m,κ,f,α,β)
plot(xint,u,mark=:dot,markersize=2,label="FD solution 1")
plot!(x,uexact.(x),label="Exact solution")
plot!(leg=:bottomright)

mvec = 2 .^ (2:9)
hvec = zeros(length(mvec))
err = zeros(length(mvec))
for (i,m) in enumerate(mvec)
    u,x,xint,h = solve(m,κ,f,α,β)
    hvec[i] = h
    err[i] = maximum(@. abs(uexact(xint) - u))
end
plot(hvec,err,marker=:dot,label="Max error")
plot!(hvec,hvec.^2,linestyle=:dash,label="O(h^2)")
plot!(xaxis=:log,yaxis=:log,legend=:bottomright)
