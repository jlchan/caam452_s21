using LinearAlgebra
using SparseArrays
using Plots
using ForwardDiff

# method of manufactured solutions: define the exact solution, then figure out
# what f,α,β should be to make it the solution to your PDE
uexact(x) = log(2+sin(pi*x))
dudx_exact(x) = ForwardDiff.derivative(uexact,x)
f(x) = -ForwardDiff.derivative(dudx_exact,x) # -d^2u/dx^2 = f

α = uexact(-1)
β = dudx_exact(1)

# The O(h) accurate method
function solve1(m,f,α,β)
    x = LinRange(-1,1,m+2) # add 2 endpoints
    xint = x[2:end-1]
    h = x[2]-x[1]
    A = (1/h^2) * spdiagm(0=>2*ones(m),-1=>-ones(m-1),1=>-ones(m-1))
    A[end,end] = 1/h^2
    b = f.(xint)
    b[1] += α/h^2

    b[end] += β/h # approach 1 - O(h) accuracy only

    u = A\b
    return u,x,xint,h
end

# The O(h^2) accurate method
function solve2(m,f,α,β)
    x = LinRange(-1,1,m+2) # add 2 endpoints
    xint = x[2:end] # leave x_{m+1} in the list of points!
    h = x[2]-x[1]
    A = (1/h^2) * spdiagm(0=>2*ones(m+1),-1=>-ones(m),1=>-ones(m))
    A[end,end] = 1/h^2
    b = f.(xint)
    b[1] += α/h^2

    b[end] = .5*f(x[end]) + β/h

    u = A\b
    return u,x,xint,h
end

m = 10
u1,x,xint1,h = solve1(m,f,α,β)
u2,x,xint2,h = solve2(m,f,α,β)
plot(xint1,u1,mark=:dot,markersize=2,label="FD solution 1")
plot!(xint2,u2,mark=:dot,markersize=2,label="FD solution 2")
plot!(x,uexact.(x),label="Exact solution")
plot!(leg=:bottomright)
