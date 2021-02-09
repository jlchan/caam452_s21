using LinearAlgebra
using SparseArrays
using Plots

"This code solves Laplace's equation with Dirichlet boundary conditions using a
finite difference method."

m = 100 # number of points
x = LinRange(-1,1,m+2)
xint = x[2:end-1]

f(x) = 5*(Float64((x > -.5) && (x <= 0.0)) - Float64((x < .5) && (x >= 0.0)))
α,β = 1.0,pi

h = x[2]-x[1]
A = (1/h^2) * spdiagm(0=>2*ones(m),-1=>-ones(m-1),1=>-ones(m-1))
b = f.(xint)
b[1] += α/h^2
b[m] += β/h^2

# solve Au = F
uu = A\b
plot(xint,uu,linewidth=2,label="Solution")
plot!(x,uexact.(x),linestyle=:dash)
# plot!(x,.1*f.(x),linecolor=:red,ls=:dash,label="f(x)")
# plot!(leg=:topleft)

# uu - uexact.(xint)
