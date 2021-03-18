using LinearAlgebra
using SparseArrays
using Plots
using ForwardDiff

"This routine solves Poisson's equation using piecewise linear finite element methods.
The matrix is assembled globally. "

m = 50 # number of points
α = 1.0
β = 2.0

# define spatial grid
x = LinRange(-1,1,m+2) # x_0, x_1, ..., x_m, x_{m+1} = x_0
xint = x[1:end-1]
h = x[2]-x[1]

# construct FEM matrix
A = spdiagm(0=>2*ones(m), 1=>-ones(m-1), -1=>-ones(m-1))
A = (1/h)*A

# assume f(x) = 1.0 for now
b = h*ones(m)
b[1] += α/h
b[m] += β/h
u = A\b

plot(x,[α;u;β])
