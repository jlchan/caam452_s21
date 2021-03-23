using LinearAlgebra
using SparseArrays
using Plots
using ForwardDiff

"This routine solves Poisson's equation -u''(x) = f(x) in weak form
(∫u'(x)*ϕi'(x) = ∫f(x)*ϕi(x) for i = 1,...,m) using piecewise linear finite element methods.
The matrix is assembled locally, and boundary nodes are not removed."

m = 50 # number of elements

# BCs
α = 1
β = 2

# define spatial grid
x = LinRange(-1,1,m+2) # x_0, x_1, ..., x_m, x_{m+1} = x_0
h = x[2]-x[1]

A = spzeros(m+2,m+2)
A_local = [1 -1;-1 1]
for e = 1:m+1 # loop over the intervals
    ids = e:e+1
    h_e = x[e+1]-x[e]
    A[ids,ids] .+= A_local * 1/h_e
end

# assume f(x) = 1
b = h*ones(m+2)

# impose Dirichlet BCs on both sides
b̃ = b - A[:,1]*α
b̃[1] = α
A[1,:] .= 0
A[:,1] .= 0
A[1,1] = 1

b̃ = b̃ - A[:,m+2]*β
b̃[m+2] = β
A[m+2,:] .= 0
A[:,m+2] .= 0
A[m+2,m+2] = 1

# convert b̃ from a SparseVector to a Vector
# (Julia doesn't currently allow "\" to be applied to SparseVectors)
u = A\Vector(b̃)

plot(x,u,legend=false)
