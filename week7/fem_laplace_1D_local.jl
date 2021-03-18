using LinearAlgebra
using SparseArrays
using Plots
using ForwardDiff

"This routine solves Poisson's equation using piecewise linear finite element methods.
The matrix is assembled locally, and boundary nodes are not removed."

m = 100 # number of elements

α = 0.0
β = 0.0
f(x) = sin(pi*x)
κ(x) = 1 + (x > 0)

# define spatial grid
x = LinRange(-1,1,m+2) # x_0, x_1, ..., x_m, x_{m+1} = x_0
xint = x[1:end-1]
h = x[2]-x[1]

# x = x + randn(size(x))/(2*m)
# x = @. x + (1+x)*(1-x)/3

# construct local FEM matrix
map_point(x,a,b) = a + (b-a) * x # maps x ∈ [0,1] to interval [a,b]
A = spzeros(m+2,m+2)
b = zeros(m+2)
A_local = [1 -1;-1 1]
for e = 1:m+1
    h_e = x[e+1] - x[e]
    ids = e:e+1

    # use 1-point quadrature to approximate integrals
    x_e = map_point(0, x[e], x[e+1])
    w_e = 2.0

    # accumulate local contributions
    A[ids,ids] += A_local * κ(x_e) / h_e
    b[ids] += h_e * w_e * f(x_e) * [.5;.5]
end

# modify for boundary conditions
function impose_Dirichlet_BC!(A,b,i,val)
    b .-= A[:,i]*val
    b[i] = val
    A[i,:] .= 0
    A[:,i] .= 0
    A[i,i] = 1.0
end
impose_Dirichlet_BC!(A,b,1,α)
# impose_Dirichlet_BC!(A,b,m+2,β)

u = A\b
plot(x,u,legend=false,mark=:dot,ms=2)
