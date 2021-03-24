using LinearAlgebra
using SparseArrays
using Plots
using ForwardDiff

"This routine solves Poisson's equation -u''(x) = f(x) in weak form
(∫κ(x)*u'(x)*ϕi'(x) = ∫f(x)*ϕi(x) for i = 1,...,m)
using piecewise linear finite element methods. The matrix is assembled locally,
and boundary nodes are not removed."

m = 50 # number of elements

# α = 0 # Dirichlet BC
# β = 1 # Neumann BC
# f(x) = sin(pi*x) #exp(2*sin(pi*x))
# κ(x) = 1 + (x > 0)*2

# Manufactured solution
uexact(x) = log(2+sin(pi*x))
κ(x) = 1 + .5*sin(pi*x)
dudx_exact(x) = ForwardDiff.derivative(uexact,x)
f(x) = -ForwardDiff.derivative(x->κ(x)*dudx_exact(x),x) # -d/dx (κ(x)*du/dx) = f
α = uexact(-1)
β = κ(1)*dudx_exact(1)

# define spatial grid
x = LinRange(-1,1,m+2) # x_0, x_1, ..., x_m, x_{m+1} = x_0
# x = @. x + randn() / (2*m) # randomly perturb nodes

A = spzeros(m+2,m+2)
A_local = [1 -1;-1 1]
b = zeros(m+2)
for e = 1:m+1 # loop over the intervals
    ids = e:e+1
    h_e = x[e+1]-x[e]

    x_midpt = .5*(x[e+1]+x[e])
    A[ids,ids] .+= κ(x_midpt)*A_local * 1/h_e

    b[ids] .+= h_e / 2.0 * f(.5*(x[e]+x[e+1])) # f(.5*(x_i+x_i+1))
end

# impose Dirichlet BCs on both sides
b̃ = b - A[:,1]*α
b̃[1] = α
A[1,:] .= 0
A[:,1] .= 0
A[1,1] = 1

# b̃ = b̃ - A[:,m+2]*β
# b̃[m+2] = β
# A[m+2,:] .= 0
# A[:,m+2] .= 0
# A[m+2,m+2] = 1

b̃[m+2] += β # impose Neumann BC on the left

# convert b̃ from a SparseVector to a Vector
# (Julia doesn't currently allow "\" to be applied to SparseVectors)
u = A\Vector(b̃)

err = abs.(u - uexact.(x))
plot(x,u,mark=:dot,legend=false,title="Max error = $(maximum(err))")
