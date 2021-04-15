using LinearAlgebra
using SparseArrays
using Plots
using ForwardDiff
using FastGaussQuadrature # more accurate quadrature

"This routine solves Poisson's equation -u''(x) = f(x) in weak form using piecewise
linear finite element methods. It's used to illustrate convergence in different norms."

m = 25 # number of elements

# Manufactured solution
uexact(x) = log(2+sin(4*pi*x))
dudx_exact(x) = ForwardDiff.derivative(uexact,x)
f(x) = -ForwardDiff.derivative(dudx_exact,x) # -d/dx (κ(x)*du/dx) = f
α = uexact(-1)
β = uexact(1)

# define spatial grid
x = LinRange(-1,1,m+2) # x_0, x_1, ..., x_m, x_{m+1} = x_0
x = @. x + randn() / (2*m) # randomly perturb nodes
x[1] = -1; x[end] = 1 # reset endpoints

# define local FEM basis
rq,wq = gausslegendre(100) # define overkill-accurate Gauss quadrature
λ(r) = [(1 .-r)./2 (1 .+r)./2]
dλ(r) = [-.5*ones(length(r)) .5*ones(length(r))]
map_point(x,a,b) = a + (b-a) * (1+x)/2 # maps x ∈ [-1,1] to interval [a,b]

"∫u' * ϕ_i' = ∫f(x)*ϕ_i(x)"
A = spzeros(m+2,m+2)
A_local = [1 -1;-1 1]
b = zeros(m+2)
for e = 1:m+1 # loop over the intervals
    ids = e:e+1
    h_e = x[e+1]-x[e]

    x_midpt = .5*(x[e+1]+x[e])
    A[ids,ids] .+= A_local * 1/h_e

    # # one point midpoint quadrature
    # b[ids] .+= h_e / 2.0 * f(.5*(x[e]+x[e+1])) # f(.5*(x_i+x_i+1))

    # higher accuracy Gauss quadrature
    xq = map_point.(rq,x[e],x[e+1])
    b[ids] .+= h_e / 2.0 * λ(rq)'*(wq.*f.(xq)) # ∑_k f(x_k) * w_k * ϕ_i(x_k)
end

# impose Dirichlet BCs on both sides
b̃ = b - vec(A[:,1]*α)
b̃[1] = α
A[1,:] .= 0
A[:,1] .= 0
A[1,1] = 1

b̃ = b̃ - vec(A[:,m+2]*β)
b̃[m+2] = β
A[m+2,:] .= 0
A[:,m+2] .= 0
A[m+2,m+2] = 1

# convert b̃ from a SparseVector to a Vector
# (Julia doesn't currently allow "\" to be applied to SparseVectors)
u = A\Vector(b̃)

nodal_err = abs.(u - uexact.(x))
plot(x,u,mark=:dot,ms=3,legend=false,title="Max error at nodes = $(maximum(nodal_err))")
xfine = LinRange(-1,1,1000)
plot!(xfine,uexact.(xfine),legend=false)
