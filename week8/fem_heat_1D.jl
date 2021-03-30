using LinearAlgebra
using SparseArrays
using Plots
using ForwardDiff

"This routine solves Poisson's equation using piecewise linear finite element methods.
The matrix is assembled locally, and boundary nodes are not removed."

m = 50 # number of elements
dt = .001
T = .2 # final time

α(t) = 1.0 * cos(10*pi*t)
β(t) = 2.0 * sin(10*pi*t)
f(x) = 0
κ(x) = 1.0 + 2.0*(x > 0)
u0(x) = exp(-25*x^2)

# define spatial grid
x = LinRange(-1,1,m+2) # x_0, x_1, ..., x_m, x_{m+1} = x_0

# construct local FEM matrix
A = spzeros(m+2,m+2)
M = spzeros(m+2,m+2)
b = zeros(m+2)
A_local = [1 -1;-1 1]
M_local = [2 1; 1 2] ./ 3
for e = 1:m+1 # loop over the intervals
    ids = e:e+1
    h_e = x[e+1]-x[e]
    x_midpt = .5*(x[e+1]+x[e])

    A[ids,ids] .+= κ(x_midpt) * A_local / h_e
    b[ids] .+= .5 * h_e * f(x_midpt) # f(.5*(x_i+x_i+1))

    M[ids,ids] .+= M_local * .5* h_e
end

# modify for boundary conditions
function impose_Dirichlet_BC!(A,b,i,val)
    b .-= A[:,i]*val
    b[i] = val
    A[i,:] .= 0
    A[:,i] .= 0
    A[i,i] = 1.0
end
function impose_Neumann_BC!(A,b,i,val)
    b[i] += val
end

Nsteps = ceil(Int,T/dt)
dt = T/Nsteps
u = u0.(x)
interval = 1
@gif for i = 1:Nsteps
    B = (M + .5*dt*A) # Crank-Nicolson matrix
    b = M*u - .5*dt*A*u # Crank-Nicolson RHS for f(x,t) = 0
    tk = (i - 1/2)*dt # in between tk, tk+1
    impose_Dirichlet_BC!(B,b,1,α(tk))
    impose_Dirichlet_BC!(B,b,m+2,β(tk))
    u .= B\b
    if i%interval==0
        println("on step $i out of $Nsteps")
        plot(x,u,legend=false,ylims=(-2,3))
    end
end every interval
