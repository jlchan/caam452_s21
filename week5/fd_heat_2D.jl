using LinearAlgebra
using SparseArrays
using Plots

"This code solves the 2D Laplace's equation with nonzero Dirichlet and Neumann
boundary conditions using a finite difference method."

m = 50 # number of points
x = LinRange(-1,1,m+2)
y = LinRange(-1,1,m+2)
xint = x[2:end-1]
yint = y[2:end-1]
h = x[2]-x[1]

T = 2.0 # final time
dt = .01 # timestep

id(i,j,m) = i + (j-1)*m # return global index from i,j
function FD_matrix_2D(h,m)
    A = spzeros(m*m,m*m)
    for i = 1:m, j = 1:m # loop thru x,y indices
        A[id(i,j,m),id(i,j,m)] = 4/h^2
        if i > 1 # avoids leftmost line of nodes
            A[id(i,j,m),id(i-1,j,m)] = -1/h^2 # x-derivative
        end
        if i < m # avoids rightmost line of nodes
            A[id(i,j,m),id(i+1,j,m)] = -1/h^2
        end
        if j > 1
            A[id(i,j,m),id(i,j-1,m)] = -1/h^2 # y-derivative
        end
        if j < m
            A[id(i,j,m),id(i,j+1,m)] = -1/h^2
        end
    end
    return A
end

A = FD_matrix_2D(h,m)

# initial condition
u0(x,y) = exp(-100*(x^2+y^2))
u0(x,y) = 0.0

# forcing
f(x,y,t) = exp(-25*((x-.25)^2 + (y-.75)^2))*exp(2*t)*(t < 1.5)
# f(x,y,t) = 0.0

F(t) = vec(f.(xint,yint',t)) # zero Dirichlet BCs
u = vec(u0.(xint,yint'))
interval = 1
Nsteps = ceil(Int,T / dt)
dt = T/Nsteps

Bsparse = (I + .5*dt*A)
#B = cholesky(I + .5*dt*A)
B = lu(I + .5*dt*A)

plot()
@gif for i = 1:Nsteps
    t = i*dt
    Favg = .5*(F(t) + F(t+dt))
    u .= B\(u + .5*dt*(Favg-A*u))
    if i%interval==0
        println("on timestep $i / $Nsteps")
        contourf(xint,yint,u,clims=(0,.5)) #vec(reshape(u,m,m)'))
    end
end every interval
