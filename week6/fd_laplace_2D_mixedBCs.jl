using LinearAlgebra
using SparseArrays
using Plots

"This code solves the 2D Laplace's equation with nonzero Dirichlet and Neumann
boundary conditions using a finite difference method."

m = 100 # number of points
x = LinRange(-1,1,m+2)
y = LinRange(-1,1,m+2)
xint = x[1:end]
yint = y[2:end-1]
h = x[2]-x[1]

id(i,j,m) = i + (j-1)*(m+2) # return global index from i,j

function FD_matrix_2D(h,m)
    A = spzeros((m+2)*m,(m+2)*m)
    for i = 1:m+2, j = 1:m # loop thru x,y indices
        A[id(i,j,m),id(i,j,m)] = 4/h^2

        # stencil at interior nodes
        if i > 1 # avoids leftmost line of nodes
            A[id(i,j,m),id(i-1,j,m)] = -1/h^2 # x-derivative
        end
        if i < m+2 # avoids rightmost line of nodes
            A[id(i,j,m),id(i+1,j,m)] = -1/h^2
        end
        if j > 1
            A[id(i,j,m),id(i,j-1,m)] = -1/h^2 # y-derivative
        end
        if j < m
            A[id(i,j,m),id(i,j+1,m)] = -1/h^2
        end

        # modify for Neumann BCs on left/right
        # 2*(u_{m+1,j} - u_{m,j})/h^2 at Neumann boundaries in the x coordinate
        if i==1 # on left
            A[id(i,j,m),id(i+1,j,m)] = -2/h^2
        end
        if i==m+2 # on right
            A[id(i,j,m),id(i-1,j,m)] = -2/h^2
        end
    end
    return A
end

A = FD_matrix_2D(h,m)

# f(x,y) = exp(-10*((x-.25)^2 + (y-.75)^2))
f(x,y) = 0.0
b = vec(f.(xint,yint'))

# impose non-zero Dirichlet BCs
u_top(x) = 1.0
u_bottom(x) = -1.0
dudx_left(y) = y^2
dudx_right(y) = 20*y

for i = 1:m+2, j = 1:m
    # modify Neumann BCs first, then add Dirichlet contributions next
    if i==1 # left
        b[id(i,j,m)] += 2*dudx_left(yint[j])/h
    end
    if i==m+2  # right
        b[id(i,j,m)] += 2*dudx_right(yint[j])/h
    end

    if j==1 # bottom
        uij = u_bottom(xint[i])
        b[id(i,j,m)] += uij/h^2
    end
    if j==m # top
        uij = u_top(xint[i])
        b[id(i,j,m)] += uij/h^2
    end
end

# solve Au = f(xi,yj)
u = A\vec(b)
contourf(xint,yint,u) #vec(reshape(u,m,m)'))
