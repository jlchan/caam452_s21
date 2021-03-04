using LinearAlgebra
using SparseArrays
using Plots

"This code solves the 2D Laplace's equation with nonzero Dirichlet and Neumann
boundary conditions using a finite difference method."

m = 100 # number of points
x = LinRange(-1,1,m+2)
y = LinRange(-1,1,m+2)
xint = x[2:end-1]
yint = y[2:end-1]
h = x[2]-x[1]

# ordering of unknowns
# u14 u24 u34 u44
# u13 u23 u33 u43
# u12 u22 u32 u42
# u11 u21 u31 u41

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

f(x,y) = exp(-10*((x-.25)^2 + (y-.75)^2))
f(x,y) = 0.0
b = vec(f.(xint,yint'))

# impose non-zero Dirichlet BCs
u_left(y) = 0.0
u_right(y) = 1.0 # 0 if y in [-1,0], 1 if y in [0,1]
u_bottom(x) = 0.0
u_top(x) = -1.0

for i = 1:m, j = 1:m
    if i==1
        b[id(i,j,m)] += u_left(yint[j])/h^2
    end
    if i==m
        b[id(i,j,m)] += u_right(yint[j])/h^2
    end
    if j==1
        b[id(i,j,m)] += u_bottom(xint[i])/h^2
    end
    if j==m
        b[id(i,j,m)] += u_top(xint[i])/h^2
    end
end

# solve Au = f(xi,yj)
u = A\b
contourf(xint,yint,u,clims=(-1,1)) #vec(reshape(u,m,m)'))
