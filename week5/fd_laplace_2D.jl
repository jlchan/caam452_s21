using LinearAlgebra
using SparseArrays
using Plots

"This code solves Laplace's equation with Dirichlet boundary conditions using a
finite difference method."

m = 100 # number of points
x1D = LinRange(-1,1,m+2)
xint1D = x1D[2:end-1]
h = x1D[2]-x1D[1]

make_grid_2D(x1D) = make_grid_2D(x1D,x1D)
make_grid_2D(x1D,y1D) = [x for x in x1D, y in y1D], [y for x in x1D, y in y1D]
x,y = make_grid_2D(x1D)
xint,yint = make_grid_2D(xint1D)

# convert (i,j) to a global index
id(i,j) = i+(j-1)*m

function FD_matrix_2D(h,m)
    A = spzeros(m*m,m*m)

    for i = 1:m, j = 1:m
        # x-derivative
        A[id(i,j),id(i,j)] = 2/h^2
        if i < m
            A[id(i,j),id(i+1,j)] = -1/h^2
        end
        if i > 1
            A[id(i,j),id(i-1,j)] = -1/h^2
        end

        # # left Neumann condition
        # if i==1
        #     A[id(i,j),id(i,j)] = 2/h^2
        #     A[id(i,j),id(i+1,j)] = -2/h^2
        # end

        # y-derivative
        A[id(i,j),id(i,j)] += 2/h^2
        if j < m
            A[id(i,j),id(i,j+1)] = -1/h^2
        end
        if j > 1
            A[id(i,j),id(i,j-1)] = -1/h^2
        end
    end
    return A
end

# function FD_matrix_1D(x)
#     m = length(x)-2
#     h = x[2]-x[1]
#     A = (1/h^2) * spdiagm(0=>2*ones(m),-1=>-ones(m-1),1=>-ones(m-1))
# end
# A1D = FD_matrix_1D(x1D)
# Ax = kron(I(m),A1D)
# Ay = kron(A1D,I(m))
# A = Ax + .01*Ay
A = FD_matrix_2D(h,m)

# f(x,y) = .25 + sin(pi*x)*sin(pi*y)
f(x,y) = exp(-10*((x+.5)^2+(y+.5)^2))
α1(x,y) = 0*(x < 0.0)
α2(x,y) = 0*sin(pi*y)
b = f.(xint,yint)
for i = 1:m
    for j = 1:m
        if j==1 # bottom face
            b[id(i,j)] += α1(x[i,j],y[i,j])/h^2
        end
        if i==1 # left face
            b[id(i,j)] += α2(x[i,j],y[i,j])/h^2
        end
    end
end
b = vec(b)

# solve Au = F
uu = A\b
contourf(xint1D,xint1D,uu)
