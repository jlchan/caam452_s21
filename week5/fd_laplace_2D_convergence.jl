using LinearAlgebra
using SparseArrays
using Plots
using ForwardDiff

"This code applies the method of manufactured solutions to the 2D Laplace's equation
with nonzero Dirichlet BCs using a finite difference method."

m = 100 # number of points

uexact(x,y) = log(2+sin(pi*x)*sin(pi*y))
dudx_exact(x,y) = ForwardDiff.derivative(x->uexact(x,y),x)
dudy_exact(x,y) = ForwardDiff.derivative(y->uexact(x,y),y)
du2dx2_exact(x,y) = ForwardDiff.derivative(x->dudx_exact(x,y),x)
du2dy2_exact(x,y) = ForwardDiff.derivative(y->dudy_exact(x,y),y)
f(x,y) = -(du2dx2_exact(x,y) + du2dy2_exact(x,y))

u_left(y)   = uexact(-1,y)
u_right(y)  = uexact(1,y)
u_bottom(x) = uexact(x,-1)
u_top(x)    = uexact(x,1)
uBC = (u_left,u_right,u_bottom,u_top)

function solve(m,f,uBC)
    u_left,u_right,u_bottom,u_top = uBC
    x = LinRange(-1,1,m+2)
    y = LinRange(-1,1,m+2)
    xint = x[2:end-1]
    yint = y[2:end-1]
    h = x[2]-x[1]

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

    b = vec(f.(xint,yint'))

    # impose non-zero Dirichlet BCs
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
    return u,h,xint,yint
end

errvec = []
hvec = []
for m in [4 8 16 32 64 128]
    u,h,xint,yint = solve(m,f,uBC)
    err = maximum(abs.(u .- vec(uexact.(xint,yint'))))
    append!(errvec,err)
    append!(hvec,h)
end
plot(hvec,errvec,mark=:dot,xaxis=:log,yaxis=:log)
plot!(hvec,hvec.^2,ls=:dash,xaxis=:log,yaxis=:log)
# contourf(xint,yint,))
