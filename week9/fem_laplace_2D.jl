using TriangleMesh
using Plots
using Triplot
using LinearAlgebra
using SparseArrays

include("./TriFEMUtils.jl") # local module
# using .TriFEMUtils

poly = polygon_unitSquare()
# poly = polygon_regular(5)
# poly = polygon_Lshape()
mesh = create_mesh(poly, set_area_max=true, quality_meshing=true)

# some renaming
VX,VY = mesh.point[1,:],mesh.point[2,:]
EToV  = mesh.cell # element-to-vertex mapping
K = mesh.n_cell
point_on_boundary = mesh.point_marker

# f(x,y) = sin(2*pi*x)*sin(2*pi*y)
f(x,y) = 1.0
uexact(x,y) = nothing
u_BC(x,y) = sin(pi*x)

λ1(r,s) = -(r+s)/2
λ2(r,s) = (1+s)/2
λ3(r,s) = (1+r)/2
V(r,s) = [λ1.(r,s) λ2.(r,s) λ3.(r,s)]
dVr() = [-.5 .5 0.0]
dVs() = [-.5 0.0 .5]
function map_triangle_pts(r,s,x,y)
    return V(r,s)*x,V(r,s)*y
end

function compute_geometric_terms(x,y)
    dxdr,dydr = dVr()*x, dVr()*y
    dxds,dyds = dVs()*x, dVs()*y
    G = [dxdr dxds; dydr dyds] # G*r = x, r = inv(G)*x
    J = det(G)
    drdx, dsdx, drdy, dsdy = inv(G)
    return J, drdx, dsdx, drdy, dsdy
end

# assemble stiffness matrix
function assemble_FE_matrix(VX,VY,EToV)
    num_vertices = length(VX)
    A = spzeros(num_vertices,num_vertices)
    b = zeros(num_vertices)
    K = size(EToV,2) # number of columns

    for e = 1:K
        ids = EToV[:,e]
        xv,yv = VX[ids],VY[ids]

        # quadrature rule
        x_mid,y_mid = map_triangle_pts(-1/3, -1/3, xv, yv)
        w_mid = 2.0

        # compute geometric mappings
        J,rx,sx,ry,sy = compute_geometric_terms(xv,yv)
        Vx = rx*dVr() + sx*dVs()
        Vy = ry*dVr() + sy*dVs()

        @. A[ids,ids] += J*w_mid*(Vx'*Vx + Vy'*Vy)
        @. b[ids] += J*w_mid*f.(x_mid,y_mid)
    end

    # impose BCs
    for i = 1:num_vertices
        if point_on_boundary[i]==1 # 1 if on boundary
            x,y = VX[i],VY[i]

            b -= A[:,i]*u_BC(x,y)
            # b[i] = u_BC(x,y)
            A[:,i] .= 0
            A[i,:] .= 0
            A[i,i] = 1.0
        end
    end
    for i = 1:num_vertices
        if point_on_boundary[i]==1
            x,y = VX[i],VY[i]
            b[i]= u_BC(x,y)
        end
    end
    return A,Vector(b)
end

A,b = assemble_FE_matrix(VX,VY,EToV)
u = A\b

# @show maximum(abs.(u .- uexact.(VX,VY)))
triplot(VX,VY,u,EToV)
