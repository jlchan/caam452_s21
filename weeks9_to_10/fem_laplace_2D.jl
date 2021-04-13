using TriangleMesh
using Plots
using Triplot
using LinearAlgebra
using SparseArrays
using ForwardDiff
include("./TriFEMUtils.jl") # plotting and misc routines

poly = polygon_unitSquare()
# poly = polygon_regular(5)
# poly = polygon_Lshape()
mesh = create_mesh(poly, quality_meshing=true, add_switches="a0.01q")
# mesh = refine_rg(mesh)
# mesh = refine_rg(mesh)
# mesh = refine_rg(mesh)

# some renaming
VX,VY = mesh.point[1,:],mesh.point[2,:]
# @. VX = 2*VX-1
# @. VY = 2*VY-1
EToV  = mesh.cell  # element-to-vertex mapping: each column contains vertex ids for a different element
point_on_boundary = mesh.point_marker # 1 if on boundary, 0 otherwise

plotTriMesh(mesh)
for i = 1:size(mesh.segment,2)
    eids = mesh.segment[:,i]
    scatter!(VX[eids],VY[eids])
end
display(plot!())

# m = 64
# VX,VY,EToV = uniform_tri_mesh(m,m)
# point_on_boundary = (@. abs(abs(VX)-1) < 10*eps()) .| (@. abs(abs(VY)-1) < 10*eps())
# iids = .!point_on_boundary
# VX[iids] .+= .25/m*randn(size(VX[iids]))
# VY[iids] .+= .25/m*randn(size(VX[iids]))
# is_Neumann_point = VX[point_on_boundary] .> 0

# f(x,y) = sin(pi*x)*sin(pi*y)
# uBC(x,y) = exp(sin(2*pi*x)*sin(1+pi*y))
# ∇u_n(x,y) = x+y

uexact(x,y) = exp(sin(1+pi*x)*sin(pi*y))
uBC(x,y) = uexact(x,y)
dudx(x,y) = ForwardDiff.derivative(x->uexact(x,y),x)
dudy(x,y) = ForwardDiff.derivative(y->uexact(x,y),y)
f(x,y) = -(ForwardDiff.derivative(x->dudx(x,y),x) +
           ForwardDiff.derivative(y->dudy(x,y),y)) # -du^2/dx^2 - du^2/dy^2


λ1(r,s) = -(r+s)/2
λ2(r,s) = (1+s)/2
λ3(r,s) = (1+r)/2
λ(r,s) = [λ1.(r,s) λ2.(r,s) λ3.(r,s)]
dλr() = [-.5 0.0 .5]
dλs() = [-.5 .5 0.0]

function map_triangle_pts(r,s,x,y)
    return λ(r,s)*x, λ(r,s)*y
end

# x,y = lists of vertices
function compute_geometric_terms(x,y)
    # dx(r,s)/dr = dλ1/dr*x1 + dλ2/dr*x2 + dλ3/dr*x3 = [dλ1/dr dλ2/dr dλ3/dr] * [x1; x2; x3]
    dxdr,dydr = dλr()*x, dλr()*y
    dxds,dyds = dλs()*x, dλs()*y
    G = [dxdr dxds; dydr dyds] # G*r = x, r = inv(G)*x
    J = det(G) # change of variables det(Jacobian) for integration
    drdx, dsdx, drdy, dsdy = inv(G)
    return J, drdx, dsdx, drdy, dsdy
end

# assemble stiffness matrix
function assemble_FE_matrix(VX,VY,EToV,point_on_boundary)
    num_vertices = length(VX)
    K = size(EToV,2) # number of columns

    A = spzeros(num_vertices, num_vertices)
    b = zeros(num_vertices)
    for e = 1:K # loop through all elements
        ids = EToV[:,e] # vertex ids = local to global index maps
        xv,yv = VX[ids],VY[ids]

        # compute geometric mappings
        J,drdx,dsdx,drdy,dsdy = compute_geometric_terms(xv,yv)
        dλdx = drdx*dλr() + dsdx*dλs() # dλr() = [dλ1/dr dλ2/dr dλ3/dr]
        dλdy = drdy*dλr() + dsdy*dλs() # dλs() = [dλ1/ds dλ2/ds dλ3/ds]
        reference_elem_area = 2.0
        @. A[ids,ids] += J*reference_elem_area*(dλdx'*dλdx + dλdy'*dλdy) # (dλdx'*dλdx)_ij = dλj/dx * dλi/dx

        # midpoint quadrature rule
        x_mid,y_mid = sum(xv)/3, sum(yv)/3 # midpoint rule = averages of vertex locations
        w_mid = 2.0
        b[ids] .+= J*w_mid*f(x_mid,y_mid)*vec(λ(-1/3,-1/3)) # λ at midpt = 1/3 constant

        # # incorporate Neumann BCs
        # boundary_points = findall(point_on_boundary[ids])
        # if length(boundary_points)==2 && all(is_Neumann_points[boundary_points]) # if 2 points lie on the boundary
        #     xb = VX[boundary_points]
        #     yb = VY[boundary_points]
        #     x_mid,y_mid = sum(xb)/2, sum(yb)/2
        #     face_length = sqrt((xb[1]-xb[2])^2  + (yb[1]-yb[2])^2)
        #     # ∇u_n(x_mid,y_mid) * λ()
        # end
    end

    # impose BCs
    for i = 1:num_vertices
        if point_on_boundary[i]==1 # 1 if on boundary
            xi,yi = VX[i],VY[i]
            b -= vec(A[:,i]*uBC(xi,yi))
            A[:,i] .= 0
            A[i,:] .= 0
            A[i,i] = 1.0
        end
    end
    for i = 1:num_vertices
        if point_on_boundary[i]==1 # 1 if on boundary
            xi,yi = VX[i],VY[i]
            b[i] = uBC(xi,yi)
        end
    end
    return A,Vector(b)
end

A,b = assemble_FE_matrix(VX,VY,EToV,point_on_boundary)
u = A\b
@show maximum(abs.(u .- uexact.(VX,VY)))
# triplot(VX,VY,u,EToV)
# triplot(VX,VY,uexact.(VX,VY),EToV)
