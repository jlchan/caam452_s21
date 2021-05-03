using TriangleMesh
using Plots
using Triplot
using LinearAlgebra
using SparseArrays
using ForwardDiff

include("./TriFEMUtils.jl") # plotting and misc routines

K1D = 16 # num elements in each direction
mesh = uniform_tri_mesh(K1D,K1D)
boundary_indices = get_boundary_info(mesh)

# manufactured solution
uexact(x,y) = exp(sin(1+pi*x)*sin(pi*y))
uBC(x,y) = uexact(x,y)
dudx(x,y) = ForwardDiff.derivative(x->uexact(x,y),x)
dudy(x,y) = ForwardDiff.derivative(y->uexact(x,y),y)
f(x,y) = -(ForwardDiff.derivative(x->dudx(x,y),x) +
           ForwardDiff.derivative(y->dudy(x,y),y)) # -(du^2/dx^2 + du^2/dy^2)

# ordering of reference element vertices
# 3
# |`.
# 1--2
reference_vertices = [[-1,1,-1],[-1,-1,1]] # stored as [r,s]
reference_face_indices = [[1,2],[2,3],[1,3]] # ordering of faces for the reference element
reference_elem_info = (;reference_face_indices,reference_vertices)

# define reference basis functions
λ1(r,s) = -(r+s)/2
λ3(r,s) = (1+r)/2
λ2(r,s) = (1+s)/2
λ(r,s) = [λ1.(r,s) λ2.(r,s) λ3.(r,s)]
dλr() = [-.5 0.0 .5]
dλs() = [-.5 .5 0.0]

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

# assemble stiffness matrix and RHS vector
function assemble_FE_matrix(mesh)
    VX,VY,EToV = unpack_mesh_info(mesh)
    num_vertices = length(VX)
    num_elements = size(EToV,2) # number of elements = of columns

    rq,sq,wq = -1/3,-1/3,2.0

    A = spzeros(num_vertices, num_vertices)
    b = zeros(num_vertices)
    for e = 1:num_elements # loop through all elements
        ids = EToV[:,e] # vertex ids = local to global index maps
        xv,yv = VX[ids],VY[ids]

        # compute geometric mappings
        J,drdx,dsdx,drdy,dsdy = compute_geometric_terms(xv,yv)
        dλdx = drdx*dλr() + dsdx*dλs() # dλr() = [dλ1/dr dλ2/dr dλ3/dr]
        dλdy = drdy*dλr() + dsdy*dλs() # dλs() = [dλ1/ds dλ2/ds dλ3/ds]
        reference_elem_area = sum(wq)
        @. A[ids,ids] += J*reference_elem_area*(dλdx'*dλdx + dλdy'*dλdy) # (dλdx'*dλdx)_ij = dλj/dx * dλi/dx

        # quadrature rule
        xq,yq = map_triangle_pts(rq,sq,xv,yv)
        b[ids] .+= J*λ(rq,sq)'*(wq.*f.(xq,yq))
    end
    return A,b
end

function impose_Dirichlet_BCs!(A,b,mesh,boundary_indices)

    VX,VY,EToV = unpack_mesh_info(mesh)

    # impose Dirichlet BCs
    for i in boundary_indices
        xi,yi = VX[i],VY[i]
        b -= Vector(A[:,i]*uBC(xi,yi))
        A[:,i] .= 0
        A[i,:] .= 0
        A[i,i] = 1.0
    end
    for i in boundary_indices
        xi,yi = VX[i],VY[i]
        b[i] = uBC(xi,yi)
    end
    return A,b
end

A,b = assemble_FE_matrix(mesh)
A,b = impose_Dirichlet_BCs!(A,b,mesh,boundary_indices)
u = A\b
VX,VY,EToV = unpack_mesh_info(mesh)
@show maximum(abs.(u .- uexact.(VX,VY)))
triplot(VX,VY,u .- uexact.(VX,VY),EToV)
# triplot(VX,VY,u,EToV)
# triplot(VX,VY,uexact.(VX,VY),EToV)


function compute_error()
    VX,VY,EToV = unpack_mesh_info(mesh)
    num_elements = size(EToV,2) # number of elements = of columns

    rq = [-2/3; 1/3; -2/3]
    sq = [-2/3; -2/3; 1/3]
    wq = 2/3 * ones(3)

    for e = 1:num_elements # loop through all elements
        ids = EToV[:,e] # vertex ids = local to global index maps
        xv,yv = VX[ids],VY[ids]

        # compute geometric mappings
        J,drdx,dsdx,drdy,dsdy = compute_geometric_terms(xv,yv)

        # quadrature rule
        xq,yq = map_triangle_pts(rq,sq,xv,yv)

        #∫f(x,y) ≈ ∑f(xi,yi)*w[i]
    end
end
