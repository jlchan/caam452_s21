using TriangleMesh
using Plots
using Triplot
using LinearAlgebra
using SparseArrays
using ForwardDiff
using UnPack
include("./TriFEMUtils.jl") # plotting and misc routines

poly = polygon_unitSquare()
# poly = polygon_regular(5)
# poly = polygon_Lshape()
num_elements_1D = 64
max_triangle_area = 1/num_elements_1D^2
mesh = create_mesh(poly, quality_meshing=true,
                   add_switches="penva"*string(max_triangle_area)*"q") # switches

err = [0.49722443836812696, 0.184642370315069, 0.04632964124545684, 0.010186295536148499,0.0029234233106564217, 0.0007100700457359929]


# manufactured solution
uexact(x,y) = exp(sin(1+pi*x)*sin(pi*y))
uBC(x,y) = uexact(x,y)
dudx(x,y) = ForwardDiff.derivative(x->uexact(x,y),x)
dudy(x,y) = ForwardDiff.derivative(y->uexact(x,y),y)
f(x,y) = -(ForwardDiff.derivative(x->dudx(x,y),x) +
           ForwardDiff.derivative(y->dudy(x,y),y)) # -(du^2/dx^2 + du^2/dy^2)

# define Dirichlet and Neumann boundaries
on_Neumann_boundary(x,y) = x ≈ 1 && y > 0 && y < 1 # right face x = 1 (excluding corners)
on_Dirichlet_boundary(x,y) = !on_Neumann_boundary(x,y) # Dirichlet boundary = everywhere else
∇u_dot_n(x,y) = dudx(x,y) # note that n = [1,0] on the Neumann face

# ordering of reference element vertices
# 3
# |`.
# 1--2
reference_vertices = [[-1,1,-1],[-1,-1,1]] # stored as [r,s]
reference_face_indices = [[1,2],[2,3],[1,3]] # ordering of faces for the reference element
reference_elem_info = (;reference_face_indices,reference_vertices)

# define reference basis functions
λ1(r,s) = -(r+s)/2
λ2(r,s) = (1+r)/2
λ3(r,s) = (1+s)/2
λ(r,s) = [λ1.(r,s) λ2.(r,s) λ3.(r,s)]
dλr() = [-.5 .5 0.0]
dλs() = [-.5 0.0 .5]

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

    A = spzeros(num_vertices, num_vertices)
    b = zeros(num_vertices)
    for e = 1:num_elements # loop through all elements
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
        b[ids] .+= J*vec(λ(-1/3,-1/3))*(w_mid.*f(x_mid,y_mid)) # λ at midpt = 1/3 constant
    end
    return A,b
end

function impose_Neumann_BCs!(b,mesh,reference_elem_info)

    @unpack reference_face_indices, reference_vertices = reference_elem_info
    VX,VY,EToV = unpack_mesh_info(mesh)
    boundary_indices, boundary_faces = get_boundary_info(reference_face_indices, mesh)
    # contributions from Neumann BCs
    for (f,e) in boundary_faces
        ref_fids = reference_face_indices[f]
        fids = EToV[ref_fids,e]
        xf,yf = VX[fids],VY[fids]
        x_mid,y_mid = sum(xf)/2,sum(yf)/2

        if on_Neumann_boundary(x_mid,y_mid)
            r,s = reference_vertices
            r_mid, s_mid = sum(r[ref_fids])/2, sum(s[ref_fids])/2
            w_mid = 2.0

            ids = EToV[:,e]
            face_length = sqrt((xf[1]-xf[2])^2 + (yf[1]-yf[2])^2)
            b[ids] .+= face_length/2 * vec(λ(r_mid,s_mid)) * w_mid * ∇u_dot_n(x_mid,y_mid) # ∫ ∇u⋅n ϕ_i on each Neumann face
        end
    end
    return b
end
function impose_Dirichlet_BCs!(A,b,mesh,reference_elem_info)

    @unpack reference_face_indices, reference_vertices = reference_elem_info
    VX,VY,EToV = unpack_mesh_info(mesh)
    boundary_indices, boundary_faces = get_boundary_info(reference_face_indices, mesh)

    # impose Dirichlet BCs
    for i in boundary_indices
        xi,yi = VX[i],VY[i]
        if on_Dirichlet_boundary(xi,yi)
            b -= Vector(A[:,i]*uBC(xi,yi))
            A[:,i] .= 0
            A[i,:] .= 0
            A[i,i] = 1.0
        end
    end
    for i in boundary_indices
        xi,yi = VX[i],VY[i]
        if on_Dirichlet_boundary(xi,yi)
            b[i] = uBC(xi,yi)
        end
    end
    return A,b
end

A,b = assemble_FE_matrix(mesh)
b   = impose_Neumann_BCs!(b,mesh,reference_elem_info)
A,b = impose_Dirichlet_BCs!(A,b,mesh,reference_elem_info)
u = A\b
VX,VY,EToV = unpack_mesh_info(mesh)
# @show maximum(abs.(u .- uexact.(VX,VY)))
# triplot(VX,VY,u .- uexact.(VX,VY),EToV)
# triplot(VX,VY,uexact.(VX,VY),EToV)

function compute_errs(u,mesh)
    L2err2 = 0.0
    H1err2 = 0.0
    rq,sq,wq = 1/3,1/3,2.0
    rq,sq,wq = [-2/3; 1/3; -2/3], [-2/3; -2/3; 1/3], 2/3 * ones(3)
    for e = 1:mesh.n_cell
        ids = EToV[:,e] # vertex ids = local to global index maps
        xv,yv = VX[ids],VY[ids]

        J,drdx,dsdx,drdy,dsdy = compute_geometric_terms(xv,yv)

        # quadrature rule
        xq,yq = map_triangle_pts(rq,sq,xv,yv)
        err = λ(rq,sq)*u[ids] - uexact.(xq,yq)
        dudr = dλr()*u[ids]
        duds = dλs()*u[ids]
        deriv_err_x = drdx*dudr + dsdx*duds .- dudx.(xq,yq)
        deriv_err_y = drdy*dudr + dsdy*duds .- dudy.(xq,yq)
        deriv_err2 = @. deriv_err_x^2 + deriv_err_y^2

        L2err2_local = J*dot(wq,err.^2)
        L2err2 += L2err2_local

        H1err2_local = J*dot(wq,deriv_err2)
        H1err2 += H1err2_local
    end
    return sqrt(L2err2), sqrt(H1err2)
end
L2err,H1err = compute_errs(u,mesh)
@show L2err,H1err
