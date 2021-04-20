using TriangleMesh
using Plots
using Triplot
using LinearAlgebra
using SparseArrays
using ForwardDiff
using NodesAndModes # for triangle quadrature rules
using UnPack  # for convenience @unpack macro
include("./TriFEMUtils.jl") # plotting and misc routines

poly = polygon_unitSquare()
# poly = polygon_regular(5)
# poly = polygon_Lshape()
max_triangle_area = .05/64
mesh = create_mesh(poly, quality_meshing=true,
                   add_switches="penva"*string(max_triangle_area)*"q") # switches

# # manufactured solution
# uexact(x,y) = exp(sin(1+pi*x)*sin(pi*y))
# dudx(x,y) = ForwardDiff.derivative(x->uexact(x,y),x)
# dudy(x,y) = ForwardDiff.derivative(y->uexact(x,y),y)
# f(x,y) = -(ForwardDiff.derivative(x->dudx(x,y),x) +
#            ForwardDiff.derivative(y->dudy(x,y),y)) # -(du^2/dx^2 + du^2/dy^2)
#
# # define Dirichlet and Neumann boundaries
# on_Neumann(x,y)   = x ≈ 1 && y > 0 && y < 1 # right face x = 1 (excluding corners)
# on_Dirichlet(x,y) = !on_Neumann(x,y) # Dirichlet boundary = everywhere else
# u_Dirichlet(x,y)  = uexact(x,y)
# ∇u_dot_n(x,y)     = dudx(x,y) # note that n = [1,0] on the Neumann face

# new problem
f(x,y) = 0.0
on_Neumann(x,y)   = x ≈ 1 && y > 0 && y < 1 # right face x = 1 (excluding corners)
on_Dirichlet(x,y) = !on_Neumann(x,y) # Dirichlet boundary = everywhere else
u_Dirichlet(x,y)  = 0.0
∇u_dot_n(x,y)     = 0.0 # note that n = [1,0] on the Neumann face
u0(x,y) = sin(pi*x)*sin(pi*y)

# ordering of reference element vertices
# 3
# |`.
# 1--2
reference_vertices = [[-1,1,-1],[-1,-1,1]] # stored as [r,s]
reference_face_indices = [[1,2],[2,3],[1,3]] # ordering of faces for the reference element
ref_elem_info = (;reference_face_indices,reference_vertices)

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

    # rq,sq,wq = 1/3,1/3,2.0
    # rq,sq,wq = quad_nodes(Tri(),1)
    rq,sq,wq = [-2/3; 1/3; -2/3], [-2/3; -2/3; 1/3], 2/3 * ones(3)

    A = spzeros(num_vertices, num_vertices)
    M = spzeros(num_vertices, num_vertices)
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

        # assemble mass matrix using quadrature
        M[ids,ids] .+= J*reference_elem_area*(λ(rq,sq)'*Diagonal(wq)*λ(rq,sq))

        # quadrature rule
        xq,yq = map_triangle_pts(rq,sq,xv,yv)
        b[ids] .+= J*λ(rq,sq)'*(wq.*f.(xq,yq))
    end
    return M,A,b
end

function compute_Neumann_BCs(mesh,ref_elem_info,on_Neumann_boundary,∇u_dot_n)

    @unpack reference_face_indices, reference_vertices = ref_elem_info
    VX,VY,EToV = unpack_mesh_info(mesh)
    boundary_indices, boundary_faces = get_boundary_info(reference_face_indices, mesh)
    # contributions from Neumann BCs

    num_vertices = length(VX)
    b = zeros(num_vertices)
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
function modify_matrix_Dirichlet_BCs!(A,mesh,ref_elem_info,on_Dirichlet_boundary)
    @unpack reference_face_indices, reference_vertices = ref_elem_info
    VX,VY,EToV = unpack_mesh_info(mesh)
    boundary_indices, boundary_faces = get_boundary_info(reference_face_indices, mesh)

    # impose Dirichlet BCs
    for i in boundary_indices
        xi,yi = VX[i],VY[i]
        if on_Dirichlet_boundary(xi,yi)
            A[:,i] .= 0
            A[i,:] .= 0
            A[i,i] = 1.0
        end
    end
end
function compute_Dirichlet_BCs(A,mesh,ref_elem_info,on_Dirichlet_boundary,u_Dirichlet)
    @unpack reference_face_indices, reference_vertices = ref_elem_info
    VX,VY,EToV = unpack_mesh_info(mesh)
    boundary_indices, boundary_faces = get_boundary_info(reference_face_indices, mesh)

    num_vertices = length(VX)
    b = zeros(num_vertices)

    # impose Dirichlet BCs
    for i in boundary_indices
        xi,yi = VX[i],VY[i]
        if on_Dirichlet_boundary(xi,yi)
            b -= Vector(A[:,i]*u_Dirichlet(xi,yi))
        end
    end
    return b
end
function constrain_Dirichlet_nodes!(b,mesh,ref_elem_info,on_Dirichlet_boundary,u_Dirichlet)
    @unpack reference_face_indices, reference_vertices = ref_elem_info
    VX,VY,EToV = unpack_mesh_info(mesh)
    boundary_indices, boundary_faces = get_boundary_info(reference_face_indices, mesh)

    for i in boundary_indices
        xi,yi = VX[i],VY[i]
        if on_Dirichlet_boundary(xi,yi)
            b[i] = u_Dirichlet(xi,yi)
        end
    end
end


M,A,b = assemble_FE_matrix(mesh)

# # Poisson
# b_Neumann = compute_Neumann_BCs(mesh,ref_elem_info,on_Neumann,∇u_dot_n)
# b_Dirichlet = compute_Dirichlet_BCs(A,mesh,ref_elem_info,on_Dirichlet,u_Dirichlet)
# b += b_Neumann + b_Dirichlet
# constrain_Dirichlet_nodes!(b,mesh,ref_elem_info,on_Dirichlet,u_Dirichlet)
# modify_matrix_Dirichlet_BCs!(A,mesh,ref_elem_info,on_Dirichlet)
# u = A\b
# VX,VY,EToV = unpack_mesh_info(mesh)
# @show maximum(abs.(u .- uexact.(VX,VY)))
# triplot(VX,VY,u .- uexact.(VX,VY),EToV)

Δt = .001
T = .2

# create/prefactor matrix
C = (M + .5*Δt*A) # Crank-Nicolson
# compute BC contributions
b_Neumann = compute_Neumann_BCs(mesh,ref_elem_info,on_Neumann,∇u_dot_n)
b_Dirichlet = compute_Dirichlet_BCs(C,mesh,ref_elem_info,on_Dirichlet,u_Dirichlet)
modify_matrix_Dirichlet_BCs!(C,mesh,ref_elem_info,on_Dirichlet)
C = cholesky(Symmetric(C))

Nsteps = ceil(T/Δt)
Δt = T/Nsteps
u = u0.(VX,VY)
plot()
VX,VY,EToV = unpack_mesh_info(mesh) # for plotting
# @gif
for i = 1:Nsteps
    global u

    b = M*u - .5*Δt*A*u # compute RHS
    @. b += b_Neumann + b_Dirichlet # add BC contributions
    constrain_Dirichlet_nodes!(b,mesh,ref_elem_info,on_Dirichlet,u_Dirichlet)
    u = C \ b
    if i%5==0
        println("on iter $i out of $Nsteps")
        # triplot(VX,VY,u,EToV)
    end
end #every 5

# triplot(VX,VY,uexact.(VX,VY),EToV)
