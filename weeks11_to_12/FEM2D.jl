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



function compute_error(u,mesh)
    VX,VY,EToV = unpack_mesh_info(mesh)
    num_elements = size(EToV,2) # number of elements = of columns

    rq = [-2/3; 1/3; -2/3]
    sq = [-2/3; -2/3; 1/3]
    wq = 2/3 * ones(3)
    L2err_squared = 0
    H1err_squared = 0 
    for e = 1:num_elements # loop through all elements
        ids = EToV[:,e] # vertex ids = local to global index maps
        xv,yv = VX[ids],VY[ids]
        switch = 1
     

        # compute geometric mappings
        J,drdx,dsdx,drdy,dsdy = compute_geometric_terms(xv,yv)

        # quadrature rule
        xq,yq = map_triangle_pts(rq,sq,xv,yv)
        u_local = λ(rq,sq)*u[ids]   #Using linear interpolation in order to obtain u at the gauss points
        
        #L2 Error = sqrt(∫(u-u_exact)^2 dA) = sqrt(Σ J∫(u-u_exact)^2 dD')
        L2err_squared += J*sum(wq.*(switch*u_local - uexact.(xq,yq)).^2)

        #H1 Error  = sqrt(∫(u'-u_exact')^2 dA)
        dudr_l = dλr()*u[ids]
        duds_l = dλs()*u[ids]
        dudx_local = dudr_l*drdx + duds_l*dsdx
        dudy_local = dudr_l*drdy + duds_l*dsdy

        H1err_squared += J*sum(wq.*(switch*dudx_local .- dudx.(xq,yq)).^2 .+ wq.*(switch*dudy_local .- dudy.(xq,yq)).^2)

    end
    L2err = sqrt(L2err_squared)
    H1err = sqrt(H1err_squared)
    return L2err, H1err
end


K1D = [16, 16*2,16*2^2,16*2^3,16*2^4]
hvec = [1/16, 1/(16*2),1/(16*2^2),1/(16*2^3),1/(16*2^4)]
L2_err = zeros(length(K1D))
H1_err = zeros(length(K1D))
for (i,Ksize) in enumerate(K1D)
    mesh = uniform_tri_mesh(Ksize,Ksize)
    boundary_indices = get_boundary_info(mesh)

    A,b = assemble_FE_matrix(mesh)
    A,b = impose_Dirichlet_BCs!(A,b,mesh,boundary_indices)
    u = A\b
    VX,VY,EToV = unpack_mesh_info(mesh)
    # @show maximum(abs.(u .- uexact.(VX,VY)))
    # triplot(VX,VY,u .- uexact.(VX,VY),EToV)
    # triplot(VX,VY,u,EToV)
    # triplot(VX,VY,uexact.(VX,VY),EToV)
    L2,H1 = compute_error(u,mesh)
    L2_err[i] = L2
    H1_err[i] = H1
end

plot(hvec,L2_err,marker=:dot,label="L2 Norm")
plot!(hvec,H1_err,marker=:dot,label="H1 Norm")
plot!(hvec,hvec,linestyle=:dash,label="O(h)")
plot!(hvec,hvec.^2,linestyle=:dash,label="O(h^2)")
plot!(xaxis=:log,yaxis=:log,legend=:bottomright)