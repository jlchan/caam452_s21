using TriangleMesh
using Plots
using Triplot

poly = polygon_unitSquare()
# poly = polygon_regular(5)
# poly = polygon_Lshape()
mesh = create_mesh(poly, set_area_max=true, quality_meshing=true)

VX,VY = mesh.point[1,:],mesh.point[2,:]
EToV  = mesh.cell # element-to-vertex mapping

function uniform_tri_nodes(N)
    r1D = LinRange(-1,1,N+1)
    Np = (N+1)*(N+2)÷2
    r,s = zeros(Np),zeros(Np)
    sk = 1
    for i = 0:N
        for j = 0:N-i
            r[sk] = r1D[i+1]
            s[sk] = r1D[j+1]
            sk += 1
        end
    end
    return r,s
end
r,s = uniform_tri_nodes(20)

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

# A = spzeros(length(VX),length(VX))
# for e = 1:mesh.n_cell
#     xv,yv = x[mesh.cell[:,e]],y[mesh.cell[:,e]]
#     J,rx,sx,ry,sy = compute_geometric_terms(xv,yv)
#
#     Vx = rx*dVr() + sx*dVs()
#     Vy = ry*dVr() + sy*dVs()
#     ids = mesh.cell[:,e]
#     @. A[ids,ids] += 2.0*J*(Vx'*Vx + Vy'*Vy)
#     # @show J
# end

scatter(VX,VY)
for e = 1:mesh.n_cell
    xv,yv = VX[mesh.cell[:,e]],VY[mesh.cell[:,e]]
    xe,ye = map_triangle_pts(r,s,xv,yv)
    zz = V(r,s)*(yv)
    scatter!(xe,ye,leg=false,ms=1)
end
display(plot!())
