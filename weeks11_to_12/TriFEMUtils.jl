using UnPack  # for convenience @unpack macro

function to_rgba(x::UInt32)
    a = ((x & 0xff000000)>>24)/255
    b = ((x & 0x00ff0000)>>16)/255
    g = ((x & 0x0000ff00)>>8)/255
    r = (x & 0x000000ff)/255
    RGBA(r, g, b, a)
end

# plotting
function triplot(x,y,z,t)
    img = to_rgba.(Triplot.rasterize(x,y,z,t)')
    plot([extrema(x)...], [extrema(y)...], img, yflip=false)
end

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

function plotTriMesh(mesh::TriMesh)
    x,y = mesh.point[1,:], mesh.point[2,:]
    plotTriMesh(x,y,mesh.cell)
end

function plotTriMesh(mesh)
    VX,VY,EToV = mesh
    xmesh = Float64[]
    ymesh = Float64[]
    for vertex_ids in eachcol(EToV)
        ids = vcat(vertex_ids, vertex_ids[1])
        append!(xmesh,[VX[ids];NaN])
        append!(ymesh,[VY[ids];NaN])
    end
    display(plot(xmesh,ymesh,linecolor=:black,legend=false,ratio=1))
end

# returns vertex coordinates and element-to-vertex connectivities
function unpack_mesh_info(mesh::TriMesh)
    VX,VY = mesh.point[1,:],mesh.point[2,:]
    EToV  = mesh.cell  # element-to-vertex mapping: each column contains vertex ids for a different element
    return VX,VY,EToV
end
unpack_mesh_info(mesh) = mesh

function map_triangle_pts(r,s,x,y)
    return λ(r,s)*x, λ(r,s)*y
end

# get list of boundary indices (vector of indices) and boundary faces (stored as list of tuples (f=face,e=element))
function get_boundary_info(reference_face_indices,mesh)
    is_point_on_boundary = mesh.point_marker # 1 if on boundary, 0 otherwise
    is_boundary_face = find_boundary_faces(reference_face_indices,mesh)
    boundary_indices = findall(vec(is_point_on_boundary) .== 1)
    boundary_faces = Tuple.(findall(is_boundary_face .== 1))
    return boundary_indices,boundary_faces
end
function get_boundary_info(mesh::Tuple)
    VX,VY,EToV = mesh
    tol = 10*eps()
    on_boundary = (@. abs(abs(VX) - 1) < tol) .| (@. abs(abs(VY) - 1) < tol)
    boundary_indices = findall(on_boundary)
end

# loop through faces and look for matches
function find_boundary_faces(reference_face_indices,mesh)
    num_elements = size(mesh.cell,2)
    is_boundary_face = zeros(Int,3,num_elements)
    for e = 1:num_elements
        for f = 1:3
            face_vertices = mesh.cell[reference_face_indices[f],e]
            if sort(face_vertices) in sort.(eachcol(mesh.segment))
                 is_boundary_face[f,e] = 1
            end
        end
    end
    return is_boundary_face
end

"""
meshgrid(vx), meshgrid(vx,vy)
Computes an (x,y)-grid from the vectors (vx,vx). For more information, see the MATLAB documentation.

Copied and pasted directly from [VectorizedRoutines.jl](https://github.com/ChrisRackauckas/VectorizedRoutines.jl/blob/master/src/matlab.jl).
Using VectorizedRoutines.jl directly causes Pkg versioning issues with SpecialFunctions.jl
"""
meshgrid(v::AbstractVector) = meshgrid(v, v)
function meshgrid(vx::AbstractVector{T}, vy::AbstractVector{T}) where {T}
    m, n = length(vy), length(vx)
    vx = reshape(vx, 1, n)
    vy = reshape(vy, m, 1)
    (repeat(vx, m, 1), repeat(vy, 1, n))
end

function uniform_tri_mesh(Kx,Ky)
        (VY, VX) = meshgrid(LinRange(-1,1,Ky+1),LinRange(-1,1,Kx+1))
        sk = 1
        EToV = zeros(Int,2*Kx*Ky,3)
        for ey = 1:Ky
                for ex = 1:Kx
                        id(ex,ey) = ex + (ey-1)*(Kx+1) # index function
                        id1 = id(ex,ey)
                        id2 = id(ex+1,ey)
                        id3 = id(ex+1,ey+1)
                        id4 = id(ex,ey+1)
                        EToV[2*sk-1,:] = [id1 id3 id2]
                        EToV[2*sk,:]   = [id3 id1 id4]
                        sk += 1
                end
        end
        return VX[:],VY[:],Matrix(EToV')
end
