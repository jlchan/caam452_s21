
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

function plotTriMesh(mesh)
    x,y = mesh.point[1,:], mesh.point[2,:]

    plot() # init plot
    for vertex_ids in eachcol(mesh.cell)
        ids = vcat(vertex_ids, vertex_ids[1])
        plot!(x[ids],y[ids],linecolor=:black)
    end
    display(plot!(legend=false,ratio=1))
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
