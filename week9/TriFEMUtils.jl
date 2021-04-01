
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
