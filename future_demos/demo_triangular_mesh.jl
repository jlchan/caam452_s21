using TriangleMesh
using Plots
using Triplot

poly = polygon_regular(5)
poly = polygon_Lshape()
mesh = create_mesh(poly, set_area_max=true, quality_meshing=true)

function plotTriMesh(mesh)
    x,y = mesh.point[1,:], mesh.point[2,:]

    plot() # init plot
    for vertex_ids in eachcol(mesh.cell)
        ids = vcat(vertex_ids, vertex_ids[1])
        plot!(x[ids],y[ids],linecolor=:black)
    end
    display(plot!(legend=false,ratio=1))
end
plotTriMesh(mesh)

function to_rgba(x::UInt32)
    a = ((x & 0xff000000)>>24)/255
    b = ((x & 0x00ff0000)>>16)/255
    g = ((x & 0x0000ff00)>>8)/255
    r = (x & 0x000000ff)/255
    RGBA(r, g, b, a)
end

# plotting
x,y = mesh.point[1,:], mesh.point[2,:]
z,t = (@. sin(pi*x)*sin(pi*y)), mesh.cell
img = to_rgba.(Triplot.rasterize(x,y,z,t)')
plot([extrema(x)...], [extrema(y)...], img, yflip=false)
