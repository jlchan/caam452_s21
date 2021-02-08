using LinearAlgebra
using SparseArrays
using Plots

m = 10
x = LinRange(0,1,m+2) # easier to define eigs on [0,1]
xint = x[2:end-1]
h = x[2]-x[1]
A = (1/h^2) * spdiagm(0=>2*ones(m),-1=>-ones(m-1),1=>-ones(m-1))

λ,V = eigen(Matrix(A))
# scatter(xint,V[:,1])

function normalize_cols!(A)
    for i = 1:size(A,2)
        A[:,i] /= maximum(A[:,i])
    end
end
normalize_cols!(V)

λex = @. 2/h^2 * (1-cos((1:m)*pi*h))
Vex = hcat((sin.(k*pi*xint) for k = 1:m)...)
normalize_cols!(Vex)

# plot(xint,V[:,1] .- Vex[:,1])
# scatter(λ - λex)
