using LinearAlgebra
using SparseArrays
using Plots

f(x) = pi^2*sin(pi*x)
uexact(x) = sin(pi*x)

mvec = 2 .^ (2:9)
hvec = zeros(length(mvec))
err_inf = zeros(length(mvec))
err_1 = zeros(length(mvec))
err_2 = zeros(length(mvec))
# Anorm = zeros(length(mvec))
for (i,m) in enumerate(mvec)
    x = LinRange(-1,1,m+2) # add 2 endpoints
    xint = x[2:end-1]
    h = x[2]-x[1]
    A = (1/h^2) * spdiagm(0=>2*ones(m),-1=>-ones(m-1),1=>-ones(m-1))
    b = f.(xint)
    u = A\b

    hvec[i] = h
    err_inf[i] = maximum(@. abs(uexact(xint) - u))
    err_1[i] = norm((@. abs(uexact(xint) - u)),1) # 1-norm
    err_2[i] = norm(@. abs(uexact(xint) - u)) # 2-norm

    # Anorm[i] = norm(inv(Matrix(A)))
end

plot(hvec,err_inf,marker=:dot,label="Max error")
plot!(hvec,err_1,marker=:dot,label="1-norm error")
plot!(hvec,err_2,marker=:dot,label="2-norm error")
plot!(hvec,hvec*1.5 ,linestyle=:dash,label="O(h)")
plot!(hvec,hvec.^2,linestyle=:dash,label="O(h^2)")
plot!(xaxis=:log,yaxis=:log,legend=:bottomright)
