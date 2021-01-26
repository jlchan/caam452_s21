using Plots
using LinearAlgebra

#####
##### Code to show convergence of errors for different finite difference methods
#####

# define forward, centered differences, D3 = some other difference approx
D1(u,x,h) = (u(x+h) - u(x))/h
D2(u,x,h) = (u(x+h) - u(x-h))/(2*h)
D3(u,x,h) = (u(x-2*h) - 6*u(x-h) + 3*u(x) + 2*u(x+h))/(6*h)

include("fdcoeffV.jl")
x = -2:2
c = fdcoeffV(1,x[3],x)
D4(u,x,h,c) = dot(c,(u(x-2*h),u(x-h),u(x),u(x+h),u(x+2*h)))/h

# function, exact derivative
u(x) = x^2 + sin(1+pi*x)
dudx(x) = 2*x + pi*cos(1+pi*x)
x = 0.0 # point at which we approximate the derivative

hvec = 2.0 .^(-(0:8))
err = (zeros(size(hvec)),zeros(size(hvec)),zeros(size(hvec)),zeros(size(hvec)))
for (i,h) in enumerate(hvec)
    err[1][i] = abs(dudx(x) - D1(u,x,h))
    err[2][i] = abs(dudx(x) - D2(u,x,h))
    err[3][i] = abs(dudx(x) - D3(u,x,h))
    err[4][i] = abs(dudx(x) - D4(u,x,h,c))
end

plot( hvec,err[1],marker=:dot,label="Forward difference")
plot!(hvec,err[2],marker=:dot,label="Centered difference")
plot!(hvec,err[3],marker=:dot,label="Some other difference")
plot!(hvec,err[4],marker=:dot,label="5-point centered difference")
for N = 1:4
    plot!(hvec,hvec.^N,linestyle=:dash,label="Order $(N)")
end
plot!(xaxis=:log,yaxis=:log,legend=:bottomright)
