using Plots
using LinearAlgebra

"
This code shows how errors for different finite difference methods converge as h->0.
"

# define forward, centered differences, D3 = some other difference approx
D1(u,x,h) = (u(x+h) - u(x))/h
D2(u,x,h) = (u(x+h) - u(x-h))/(2*h)
D3(u,x,h) = (u(x-2*h) - 6*u(x-h) + 3*u(x) + 2*u(x+h))/(6*h)

include("fdcoeffV.jl")
x = -2:2
a = fdcoeffV(1,x[3],x) # 1 = order of derivative, x̄, stencil points x
D4(u,x,h,c) = dot(c,(u(x-2*h),u(x-h),u(x),u(x+h),u(x+2*h)))/h

# function, exact derivative
u(x) = x^2 + sin(1+pi*x)
dudx(x) = 2*x + pi*cos(1+pi*x)
x = 0.0 # point at which we approximate the derivative

# measure how error changes as h decreases
hvec = 2.0 .^(-(0:8))
err = (zeros(size(hvec)),zeros(size(hvec)),zeros(size(hvec)),zeros(size(hvec)))
for (i,h) in enumerate(hvec)
    err[1][i] = abs(dudx(x) - D1(u,x,h))
    err[2][i] = abs(dudx(x) - D2(u,x,h))
    err[3][i] = abs(dudx(x) - D3(u,x,h))
    err[4][i] = abs(dudx(x) - D4(u,x,h,a))
end

plot(xaxis=:log,yaxis=:log,legend=:bottomright)
labels = ("Forward diff","Centered diff","Some other 4-point diff","5-point centered diff")
for i = 1:4
    plot!(hvec,err[i],marker=:dot,label=labels[i])
    plot!(hvec,hvec.^i,linestyle=:dash,label="Order $(i)")
end
display(plot!())
