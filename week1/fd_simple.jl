"This code defines some simple finite difference approximations, applies them to
approximate u'(x), and compares the result to the exact solution."

# run file by typing "include("fd_simple.jl")", copying and pasting into the REPL, or
# selecting code and pressing Ctrl/Command-Enter

Dfwd(u,x,h) = (u(x+h) - u(x))/h
Dback(u,x,h) = (u(x) - u(x-h))/(h)
D2(u,x,h) = (u(x+h) - u(x-h))/(2*h)

u(x) = sin(1+pi*x)
dudx(x) = pi*cos(1+pi*x)

x̄ = .5
h = .001
@show Dfwd(u,x̄,h) - dudx(x̄)
@show Dback(u,x̄,h) - dudx(x̄)
@show D2(u,x̄,h) - dudx(x̄)
