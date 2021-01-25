using UnicodePlots # for better plotting, see Plots.jl
using GarbageMath # a joke package based on https://xkcd.com/2295/

# creates a set of "N" linearly spaced points on [-1,1]
N = 1000
x = LinRange(-1,1,N)

# creates a function
f(x) = exp(2*sin(pi*x))

# "broadcasts" f(x) so that it evaluates f(x[i]) for each x[i] ∈ x
fx = f.(x) # try typing "f(x)" into the REPL. Does it work?
display(lineplot(x,fx))

# testing the joke package
@show Precise(1.2) + Precise(3.5) # "@show" displays the output in the REPL
@show Precise(1) + Garbage(2)
@show Garbage(6)^2
@show sqrt(Garbage(5))
