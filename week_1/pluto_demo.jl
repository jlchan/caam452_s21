### A Pluto.jl notebook ###
# v0.12.18

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : missing
        el
    end
end

# ╔═╡ 91615676-4eb1-11eb-2599-9f94046c2a5d
begin
	let
		env = mktempdir()
		import Pkg
		Pkg.activate(env)
		Pkg.Registry.update()
		Pkg.add(Pkg.PackageSpec(;name="PlutoUI", version="0.6.7-0.6"))
	end
	using PlutoUI
end

# ╔═╡ f381d0ce-3faf-11eb-2586-ffe002ea8d9d
begin 
	using Plots
	using LinearAlgebra
	using SparseArrays
end

# ╔═╡ 6ce54252-3fb0-11eb-34e6-978f07a94d99
md"""

# Simulation of a model PDE 

Let's try to build some intuition for how a PDE behaves. Suppose we have a PDE on ``[-1,1]``

```math
\begin{gather*}
\frac{\partial u}{\partial t} + a\frac{\partial u}{\partial x} - b\frac{\partial^2 u}{\partial x^2} = f(x,t)
\end{gather*}
```
with left and right boundary conditions
```math
u(-1,t) = u_L(t), \qquad u'(1,t) = 0
```
and initial condition ``u(x,0) = u_0(x)``.


Different choices of ``a,b,c`` transition between different types of PDEs. 

## Physical parameters

Try choosing different values for each parameter. 

``a =`` $(@bind a NumberField(0:.1:10, default = 1))
``b =`` $(@bind b NumberField(0:.1:10, default = 1))

Different parameter values can yield different PDEs. For example:
- ``a=1, b = 0`` results in the advection equation
- ``a=0, b = 1`` results in the heat equation 
"""

# ╔═╡ e6a60308-3fb8-11eb-090f-9d533d92759b
md"Let's now define a forcing function `f(x,t)`, left boundary data `uL(t)`, and an initial condition `u0(x)`. Feel free to change these (set them to zero to remove)."

# ╔═╡ eb538830-3fb8-11eb-1d93-230874209dc7
begin
	u0(x)  = (.5*sin(pi*x) - .25*sin(2*pi*x) + .25*sin(4*pi*x))
	f(x,t) = 0*(exp(-100*(x+.75)^2))*(.5*t + sin(2*t))*exp(-.1*t)
	uL(t)  = 0*.75*sin(1*pi*t)
end;

# ╔═╡ 6bab77f0-6006-11eb-08b4-6b0a33f38472
md"Let's plot the initial condition below"

# ╔═╡ 1afcd228-3fc1-11eb-127a-b5059e1a0029
md"""
# Discretization parameters
"""

# ╔═╡ 952ca744-3fb7-11eb-20ab-a90257510dfb
md"""
Number of nodes `N` and timestep `dt`

`N =` $(@bind N NumberField(3:2:200, default = 201))

`dt =` $(@bind dt NumberField(.0001:.0001:10, default = .01))

Final time = $(@bind FinalTime NumberField(0:.0001:10, default = .25))
"""

# ╔═╡ d323417c-3fb5-11eb-0cdb-655211612ff0
begin 
	# Here is the code to construct discretization matrices for our simulation
	x = LinRange(-1,1,N)
	h = 2/(N-1) # mesh size
	K = spdiagm(-1=>-ones(N-1),1=>-ones(N-1),0=>2*ones(N))
	K[1,1] = -1  
	K[N,N] = 1 # corrections at the left/right endpoints
	K = (@. K/h^2)
		
	Q = spdiagm(-1=>-ones(N-1),1=>ones(N-1))
	Q[end,end] = 1
	Q = (@. Q/h)
	
	# modify matrices for Dirichlet BCs
	K[1,:] .= 0
	K[1,1] = 1
	Q[1,:] .= 0
	Q[1,1] = 1
end;

# ╔═╡ 73588218-6006-11eb-095e-e9c90ff9bbb8
begin 
	plot(x,u0.(x),xlim=(-1,1),ylim=(-1,1),lw=3)
	plot!(title="Initial condition",ratio=1,legend=false)
	scatter!(x,u0.(x))
end

# ╔═╡ 3b1b2ad0-3fbe-11eb-0745-2f8aeccfea54
md"Now we'll solve our PDE until final time $(FinalTime) and animate the results."

# ╔═╡ 4b1df9c6-3fbe-11eb-1dcf-3f883d7d92de
begin
	B = (a*Q + b*K) 
	A = I + .5*dt*B # Crank-Nicholson in time
	Nsteps = ceil(Int,FinalTime/dt)
	u = u0.(x) 
	
	anim = @animate for i = 1:Nsteps
		global u
		t = i*dt

		bb = u + dt*f.(x,t) - .5*dt*B*u 
		bb[1] = uL(t) # impose BC at left endpoint
		u = A\bb

		plot(x,u,xlim=(-1,1),ylim=(-1,1),lw=3)
		scatter!(x,u,label="")
		plot!(title="Time=$t",ratio=1,legend=false)
	end	every 1
	gif(anim, "fem.gif", fps = 10)
end

# ╔═╡ Cell order:
# ╟─6ce54252-3fb0-11eb-34e6-978f07a94d99
# ╟─e6a60308-3fb8-11eb-090f-9d533d92759b
# ╠═eb538830-3fb8-11eb-1d93-230874209dc7
# ╟─6bab77f0-6006-11eb-08b4-6b0a33f38472
# ╟─73588218-6006-11eb-095e-e9c90ff9bbb8
# ╟─1afcd228-3fc1-11eb-127a-b5059e1a0029
# ╟─952ca744-3fb7-11eb-20ab-a90257510dfb
# ╟─d323417c-3fb5-11eb-0cdb-655211612ff0
# ╟─3b1b2ad0-3fbe-11eb-0745-2f8aeccfea54
# ╟─4b1df9c6-3fbe-11eb-1dcf-3f883d7d92de
# ╟─91615676-4eb1-11eb-2599-9f94046c2a5d
# ╟─f381d0ce-3faf-11eb-2586-ffe002ea8d9d
