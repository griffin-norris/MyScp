module MyScp

using JuMP, ECOS
using LinearAlgebra

greet() = println("Hello World!")

const Func = Union{Nothing,Function}

include("OptimizationProblem.jl")

pbm = OptimizationProblem()

_tf = 1
_x0 = [0.1; 0; 0]   # Initial state
_xf = [0; 4; 0]     # Terminal state
_ro = 0.35          # KOZ radius
_co = [-0.1; 1]     # KOZ center
_carw = 0.1         # Car width

greet()

# Define nodes
pbm.N = 10

# Define variable sizes
pbm.nx = 3
pbm.nu = 2
pbm.np = 1

# Define cost
pbm.Γ = (t, k, x, u, p, pbm) -> u'*u

# Define dynamics
pbm.f = (t, k, x, u, p, pbm) -> [u[1]*sin(x[3]); u[1]*cos(x[3]); u[2]]*_tf
pbm.A = (t, k, x, u, p, pbm) ->
        [0 0 u[1]*cos(x[3]);
         0 0 -u[1]*sin(x[3]);
         0 0 0]*_tf
pbm.B = (t, k, x, u, p, pbm) ->
        [sin(x[3]) 0;
         cos(x[3]) 0;
         0 1]*_tf
pbm.F = (t, k, x, u, p, pbm) -> zeros(pbm.nx, pbm.np)

# Define constraints
pbm.s = (t, k, x, u, p, pbm) -> [(_ro+_carw/2)^2-(x[1]-_co[1])^2-(x[2]-_co[2])^2]
pbm.C = (t, k, x, u, p, pbm) -> collect([-2*(x[1]-_co[1]); -2*(x[2]-_co[2]); 0]')
pbm.D = (t, k, x, u, p, pbm) -> zeros(1, pbm.nu)
pbm.G = (t, k, x, u, p, pbm) -> zeros(1, pbm.np)

A_d = exp(pbm.A(0, 0, [0 0 0], [1 1], 1, pbm) * 1 / pbm.N)
print(A_d)

# Define boundary conditions
pbm.g₀ = (x, p, pbm) -> x-_x0
pbm.H₀ = (x, p, pbm) -> I(pbm.nx)
pbm.ℓ₀ = (x₀, x_bar, p_bar, pbm) -> (
    pbm.g₀(x_bar, p_bar, pbm) - pbm.H₀(x_bar, p_bar, pbm) * x₀ - pbm.K₀(x, p, pbm) * p_bar
)
pbm.g₁ = (x, p, pbm) -> x-_xf
pbm.H₁ = (x, p, pbm) -> I(pbm.nx)
pbm.ℓ₁ = (x₁, x_bar, p_bar, pbm) -> (
    pbm.g₁(x_bar, p_bar, pbm) - pbm.H₁(x_bar, p_bar, pbm) * x₁ - pbm.K₁(x, p, pbm) * p_bar
)

println(pbm.f)

# Test JuMP, ECOS
model = JuMP.Model(ECOS.Optimizer)
@variable(model, x >= 0)
@variable(model, 0 <= y <= 3)
@objective(model, Min, 12x + 20y)
@constraint(model, c1, 6x + 8y >= 100)
@constraint(model, c2, 7x + 12y >= 120)
print(model)
optimize!(model)
println(termination_status(model))
println(primal_status(model))
println(dual_status(model))
println(objective_value(model))
println(value(x))
println(value(y))
println(shadow_price(c1))
println(shadow_price(c2))

E = I(pbm.nx)

# SCvx algorithm modifications
f_scvx_ct = (t, k, x, u, p, pbm, x_bar, u_bar, p_bar, ν, νₛ, ν₀) -> (
    pbm.A(t, k, x_bar, u_bar, p, pbm) * x +
    pbm.B(t, k, x_bar, u_bar, p, pbm) * u +
    pbm.F(t, k, x_bar, u_bar, p, pbm) * p +
    pbm.r(t, k, x_bar, u_bar, p, pbm) +
    E*ν
)
s_scvx_ct = (t, k, x, u, p, pbm, x_bar, u_bar, p_bar, ν, νₛ, ν₀) -> (
    pbm.C(t, k, x_bar, u_bar, p_bar, pbm) * x +
    pbm.D(t, k, x_bar, u_bar, p_bar, pbm) * u +
    pbm.G(t, k, x_bar, u_bar, p_bar, pbm) * p +
    pbm.r′(t, k, x_bar, u_bar, p_bar, pbm) -
    νₛ
)
ic_scvx_ct = (t, k, x, u, p, pbm, x_bar, u_bar, p_bar, ν, νₛ, ν₀) -> (
    pbm.H₀(t, k, x_bar, u_bar, p_bar, pbm) * x
    + pbm.K₀(t, k, x_bar, u_bar, p_bar, pbm) * p
    + pbm.ℓ₀(t, k, x_bar, u_bar, p_bar, pbm) + ν₀
)
tc_scvx_ct = (t, k, x, u, p, pbm, x_bar, u_bar, p_bar, ν, νₛ, ν₀) -> (
    pbm.H₁(t, k, x_bar, u_bar, p_bar, pbm) * x
    + pbm.K₁(t, k, x_bar, u_bar, p_bar, pbm) * p
    + pbm.ℓ₁(t, k, x_bar, u_bar, p_bar, pbm) + ν₀
)
trust_region = (t, k, x, u, p, pbm, x_bar, u_bar, p_bar, ν, νₛ, ν₀) -> (
    norm2(x-x_bar) + norm2(u-u_bar) + norm2(p-p_bar) - η
)

# SCvx objective function
P = (x, y) -> (norm1(x) + norm1(y))

ϕλ = (t, k, x, u, p, pbm, x_bar, u_bar, p_bar, ν, νₛ, ν₀) -> (
    pbm.ϕ(t, k, x, u, p, pbm) + λ*P(0,ν₀) + λ*P(0,ν₁)
)
Γλ = (t, k, x, u, p, pbm, x_bar, u_bar, p_bar, ν, νₛ, ν₀) -> (
    pbm.Γ(t, k, x, u, p, pbm) + λ*P(E*ν, νₛ)
)

# Discretized system

Aₖ = (t, k, x_bar, u_bar, p, pbm) -> (
    exp(pbm.A(t, k, x_bar, u_bar, p, pbm) * 1/pbm.N)
)
Bₖ = (t, k, x_bar, u_bar, p, pbm) -> (
    exp(pbm.B(t, k, x_bar, u_bar, p, pbm) * 1/pbm.N)
)
Fₖ = (t, k, x_bar, u_bar, p, pbm) -> (
    exp(pbm.F(t, k, x_bar, u_bar, p, pbm) * 1/pbm.N)
)
# rₖ TODO
Eₖ = (t, k, x_bar, u_bar, p, pbm) -> (
    exp(E * 1/pbm.N)
)
Cₖ = (t, k, x_bar, u_bar, p, pbm) -> (
    exp(pbm.C(t, k, x_bar, u_bar, p, pbm) * 1/pbm.N)
)
Dₖ = (t, k, x_bar, u_bar, p, pbm) -> (
    exp(pbm.D(t, k, x_bar, u_bar, p, pbm) * 1/pbm.N)
)
Gₖ = (t, k, x_bar, u_bar, p, pbm) -> (
    exp(pbm.G(t, k, x_bar, u_bar, p, pbm) * 1/pbm.N)
)
# r′ₖ TODO

# Discretized cost
L = (t, k, x, u, p, pbm, x_bar, u_bar, p_bar, ν, νₛ, ν₀) -> (
    ϕλ(t, k, x, u, p, pbm, x_bar, u_bar, p_bar, ν, νₛ, ν₀) + 
    for k in 1:N-1
        
    end
)

end # module MyScp
