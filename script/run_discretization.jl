using LinearAlgebra

include("../src/MyScp.jl")

using .MyScp

n_x = 13
n_u = 6
n_nodes = 10

drone_params = Dict(
    :m => 1.0,
    :g => -9.81,
    :J_b => [1.0, 1.0, 1.0],
    :n_x_aug => n_x,
    :n_u => n_u,
    :n_nodes => n_nodes,
    :dt_ss => 1.0,
)

t = 1.0
state = zeros(n_x)
state[7] = 1 # ensure quat is physical
control = ones(n_u)

V0 = [state; zeros(n_x * n_x + n_x * n_u + n_x * n_u + n_x)]

dV = dVdt(
    t,
    V0,
    zeros(n_u),
    zeros(n_u),
    nonlinear_drone_dynamics,
    jacobian_A,
    jacobian_B,
    drone_params
)

display(dV)

# Generate vectors of the matrices to test code for time varying matrices
u_vecs = zeros(n_u, n_nodes)
x_vecs = zeros(n_x, n_nodes)
for k in 1:n_nodes
    x_vecs[:, k] = state
    u_vecs[:, k] = control
end

A_bar, B_bar, C_bar, z_bar = calculate_discretization(
    x_vecs,
    u_vecs,
    nonlinear_drone_dynamics,
    jacobian_A,
    jacobian_B,
    drone_params
)

display(A_bar)