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
    :n_x => n_x,
    :n_u => n_u,
    :n_nodes => n_nodes,
    :dt_ss => 1.0,
    :dt_sim => 0.05,
    :total_time => 10.0,
)

t = 1.0
state_initial = zeros(n_x)
state_initial[7] = 1 # ensure quat is physical
# control = ones(n_u)
control = zeros(n_u)
control[3] = 9.80
control[1] = 1

u_vecs = zeros(n_u, n_nodes+1)
for k in 1:n_nodes+1
    u_vecs[:, k] = (-1)^(k+1) * control
end

display(simulate_nonlinear(nonlinear_drone_dynamics, state_initial, u_vecs, drone_params))

