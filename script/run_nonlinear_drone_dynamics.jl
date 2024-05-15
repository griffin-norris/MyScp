using LinearAlgebra
# using Revise

include("../src/MyScp.jl")

using .MyScp

n_x = 13
n_u = 6

drone_params = Dict(
    :m => 1.0,
    :g => -9.81,
    :J_b => [1.0, 1.0, 1.0],
    :n_x => n_x,
    :n_u => n_u,
    :dt_ss => 1.0,
)

state = zeros(n_x)
state[7] = 1 # ensure quat is physical
control = ones(n_u)

display(nonlinear_drone_dynamics(state, control, drone_params))

display(jacobian_A(state, control, drone_params))
display(jacobian_B(state, control, drone_params))