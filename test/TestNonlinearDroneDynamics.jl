include("../src/MyScp.jl")

using .MyScp
using LinearAlgebra

drone_params = Dict(
    :m => 1.0,
    :g => -9.81,
    :J_b => [1.0, 1.0, 1.0],
    :n_x => 13,
    :n_u => 6,
    :dt_ss => 1.0,
)

t = 1.0
state = vec(zeros(13, 1))
state[7] = 1 # ensure quat is physical
control = vec(zeros(6, 1))

state_new = MyScp.nonlinear_drone_dynamics(state, control, drone_params)
state_should = zeros(13)
state_should[6] = -9.81

@test state_new == state_should