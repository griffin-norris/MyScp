# using Revise
using LinearAlgebra
import JuMP as jp

include("../src/MyScp.jl")

using .MyScp

# state =   [px, py, vx, vy]
# control = [ax, ay]

n_x = 4
n_x_aug = 5
n_u = 2

n_nodes = 11
dt = 1.0 / n_nodes

S_x = Matrix(I(n_x_aug))
c_x = vec(zeros(n_x_aug, 1))
S_u = Matrix(I(n_u))
c_u = vec(zeros(n_u, 1))

x_initial = [0.5, 0.5, 0.0, 0.0]
x_final = [0.0, 0.0, 0.0, 0.0]

x_max = 1.0 * ones(n_x_aug)
x_min = -x_max
u_max = 10.0 * ones(n_u)
u_min = -u_max

params = Dict(
    :n_nodes => n_nodes,
    :n_x => n_x,
    :n_x_aug => n_x_aug,
    :n_u => n_u,
    :S_x => S_x,
    :c_x => c_x,
    :S_u => S_u,
    :c_u => c_u,
    :x_initial => x_initial,
    :x_final => x_final,
    :x_max => x_max,
    :x_min => x_min,
    :u_max => u_max,
    :u_min => u_min,
    :w_tr => 1.0,
    :λ_fuel => 1.0,
    :λ_vc => 1.0,
    :λ_vc_ctcs => 1.0,
)

A_d = [1 0 dt 0;
    0 1 0 dt;
    0 0 1 0;
    0 0 0 1]
B_d = [0 0;
    0 0;
    10 0;
    0 10] * dt
C_d = [0 0;
    0 0;
    0 0;
    0 0] * dt
z_d = [0, 0, 0, 0]

A_aug = [A_d zeros(n_x); zeros(n_x)' 1]
B_aug = [B_d; zeros(n_u)']
C_aug = [C_d; zeros(n_u)']
z_aug = [z_d; 0]

# Generate vectors of the matrices to test code for time varying matrices
A_vectors = zeros(n_x_aug * n_x_aug, n_nodes - 1)
B_vectors = zeros(n_x_aug * n_u, n_nodes - 1)
C_vectors = zeros(n_x_aug * n_u, n_nodes - 1)
z_vectors = zeros(n_x_aug, n_nodes)
for k in 1:n_nodes-1
    A_vectors[:, k] = vec(A_aug)
    B_vectors[:, k] = vec(B_aug)
    C_vectors[:, k] = vec(C_aug)
    z_vectors[:, k] = vec(z_aug)
end

traj_bar = Trajectory(
    linterp(n_nodes, [x_initial; 0.0], [x_final; 0.0]),
    linterp(n_nodes, [0, 0], [0, 0]),
)

sys = LinearizedDiscretizedFohSystem(A_vectors, B_vectors, C_vectors, z_vectors)

ocp = OCP(params, sys, traj_bar)
print(ocp)

jp.optimize!(ocp)

if jp.is_solved_and_feasible(ocp)
    println("state:")
    display(jp.value.(ocp[:x]))
    println("control:")
    display(jp.value.(ocp[:u]))
    println("virtual control:")
    display(jp.value.(ocp[:nu]))
end