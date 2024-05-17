using LinearAlgebra
using Plots

include("../src/MyScp.jl")

using .MyScp

n_x = 13
n_x_aug = n_x + 1
n_u = 6
x_initial = zeros(n_x)
x_initial[1] = -10
x_initial[7] = 1
x_final = copy(x_initial)
x_final[1] = 10.0
x_max = [10, 10, 10, 50, 50, 50, 1, 1, 1, 1, 10, 10, 10]
x_min = -x_max
x_max = [x_max; 1]
x_min = [x_min; 0]
u_max = [0.0, 0.0, 200.0, 10.0, 10.0, 1.0]
u_min = [0.0, 0.0, 0.0, -10.0, -10.0, -1.0]

obstacle_centers = [
    [-5.0, 0.01, 0.0],
    [0.0, 0.01, 0.0],
    [5.0, 0.01, 0.0],
]

# Must be orthogonal unit vectors
# can use function `generate_orthogonal_unit_vectors`
obstacle_axes = [
    I(3),
    I(3),
    I(3),
]

obstacle_radius = [
    1.0 * ones(3),
    1.0 * ones(3),
    1.0 * ones(3),
]

# S_x = I(n_x_aug)
# S_u = I(n_u)
# c_x = zeros(n_x_aug)
# c_u = zeros(n_u)

S_x = Diagonal(max.(ones(n_x_aug), abs.(x_min - x_max) / 2))
c_x = (x_max + x_min) / 2

S_u = Diagonal(max.(ones(n_u), abs.(u_min - u_max) / 2))
c_u = (u_max + u_min) / 2

w_tr = 1e2
λ_fuel = 1e1
λ_vc = 1e2
λ_vc_ctcs = 1e1
scale = max(w_tr, λ_fuel, λ_vc, λ_vc_ctcs)

total_time = 10.0
dt_ss = 0.5
dt_sim = 0.05

n_nodes = Int(total_time / dt_ss) + 1

params = Dict(
    :m => 1.0,
    :g => -9.81,
    :J_b => [1.0, 1.0, 1.0],
    :total_time => total_time,
    :dt_ss => dt_ss,
    :dt_sim => dt_sim,
    :k_max => 50,
    :n_nodes => n_nodes,
    :n_x => n_x,
    :n_x_aug => n_x_aug,
    :n_u => n_u,
    :n_obs => 3,
    :obstacle_centers => obstacle_centers,
    :obstacle_axes => obstacle_axes,
    :obstacle_radius => obstacle_radius,
    :S_x => S_x,
    :S_u => S_u,
    :c_x => c_x,
    :c_u => c_u,
    :w_tr => w_tr / scale,
    :λ_fuel => λ_fuel / scale,
    :λ_vc => λ_vc / scale,
    :λ_vc_ctcs => λ_vc_ctcs / scale,
    :ϵ_tr => 1E-3,
    :ϵ_vc => 1E-7,
    :x_initial => x_initial,
    :x_final => x_final,
    :x_max => x_max,
    :x_min => x_min,
    :u_max => u_max,
    :u_min => u_min,
)

result = ctcs_main(params)

# Time vector assuming each column is a time step
time_steps = 1:size(result[:x], 2)

positions = result[:x][1:3, :]
velocities = result[:x][4:6, :]


# Create plots
# plotlyjs()

p = plot(layout=(3, 2), size=(1600, 1000))

# Plot the history of trajectory iterations
for x_hist in result[:x_hist]
    plot!(
        p[1],
        x_hist[1:3, :]',
        alpha=0.3,
        linewidth=1,
        label=false
    )
end

plot!(
    p[1],
    result[:x][1:3, :]',
    ylims=(
        1.2 * minimum(params[:x_min][1:3]),
        1.2 * maximum(params[:x_max][1:3]),
    ),
    title="Position (Inertial Frame)",
    ylabel="Values",
    label=["px" "py" "pz"]
)

plot!(
    p[3],
    result[:x][4:6, :]',
    # ylims=(
    #     1.2 * minimum(params[:x_min][4:6]),
    #     1.2 * maximum(params[:x_max][4:6]),
    # ),
    title="Velocity (Inertial Frame)",
    ylabel="Values",
    label=["vx" "vy" "vz"]
)

plot!(
    p[2],
    result[:x][7:10, :]',
    ylims=(
        1.2 * minimum(params[:x_min][7:10]),
        1.2 * maximum(params[:x_max][7:10]),
    ),
    title="Quaternion",
    ylabel="Values",
    label=["qw" "qx" "qy" "qz"]
)

plot!(
    p[4],
    result[:x][11:13, :]',
    ylims=(
        1.2 * minimum(params[:x_min][11:13]),
        1.2 * maximum(params[:x_max][11:13]),
    ),
    title="Rotation Rate (Body Frame)",
    ylabel="Values",
    label=["ω1" "ω2" "ω3"]
)

plot!(
    p[5],
    result[:u][1:3, :]',
    # ylims=(
    #     1.2 * minimum(params[:u_min][1:3]),
    #     1.2 * maximum(params[:u_max][1:3]),
    # ),
    title="Force (Body Frame)",
    ylabel="Values",
    label=["Fb1" "Fb2" "Fb3"]
)

plot!(
    p[6],
    result[:u][4:6, :]',
    # ylims=(
    #     1.2 * minimum(params[:u_min][4:6]),
    #     1.2 * maximum(params[:u_max][4:6]),
    # ),
    title="Torque (Body Frame)",
    ylabel="Values",
    label=["τ1" "τ2" "τ3"]
)

# Display the plot
display(p)

# p3d = plot3d(size=(1600, 1000))
# plot!(
#     p3d,
#     result[:x][1, :],
#     result[:x][2, :],
#     result[:x][3, :],
#     title="3D Trajectory Plot",
#     xlabel="X Position",
#     ylabel="Y Position",
#     zlabel="Z Position",
#     legend=false,
#     linewidth=2,
#     marker=:circle,
#     ylims=(params[:x_min][2], params[:x_max][2]),
#     zlims=(params[:x_min][3], params[:x_max][3]),
# )

# # Display the plot
# display(p3d)