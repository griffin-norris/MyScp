using LinearAlgebra
using Plots
using Random

include("../src/MyScp.jl")

using .MyScp

Random.seed!(0)

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
    # I(3),
    # I(3),
    # I(3),
    generate_orthogonal_unit_vectors(),
    generate_orthogonal_unit_vectors(),
    generate_orthogonal_unit_vectors(),
]

obstacle_radius = [
    # 1.0 * ones(3),
    # 1.0 * ones(3),
    # 1.0 * ones(3),
    rand(3) + 0.5 * ones(3),
    rand(3) + 0.5 * ones(3),
    rand(3) + 0.5 * ones(3),
]

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
dt_ss = 1.0
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
    :w_tr_adapt_factor => 1.1,
    :λ_fuel => λ_fuel / scale,
    :λ_vc => λ_vc / scale,
    :λ_vc_ctcs => λ_vc_ctcs / scale,
    :ϵ_tr => 1E-3,
    :ϵ_vc => 1E-7,
    :ϵ_vc_ctcs => 1E-4,
    :x_initial => x_initial,
    :x_final => x_final,
    :x_max => x_max,
    :x_min => x_min,
    :u_max => u_max,
    :u_min => u_min,
    :dis_int_high_order => false,
)

result = ctcs_main(params)

# Time vector assuming each column is a time step
time_steps = 1:size(result[:x], 2)

positions = result[:x][1:3, :]
velocities = result[:x][4:6, :]


n_iterations = length(result[:x_hist])
# Create Plots
gr()

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

plotlyjs()
p3d = plot(
    size=(1600, 1000),
    palette=palette(:inferno, 1:n_iterations+1, rev=true),
)


# Define function to generate ellipsoid points
function generate_ellipsoid_points(center, radii, axes, n=40)
    u = range(0, stop=2π, length=n)
    v = range(0, stop=π, length=n)

    x = [1 / radii[1] * cos(ui) * sin(vi) for ui in u, vi in v]
    y = [1 / radii[2] * sin(ui) * sin(vi) for ui in u, vi in v]
    z = [1 / radii[3] * cos(vi) for ui in u, vi in v]

    points = [x[:] y[:] z[:]]'
    rotated_points = axes * points
    translated_points = rotated_points .+ center

    X = reshape(translated_points[1, :], n, n)
    Y = reshape(translated_points[2, :], n, n)
    Z = reshape(translated_points[3, :], n, n)

    return X, Y, Z
end

obstacles = []
for k in 1:params[:n_obs]
    push!(
        obstacles,
        EllipsoidalObstacle(
            params[:obstacle_centers][k],
            params[:obstacle_axes][k],
            params[:obstacle_radius][k],
        )
    )
end

# Plot ellipsoidal obstacles
for obs in obstacles
    X, Y, Z = generate_ellipsoid_points(obs.center, obs.radius, obs.axes)
    plot!(
        p3d,
        X, Y, Z,
        seriestype=:surface,
        opacity=0.5,
        legend=false,
        showscale=false,
    )
end

for (i, x_hist) in enumerate(result[:x_hist])
    plot!(
        p3d,
        x_hist[1, :],
        x_hist[2, :],
        x_hist[3, :],
        alpha=1.0,
        linewidth=1,
    )
end
plot!(
    p3d,
    result[:x][1, :],
    result[:x][2, :],
    result[:x][3, :],
    title="3D Trajectory Plot",
    xlabel="X Position",
    ylabel="Y Position",
    zlabel="Z Position",
    legend=false,
    linewidth=1,
    # marker=:circle,
    ylims=(params[:x_min][2], params[:x_max][2]),
    zlims=(params[:x_min][3], params[:x_max][3]),
)

plot_full_trajectory = false
if plot_full_trajectory
    plot!(
        p3d,
        result[:x_full][1, :],
        result[:x_full][2, :],
        result[:x_full][3, :],
        linewidth=10,
        # color=:inferno,
        # marker=:circle,
        ylims=(params[:x_min][2], params[:x_max][2]),
        zlims=(params[:x_min][3], params[:x_max][3]),
    )
end

# TODO: plot cost in log scale over nodes 

# Display the plot
display(p3d)


album_plot = false
if album_plot
    gr()

    # Create a new plot
    albumplt = plot(
        dpi=72,
        size=(2400, 2400),
        legend=false,
        # ticks = false,
        # showaxis=false,
        framestyle=:none,
        background=:black,
        foreground=:black,
        palette=palette(:managua, 1:n_iterations+1, rev=true),
    )

    # Plot trajectory history in x-z plane
    for (i, x_hist) in enumerate(result[:x_hist])
        plot!(
            albumplt,
            x_hist[1, :], x_hist[3, :],
            alpha=0.5,
            linewidth=4,
        )
    end

    # Plot final trajectory in x-z plane
    plot!(
        albumplt,
        result[:x][1, :], result[:x][3, :],
        linewidth=6,
    )

    # Save or display the plot
    savefig(albumplt, "minimalist_trajectory_xz.pdf")
    savefig(albumplt, "minimalist_trajectory_xz.png")
    display(albumplt)
end