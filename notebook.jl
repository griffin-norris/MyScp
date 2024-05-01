### A Pluto.jl notebook ###
# v0.19.41

using Markdown
using InteractiveUtils

# ╔═╡ ca1eadcf-dcf4-4650-8782-929fdd7cc6a2
using JuMP, ECOS, LinearAlgebra, Symbolics


# ╔═╡ dfc273e3-33e9-4168-a911-8e12c4fd7440
md"""
## Imports
"""

# ╔═╡ 557b7b8d-f894-408d-980a-2344596dcce7
md"""
## Utility functions
"""

# ╔═╡ ec71cdab-1a87-4e24-9d82-33aa7c6d5dfe
function qdcm(q::Vector{Float64})::Matrix{Float64} 
	q_norm = (q[1]^2 + q[2]^2 + q[3]^2 + q[4]^2)^0.5
	w, x, y, z = q/q_norm
	return [1-2*(y^2+z^2) 2*(x*y-z*w) 2*(x*z+y*w);
			2*(x*y+z*w) 1-2*(x^2+z^2) 2*(y*z-x*w);
			2*(x*z-y*w) 2*(y*z+x*w) 1-2*(x^2+y^2)]
end

# ╔═╡ feb186c0-55cd-420b-9315-4918d4b02904
function skew_symetric_matrix_quat(w::Vector{Float64})::Matrix{Float64}
	x, y, z = w
	return [0 -x -y -z;
		 	x 0 z -y;
		 	y -z 0 x;
		 	z y -x 0]
end

# ╔═╡ eb7f5dc1-1282-49ff-92b3-45f150686e3a
function skew_symetric_matrix(w::Vector{Float64})::Matrix{Float64}
    x, y, z = w
    return [0 -z y;
			z 0 -x;
			-y x 0]
end

# ╔═╡ 5e7a12e6-c202-4fc2-b310-28700f504aa7
md"""
## Dynamics
"""

# ╔═╡ 7b9d4b5d-7dc9-42e1-8e4e-a7d1cd9a8249
"""
    nonlinear_dynamics(t, state, control_slope, t_start, prop, params)

Calculate the time derivatives of the state variables for a nonlinear dynamic system with control inputs.

# Arguments
- `t::Float64`: Current time.
- `state::Vector{Float64}`: Current state vector containing the position, velocity, attitude (quaternion), and angular rate.
- `control_slope::Vector{Float64}`: Control input vector that includes both applied forces and torques.
- `t_start::Float64`: Start time of the control.
- `prop`: ...
- `params`: Dictionary containing parameters such as mass (`m`), gravitational acceleration (`g`), and inertia matrix (`J_b`).

# Returns
- `Vector{Float64}`: The time derivatives of the state variables, concatenated in a single vector. The vector is structured as follows:
  - Position derivative (`p_i_dot`), which is the current velocity.
  - Velocity derivative (`v_i_dot`), calculated based on the applied force, attitude, and gravitational forces.
  - Quaternion derivative (`q_bi_dot`), describing the attitude's rate of change due to angular velocity.
  - Angular rate derivative (`w_b_dot`), which accounts for the applied torque and gyroscopic effects.

# Example
```jl
# Define initial conditions and parameters
t = 0.0
state = [1.0, 2.0, 3.0, 0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0, 0.01, 0.02, 0.03]
control_slope = [0.5, 0.5, 0.5, 0.1, 0.1, 0.1]
t_start = 0.0
params = Dict("m" => 1.0, "g" => 9.81, "J_b" => [1.0, 1.0, 1.0])

# Compute derivatives
derivatives = nonlinear_dynamics(t, state, control_slope, t_start, prop, params)
```
"""
function nonlinear_dynamics(
	t::Float64,
	state::Vector{Float64},
	control_slope::Vector{Float64},
	t_start::Float64,
	prop,
	params,
)::Vector{Float64}
	# State
    r_i = state[1:3] # Position
    v_i = state[4:6] # Velocity
    q_bi = state[7:10] # Attitude (Quaternion)
    w_b = state[11:end] # Angular Rate

    u_mB = control[1:3] # Applied Force
    tau_i = control[4:end] # Applied Torque

    # Ensure that the quaternion is normalized
    q_norm = LinearAlgebra.norm(q_bi)
    q_bi = q_bi/q_norm

    p_i_dot = v_i
    v_i_dot = (1 / params["m"]) * qdcm(q_bi) * (u_mB) + [0, 0, params["g"]]

    q_bi_dot = 0.5 * skew_symetric_matrix_quat(w_b) * q_bi
    w_b_dot = LinearAlgebra.Diagonal(1/params["J_b"]) * (tau_i - skew_symetric_matrix(w_b) * LinearAlgebra.Diagonal(params["J_b"]) * w_b)
    return [p_i_dot; v_i_dot; q_bi_dot; w_b_dot]
end

# ╔═╡ a20dd084-9b66-457d-9d8c-c2848b69c9e5
function analytical_jacobians(
	params::Dict{String,Float64},
)
    Symbolics.@variables r[1:3,1] v[1:3,1] q[1:4,1] w[1:3,1]
    Symbolics.@variables f[1:3,1] tau[1:3,1]

    r_dot = v
    v_dot = (1 / params["m"]) * qdcm(q) * f + [0, 0, params["g"]]

    q_dot = 0.5 * skew_symmetric_matrix_quat(w) * q
    w_dot = LinearAlgebra.Diagonal(1 ./ params["J_b"]) * (tau - skew_symmetric_matrix(w) * LinearAlgebra.Diagonal(params["J_b"]) * w)

    state_dot = [r_dot; v_dot; q_dot; w_dot]
    A_expr = Symbolics.jacobian(state_dot, [r; v; q; w])
    B_expr = Symbolics.jacobian(state_dot, [f; tau])

    A = Symbolics.build_function(A_expr, [r; v; q; w; f; tau])
    B = Symbolics.build_function(B_expr, [r; v; q; w; f; tau])

    return A, B
end

# ╔═╡ 96e560ce-386c-4b57-92c7-4595b9d8804a
md"""
## Discretization
"""

# ╔═╡ fdc8208f-2e51-49aa-a398-570531b608e4
"""
    zoh_discretized_dynamics(dt, state, control, A, B)

Compute the discretized system and control matrices (`A_d` and `B_d`) using the Zero Order Hold (ZOH) method over a specified time step.

# Arguments
- `dt::Float64`: Time step for discretization.
- `state::Vector{Float64}`: Current state vector, which includes position `p`, velocity `v`, attitude `q` (quaternion), and angular velocity `w`.
- `control::Vector{Float64}`: Control input vector, split into forces `f` and torques `tau`.
- `A::Function`: Function that calculates the continuous-time system matrix `A` based on the state and control.
- `B::Function`: Function that calculates the continuous-time control matrix `B` based on the state and control.

# Returns
- `(Matrix{Float64}, Matrix{Float64})`: A tuple of matrices `(A_d, B_d)` where:
  - `A_d`: Discretized system matrix.
  - `B_d`: Discretized control matrix.

# Description
The function takes the continuous-time state dynamics and control matrices from the provided functions `A` and `B` and discretizes them using the Zero Order Hold (ZOH) assumption over the interval `dt`. This involves creating a compound matrix `Xi`, computing its matrix exponential, and extracting the discretized matrices for system dynamics (`A_d`) and control (`B_d`).

The matrix `Xi` is structured to consider the coupling between different parts of the state and control transformations, ensuring that the discrete-time equivalents are accurately derived for the simulation or control over the specified `dt`.

# Example
```jl
# Define state and control inputs
state = [1.0, 0.0, 0.0, 0.1, 0.1, 0.1, 1.0, 0.0, 0.0, 0.0, 0.01, 0.01, 0.01]
control = [0.5, 0.5, 0.5, 0.1, 0.1, 0.1]

# Define system and control matrix functions
A_func = (p, v, q, w, f, tau) -> ...  # Fill in with system-specific matrix
B_func = (p, v, q, w, f, tau) -> ...  # Fill in with control-specific matrix

# Compute discretized dynamics matrices
A_d, B_d = zoh_discretized_dynamics(0.1, state, control, A_func, B_func)
```
"""
function zoh_discretized_dynamics(
	dt::Float64,
	state::Vector{Float64},
	control::Vector{Float64},
	A::Function,
	B::Function,
)
    p = state[1:3]
    v = state[4:6]
    q = state[7:10]
    w = state[11:end]
    f = control[1:3]
    tau = control[4:end]
    
    A = A(p, v, q, w, f, tau)
    B = B(p, v, q, w, f, tau)
    
    n = size(A, 1)
    m = size(B, 2)
    
    Xi = zeros(Float64, n*3 + m, n*3 + m)
    Xi[1:n, 1:n] = A
    Xi[n+1:2n, n+1:2n] = -A'
    Xi[2n+1:3n, 2n+1:3n] = A
    Xi[2n+1:3n, 3n+1:end] = B
    
    Y = exp(Xi * dt)
    A_d = Y[1:n, 1:n]
    B_d = Y[2n+1:3n, 3n+1:end]
    return A_d, B_d
end

# ╔═╡ bb115df2-9f5b-402d-9a25-6c096d913d63
"""
    dVdt(t, V, control_current, control_next, A, B, params)

Compute the derivative of the state and control trajectory matrices with respect to time 
using a given system model and control input, under different discretization types.

# Arguments
- `t::Float64`: Current time.
- `V::Vector{Float64}`: Vector containing current state variables and trajectory matrices.
- `control_current::Vector{Float64}`: Current control input vector.
- `control_next::Vector{Float64}`: Next control input vector, used for interpolation in first-order hold (FOH) discretization.
- `A::Function`: Function that returns the system matrix `A` based on the current state and control.
- `B::Function`: Function that returns the control matrix `B` based on the current state and control.
- `params`: Dictionary containing parameters like the number of states (`n_states`), the number of controls (`n_controls`), the discretization type (`dis_type`), and the time step size (`dt_ss`).

# Returns
- `Vector{Float64}`: The derivative of the state trajectory matrix `V`. This includes:
  - The state derivative from the dynamics function.
  - The reshaped derivatives of state-to-state, state-to-control, and control-to-control matrices, adjusted for the current and next control inputs.

# Details
The function handles different types of discretization for integrating control inputs:
- Zero-order hold (ZOH): Assumes control inputs are constant over the time step.
- First-order hold (FOH): Linearly interpolates between `control_current` and `control_next` based on the time `t`.

This approach adjusts the state and control matrices `A` and `B` to accommodate for the dynamics provided by the `nonlinear_dynamics` function and the specific discretization strategy. 

# Example
```jl
# Define parameters and initial conditions
t = 0.0
V = [0.0, 1.0, 2.0, ..., 0.1]  # Example vector with appropriate dimensions
control_current = [1.0, 2.0, 3.0]
control_next = [1.1, 2.1, 3.1]
params = Dict("n_states" => 13, "n_controls" => 6, "dis_type" => "ZOH", "dt_ss" => 0.1)

# Define system and control matrices functions
A = (p, v, q, w, f, tau) -> ...  # Define based on the system
B = (p, v, q, w, f, tau) -> ...  # Define based on the system

# Compute derivatives
derivatives = dVdt(t, V, control_current, control_next, A, B, params)
```
"""
function dVdt(
	t::Float64,
	V::Vector{Float64},
	control_current::Vector{Float64},
	control_next::Vector{Float64},
	A::Function,
	B::Function,
	params,
)::Vector{Float64}
	n_x = params["n_states"]
    n_u = params["n_controls"]
	
	i0 = 1
	i1 = n_x
	i2 = i1 + n_x * n_x
	i3 = i2 + n_x * n_u
	i4 = i3 + n_x * n_u
	i5 = i4 + n_x

	x = V[i0:i1]

	if params["dis_type"] == "ZOH"
		beta = 0
	elseif params["dis_type"] == "FOH"
		beta = t / params["dt_ss"]
	end
	alpha = 1 - beta

	u = control_current + beta * (control_next - control_current)

	p = x[1:3]
    v = x[4:6]
    q = x[7:10]
    w = x[11:end]
    f = u[1:3]
    tau = u[4:end]

	A_subs = A(p, v, q, w, f, tau)
	B_subs = B(p, v, q, w, f, tau)
	f_subs = nonlinear_dynamics(t, x, u, nothing, nothing, true, params)

	z_t = f_subs - A_subs * x - B_subs * u

	dVdt = [
		f_subs
		vec(A_subs * reshape(V[i1:i2], n_x, n_x))
		A_subs * reshape(V[i2:i3], n_x, n_u) + B_subs .* alpha
		A_subs * reshape(V[i3:i4], n_x, n_u) + B_subs .* beta
		A_subs * V[i4:i5] .+ z_t
	]

	return dVdt
end

# ╔═╡ 79facf78-a456-4b29-b18d-ae1dc23c0e6e
function calculate_discretization(
	x,
	u,
	A,
	B,
	params
)
	n_x = params["n_states"]
    n_u = params["n_controls"]
	
	i0 = 1
	i1 = n_x
	i2 = i1 + n_x * n_x
	i3 = i2 + n_x * n_u
	i4 = i3 + n_x * n_u
	i5 = i4 + n_x
	
	V0 = zeros(i5)
	V0 .= reshape(I(n_x), n_x*n_x, 1)

	f_bar = zeros((n_x, params["n"]-1))
    A_bar = zeros((n_x * n_x, params["n"]-1))
    B_bar = zeros((n_x * n_u, params["n"]-1))
    C_bar = zeros((n_x * n_u, params["n"]-1))
    z_bar = zeros((n_x, params["n"]-1))

	for k in 1:(params["n"] - 1)
		V0[i0:i1] .= x[:,k]
		dt = params["dt_ss"]
		
		k1 = dVdt(t, V0, u[:, k], u[:, k+1], A, B, params)
		k2 = dVdt(t + dt / 2, V0 + dt / 2 * k1, u[:, k], u[:, k+1], A, B, params)
		k3 = dVdt(t + dt / 2, V0 + dt / 2 * k2, u[:, k], u[:, k+1], A, B, params)
		k4 = dVdt(t + dt, V0 + dt * k3, u[:, k], u[:, k+1], A, B, params)
		V = V0 + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

		f_bar[:, k] .= V[i0:i1]
		Phi = reshape(V[i1:i2], n_x, n_x)
		A_bar[:, k] = vec(Phi)
		B_bar[:, k] .= V[i2:i3]
		C_bar[:, k] .= V[i3:i4]
		z_bar[:, k] .= V[i4:i5]
	end
end

# ╔═╡ b522641d-cfb5-4dac-abf0-28ffe038ed14
md"""
## Integration
"""

# ╔═╡ 299dd5b6-a211-4c2a-9e65-b9a80664cd10
md"""
## PTR -- Penalized Trust Region
"""

# ╔═╡ 0480ce3a-8763-4ece-8e9b-b931874e9f38
md"""
## CTCS -- Continuous-Time Constraint Satisfaction
"""

# ╔═╡ cdf8230b-5355-4ba9-924b-f2919ebec026
function OCP(
	params,
)
	mdl = JuMP.Model(ECOS.Optimizer)
	A_prop = ones(n_x*n_x, params["n"])
	B_prop = ones(n_x*n_u, params["n"])
	C_prop = ones(n_x*n_u, params["n"])
	z_prop = ones(n_x, params["n"])

	# Parameters
	# NOTE: the `Parameter` set must be set externally
	JuMP.@variables(
		mdl,
		begin
			# Trust region weight
			w_tr in Parameter(1)
		end
	)

	# State
	JuMP.@variables(
		mdl,
		begin
			x[1:params["n_states"], 1:params["n"]], (start = 0)
	        dx[1:params["n_states"], 1:params["n"]], (start = 0)
	        x_bar[1:params["n_states"], 1:params["n"]], (start = 0)
		end
	)
	
	# Control
	JuMP.@variables(
		mdl,
		begin
	        u[1:params["n_controls"], 1:params["n"]], (start = 0)
	        du[1:params["n_controls"], 1:params["n"]], (start = 0)
			u_bar[1:params["n_controls"], 1:params["n"]], (start = 0)
		end
	)
	
	# Slack
	JuMP.@variables(
		mdl,
		begin
	        # Virtual control for linearized augmented dynamics constraints
	        nu[1:params["n_states"], 1:params["n"] - 1], (start = 0)
		end
	)

	# Scaling
	S_x = params["S_x"]
	c_x = params["c_x"]
	S_u = params["S_u"]
	c_u = params["c_u"]

	x_nonscaled = []
	u_nonscaled = []

	for k in (1:params["n"])
		push!(x_nonscaled, S_x * x[:,k] + c_x)
		push!(u_nonscaled, S_u * u[:,k] + c_u)
	end

	# Boundary Constraints
	n_states = params["n_states"]
	n_obs = params["n_obs"]
	JuMP.@constraints(
		mdl, 
		begin
			x_nonscaled[1][1:n_states-n_obs] .== params["initial_state"] # Initial condition
			x_nonscaled[end][1:n_states-n_obs] .== params["final_state"] # Terminal Condition
			x_nonscaled[1] - x_bar[:, 1] - dx[:, 1] .== 0 # Initial state error
			u_nonscaled[1] - u_bar[:, 1] - du[:, 1] .== 0 # Initial control error
		end
	)

	# Cost initialization
	cost_components = []

	# SOCP variable initialization for cost
	JuMP.@variables(
		mdl,
		begin
			t_fuel[1:params["n"]]
			t_tr[1:params["n"]]
			t_vc[1:n_x, 1:params["n"]]
		end
	)

	# Loop
	for k in (2:params["n"])
		JuMP.@constraints(
			mdl,
			begin
				inv(S_x) * (x_nonscaled[k] - x_bar[:, k] - dx[:, k]) .== 0
				inv(S_u) * (u_nonscaled[k] - u_bar[:, k] - du[:, k]) .== 0
			end
		)

		# Running cost
        JuMP.@constraint(
			mdl, 
			[
				t_fuel[k]; 
				inv(S_u) * u_nonscaled[k][:]
			] in SecondOrderCone()
		)

		# Trust region cost
        JuMP.@constraint(
			mdl,
			[
				t_tr[k]; 
				[inv(S_x) zeros(n_x, n_u); zeros(n_u, n_x) inv(S_u)] * [dx[:,k]; du[:,k]]
			] in SecondOrderCone()
		)

		# Virtual control cost
		# λ_vc ||ν||₁
        JuMP.@constraints(
			mdl, 
			begin
				t_vc[:,k] >= -nu[:,k]
				t_vc[:,k] >= nu[:,k]
			end
		)

		# Weighted sum of cost components
        cost = (
			params["lam_fuel"] * t_fuel[k] 
			+ w_tr * t_tr[k] 
			+ params["lam_vc"] * sum(t_vc[:,k])
		)
        push!(cost_components, cost)

		# Dynamics constraints
		Ak = reshape(A_prop[:, k-1], n_x, n_x)
		Bk = reshape(B_prop[:, k-1], n_x, n_u)
		Ck = reshape(C_prop[:, k-1], n_x, n_u)
		for i in 1:n_x
			JuMP.@constraint(
				mdl, 
				x_nonscaled[k][i] == sum([
					sum(Ak[i,j] * x_nonscaled[k-1][j] for j in 1:n_x),
					sum(Bk[i,j] * u_nonscaled[k-1][j] for j in 1:n_u),
					sum(Ck[i,j] * u_nonscaled[k][j] for j in 1:n_u),
					z_prop[i, k-1],
					nu[i, k-1],
				])
			)
		end

		# State & control set constraints
		JuMP.@constraints(
			mdl,
			begin
				x_nonscaled[k] <= params["max_state"]
				x_nonscaled[k] >= params["min_state"]
				u_nonscaled[k-1] <= params["max_control"]
				u_nonscaled[k-1] >= params["min_control"]
			end
		)
		
	end

	JuMP.@objective(mdl, Min, sum(cost_components))
	return mdl
end

# ╔═╡ 57b1dbeb-4365-4a7d-9f9b-ddf18067cac5
params = Dict(
	"n_states" => 3,
	"n_controls" => 2,
	"n" => 10,
	"n_obs" => 2,
	"S_x" => I(3),
	"c_x" => zeros(3),
	"S_u" => I(2),
	"c_u" => zeros(2),
	"initial_state" => zeros(3),
	"final_state" => zeros(3),
	"lam_fuel" => 1.0,
)

# ╔═╡ 7553603d-343b-4df1-955d-044736ba61e0
mdl = OCP(params)

# ╔═╡ 4c4e04ba-1ab4-43d9-858d-b2108ab596a2
function PTR_subproblem(
	x_bar,
	u_bar,
	A,
	B,
	obstacles,
	model,
	params,
)
    J_vb_vec = []
    J_vc_vec = []
    J_vc_ctcs_vec = []
    J_tr_vec = []

    A_d = zeros(params["n_states"], params["n_states"] * params["n"] - 1)
    B_d = zeros(params["n_states"], params["n_controls"] * params["n"] - 1)
    x_prop = zeros(params["n_states"], params["n"] - 1)

    JuMP.set_value(model[:x_bar], x_bar)
    JuMP.set_value(model[:u_bar], u_bar)

    i_obs = 0

    if params["dis_exact"]
        A_bar, B_bar, C_bar, z_bar = calculate_discretization(x_bar, u_bar, A, B, obstacles, params)
        JuMP.set_value(model[:A_d], A_bar)
        JuMP.set_value(model[:B_d], B_bar)
        JuMP.set_value(model[:C_d], C_bar)
        JuMP.set_value(model[:z_d], z_bar)
    else
        for k in 2:params["n"]
            A_d[:, params["n_states"] * (k - 2) + 1 : params["n_states"] * (k - 1)], B_d[:, params["n_controls"] * (k - 2) + 1 : params["n_controls"] * (k - 1)] = zoh_discretized_dynamics(params["dt_ss"], x_bar[:, k - 1], u_bar[:, k - 1], A, B)
            x_prop[:, k - 1] = simulate_nonlinear(x_bar[:, k - 1], u_bar[:, k - 1], params, params["dt_ss"])
        end
        JuMP.set_value(model[:A_d], A_d)
        JuMP.set_value(model[:B_d], B_d)
        JuMP.set_value(model[:x_prop], x_prop)
    end

    if !params["ctcs"]
        for obs in obstacles
            JuMP.set_value(model[Symbol("g_bar_obs_", i_obs)], obs.g_bar_obs(x_bar[1:3, 2:end]))
            JuMP.set_value(model[Symbol("grad_g_bar_obs_", i_obs)], obs.grad_g_bar_obs(x_bar[1:3, 2:end]))
            i_obs += 1
        end
    end

    JuMP.set_value(model[:w_tr], params["w_tr"])

    optimize!(model)

    x = params["S_x"] * value.(model[:x]) .+ params["c_x"]
    u = params["S_u"] * value.(model[:u]) .+ params["c_u"]

    push!(J_tr_vec, norm(inv(params["S_x"]) * (x[:, 1] - x_bar[:, 1])))
    for k in 2:params["n"]
        if !params["dis_exact"]
            push!(J_vc_vec, sum(abs.(x[:, k] - x_prop[:, k - 1] - A_d[:, 13 * (k - 2) + 1 : 13 * (k - 1)] * (x[:, k - 1] - x_bar[:, k - 1]) - B_d[:, 6 * (k - 2) + 1 : 6 * (k - 1)] * (u[:, k - 1] - u_bar[:, k - 1]))))
        end
        push!(J_tr_vec, norm(inv(blockdiag(params["S_x"], params["S_u"])) * (hcat(x[:, k], u[:, k - 1]) - hcat(x_bar[:, k], u_bar[:, k - 1]))))

        J_vb = 0
        if !params["ctcs"]
            for obs in obstacles
                J_vb += max(0, (obs.g_bar_obs(x_bar[1:3, k]) + obs.grad_g_bar_obs(x_bar[1:3, k])' * (x[1:3, k] - x_bar[1:3, k]))[1])
            end
        end
        push!(J_vb_vec, J_vb)
        if params["dis_exact"]
            push!(J_vc_ctcs_vec, sum(abs.(value.(model[:nu])[params["n_states"] - params["n_obs"] + 1:end, k - 1])))
            push!(J_vc_vec, sum(abs.(value.(model[:nu])[1:params["n_states"] - params["n_obs"], k - 1])))
        end
    end
	return x, u, objective_value(model), J_vb_vec, J_vc_vec, J_tr_vec, J_vc_ctcs_vec
end


# ╔═╡ 4073f275-589c-465c-a8c2-2b3f99a2c27b
function PTR_main(
	params,
)
    J_vb = 1e2
    J_vc = 1e2
    J_tr = 1e2

    prob = OCP(params)  # Initialize the problem
    A, B = analytical_jacobians(params)  # Analytical Jacobians

    x_bar, u_bar = straight_line_init(
		params["initial_state"], 
		params["initial_control"], 
		params["final_state"], 
		params["n"], 
		params
	)

    scp_trajs = [x_bar]
    scp_controls = [u_bar]

    obstacles = []
    for i in 1:params["n_obs"]
        push!(obstacles, Ellipsoidal_Obstacle(params["obstacle_centers"][i]))
    end

    println("Iter | J_total |  J_tr   |  J_vb   |  J_vc   | J_vc_ctcs ")
    println("--------------------------------------------------------")

    k = 0
    while k <= params["k_max"] && ((J_tr >= params["ep_tr"]) || (J_vb >= params["ep_vb"]) || (J_vc >= params["ep_vc"]))
        x, u, J_total, J_vb_vec, J_vc_vec, J_tr_vec, J_vc_ctcs_vec = PTR_subproblem(x_bar, u_bar, A, B, obstacles, prob, params)

        x_bar = x
        u_bar = u

        J_tr = sum(J_tr_vec)
        J_vb = sum(J_vb_vec)
        J_vc = sum(J_vc_vec)
        J_vc_ctcs = sum(J_vc_ctcs_vec)
        push!(scp_trajs, x)
        push!(scp_controls, u)

        params["w_tr"] *= 1.1

        println(
			lpad(k, 4), " | ",
            string(round(params[:J_total], digits=1), "e"), " | ",
            string(round(params[:J_tr], digits=1), "e"), " | ",
            string(round(params[:J_vb], digits=1), "e"), " | ",
            string(round(params[:J_vc], digits=1), "e"), " | ",
            string(round(params[:J_vc_ctcs], digits=1), "e")
		)
        k += 1
    end

    x_full = simulate_nonlinear(x[1:end - params["n_obs"], 1], u, params, params["dt_sim"])

    scp_trajs_interp = scp_traj_interp(scp_trajs, params)

    result = Dict(
        :times => 0:params["dt_sim"]:params["total_time"],
        :drone_state => x_full,
        :drone_positions => x_full[:, 1:3],
        :drone_attitudes => x_full[:, 6:10],
        :drone_forces => u[:, 1:3],
        :drone_controls => u,
        :scp_trajs => scp_trajs,
        :scp_controls => scp_controls,
        :obstacles => obstacles,
        :scp_interp => scp_trajs_interp,
        :J_tr_vec => J_tr_vec,
        :J_vb_vec => J_vb_vec,
        :J_vc_vec => J_vc_vec,
        :J_vc_ctcs_vec => J_vc_ctcs_vec
    )
    return result
end


# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
ECOS = "e2685f51-7e38-5353-a97d-a921fd2c8199"
JuMP = "4076af6c-e467-56ae-b986-b466b2749572"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Symbolics = "0c5d862f-8b57-4792-8d23-62f2024744c7"

[compat]
ECOS = "~1.1.2"
JuMP = "~1.21.1"
Symbolics = "~5.14.1"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.9.3"
manifest_format = "2.0"
project_hash = "ab177a5f66e2d62b1e88565f3a2d1f2863bdd578"

[[deps.ADTypes]]
git-tree-sha1 = "016833eb52ba2d6bea9fcb50ca295980e728ee24"
uuid = "47edcb42-4c32-4615-8424-f2b9edc5f35b"
version = "0.2.7"

[[deps.AbstractAlgebra]]
deps = ["GroupsCore", "InteractiveUtils", "LinearAlgebra", "MacroTools", "Preferences", "Random", "RandomExtensions", "SparseArrays", "Test"]
git-tree-sha1 = "d7832de8cf7af26abac741f10372080ac6cb73df"
uuid = "c3fe647b-3220-5bb0-a1ea-a7954cac585d"
version = "0.34.7"

[[deps.AbstractTrees]]
git-tree-sha1 = "2d9c9a55f9c93e8887ad391fbae72f8ef55e1177"
uuid = "1520ce14-60c1-5f80-bbc7-55ef81b5835c"
version = "0.4.5"

[[deps.Adapt]]
deps = ["LinearAlgebra", "Requires"]
git-tree-sha1 = "cde29ddf7e5726c9fb511f340244ea3481267608"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.7.2"
weakdeps = ["StaticArrays"]

    [deps.Adapt.extensions]
    AdaptStaticArraysExt = "StaticArrays"

[[deps.AliasTables]]
deps = ["Random"]
git-tree-sha1 = "07591db28451b3e45f4c0088a2d5e986ae5aa92d"
uuid = "66dad0bd-aa9a-41b7-9441-69ab47430ed8"
version = "1.1.1"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.ArrayInterface]]
deps = ["Adapt", "LinearAlgebra", "Requires", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "c5aeb516a84459e0318a02507d2261edad97eb75"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "7.7.1"

    [deps.ArrayInterface.extensions]
    ArrayInterfaceBandedMatricesExt = "BandedMatrices"
    ArrayInterfaceBlockBandedMatricesExt = "BlockBandedMatrices"
    ArrayInterfaceCUDAExt = "CUDA"
    ArrayInterfaceGPUArraysCoreExt = "GPUArraysCore"
    ArrayInterfaceStaticArraysCoreExt = "StaticArraysCore"
    ArrayInterfaceTrackerExt = "Tracker"

    [deps.ArrayInterface.weakdeps]
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"
    BlockBandedMatrices = "ffab5731-97b5-5995-9138-79e8c1846df0"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    GPUArraysCore = "46192b85-c4d5-4398-a991-12ede77f4527"
    StaticArraysCore = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.BenchmarkTools]]
deps = ["JSON", "Logging", "Printf", "Profile", "Statistics", "UUIDs"]
git-tree-sha1 = "f1dff6729bc61f4d49e140da1af55dcd1ac97b2f"
uuid = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
version = "1.5.0"

[[deps.Bijections]]
git-tree-sha1 = "c9b163bd832e023571e86d0b90d9de92a9879088"
uuid = "e2ed5e7c-b2de-5872-ae92-c73ca462fb04"
version = "0.1.6"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9e2a6b69137e6969bab0152632dcb3bc108c8bdd"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+1"

[[deps.CEnum]]
git-tree-sha1 = "389ad5c84de1ae7cf0e28e381131c98ea87d54fc"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.5.0"

[[deps.Calculus]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f641eb0a4f00c343bbc32346e1217b86f3ce9dad"
uuid = "49dc2e85-a5d0-5ad3-a950-438e2897f1b9"
version = "0.5.1"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra"]
git-tree-sha1 = "575cd02e080939a33b6df6c5853d14924c08e35b"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.23.0"
weakdeps = ["SparseArrays"]

    [deps.ChainRulesCore.extensions]
    ChainRulesCoreSparseArraysExt = "SparseArrays"

[[deps.CodecBzip2]]
deps = ["Bzip2_jll", "Libdl", "TranscodingStreams"]
git-tree-sha1 = "9b1ca1aa6ce3f71b3d1840c538a8210a043625eb"
uuid = "523fee87-0ab8-5b00-afb7-3ecf72e48cfd"
version = "0.8.2"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "59939d8a997469ee05c4b4944560a820f9ba0d73"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.4"

[[deps.Combinatorics]]
git-tree-sha1 = "08c8b6831dc00bfea825826be0bc8336fc369860"
uuid = "861a8166-3701-5b0c-9a16-15d98fcdc6aa"
version = "1.0.2"

[[deps.CommonSolve]]
git-tree-sha1 = "0eee5eb66b1cf62cd6ad1b460238e60e4b09400c"
uuid = "38540f10-b2f7-11e9-35d8-d573e4eb0ff2"
version = "0.2.4"

[[deps.CommonSubexpressions]]
deps = ["MacroTools", "Test"]
git-tree-sha1 = "7b8a93dba8af7e3b42fecabf646260105ac373f7"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.0"

[[deps.Compat]]
deps = ["TOML", "UUIDs"]
git-tree-sha1 = "c955881e3c981181362ae4088b35995446298b80"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.14.0"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.0.5+0"

[[deps.CompositeTypes]]
git-tree-sha1 = "bce26c3dab336582805503bed209faab1c279768"
uuid = "b152e2b5-7a66-4b01-a709-34e65c35f657"
version = "0.1.4"

[[deps.ConstructionBase]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "260fd2400ed2dab602a7c15cf10c1933c59930a2"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.5.5"
weakdeps = ["IntervalSets", "StaticArrays"]

    [deps.ConstructionBase.extensions]
    ConstructionBaseIntervalSetsExt = "IntervalSets"
    ConstructionBaseStaticArraysExt = "StaticArrays"

[[deps.Crayons]]
git-tree-sha1 = "249fe38abf76d48563e2f4556bebd215aa317e15"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.1.1"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "1d0a14036acb104d9e89698bd408f63ab58cdc82"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.20"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DiffResults]]
deps = ["StaticArraysCore"]
git-tree-sha1 = "782dd5f4561f5d267313f23853baaaa4c52ea621"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.1.0"

[[deps.DiffRules]]
deps = ["IrrationalConstants", "LogExpFunctions", "NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "23163d55f885173722d1e4cf0f6110cdbaf7e272"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.15.1"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.Distributions]]
deps = ["AliasTables", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SpecialFunctions", "Statistics", "StatsAPI", "StatsBase", "StatsFuns"]
git-tree-sha1 = "22c595ca4146c07b16bcf9c8bea86f731f7109d2"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.108"

    [deps.Distributions.extensions]
    DistributionsChainRulesCoreExt = "ChainRulesCore"
    DistributionsDensityInterfaceExt = "DensityInterface"
    DistributionsTestExt = "Test"

    [deps.Distributions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DensityInterface = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "2fb1e02f2b635d0845df5d7c167fec4dd739b00d"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.3"

[[deps.DomainSets]]
deps = ["CompositeTypes", "IntervalSets", "LinearAlgebra", "Random", "StaticArrays", "Statistics"]
git-tree-sha1 = "51b4b84d33ec5e0955b55ff4b748b99ce2c3faa9"
uuid = "5b8099bc-c8ec-5219-889f-1d9e522a28bf"
version = "0.6.7"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.DualNumbers]]
deps = ["Calculus", "NaNMath", "SpecialFunctions"]
git-tree-sha1 = "5837a837389fccf076445fce071c8ddaea35a566"
uuid = "fa6b7ba4-c1ee-5f82-b5fc-ecf0adba8f74"
version = "0.6.8"

[[deps.DynamicPolynomials]]
deps = ["Future", "LinearAlgebra", "MultivariatePolynomials", "MutableArithmetics", "Pkg", "Reexport", "Test"]
git-tree-sha1 = "0c056035f7de73b203a5295a22137f96fc32ad46"
uuid = "7c1d4256-1411-5781-91ec-d7bc3513ac07"
version = "0.5.6"

[[deps.ECOS]]
deps = ["CEnum", "ECOS_jll", "MathOptInterface"]
git-tree-sha1 = "ea9f95d78d94af14e0f50973267c9c2209338079"
uuid = "e2685f51-7e38-5353-a97d-a921fd2c8199"
version = "1.1.2"

[[deps.ECOS_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "5f84034ddd642cf595e57d46ea2f085321c260e4"
uuid = "c2c64177-6a8e-5dca-99a7-64895ad7445f"
version = "200.0.800+0"

[[deps.EnumX]]
git-tree-sha1 = "bdb1942cd4c45e3c678fd11569d5cccd80976237"
uuid = "4e289a0a-7415-4d19-859d-a7e5c4648b56"
version = "1.0.4"

[[deps.ExprTools]]
git-tree-sha1 = "27415f162e6028e81c72b82ef756bf321213b6ec"
uuid = "e2ba6199-217a-4e67-a87a-7c52f15ade04"
version = "0.1.10"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FillArrays]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "57f08d5665e76397e96b168f9acc12ab17c84a68"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.10.2"
weakdeps = ["PDMats", "SparseArrays", "Statistics"]

    [deps.FillArrays.extensions]
    FillArraysPDMatsExt = "PDMats"
    FillArraysSparseArraysExt = "SparseArrays"
    FillArraysStatisticsExt = "Statistics"

[[deps.Format]]
git-tree-sha1 = "9c68794ef81b08086aeb32eeaf33531668d5f5fc"
uuid = "1fa38f19-a742-5d3f-a2b9-30dd87b9d5f8"
version = "1.3.7"

[[deps.ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions"]
git-tree-sha1 = "cf0fe81336da9fb90944683b8c41984b08793dad"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "0.10.36"
weakdeps = ["StaticArrays"]

    [deps.ForwardDiff.extensions]
    ForwardDiffStaticArraysExt = "StaticArrays"

[[deps.FunctionWrappers]]
git-tree-sha1 = "d62485945ce5ae9c0c48f124a84998d755bae00e"
uuid = "069b7b12-0de2-55c6-9aab-29f3d0a68a2e"
version = "1.1.3"

[[deps.FunctionWrappersWrappers]]
deps = ["FunctionWrappers"]
git-tree-sha1 = "b104d487b34566608f8b4e1c39fb0b10aa279ff8"
uuid = "77dc65aa-8811-40c2-897b-53d922fa7daf"
version = "0.1.3"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[deps.GPUArraysCore]]
deps = ["Adapt"]
git-tree-sha1 = "2d6ca471a6c7b536127afccfa7564b5b39227fe0"
uuid = "46192b85-c4d5-4398-a991-12ede77f4527"
version = "0.1.5"

[[deps.Groebner]]
deps = ["AbstractAlgebra", "Combinatorics", "ExprTools", "Logging", "MultivariatePolynomials", "PrecompileTools", "PrettyTables", "Primes", "Printf", "Random", "SIMD", "TimerOutputs"]
git-tree-sha1 = "6b505ef15e55bdc5bb3ddbcfebdff1c9e67081e8"
uuid = "0b43b601-686d-58a3-8a1c-6623616c7cd4"
version = "0.5.1"

[[deps.GroupsCore]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "6df9cd6ee79fc59feab33f63a1b3c9e95e2461d5"
uuid = "d5909c97-4eac-4ecc-a3dc-fdd0858a4120"
version = "0.4.2"

[[deps.HypergeometricFunctions]]
deps = ["DualNumbers", "LinearAlgebra", "OpenLibm_jll", "SpecialFunctions"]
git-tree-sha1 = "f218fe3736ddf977e0e772bc9a586b2383da2685"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.23"

[[deps.IfElse]]
git-tree-sha1 = "debdd00ffef04665ccbb3e150747a77560e8fad1"
uuid = "615f187c-cbe4-4ef1-ba3b-2fcf58d6d173"
version = "0.1.1"

[[deps.IntegerMathUtils]]
git-tree-sha1 = "b8ffb903da9f7b8cf695a8bead8e01814aa24b30"
uuid = "18e54dd8-cb9d-406c-a71d-865a43cbb235"
version = "0.1.2"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.IntervalSets]]
git-tree-sha1 = "dba9ddf07f77f60450fe5d2e2beb9854d9a49bd0"
uuid = "8197267c-284f-5f27-9208-e0e47529a953"
version = "0.7.10"
weakdeps = ["Random", "RecipesBase", "Statistics"]

    [deps.IntervalSets.extensions]
    IntervalSetsRandomExt = "Random"
    IntervalSetsRecipesBaseExt = "RecipesBase"
    IntervalSetsStatisticsExt = "Statistics"

[[deps.IrrationalConstants]]
git-tree-sha1 = "630b497eafcc20001bba38a4651b327dcfc491d2"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.2"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "7e5d6779a1e09a36db2a7b6cff50942a0a7d0fca"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.5.0"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.JuMP]]
deps = ["LinearAlgebra", "MacroTools", "MathOptInterface", "MutableArithmetics", "OrderedCollections", "PrecompileTools", "Printf", "SparseArrays"]
git-tree-sha1 = "07385c772da34d91fc55d6c704b6224302082ba0"
uuid = "4076af6c-e467-56ae-b986-b466b2749572"
version = "1.21.1"

    [deps.JuMP.extensions]
    JuMPDimensionalDataExt = "DimensionalData"

    [deps.JuMP.weakdeps]
    DimensionalData = "0703355e-b756-11e9-17c0-8b28908087d0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "50901ebc375ed41dbf8058da26f9de442febbbec"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.1"

[[deps.LabelledArrays]]
deps = ["ArrayInterface", "ChainRulesCore", "ForwardDiff", "LinearAlgebra", "MacroTools", "PreallocationTools", "RecursiveArrayTools", "StaticArrays"]
git-tree-sha1 = "d1f981fba6eb3ec393eede4821bca3f2b7592cd4"
uuid = "2ee39098-c373-598a-b85f-a56591580800"
version = "1.15.1"

[[deps.LambertW]]
git-tree-sha1 = "c5ffc834de5d61d00d2b0e18c96267cffc21f648"
uuid = "984bce1d-4616-540c-a9ee-88d1112d94c9"
version = "0.4.6"

[[deps.Latexify]]
deps = ["Format", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "OrderedCollections", "Requires"]
git-tree-sha1 = "e0b5cd21dc1b44ec6e64f351976f961e6f31d6c4"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.16.3"

    [deps.Latexify.extensions]
    DataFramesExt = "DataFrames"
    SymEngineExt = "SymEngine"

    [deps.Latexify.weakdeps]
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    SymEngine = "123dc426-2d89-5057-bbad-38513e3affd8"

[[deps.Lazy]]
deps = ["MacroTools"]
git-tree-sha1 = "1370f8202dac30758f3c345f9909b97f53d87d3f"
uuid = "50d2b5c4-7a5e-59d5-8109-a42b560f39c0"
version = "0.15.1"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.3"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "7.84.0+0"

[[deps.LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.10.2+0"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "18144f3e9cbe9b15b070288eef858f71b291ce37"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.27"

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

    [deps.LogExpFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ChangesOfVariables = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "2fa9ee3e63fd3a4f7a9a4f4744a52f4856de82df"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.13"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MathOptInterface]]
deps = ["BenchmarkTools", "CodecBzip2", "CodecZlib", "DataStructures", "ForwardDiff", "JSON", "LinearAlgebra", "MutableArithmetics", "NaNMath", "OrderedCollections", "PrecompileTools", "Printf", "SparseArrays", "SpecialFunctions", "Test", "Unicode"]
git-tree-sha1 = "9cc5acd6b76174da7503d1de3a6f8cf639b6e5cb"
uuid = "b8f27783-ece8-5eb3-8dc8-9495eed66fee"
version = "1.29.0"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.2+0"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "ec4f7fbeab05d7747bdf98eb74d130a2a2ed298d"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.2.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2022.10.11"

[[deps.MultivariatePolynomials]]
deps = ["ChainRulesCore", "DataStructures", "LinearAlgebra", "MutableArithmetics"]
git-tree-sha1 = "769c9175942d91ed9b83fa929eee4fe6a1d128ad"
uuid = "102ac46a-7ee4-5c85-9060-abc95bfdeaa3"
version = "0.5.4"

[[deps.MutableArithmetics]]
deps = ["LinearAlgebra", "SparseArrays", "Test"]
git-tree-sha1 = "a3589efe0005fc4718775d8641b2de9060d23f73"
uuid = "d8a4904e-b15c-11e9-3269-09a3773c0cb0"
version = "1.4.4"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "0877504529a3e5c3343c6f8b4c0381e57e4387e4"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.2"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.21+4"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "dfdf5519f235516220579f949664f1bf44e741c5"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.6.3"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "949347156c25054de2db3b166c52ac4728cbad65"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.31"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "8489905bcdbcfac64d1daa51ca07c0d8f0283821"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.1"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.9.2"

[[deps.PreallocationTools]]
deps = ["Adapt", "ArrayInterface", "ForwardDiff", "Requires"]
git-tree-sha1 = "01ac95fca7daabe77a9cb705862bd87016af9ddb"
uuid = "d236fae5-4411-538c-8e31-a6e3d9e00b46"
version = "0.4.13"

    [deps.PreallocationTools.extensions]
    PreallocationToolsReverseDiffExt = "ReverseDiff"

    [deps.PreallocationTools.weakdeps]
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "5aa36f7049a63a1528fe8f7c3f2113413ffd4e1f"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.2.1"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "9306f6085165d270f7e3db02af26a400d580f5c6"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.4.3"

[[deps.PrettyTables]]
deps = ["Crayons", "LaTeXStrings", "Markdown", "PrecompileTools", "Printf", "Reexport", "StringManipulation", "Tables"]
git-tree-sha1 = "88b895d13d53b5577fd53379d913b9ab9ac82660"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "2.3.1"

[[deps.Primes]]
deps = ["IntegerMathUtils"]
git-tree-sha1 = "cb420f77dc474d23ee47ca8d14c90810cafe69e7"
uuid = "27ebfcd6-29c5-5fa9-bf4b-fb8fc14df3ae"
version = "0.5.6"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.Profile]]
deps = ["Printf"]
uuid = "9abbd945-dff8-562f-b5e8-e1ebf5ef1b79"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "9b23c31e76e333e6fb4c1595ae6afa74966a729e"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.9.4"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.RandomExtensions]]
deps = ["Random", "SparseArrays"]
git-tree-sha1 = "b8a399e95663485820000f26b6a43c794e166a49"
uuid = "fb686558-2515-59ef-acaa-46db3789a887"
version = "0.4.4"

[[deps.RecipesBase]]
deps = ["PrecompileTools"]
git-tree-sha1 = "5c3d09cc4f31f5fc6af001c250bf1278733100ff"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.3.4"

[[deps.RecursiveArrayTools]]
deps = ["Adapt", "ArrayInterface", "DocStringExtensions", "GPUArraysCore", "IteratorInterfaceExtensions", "LinearAlgebra", "RecipesBase", "Requires", "SparseArrays", "StaticArraysCore", "Statistics", "SymbolicIndexingInterface", "Tables"]
git-tree-sha1 = "27ee1c03e732c488ecce1a25f0d7da9b5d936574"
uuid = "731186ca-8d62-57ce-b412-fbd966d074cd"
version = "3.3.3"

    [deps.RecursiveArrayTools.extensions]
    RecursiveArrayToolsFastBroadcastExt = "FastBroadcast"
    RecursiveArrayToolsMeasurementsExt = "Measurements"
    RecursiveArrayToolsMonteCarloMeasurementsExt = "MonteCarloMeasurements"
    RecursiveArrayToolsTrackerExt = "Tracker"
    RecursiveArrayToolsZygoteExt = "Zygote"

    [deps.RecursiveArrayTools.weakdeps]
    FastBroadcast = "7034ab61-46d4-4ed7-9d0f-46aef9175898"
    Measurements = "eff96d63-e80a-5855-80a2-b1b0885c5ab7"
    MonteCarloMeasurements = "0987c9cc-fe09-11e8-30f0-b96dd679fdca"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"
    Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "f65dcb5fa46aee0cf9ed6274ccbd597adc49aa7b"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.7.1"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6ed52fdd3382cf21947b15e8870ac0ddbff736da"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.4.0+0"

[[deps.RuntimeGeneratedFunctions]]
deps = ["ExprTools", "SHA", "Serialization"]
git-tree-sha1 = "6aacc5eefe8415f47b3e34214c1d79d2674a0ba2"
uuid = "7e49a35a-f44a-4d26-94aa-eba1b4ca6b47"
version = "0.5.12"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.SIMD]]
deps = ["PrecompileTools"]
git-tree-sha1 = "d8911cc125da009051fb35322415641d02d9e37f"
uuid = "fdea26ae-647d-5447-a871-4b548cad5224"
version = "3.4.6"

[[deps.SciMLBase]]
deps = ["ADTypes", "ArrayInterface", "CommonSolve", "ConstructionBase", "Distributed", "DocStringExtensions", "EnumX", "FillArrays", "FunctionWrappersWrappers", "IteratorInterfaceExtensions", "LinearAlgebra", "Logging", "Markdown", "PrecompileTools", "Preferences", "Printf", "RecipesBase", "RecursiveArrayTools", "Reexport", "RuntimeGeneratedFunctions", "SciMLOperators", "StaticArraysCore", "Statistics", "SymbolicIndexingInterface", "Tables", "TruncatedStacktraces"]
git-tree-sha1 = "09324a0ae70c52a45b91b236c62065f78b099c37"
uuid = "0bca4576-84f4-4d90-8ffe-ffa030f20462"
version = "2.15.2"

    [deps.SciMLBase.extensions]
    SciMLBaseChainRulesCoreExt = "ChainRulesCore"
    SciMLBasePartialFunctionsExt = "PartialFunctions"
    SciMLBasePyCallExt = "PyCall"
    SciMLBasePythonCallExt = "PythonCall"
    SciMLBaseRCallExt = "RCall"
    SciMLBaseZygoteExt = "Zygote"

    [deps.SciMLBase.weakdeps]
    ChainRules = "082447d4-558c-5d27-93f4-14fc19e9eca2"
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    PartialFunctions = "570af359-4316-4cb7-8c74-252c00c2016b"
    PyCall = "438e738f-606a-5dbb-bf0a-cddfbfd45ab0"
    PythonCall = "6099a3de-0909-46bc-b1f4-468b9a2dfc0d"
    RCall = "6f49c342-dc21-5d91-9882-a32aef131414"
    Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[[deps.SciMLOperators]]
deps = ["ArrayInterface", "DocStringExtensions", "Lazy", "LinearAlgebra", "Setfield", "SparseArrays", "StaticArraysCore", "Tricks"]
git-tree-sha1 = "51ae235ff058a64815e0a2c34b1db7578a06813d"
uuid = "c0aeaf25-5076-4817-a8d5-81caf7dfa961"
version = "0.3.7"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.Setfield]]
deps = ["ConstructionBase", "Future", "MacroTools", "StaticArraysCore"]
git-tree-sha1 = "e2cc6d8c88613c05e1defb55170bf5ff211fbeac"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "1.1.1"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "66e0a8e672a0bdfca2c3f5937efb8538b9ddc085"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.1"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "e2cfc4012a19088254b3950b85c3c1d8882d864d"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.3.1"
weakdeps = ["ChainRulesCore"]

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "PrecompileTools", "Random", "StaticArraysCore"]
git-tree-sha1 = "bf074c045d3d5ffd956fa0a461da38a44685d6b2"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.9.3"
weakdeps = ["ChainRulesCore", "Statistics"]

    [deps.StaticArrays.extensions]
    StaticArraysChainRulesCoreExt = "ChainRulesCore"
    StaticArraysStatisticsExt = "Statistics"

[[deps.StaticArraysCore]]
git-tree-sha1 = "36b3d696ce6366023a0ea192b4cd442268995a0d"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.2"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.9.0"

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1ff449ad350c9c4cbc756624d6f8a8c3ef56d3ed"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.7.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "5cf7606d6cef84b543b483848d4ae08ad9832b21"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.3"

[[deps.StatsFuns]]
deps = ["HypergeometricFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "cef0472124fab0695b58ca35a77c6fb942fdab8a"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.3.1"

    [deps.StatsFuns.extensions]
    StatsFunsChainRulesCoreExt = "ChainRulesCore"
    StatsFunsInverseFunctionsExt = "InverseFunctions"

    [deps.StatsFuns.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.StringManipulation]]
deps = ["PrecompileTools"]
git-tree-sha1 = "a04cabe79c5f01f4d723cc6704070ada0b9d46d5"
uuid = "892a3eda-7b42-436c-8928-eab12a02cf0e"
version = "0.3.4"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "Pkg", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "5.10.1+6"

[[deps.SymbolicIndexingInterface]]
git-tree-sha1 = "be414bfd80c2c91197823890c66ef4b74f5bf5fe"
uuid = "2efcf032-c050-4f8e-a9bb-153293bab1f5"
version = "0.3.1"

[[deps.SymbolicUtils]]
deps = ["AbstractTrees", "Bijections", "ChainRulesCore", "Combinatorics", "ConstructionBase", "DataStructures", "DocStringExtensions", "DynamicPolynomials", "IfElse", "LabelledArrays", "LinearAlgebra", "MultivariatePolynomials", "NaNMath", "Setfield", "SparseArrays", "SpecialFunctions", "StaticArrays", "SymbolicIndexingInterface", "TimerOutputs", "Unityper"]
git-tree-sha1 = "669e43e90df46fcee4aa859b587da7a7948272ac"
uuid = "d1185830-fcd6-423d-90d6-eec64667417b"
version = "1.5.1"

[[deps.Symbolics]]
deps = ["ArrayInterface", "Bijections", "ConstructionBase", "DataStructures", "DiffRules", "Distributions", "DocStringExtensions", "DomainSets", "DynamicPolynomials", "Groebner", "IfElse", "LaTeXStrings", "LambertW", "Latexify", "Libdl", "LinearAlgebra", "LogExpFunctions", "MacroTools", "Markdown", "NaNMath", "PrecompileTools", "RecipesBase", "Reexport", "Requires", "RuntimeGeneratedFunctions", "SciMLBase", "Setfield", "SparseArrays", "SpecialFunctions", "StaticArrays", "SymbolicIndexingInterface", "SymbolicUtils"]
git-tree-sha1 = "8d28ebc206dec9e250e21b9502a2662265897650"
uuid = "0c5d862f-8b57-4792-8d23-62f2024744c7"
version = "5.14.1"

    [deps.Symbolics.extensions]
    SymbolicsSymPyExt = "SymPy"

    [deps.Symbolics.weakdeps]
    SymPy = "24249f21-da20-56a4-8eb1-6a02cf4ae2e6"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "OrderedCollections", "TableTraits"]
git-tree-sha1 = "cb76cf677714c095e535e3501ac7954732aeea2d"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.11.1"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.TimerOutputs]]
deps = ["ExprTools", "Printf"]
git-tree-sha1 = "f548a9e9c490030e545f72074a41edfd0e5bcdd7"
uuid = "a759f4b9-e2f1-59dc-863e-4aeb61b1ea8f"
version = "0.5.23"

[[deps.TranscodingStreams]]
git-tree-sha1 = "71509f04d045ec714c4748c785a59045c3736349"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.10.7"
weakdeps = ["Random", "Test"]

    [deps.TranscodingStreams.extensions]
    TestExt = ["Test", "Random"]

[[deps.Tricks]]
git-tree-sha1 = "eae1bb484cd63b36999ee58be2de6c178105112f"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.8"

[[deps.TruncatedStacktraces]]
deps = ["InteractiveUtils", "MacroTools", "Preferences"]
git-tree-sha1 = "ea3e54c2bdde39062abf5a9758a23735558705e1"
uuid = "781d530d-4396-4725-bb49-402e4bee1e77"
version = "1.4.0"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.Unityper]]
deps = ["ConstructionBase"]
git-tree-sha1 = "25008b734a03736c41e2a7dc314ecb95bd6bbdb0"
uuid = "a7c27f48-0311-42f6-a7f8-2c11e75eb415"
version = "0.1.6"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.8.0+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.48.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+0"
"""

# ╔═╡ Cell order:
# ╟─dfc273e3-33e9-4168-a911-8e12c4fd7440
# ╠═ca1eadcf-dcf4-4650-8782-929fdd7cc6a2
# ╟─557b7b8d-f894-408d-980a-2344596dcce7
# ╠═ec71cdab-1a87-4e24-9d82-33aa7c6d5dfe
# ╠═feb186c0-55cd-420b-9315-4918d4b02904
# ╠═eb7f5dc1-1282-49ff-92b3-45f150686e3a
# ╟─5e7a12e6-c202-4fc2-b310-28700f504aa7
# ╠═7b9d4b5d-7dc9-42e1-8e4e-a7d1cd9a8249
# ╠═a20dd084-9b66-457d-9d8c-c2848b69c9e5
# ╟─96e560ce-386c-4b57-92c7-4595b9d8804a
# ╠═fdc8208f-2e51-49aa-a398-570531b608e4
# ╠═bb115df2-9f5b-402d-9a25-6c096d913d63
# ╠═79facf78-a456-4b29-b18d-ae1dc23c0e6e
# ╟─b522641d-cfb5-4dac-abf0-28ffe038ed14
# ╟─299dd5b6-a211-4c2a-9e65-b9a80664cd10
# ╟─0480ce3a-8763-4ece-8e9b-b931874e9f38
# ╠═cdf8230b-5355-4ba9-924b-f2919ebec026
# ╠═57b1dbeb-4365-4a7d-9f9b-ddf18067cac5
# ╠═7553603d-343b-4df1-955d-044736ba61e0
# ╠═4c4e04ba-1ab4-43d9-858d-b2108ab596a2
# ╠═4073f275-589c-465c-a8c2-2b3f99a2c27b
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
