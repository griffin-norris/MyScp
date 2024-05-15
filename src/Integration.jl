using DifferentialEquations

export simulate_nonlinear

"""
    simulate_nonlinear(f, x_initial, u_vecs, params)

Simulate a nonlinear system using the provided dynamics function `f`. This function integrates the system over a specified time span using control inputs provided in `u_vecs`.

# Arguments
- `f`: A function representing the system dynamics. It should take the current state, control input, and parameters as arguments and return the derivative of the state.
- `x_initial`: An array representing the initial state of the system.
- `u_vecs`: A matrix where each column represents the control input vector at a specific time step.
- `params`: A dictionary containing simulation parameters. Expected keys include:
  - `:dt_ss`: Step size for updating the control inputs.
  - `:total_time`: Total simulation time.
  - `:dt_sim`: Time step size for the solver.

# Returns
- `Array`: A matrix where each column represents the state of the system at a specific time step, resulting from the simulation.

# Example
```julia
f = (x, u, p) -> x + u + p
x_initial = [0.0]
u_vecs = [1 2 3; 4 5 6]
params = Dict(:dt_ss => 1.0, :total_time => 3.0, :dt_sim => 0.1)
simulate_nonlinear(f, x_initial, u_vecs, params)
```
"""
function simulate_nonlinear(
    f,
    x_initial,
    u_vecs,
    params,
)
    # Initial state array
    states = [x_initial]

    # Time evaluation options for multiple control inputs
    t_eval_opt = 0:params[:dt_ss]:(params[:total_time]-params[:dt_ss])

    for (i, t) in enumerate(t_eval_opt)
        u_start = u_vecs[:, i]
        u_next = u_vecs[:, i+1]
        u_slope = (u_next - u_start) / params[:dt_ss]
        t_eval = t:params[:dt_sim]:(t+params[:dt_ss])

        prob = ODEProblem(
            (state, p, t) -> f(
                state, # State
                p[1] + p[2] * (t - p[3]), # Control (FOH)
                p[4] # Params
            ),
            states[end],
            (t, t + params[:dt_ss]),
            (u_start, u_slope, t, params)
        )
        sol = solve(prob, reltol=1e-6, abstol=1e-6, saveat=t_eval)
        for k in 2:length(sol.u)
            push!(states, sol.u[k])
        end
    end

    return hcat(states...)
end
