import JuMP as jp

export LinearizedDiscretizedFohSystem, Trajectory

mutable struct LinearizedDiscretizedFohSystem
    A_vectors
    B_vectors
    C_vectors
    z_vectors
end

mutable struct Trajectory
    x_bar
    u_bar
end

export linterp, OCP

function linterp(nodes, init, final)
    linear = [init .+ k * (final .- init) / (nodes - 1) for k in 0:nodes-1]
    return hcat(linear...)
end

function OCP(
    params,
    sys::LinearizedDiscretizedFohSystem,
    traj_prev::Trajectory;
    verbose=false
)

    N = params[:n_nodes]
    n_x = params[:n_x]
    n_x_aug = params[:n_x_aug]
    n_u = params[:n_u]
    S_x = params[:S_x]
    c_x = params[:c_x]
    S_u = params[:S_u]
    c_u = params[:c_u]

    model = jp.Model(OSQP.Optimizer)
    set_optimizer_attribute(model, "verbose", 0)

    # Variables
    (
        model[:x],
        model[:dx],
        model[:u],
        model[:du],
        model[:nu],
    ) = jp.@variables(
        model,
        begin
            x[1:n_x_aug, 1:N], (start = 0) # State
            dx[1:n_x_aug, 1:N], (start = 0)
            u[1:n_u, 1:N], (start = 0) # Control
            du[1:n_u, 1:N], (start = 0)
            nu[1:n_x_aug, 1:N-1], (start = 0) # Virtual control
        end
    )

    x_nonscaled = []
    u_nonscaled = []

    for k in (1:N)
        push!(x_nonscaled, S_x * x[:, k] + c_x)
        push!(u_nonscaled, S_u * u[:, k] + c_u)
    end

    # Boundary constraints
    model[Symbol("constraint_bc")] = jp.@constraints(
        model,
        begin
            x_nonscaled[1][1:n_x] .== params[:x_initial] # Initial condition
            x_nonscaled[end][1:n_x] .== params[:x_final] # Terminal Condition
        end
    )

    # Dynamics constraints
    for k in 1:N-1
        model[Symbol("constraint_dynamics_$(k)")] = jp.@constraint(
            model,
            reshape(sys.A_vectors[:, k], n_x_aug, n_x_aug) * x_nonscaled[k]
            + reshape(sys.B_vectors[:, k], n_x_aug, n_u) * u_nonscaled[k]
            + reshape(sys.C_vectors[:, k], n_x_aug, n_u) * u_nonscaled[k+1]
            + sys.z_vectors[:, k]
            + nu[:, k] == x_nonscaled[k+1]
        )
    end

    # CTCS constraint
    for k in 2:N
        model[Symbol("constraint_ctcs_$(k)")] = jp.@constraints(
            model,
            begin
                x_nonscaled[k][end] - x_nonscaled[k-1][end] <= 1e-4
                x_nonscaled[k-1][end] - x_nonscaled[k][end] <= 1e-4
            end
        )
    end

    # State constraints
    for k in 2:N
        model[Symbol("constraint_x_$(k)")] = jp.@constraints(
            model,
            begin
                x_nonscaled[k] <= params[:x_max]
                x_nonscaled[k] >= params[:x_min]
            end
        )
    end

    # Control constraints
    for k in 1:N
        model[Symbol("constraint_u_$(k)")] = jp.@constraints(
            model,
            begin
                u_nonscaled[k] <= params[:u_max]
                u_nonscaled[k] >= params[:u_min]
            end
        )
    end

    # State and control deltas
    for k in 1:N
        model[Symbol("constraint_deltas_$(k)")] = jp.@constraints(
            model,
            begin
                inv(S_x) * (x_nonscaled[k] - traj_prev.x_bar[:, k] - dx[:, k]) .== 0
                inv(S_u) * (u_nonscaled[k] - traj_prev.u_bar[:, k] - du[:, k]) .== 0
            end
        )
    end

    # Define extra variable for trick to model virtual control cost
    # 	||x||₁ = ∑ᵢ|xᵢ| -> NormOneCone
    model[:t_vc] = jp.@variable(model, t_vc[1:N-1], (start = 0))
    for k in 1:N-1
        model[Symbol("constraint_vc_$(k)")] = jp.@constraint(model, [t_vc[k]; nu[1:n_x, k]] in MOI.NormOneCone(1 + length(nu[1:n_x, k])))
    end
    model[:t_vc_ctcs] = jp.@variable(model, t_vc_ctcs[1:N-1], (start = 0))
    for k in 1:N-1
        model[Symbol("constraint_vc_ctcs_$(k)")] = jp.@constraint(model, [t_vc_ctcs[k]; nu[end, k]] in MOI.NormOneCone(1 + length(nu[end, k])))
    end

    # Cost 
    model[:Objective] = jp.@objective(
        model,
        Min,
        params[:λ_fuel] * sum(
            [transpose(inv(S_u) * u) * I(n_u) * (inv(S_u) * u) for (k, u) in enumerate(u_nonscaled)]
        )
        + params[:w_tr] * transpose(inv(S_x) * dx[:, 1]) * I(n_x_aug) * (inv(S_x) * dx[:, 1])
        + params[:w_tr] * sum(
            [transpose(
                 [inv(S_x) zeros(n_x_aug, n_u); zeros(n_u, n_x_aug) inv(S_u)] * [dx[:, k]; du[:, k-1]]
             ) * I(n_x_aug + n_u) * (
                 [inv(S_x) zeros(n_x_aug, n_u); zeros(n_u, n_x_aug) inv(S_u)] * [dx[:, k]; du[:, k-1]]
             ) for k in 2:N]
        )
        + params[:λ_vc] * sum(t_vc)
        + params[:λ_vc_ctcs] * sum(t_vc_ctcs)
    )

    if verbose
        println(" === Objective Function")
        display(jp.objective_function(model))

        println(" === Constraints Dynamics")
        for k in 1:N-1
            display(model[Symbol("constraint_dynamics_$(k)")])
        end

        println(" === Constraints State & Control Deltas")
        for k in 1:N
            display(model[Symbol("constraint_deltas_$(k)")])
        end

        println(" === Constraints CTCS")
        for k in 2:N
            display(model[Symbol("constraint_ctcs_$(k)")])
        end

        println(" === Constraints VC")
        for k in 1:N-1
            display(model[Symbol("constraint_vc_$(k)")])
            display(model[Symbol("constraint_vc_ctcs_$(k)")])
        end

    end

    return model
end