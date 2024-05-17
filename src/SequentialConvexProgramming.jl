import JuMP as jp
using LinearAlgebra
using Printf

export ctcs_main

function ctcs_subproblem(x_bar, u_bar, f_aug, A_aug, B_aug, params; verbose=false)
    n_x = params[:n_x]
    n_x_aug = params[:n_x_aug]
    n_u = params[:n_u]
    N = params[:n_nodes]
    S_x = params[:S_x]
    S_u = params[:S_u]
    c_x = params[:c_x]
    c_u = params[:c_u]

    A_bar = zeros(n_x_aug * n_x_aug, N - 1)
    B_bar = zeros(n_x_aug * n_u, N - 1)
    C_bar = zeros(n_x_aug * n_u, N - 1)
    z_bar = zeros(n_x_aug, N - 1)

    A_bar, B_bar, C_bar, z_bar = calculate_discretization(
        x_bar,
        u_bar,
        f_aug,
        A_aug,
        B_aug,
        params
    )

    sys = LinearizedDiscretizedFohSystem(A_bar, B_bar, C_bar, z_bar)
    traj_prev = Trajectory(x_bar, u_bar)

    # TODO: ideally would define model outside of this function
    # can then just overwrite dynamics constraints here
    model = OCP(params, sys, traj_prev)
    jp.optimize!(model)

    x = S_x * jp.value.(model[:x]) .+ c_x
    u = S_u * jp.value.(model[:u]) .+ c_u

    nu = jp.value.(model[:nu])

    J_vc = params[:λ_vc] *sum(norm(nu[1:n_x, k], 1) for k in size(nu, 2))
    J_vc_ctcs = params[:λ_vc_ctcs] * sum(norm(nu[end, k], 1) for k in size(nu, 2))
    J_tr = params[:w_tr] * sum(
    # norm(
    #     inv(
    #         [S_x zeros(n_x_aug, n_u); zeros(n_u, n_x_aug) S_u]
    #     ) * (
    #         [x[:, k]; u[:, k-1]] - [x_bar[:, k]; u_bar[:, k-1]]
    #     ), 2
    # ) for k in 2:N
        [transpose(
             [inv(S_x) zeros(n_x_aug, n_u); zeros(n_u, n_x_aug) inv(S_u)] * ([x[:, k]; u[:, k-1]] - [x_bar[:, k]; u_bar[:, k-1]])
         ) * I(n_x_aug + n_u) * (
             [inv(S_x) zeros(n_x_aug, n_u); zeros(n_u, n_x_aug) inv(S_u)] * ([x[:, k]; u[:, k-1]] - [x_bar[:, k]; u_bar[:, k-1]])
         ) for k in 2:N]
    )

    if verbose
        display(maximum(jp.value.(model[:dx])))
    end

    return x, u, jp.objective_value(model), J_vc, J_vc_ctcs, J_tr
end

function ctcs_main(params)
    J_vc = 1e2
    J_tr = 1e2

    # TODO: kind of disgusting augmentation of x here
    x_bar = linterp(params[:n_nodes], [params[:x_initial]; 0.0], [params[:x_final]; 0.0])
    u_guess = zeros(params[:n_u])
    u_guess[3] = 9.81
    u_bar = linterp(params[:n_nodes], u_guess, u_guess)

    obstacles = []
    for k in params[:n_obs]
        push!(
            obstacles,
            EllipsoidalObstacle(
                params[:obstacle_centers][k],
                params[:obstacle_axes][k],
                params[:obstacle_radius][k],
            )
        )
    end

    k = 0

    x = copy(x_bar)
    u = copy(u_bar)

    println("")
    println("Iter | J_total |  J_tr   |  J_vc   | J_vc_ctcs ")
    println("----------------------------------------------")

    while k <= params[:k_max] && ((J_tr >= params[:ϵ_tr]) || (J_vc >= params[:ϵ_vc]))
        x, u, J_total, J_vc, J_vc_ctcs, J_tr = ctcs_subproblem(
            x_bar,
            u_bar,
            (x, u, p) -> nonlinear_dynamics_aug(nonlinear_drone_dynamics, x, u, p, obstacles),
            (x, u, p) -> jacobian_A_aug(jacobian_A, x, u, p, obstacles),
            (x, u, p) -> jacobian_B_aug(jacobian_B, x, u, p, obstacles),
            params,
        )

        x_bar = x
        u_bar = u

        params[:w_tr] = params[:w_tr] * 1.1

        @printf("%4d | %.1e | %.1e | %.1e | %.1e\n", k, J_total, J_tr, J_vc, J_vc_ctcs)
        k += 1
    end

    x_full = simulate_nonlinear(nonlinear_drone_dynamics, x[1:params[:n_x], 1], u, params)

    result = Dict(
        :x => x,
        :u => u,
        :x_full => x_full,
    )

    return result
end