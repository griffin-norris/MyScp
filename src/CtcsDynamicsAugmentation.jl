export nonlinear_dynamics_aug, jacobian_A_aug, jacobian_B_aug

function nonlinear_dynamics_aug(f, x, u, params, obstacles)
    x_dot = f(x[1:params[:n_x]], u, params)
    return [
        x_dot;
        sum(max(0, ellipsoid_g_bar_ctcs(obs, x[1:3]))^2 for obs in obstacles)
    ]
end

function jacobian_A_aug(A, x, u, params, obstacles)
    return [
        A(x[1:params[:n_x]], u, params) zeros(params[:n_x]);
        sum(ellipsoid_grad_g_bar_ctcs(obs, x[1:3]) for obs in obstacles) zeros(params[:n_x] - 3 + 1)';
    ]
end

function jacobian_B_aug(B, x, u, params, obstacles)
    return [
        B(x[1:params[:n_x]], u, params);
        zeros(params[:n_u])';
    ]
end