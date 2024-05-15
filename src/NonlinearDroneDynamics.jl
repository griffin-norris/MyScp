using LinearAlgebra, ForwardDiff

export nonlinear_drone_dynamics, jacobian_A, jacobian_B

function nonlinear_drone_dynamics(
    x,
    u,
    params,
)

    # State
    r_i = x[1:3] # Position
    v_i = x[4:6] # Velocity
    q_bi = x[7:10] # Attitude (Quaternion)
    w_b = x[11:end] # Angular Rate

    # Control
    u_mB = u[1:3] # Applied Force
    tau_i = u[4:end] # Applied Torque

    # Ensure that the quaternion is normalized
    q_norm = LinearAlgebra.norm(q_bi)
    q_bi = q_bi / q_norm

    p_i_dot = v_i
    v_i_dot = (1 / params[:m]) * qdcm(q_bi) * (u_mB) + [0, 0, params[:g]]

    q_bi_dot = 0.5 * skew_symetric_matrix_quat(w_b) * q_bi
    w_b_dot = LinearAlgebra.Diagonal(1 ./ params[:J_b]) * (tau_i - skew_symetric_matrix(w_b) * LinearAlgebra.Diagonal(params[:J_b]) * w_b)
    return [p_i_dot; v_i_dot; q_bi_dot; w_b_dot]
end

function jacobian_A(x, u, params)
    return ForwardDiff.jacobian(dx -> nonlinear_drone_dynamics(dx, u, params), x)
end

function jacobian_B(x, u, params)
    return ForwardDiff.jacobian(du -> nonlinear_drone_dynamics(x, du, params), u)
end