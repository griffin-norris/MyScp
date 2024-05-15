export dVdt, calculate_discretization

struct VIndices
    x
    A
    B
    C
    z

    function VIndices(n_x, n_u)
        idx_x = 1:n_x
        idx_A = idx_x[end]+1:idx_x[end]+n_x*n_x
        idx_B = idx_A[end]+1:idx_A[end]+n_x*n_u
        idx_C = idx_B[end]+1:idx_B[end]+n_x*n_u
        idx_z = idx_C[end]+1:idx_C[end]+n_x
        new(idx_x, idx_A, idx_B, idx_C, idx_z)
    end
end

function dVdt(
    t::Float64,
    V::Vector{Float64},
    u_curr::Vector{Float64},
    u_next::Vector{Float64},
    f::Function,
    A::Function,
    B::Function,
    params,
)::Vector{Float64}
    n_x_aug = params[:n_x_aug]
    n_u = params[:n_u]
    dt = params[:dt_ss]

    idx = VIndices(n_x_aug, n_u)

    x = V[idx.x]
    beta = t / dt
    alpha = 1 - beta
    u = u_curr + beta * (u_next - u_curr)

    A_subs = A(x, u, params)
    B_subs = B(x, u, params)
    f_subs = f(x, u, params)

    z_t = f_subs - A_subs * x - B_subs * u

    dVdt = [
        f_subs
        vec(A_subs * reshape(V[idx.A], n_x_aug, n_x_aug))
        vec(A_subs * reshape(V[idx.B], n_x_aug, n_u) + B_subs .* alpha)
        vec(A_subs * reshape(V[idx.C], n_x_aug, n_u) + B_subs .* beta)
        vec(A_subs * V[idx.z] .+ z_t)
    ]

    return dVdt
end

function calculate_discretization(
    x,
    u,
    f,
    A,
    B,
    params
)
    n_x_aug = params[:n_x_aug]
    n_u = params[:n_u]
    N = params[:n_nodes]
    dt = params[:dt_ss]

    idx = VIndices(n_x_aug, n_u)

    V0 = zeros(idx.z[end])
    V0[idx.A] .= vec(I(n_x_aug))

    f_bar = zeros((n_x_aug, N - 1))
    A_bar = zeros((n_x_aug * n_x_aug, N - 1))
    B_bar = zeros((n_x_aug * n_u, N - 1))
    C_bar = zeros((n_x_aug * n_u, N - 1))
    z_bar = zeros((n_x_aug, N - 1))

    for k in 1:(N-1)
        V0[idx.x] .= x[:, k]

        k1 = dVdt(0.0, V0, u[:, k], u[:, k+1], f, A, B, params)
        k2 = dVdt(dt / 2, V0 + (dt / 2) .* k1, u[:, k], u[:, k+1], f, A, B, params)
        k3 = dVdt(dt / 2, V0 + (dt / 2) .* k2, u[:, k], u[:, k+1], f, A, B, params)
        k4 = dVdt(dt, V0 + dt .* k3, u[:, k], u[:, k+1], f, A, B, params)
        V = V0 + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        f_bar[:, k] .= V[idx.x]
        A_bar[:, k] .= V[idx.A]
        B_bar[:, k] .= V[idx.B]
        C_bar[:, k] .= V[idx.C]
        z_bar[:, k] .= V[idx.z]
    end

    return A_bar, B_bar, C_bar, z_bar
end