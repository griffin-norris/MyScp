"""
    Describes generic optimization problem:

                            1
        min. [ϕ(x(t), p) + ∫  Γ(x(t), u(t), p)dt]
        u,p                 0

        s.t     x_dot(t) = f(t, x(t), u(t), p),     Dynamics
                (x(t), p) ∈ X,                      State constraints
                (u(t), p) ∈ U,                      Input constraints
                s(t, x(t), u(t), p) ≤ 0,            State inequality constraints
                g_ic(x(0), p) = 0,                  Initial conditions
                g_tc(x(1), p) = 0,                  Terminal condition
"""
mutable struct OptimizationProblem
    # ..:: Dimensions ::..
    N::Int      # Nodes
    nx::Int     # State
    nu::Int     # Input
    np::Int     # Parameters
    # ..:: Cost ::..
    ϕ::Func     # Terminal cost
    Γ::Func     # Running cost
    # ..:: Dynamics ::..
    f::Func     # Continuous dynamics
    A::Func     # df/dx
    B::Func     # df/du
    F::Func     # df/dp
    # ..:: Constraints ::..
    X::Func     # (x(t), p) ∈ X
    U::Func     # (u(t), p) ∈ U
    s::Func     # s(t, x(t), u(t), p) ≤ 0
    C::Func     # ds/dx
    D::Func     # ds/du
    G::Func     # ds/dp
    # ..:: Boundary conditions ::..
    g_ic::Func  # Initial condition
    g_tc::Func  # Terminal condition
    H_0::Func   # dg_ic/dx
    K_0::Func   # dg_ic/dp
    H_f::Func   # dg_tc/dx
    K_f::Func   # dg_tc/dp
end

"""
    Empty constructor for OptimizationProblem
"""
function OptimizationProblem()::OptimizationProblem
    # ..:: Dimensions ::..
    nx = 0
    nu = 0
    np = 0
    # ..:: Cost ::..
    ϕ = nothing
    Γ = nothing
    # ..:: Dynamics ::..
    f = nothing
    A = nothing
    B = nothing
    F = nothing
    # ..:: Constraints ::..
    X = nothing
    U = nothing
    s = nothing
    C = nothing
    D = nothing
    G = nothing
    # ..:: Boundary conditions ::..
    g_ic = nothing
    g_tc = nothing
    H_0 = nothing
    K_0 = nothing
    H_f = nothing
    K_f = nothing

    return OptimizationProblem(
        nx,
        nu,
        np,
        ϕ,
        Γ,
        f,
        A,
        B,
        F,
        X,
        U,
        s,
        C,
        D,
        G,
        g_ic,
        g_tc,
        H_0,
        K_0,
        H_f,
        K_f,
    )
end