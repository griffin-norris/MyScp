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
    nx::Int # state
    nu::Int # input
    np::Int # parameters
    # ..:: Cost ::..
    ϕ::Func # terminal cost
    Γ::Func # running cost
    # ..:: Dynamics ::..
    f::Func # continuous dynamics
    A::Func # df/dx
    B::Func # df/du
    F::Func # df/dp
    # ..:: Constraints ::..
    X::Func # (x(t), p) ∈ X
    U::Func # (u(t), p) ∈ U
    s::Func # s(t, x(t), u(t), p) ≤ 0
    C::Func # ds/dx
    D::Func # ds/du
    G::Func # ds/dp
    # ..:: Boundary conditions ::..
    g_ic::Func # initial condition
    g_tc::Func # terminal condition
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
    )
end