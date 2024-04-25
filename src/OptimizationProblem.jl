"""
    Describes generic optimization problem:

                            1
        min. [ϕ(x(t), p) + ∫  Γ(x(t), u(t), p)dt]
        u,p                 0

        s.t     ̇x(t) = f(t, x(t), u(t), p),     Dynamics
                (x(t), p) ∈ X,                  State constraints
                (u(t), p) ∈ U,                  Input constraints
                s(t, x(t), u(t), p) ≤ 0,        State inequality constraints
                g₀(x(0), p) = 0,                Initial conditions
                g₁(x(1), p) = 0,                Terminal condition
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
    A::Func     # ∇x f(t, ̄x(t), ̄u(t), ̄p)
    B::Func     # ∇u f(t, ̄x(t), ̄u(t), ̄p)
    F::Func     # ∇p f(t, ̄x(t), ̄u(t), ̄p)
    r::Func     # r(t) = f(t, ̄x(t), ̄u(t), ̄p) - A*̄x(t) - B*̄u(t) - F*̄p
    # ..:: Constraints ::..
    X::Func     # (x(t), p) ∈ X
    U::Func     # (u(t), p) ∈ U
    s::Func     # s(t, x(t), u(t), p) ≤ 0
    C::Func     # ∇x s(t, ̄x(t), ̄u(t), ̄p)
    D::Func     # ∇u s(t, ̄x(t), ̄u(t), ̄p)
    G::Func     # ∇p s(t, ̄x(t), ̄u(t), ̄p)
    r′::Func    # r′(t) ≔ s(t, ̄x(t), ̄u(t), ̄p) - C*̄x(t) - D*̄u(t) - G*̄p
    # ..:: Boundary conditions ::..
    g₀::Func    # Initial condition
    g₁::Func    # Terminal condition
    H₀::Func    # ∇x g₀(̄x(0), ̄p)
    K₀::Func    # ∇u g₀(̄x(0), ̄p)
    ℓ₀::Func    # g₀(̄x(0), ̄p) - H₀*̄x(0) - K₀*̄p
    H₁::Func    # ∇x g₁(̄x(1), ̄p)
    K₁::Func    # ∇u g₁(̄x(1), ̄p)
    ℓ₁::Func    # g₁(̄x(1), ̄p) - H₁*̄x(1) - K₁*̄p
end

# ̇x(t)      = A*x(t) + B*u(t) + F*p + r(t) + E*ν(t)
# r(t)      = f(t, ̄x(t), ̄u(t), ̄p) - A*̄x(t) - B*̄u(t) - F*̄p
# ⇒ ̇x(t)    = A*x(t) + B*u(t) + F*p + f(t, ̄x(t), ̄u(t), ̄p) - A*̄x(t) - B*̄u(t) - F*̄p + E*ν(t)
#           = A(x - ̄x) + B(u - ̄u) + F(p - ̄p) + f(t, ̄x, ̄u, ̄p) + E*ν
#           ↑
#           First order Taylor series approximation of f(t, x, u, p) 
#           around previous trajectory (̄x, ̄u, ̄p)

# xₖ₊₁  = Aₖ*xₖ + Bₖ*uₖ +Fₖ*pₖ + rₖ + Eₖ*νₖ

"""
    Empty constructor for OptimizationProblem
"""
function OptimizationProblem()::OptimizationProblem
    # ..:: Dimensions ::..
    N = 0
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
    r = nothing
    # ..:: Constraints ::..
    X = nothing
    U = nothing
    s = nothing
    C = nothing
    D = nothing
    G = nothing
    r′ = nothing
    # ..:: Boundary conditions ::..
    g₀ = nothing
    g₁ = nothing
    H₀ = nothing
    K₀ = nothing
    ℓ₀ = nothing
    H₁ = nothing
    K₁ = nothing
    ℓ₁ = nothing

    return OptimizationProblem(
        N,
        nx,
        nu,
        np,
        ϕ,
        Γ,
        f,
        A,
        B,
        F,
        r,
        X,
        U,
        s,
        C,
        D,
        G,
        r′,
        g₀,
        g₁,
        H₀,
        K₀,
        ℓ₀,
        H₁,
        K₁,
        ℓ₁,
    )
end