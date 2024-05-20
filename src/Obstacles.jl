using LinearAlgebra
using Random

export EllipsoidalObstacle, ellipsoid_g_bar_ctcs, ellipsoid_grad_g_bar_ctcs, generate_orthogonal_unit_vectors

struct EllipsoidalObstacle
    center::Array{Float64,1}
    axes::Array{Float64,2}
    radius::Array{Float64,1}
    A::Array{Float64,2}

    function EllipsoidalObstacle(center::Array{Float64,1}, axes, radius)
        A = axes * Diagonal(radius .^ 2) * axes'
        new(center, axes, radius, A)
    end
end

function generate_orthogonal_unit_vectors()
    vectors = rand(3, 3)
    Q, R = qr(vectors)
    return Q
end

function ellipsoid_g_bar_ctcs(obs::EllipsoidalObstacle, p::Array{Float64,1})
    return 1 - (p - obs.center)' * obs.A * (p - obs.center)
end

function ellipsoid_grad_g_bar_ctcs(obs::EllipsoidalObstacle, p::Array{Float64,1})
    g = ellipsoid_g_bar_ctcs(obs, p)
    return -2 * max(0, g) * ((p - obs.center)' * (obs.A + obs.A'))
end
