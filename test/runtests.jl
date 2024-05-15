using Test

@testset "Main" begin
    @testset "OptimizationProblem" begin
        include("TestOptimizationProblem.jl")
    end
    @testset "NonlinearDroneDynamics" begin
        include("TestNonlinearDroneDynamics.jl")
    end
end