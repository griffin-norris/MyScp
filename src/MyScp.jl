module MyScp

using JuMP, OSQP
using LinearAlgebra

include("Utils.jl")
include("NonlinearDroneDynamics.jl")
include("Discretization.jl")
include("Integration.jl")
include("Obstacles.jl")
include("CtcsDynamicsAugmentation.jl")
include("OptimizationProblem.jl")
include("SequentialConvexProgramming.jl")

end # module MyScp
