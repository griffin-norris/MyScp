module MyScp

greet() = println("Hello World!")

const Func = Union{Nothing,Function}

include("OptimizationProblem.jl")

pbm = OptimizationProblem()

greet()
println(pbm.nx)
pbm.nx = 1
println(pbm.nx)

end # module MyScp
