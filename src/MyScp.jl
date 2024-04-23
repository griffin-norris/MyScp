module MyScp

greet() = println("Hello World!")

const Func = Union{Nothing,Function}

include("OptimizationProblem.jl")

pbm = OptimizationProblem()

_tf = 1

greet()

# Define variable sizes
pbm.nx = 3
pbm.nu = 2
pbm.np = 1

# Define dynamics
pbm.f = (t, k, x, u, p) -> [u[1]*sin(x[3]); u[1]*cos(x[3]); u[2]]*_tf
pbm.A = (t, k, x, u, p, pbm) ->
        [0 0 u[1]*cos(x[3]);
         0 0 -u[1]*sin(x[3]);
         0 0 0]*_tf
pbm.B = (t, k, x, u, p, pbm) ->
        [sin(x[3]) 0;
         cos(x[3]) 0;
         0 1]*_tf
pbm.F = (t, k, x, u, p, pbm) -> zeros(pbm.nx, pbm.np)

# Define constraints
# TODO

# Define boundary conditions
# TODO

println(pbm.f)

end # module MyScp
