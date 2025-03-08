using LinearAlgebra

## Centering step of standard-form LP solver

function backtracking_line_search(A, b, c, x, ν, Δx, Δν; α = 0.1, β = 0.7)
    t = 1.0
    # assert feasibility of x + t * Δx
    while any(x + t * Δx .≤ 0.)
        t *= β
    end
    # primal-dual Armijo condition
    while norm([
            c - 1. ./ (x + t * Δx) + A' * (ν + t * Δν);
            A * (x + t * Δx) - b
        ]) > (1 - α * t) * norm([
            c - 1. ./ x + A' * ν;
            A * x - b
        ])
        t *= β
    end
    return t
end

function newton_method(A, b, c, x0; ϵ = 1e-5, α = 0.1, β = 0.7, max_iters = 1000)
    m, n = size(A)
    x = copy(x0)

    λ² = Inf
    ν = ones(m) # don't know if I can do this
    num_iter = 0

    while λ² / 2 > ϵ && num_iter < max_iters
        ∇f = c - 1 ./ x # this assumes strictly feasible x
        ∇²f = Diagonal(x.^(-2))   # this too
        g = ∇f + A' * ν
        h = A * x - b
        H = ∇²f
        
        # solving the KKT system via subblock elimination
        Δν = (A / H * A') \ (h - A / H * g)
        Δx = - H \ (g + A' * Δν)
        
        λ² = - Δx' * ∇f
        
        t = backtracking_line_search(A, b, c, x, ν, Δx, Δν; α, β)
        
        x += t * Δx
        ν += t * Δν
        
        num_iter += 1
    end
    
    return x, ν, num_iter, num_iter == max_iters ? :max_iters : :converged
end

## LP solver with strictly feasible initial point

function barrier_method(A, b, c, x0; μ = 15, ϵ = 1e-5, α = 0.1, β = 0.7, max_iters = 1000)
    t = 0.1
    x = copy(x0)
    n = length(x)
    history = Matrix{Float64}(undef, 2, 0)
    num_iter = 0
    while num_iter < max_iters
        x_opt, ν_opt, iters, _ = newton_method(A, b, t*c, ones(n); α, β, max_iters)
        duality_gap = n / t
        history = [history [iters; duality_gap]]
        if duality_gap < ϵ
            return x_opt, ν_opt, history
        end 
        t *= μ
        num_iter += 1
    end
end

# test
using Random
Random.seed!(123)

m, n = 3, 5
A = rand(m, n)
x0 = rand(n)
b = A * x0
# x0 += 1e-1 * rand(n)    # make it infeasible
c = rand(n)

# x, ν, num_iter, status = newton_method(A, b, c, x0)
x, ν, history = barrier_method(A, b, c, x0; μ = 15)

# using Plots
# plot!(cumsum(history[1,:]), history[2,:], yaxis=:log10, linetype=:steppost)