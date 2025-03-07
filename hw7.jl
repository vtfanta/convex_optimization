## Task 9.30    Gradient and Newton methods
using LinearAlgebra
using Random

Random.seed!(123)
n = 100
m = 200
A = randn(m, n)

# first part of cost function
fₐ(x) = -sum(log.(1 .- A * x))
∇fₐ(x) = sum([(1 - A[k,:]' * x) \ A[k,:] for k = 1:m])
∇²fₐ(x) = sum([A[k,:] * A[k,:]' / (1 - A[k,:]' * x)^2 for k = 1:m])

# second part of cost function
fᵦ(x) = -sum(log.(1 .- x.^2))
∇fᵦ(x) = 2x ./ (1 .- x.^2)
∇²fᵦ(x) = 2diagm(1 ./ (1 .- x.^2)) + 2diagm(2x.^2 ./ (1 .- x.^2).^2) |> Diagonal

# full cost function
f(x) = fₐ(x) + fᵦ(x)
∇f(x) = ∇fₐ(x) + ∇fᵦ(x)
∇²f(x) = ∇²fₐ(x) + ∇²fᵦ(x)

g(x) = A' * (1. ./ (1 .- A * x)) - 1 ./ (1 .+ x) + 1 ./ (1 .- x)

function backtracking_line_search(f, ∇f, x, Δx; α = 0.2, β = 0.7)
    t = 1.
    # check domain and Armijo condition
    while maximum(abs.(x + t*Δx)) ≥ 1. || maximum(A*(x + t*Δx)) ≥ 1. || f(x + t*Δx) ≥ f(x) + α*t*∇f(x)'*Δx
        t *= β
    end
    return t
end

function gradient_descent_step(f, ∇f, x; α = 0.2, β = 0.7)
    Δx = -∇f(x)
    t = backtracking_line_search(f, ∇f, x, Δx; α, β)
    return x + t * Δx, t
end

function gradient_descent!(f, ∇f, x; ϵ = 1e-4, α = 0.2, β = 0.7, max_iter = 1000)
    k = 0
    while norm(∇f(x)) > ϵ && k < max_iter
        x⁺, _ = gradient_descent_step(f, ∇f, x; α, β)
        x .= x⁺
        k += 1
    end
    x, k == max_iter ? :max_iter : :converged
end

function newton_method_step(f, ∇f, ∇²f, x; α = 0.2, β = 0.7)
    Δx = -∇²f(x) \ ∇f(x)
    λ² = Δx' * ∇²f(x) * Δx
    t = backtracking_line_search(f, ∇f, x, Δx; α, β)
    return x + t * Δx, λ², t
end

function newton_method!(f, ∇f, ∇²f, x; ϵ = 1e-4, α = 0.2, β = 0.7, max_iter = 1000)
    k = 0
    λ² = Inf
    while λ²/2 > ϵ && k < max_iter
        x⁺, λ²⁺, _ = newton_method_step(f, ∇f, ∇²f, x; α, β)
        x .= x⁺
        λ² = λ²⁺
        k += 1
    end
    x, k == max_iter ? :max_iter : :converged
end

# initial points
x = zeros(n)
x_opt, status = gradient_descent!(f, ∇f, x, ϵ = 1e-4, α = 0.4)
x_opt_newton, status_newton = newton_method!(f, ∇f, ∇²f, x, ϵ = 1e-4, α = 0.4)

function time_gradient_descent(f, ∇f, x0)
    x = copy(x0)
    gradient_descent!(f, ∇f, x)
end

# x, t = gradient_descent_step(f, ∇f, x)
function time_newton_method(f, ∇f, ∇²f, x0)
    x = copy(x0)
    newton_method!(f, ∇f, ∇²f, x)
end

# using BenchmarkTools
# @benchmark time_gradient_descent(f, ∇f, x)
# @benchmark time_newton_method(f, ∇f, ∇²f, x)

## Try solving algebraic continuous-time Ricatti equation using Newton's method
A = [1 -2 3;
     -4 5 6;
     7 8 9]
B = [5, 6, -7]
C = [7 -8 9]
Q = C'*C
R = I(size(B, 2))

p2P(p) = [p[1] p[2] p[3];
          p[2] p[4] p[5];
          p[3] p[5] p[6]]

P2p(P) = [P[1,1], P[1,2], P[1,3], P[2,2], P[2,3], P[3,3]]

function ricatti(p)
    P = p2P(p)
    A'*P + P*A - P*B/R*B'*P + Q
end

function ∇ricatti(p)
    P = p2P(p)
    A' + A - B/R*B'*P + P*B/R*B'
end


p = ones(6)
P = p2P(p) - ∇ricatti(p) \ ricatti(p)
p = P2p(P)
# does not seem to converge