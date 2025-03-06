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

x = zeros(n)