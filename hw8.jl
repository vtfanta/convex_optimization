using LinearAlgebra

## Centering step of standard-form LP solver, HW8

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

    while λ²[1] / 2 > ϵ && num_iter < max_iters
        ∇f = c - 1 ./ x # this assumes strictly feasible x
        ∇²f = Diagonal(x.^(-2))   # this too
        g = ∇f + A' * ν
        h = A * x - b
        H = ∇²f
        
        # solving the KKT system via subblock elimination
        # σs = svdvals(A / H * A' + 1e-1I(size(A, 1)))
        # condition_number = σs[1] / σs[end]
        # @show size(H), condition_number
        # correction to make regularity more probable
        # if condition_number > 20.
            # Δν = (A / H * A' + 1e-1I(size(A, 1))) \ (h - A / H * g)
        # else
            Δν = ((A / H * A') \ (h - A / H * g))[:]
        # end
        Δx = (- H \ (g + A' * Δν))[:]
        
        λ² = (- Δx' * ∇f)[1]
        # @show λ², Δx |> size, A |> size, A / H * A'|>size
        
        t = backtracking_line_search(A, b, c, x, ν, Δx, Δν; α, β)
        
        x += t * Δx
        ν += t * Δν
        
        num_iter += 1
    end
    
    return x, ν, num_iter, num_iter == max_iters ? :max_iters : :converged
end

## LP solver with strictly feasible initial point

function barrier_method(A, b, c, x0; μ = 15, ϵ = 1e-3, α = 0.1, β = 0.7, max_iters = 1000)
    t = 0.1
    x = copy(x0)
    n = length(x)
    history = Matrix{Float64}(undef, 2, 0)
    num_iter = 0
    while num_iter < max_iters
        x_opt, ν_opt, iters, _ = newton_method(A, b, t*c, x; α, β, max_iters)
        duality_gap = n / t
        history = [history [iters; duality_gap]]
        if duality_gap < ϵ
            return x_opt, ν_opt, history
        end 
        t *= μ
        num_iter += 1
    end
end

## LP solver
function lp_solver(A, b, c; ϵ = 1e-4)
    m, n = size(A)
    x0 = A \ b
    min_x0 = minimum(x0)

    if min_x0 ≤ 0
        # phase I to find strictly feasible initial point
        t0 = 2 - min_x0
        # z = x + (t - 1) * ones(n)
        # x = z - (t - 1) * ones(n)
        # t * ones(n) = z - x
        z0 = x0 + (t0 - 1) * ones(n)    
        Ā = [-A * ones(n) A]
        b̄ = b - A * ones(n)
        c̄ = [1; ones(n)]
        @info "Starting phase I with t0 = $t0."
        t_z_opt, _, _ = barrier_method(Ā, b̄, c̄, [t0; z0])
        t_opt, z_opt = t_z_opt[1], t_z_opt[2:end]
        @info "Phase I completed with t = $t_opt."
        if t_opt ≥ 1
            return NaN, NaN, :infeasible
        end
        x0 = z_opt + (1 - t_opt) * ones(n)
    end
    # phase II
    @info "Starting phase II."
    x, ν, _ = barrier_method(A, b, c, x0; ϵ = ϵ)
    return x, c'*x, :optimal
end
    

## test
using Clarabel
using Convex
using Random
# Random.seed!(1)

m, n = 3, 5
A = randn(m, n)
x0 = rand(n)
# A = [1 1 0 0 3;
#      1 7 0 0 3;
#      0 6 1 5 0]
# b = [5, 11, 13]
b = A * rand(n)
# x0 += 1e-1 * rand(n)    # make it infeasible
c = A' * randn(m) + rand(n)

# x, ν, num_iter, status = newton_method(A, b, c, x0)
# x, ν, history = barrier_method(A, b, c, A\b; μ = 15)
x, ν, status = lp_solver(A, b, c, ϵ = 1e-5)
# using BenchmarkTools
# @btime lp_solver($A, $b, $c, ϵ = 1e-5)

x_var = Variable(n)
prob = minimize(c' * x_var, [A * x_var == b, x_var ≥ 0])
solve!(prob, Clarabel.Optimizer)
x_clarabel = evaluate(x_var)

@show norm(x - x_clarabel)

# using Plots
# plot!(cumsum(history[1,:]), history[2,:], yaxis=:log10, linetype=:steppost)
#

## Try on HW1 from ORR
using Clarabel
using Convex

d = [3, 7, 10]
r = [15., 25, 30]
m = [6., 6, 4]
a = [11.6, 11.9, 10.6, 8.8, 8., 8.8, 10.6, 11.9, 11.6, 10.]
c = [.58, .72, .92, .68, .54, .78, .64, .57, .74, .74]
# a = [30.5, 50.2, 40.1, 20.3, 50.0, 30.0, 40.0, 20.0, 30.0, 40.0, 50.0, 20.0, 30.0, 40.0, 50.0, 20.0, 30.0, 40.0, 50.0, 20.0, 30.0, 40.0, 50.0, 20.0]
# c = [0.03, 0.94, 0.86, 0.34, 0.96, 0.53, 0.91, 0.03, 0.29, 0.11, 0.91, 0.38, 0.43, 0.89, 0.45, 0.51, 0.46, 0.88, 0.62, 0.65, 0.42, 0.63, 0.80, 0.25]
# m = [7.0, 3.0, 8.0, 12.0, 5.0, 6.0, 4.0, 9.0, 10.0, 11.0]
# r = [10.0, 27.0, 34.0, 20.0, 15.0, 25.0, 40.0, 20.0, 6.0, 18.0]
# d = [12, 15, 18, 21, 23, 16, 19, 22, 23, 22]

N, K = length(d), length(a)
num_cars = N
num_hours = K

deadlines = d
wanted_energy = r
max_power_to_car = m
energy_cost_per_hour = c
available_energy_per_hour = a

X = Variable(num_cars, num_hours)
constrs = []
push!(constrs, X ≥ 0)
for car in 1:num_cars
    push!(constrs, sum(X[car, 1:deadlines[car]]) ≥ wanted_energy[car])
    push!(constrs, sum(X[car, deadlines[car]+1:end]) == 0)
    push!(constrs, X[car, :] ≤ max_power_to_car[car])  
end
push!(constrs, sum(X, dims=1) ≤ available_energy_per_hour')
obj = sum(X * energy_cost_per_hour)
prob = minimize(obj, constrs)
solve!(prob, Clarabel.Optimizer)

# convert to standard form LP
b₁ = copy(wanted_energy)
b₂ = zeros(num_cars)
b₃ = vcat([ones(num_hours) * max_power_to_car[i] for i in 1:num_cars]...)
b₄ = copy(available_energy_per_hour)

A₁ = fill(NaN, num_cars, 2*N*K+N+K)
A₂ = fill(NaN, num_cars, 2*N*K+N+K)
A₃ = fill(NaN, N*K, 2*N*K+N+K)
A₄ = fill(NaN, K, 2*N*K+N+K)

A₄[:, :] .= [kron(ones(1, N), I(K)) zeros(K, N) zeros(K, N*K) I(K)]

for n = 1:num_cars
    if n == 1
        A₁[n, :] .=
            [ones(deadlines[n]); zeros(K-deadlines[n]); zeros(K*(N-1)); -1; zeros(N-1); zeros(K*(N+1))];
            
        A₂[n, :] .=
            [zeros(deadlines[n]); ones(K - deadlines[n]); zeros(K*(N-1)); zeros(N*K+N+K)]
            
        A₃[1:K, :] .=
            [I(K) zeros(K, K*(N-1)) zeros(K, N) I(K) zeros(K, K*(N-1)) zeros(K, K)]
    elseif n == num_cars
        A₁[n, :] .=
            [zeros(K*(N-1)); ones(deadlines[n]); zeros(K-deadlines[n]); zeros(N-1); -1; zeros(K*(N+1))]
            
        A₂[n, :] .=
            [zeros(K*(N-1)); zeros(deadlines[n]); ones(K-deadlines[n]); zeros(N*K+N+K)]
            
        A₃[(n-1)*K+1:n*K, :] .=
            [zeros(K, K*(N-1)) I(K) zeros(K, N) zeros(K, K*(N-1)) I(K) zeros(K, K)]
    else
        A₁[n, :] .=
            [zeros(K*(n-1)); ones(deadlines[n]); zeros(K-deadlines[n]); zeros(K*(N-n)); zeros(n-1); -1; zeros(N-n); zeros(K*(N+1))]
            
        A₂[n, :] .=
            [zeros(K*(n-1)); zeros(deadlines[n]); ones(K-deadlines[n]); zeros(K*(N-n)); zeros(N*K+N+K)]
            
        A₃[(n-1)*K+1:n*K, :] .=
            [zeros(K, K*(n-1)) I(K) zeros(K, K*(N-n)) zeros(K, N) zeros(K, K*(n-1)) I(K) zeros(K, K*(N-n)) zeros(K, K)]
    end
end

# Abig = [A₁; A₂; A₃; A₄]
# bbig = vcat(b₁, b₂, b₃, b₄)
# try omitting the zero power after deadline, perhaps improves conditioning
Abig = [A₁; A₃; A₄]
bbig = vcat(b₁, b₃, b₄)
cbig = [kron(ones(N, 1), c); zeros(N*K+N+K)]

z = Variable(length(cbig))
constrs = []
push!(constrs, Abig * z == bbig)
push!(constrs, z ≥ 0)
prob = minimize(cbig' * z, constrs)
solve!(prob, Clarabel.Optimizer)

x_opt, opt_val, status = lp_solver(Abig, bbig, cbig)