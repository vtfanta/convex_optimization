## Task a.1
using Clarabel, Convex
import Convex: MOI
k = 201
t = [-3 + 6(i-1)/(k-1) for i in 1:k]
y = exp.(t)


γ̅ = 100
γ̲ = 0.001
optval = Inf
aopt = zeros(3)
bopt = zeros(2)
while γ̅ -γ̲ > 1e-3
    # formulate quasiconvex generalized linear-fractional program as
    # a convex feasibility problem
    γ = (γ̅ + γ̲) / 2
    a = Variable(3)
    b = Variable(2)
    constrs = []
    for i in 1:k
        p = a[1] + a[2] * t[i] + a[3] * t[i]^2
        q = 1 + b[1] * t[i] + b[2] * t[i]^2
        # positive abs
        push!(constrs, p - y[i] * q ≤ γ * q)
        # negative abs
        push!(constrs, -p + y[i] * q ≤ γ * q)
    end
    prob = minimize(1, constrs)
    solve!(prob, Clarabel.Optimizer, silent=true)
    if prob.status == MOI.OPTIMAL
        aopt = evaluate(a)
        bopt = evaluate(b)
        optval = γ
        γ̅ = γ
    else
        γ̲ = γ
    end
end

# plots
using Plots
f̂(t) = (aopt[1] + aopt[2] * t + aopt[3] * t^2) / (1 + bopt[1] * t + bopt[2] * t^2)
plot(t, y, label="y", legend=:topleft)
plot!(f̂, -3, 3, label="f̂", legend=:topleft)

plot(t, y - f̂.(t), label="y - f̂", legend=:topleft)

## Task a.2

## Task a.3
# load variables x and y
include("./data/Matlab-pwl_fit_data.m")

# knot points
a = range(0, 1, length=5)
num_segments = length(a) - 1

αs = Variable(num_segments)
βs = Variable(num_segments)

# border indices of x between knot points
ms = [findlast(x .≤ a[k]) for k in 2:length(a)]

constrs = []
obj = sumsquares(αs[1] * x[1:ms[1]] + βs[1] - y[1:ms[1]])
if num_segments > 1
    for k in 1:num_segments-1
        push!(constrs, 
            αs[k] * x[ms[k]] + βs[k] == αs[k+1] * x[ms[k+1]] + βs[k+1])
        obj += sumsquares(αs[k] .* x[ms[k]+1:ms[k+1]] + βs[k] - y[ms[k]+1:ms[k+1]])
    end
end
prob = minimize(obj, constrs)
solve!(prob, Clarabel.Optimizer)
αopt = evaluate(αs)
βopt = evaluate(βs)

function f(x)
    maximum([αopt[k] * x + βopt[k] for k in 1:num_segments])
end

plot(x, y, label="y", legend=:topleft)
plot!(x, f.(x), label="f̂", legend=:topleft)
# this is probably not correct

## Task a.4
using Clarabel
using Convex
using LinearAlgebra

Ā = [60 45 -8
     90 30 -30
     0 -8 -4
     30 10 -10]
R = fill(0.05, 4, 3)
b = [-6, -3, 18, -9]
xₗₛ = Ā \ b
ls_nominal_residual_norm = norm(Ā * xₗₛ - b)

x = Variable(size(Ā, 2))
abs_x = Variable(size(x))
y = Variable(size(Ā, 1))
obj = sumsquares(y)
constrs = [
    Ā * x + R * abs_x - b ≤ y,
    Ā * x - R * abs_x - b ≥ -y,
    abs_x ≥ x,
    abs_x ≥ -x
]
prob = minimize(obj, constrs)
solve!(prob, Clarabel.Optimizer)
# this is not correct somehow

# second variant from the solution
obj = norm(abs(Ā * x - b) + R * abs(x), 2)
prob = minimize(obj, [])
solve!(prob, Clarabel.Optimizer)

wc_nominal_residual_norm = norm(Ā * evaluate(x) - b)
wc_robust_residual_norm = prob.optval
fix!(x, xₗₛ)
solve!(prob, Clarabel.Optimizer)
ls_robust_residual_norm = prob.optval

println("Nominal residual norm: LS $(ls_nominal_residual_norm), WC $(wc_nominal_residual_norm)")
println("Robust residual norm: LS $(ls_robust_residual_norm), WC $(wc_robust_residual_norm)")