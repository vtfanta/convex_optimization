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