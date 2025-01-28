using Clarabel
using Convex

## task 4 in additional exercises, log-optimal investment strategy


## task 6 in additional exercises, heuristic suboptimal solution for Boolean LP

using HiGHS
using Random
Random.seed!(0)   # for reproducibility

n = 100
m = 300
A = rand(m, n)
b = A * ones(n) / 2
c = -rand(n)

# Original ILP
xILP = Variable(n, BinVar)
problemILP = minimize(dot(c, xILP), A*xILP ≤ b)
solve!(problemILP, HiGHS.Optimizer)

ILP_val = problemILP.optval
ILP_x = evaluate(xILP)

## LP relaxation
xLP = Variable(n)
constrs = []
push!(constrs, A*xLP ≤ b)
push!(constrs, 0 ≤ xLP)
push!(constrs, xLP ≤ 1)
problemLP = minimize(dot(c, xLP), constrs)
solve!(problemLP, Clarabel.Optimizer)

LP_val = problemLP.optval
LP_x = evaluate(xLP)
LP_maxviol = maximum(A*LP_x - b)

# Heuristic suboptimal solution
ts = range(0, 1, length=100)
hLP_vals = []
hLP_maxviol = []
for t in ts
    xhLP = Variable(n)
    constrs = []
    push!(constrs, A*xhLP ≤ b)
    push!(constrs, 0 ≤ xhLP)
    push!(constrs, xhLP ≤ 1)
    prob = minimize(dot(c, xhLP), constrs)
    solve!(prob, Clarabel.Optimizer)
    x_tmp = evaluate(xhLP)
    x_tmp = [x ≥ t ? 1 : 0  for x in x_tmp]
    push!(hLP_vals, prob.optval)
    push!(hLP_maxviol, maximum(A*x_tmp - b))
end

# Plot 
using Plots
plot(ts, hLP_vals, label="Heuristic LP", xlabel="Threshold", ylabel="Objective value", legend=:topleft)
plot(ts, hLP_maxviol, label="Heuristic LP", xlabel="Threshold", ylabel="Max violation", legend=:topleft)