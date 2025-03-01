using Clarabel
using Convex
using MAT

## task 4 in additional exercises, log-optimal investment strategy
P = [3.5000    1.1100    1.1100    1.0400    1.0100;
     0.5000    0.9700    0.9800    1.0500    1.0100;
     0.5000    0.9900    0.9900    0.9900    1.0100;
     0.5000    1.0500    1.0600    0.9900    1.0100;
     0.5000    1.1600    0.9900    1.0700    1.0100;
     0.5000    0.9900    0.9900    1.0600    1.0100;
     0.5000    0.9200    1.0800    0.9900    1.0100;
     0.5000    1.1300    1.1000    0.9900    1.0100;
     0.5000    0.9300    0.9500    1.0400    1.0100;
     3.5000    0.9900    0.9700    0.9800    1.0100];
m, n = size(P)
x_unif = ones(n) / n
probabilities = ones(m) / m

x = Variable(n)
constraints = []
push!(constraints, x ≥ 0)
push!(constraints, sum(x) == 1)
criterion = [probabilities[j] * log(dot(P[j, :], x)) for j in 1:m] |> sum
problem = maximize(criterion, constraints)
solve!(problem, Clarabel.Optimizer)
x_opt = evaluate(x)

# simulate the investment strategy
using Random
Random.seed!(0)
N = 10  # number of simulations
T = 200 # number of time steps
w_opt = []
w_unif = []
for i = 1:N
    events = reshape(ceil.(Int, rand(1, T) * m), T)
    P_event = P[events, :]
    w_opt = vcat(w_opt, [1; cumprod(P_event * x_opt)])
    w_unif = vcat(w_unif, [1; cumprod(P_event * x_unif)])
end

# plot
using Plots
plot(w_opt, color=:green, yscale=:log10, label="log-optimal")
plot!(w_unif, color=:red, style=:dash, yscale=:log10, label="uniform")

## task 5 in additional exercises, maximizing house profit in a gable and imputed probabilities
n = 5
m = 5
A = [1 1 0 0 0; # my A is transpose of the one in the official solution
     0 0 0 1 0;
     1 0 0 1 1;
     0 1 0 0 1;
     0 0 1 0 0]
p = [.5, .6, .6, .6, .2]
q = [10, 5, 5, 20, 10]
x = Variable(n)
t = Variable()
constraints = []
for j = 1:m
    push!(constraints, A[:, j]' * x ≤ t) 
end
push!(constraints, x ≥ 0)
push!(constraints, x ≤ q)
criterion = dot(p, x) - t
problem = maximize(criterion, constraints)
solve!(problem, Clarabel.Optimizer)
x_opt = evaluate(x)
t_opt = evaluate(t)
opt_val = problem.optval
imputed_probabilities = -getproperty.(problem.constraints[1:m], :dual)
@show opt_val

# if all is accepted
t = Variable()
constraints = []
for j = 1:m
    push!(constraints, A[:, j]' * q ≤ t) 
end
problem = maximize(dot(p, q) - t, constraints)
solve!(problem, Clarabel.Optimizer)
@show problem.optval, opt_val


## task 6 in additional exercises, heuristic suboptimal solution for Boolean LP

using HiGHS
using Random
Random.seed!(0)   # for reproducibility

n = 100
m = 300
A = rand(m, n)
b = A * ones(n) / 2
c = -rand(n)

# original ILP
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

# heuristic suboptimal solution
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

# plot 
using Plots
plot(ts, hLP_vals, label="Heuristic LP", xlabel="Threshold", ylabel="Objective value", legend=:topleft)
plot(ts, hLP_maxviol, label="Heuristic LP", xlabel="Threshold", ylabel="Max violation", legend=:topleft)