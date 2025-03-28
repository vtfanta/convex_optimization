using Clarabel
using Convex

# problem data
A = [1 2 0 1;
     0 0 3 1;
     0 3 1 1;
     2 1 2 5;
     1 0 3 2]
cᵐ = repeat([100], 5)
p = [3, 2, 7, 6]
pᵈ = [2, 1, 4, 2]
q = [4, 10, 5, 10]

# variables
n = size(A, 2)
x = Variable(n)
t = Variable(n)

constraints = []
for j ∈ 1:n
    push!(constraints, x[j] ≥ 0)
    push!(constraints, t[j] ≥ 0)
    push!(constraints, t[j] ≤ p[j] * x[j])
    push!(constraints, t[j] ≤ p[j] * q[j] + pᵈ[j] * (x[j] - q[j]))
end

for i ∈ 1:size(A, 1)
    push!(constraints, dot(A[i, :], x) ≤ cᵐ[i])
end

objective = -sum(t)
prob = minimize(objective, constraints)
solve!(prob, Clarabel.Optimizer)

x_opt = evaluate(x)

## second variant (without reformulation and leave Convex.jl to handle it)
x2 = Variable(n)
r1 = Variable(n, Positive())
r2 = Variable(n, Positive())
constraints2 = []
for j ∈ 1:n
    push!(constraints2, x2[j] ≥ 0)
    push!(constraints2, dot(A[j, :], x2) ≤ cᵐ[j])
    push!(constraints2, r1[j] == p[j] * x2[j])
    push!(constraints2, r2[j] == p[j] * q[j] + pᵈ[j] * (x2[j] - q[j]))
end
for i ∈ 1:size(A, 1)
    push!(constraints2, dot(A[i, :], x2) ≤ cᵐ[i])
end
objective2 = sum([min(r1[j], r2[j]) for j ∈ 1:n])
prob2 = maximize(objective2, constraints2)
solve!(prob2, Clarabel.Optimizer)
x2_opt = evaluate(x2)

# revenue generated by each activity
r1, r2 = evaluate(r1), evaluate(r2)
revenues = min.(r1, r2)
avg_price_per_unit = revenues ./ x2_opt

## task 3
using MAT
file = matopen("illum_data.mat")
A = read(file, "A")
close(file)

n_patches, n_lamps = size(A)

# equal lamp power
f₀(p, A) = A*p .|> log .|> abs |> maximum
γ = 0.4
p_equal = repeat([γ], n_lamps)
f₀_equal = f₀(p_equal, A)

# least squares with saturations
p_ls = A\ones(n_patches)
p_ls = clamp.(p_ls, 0, 1)
f₀_ls = f₀(p_ls, A)

# regularized least squares
using LinearAlgebra
ρ = 0.1
p_reg = [A; sqrt(ρ) * I(n_lamps)] \ [ones(n_patches); sqrt(ρ)/2 * ones(n_lamps)]
for k = 1:100
    if sum(p_reg .> 1 .|| p_reg .< 0) == 0
        break
    else
        ρ += 0.1
    end
    p_reg = [A; sqrt(ρ) * I(n_lamps)] \ [ones(n_patches); sqrt(ρ)/2 * ones(n_lamps)]
end
f₀_reg = f₀(p_reg, A)

# Chebushev approximation
p = Variable(n_lamps)
cons = []
for i = 1:n_lamps
    push!(cons, p[i] ≥ 0)
    push!(cons, p[i] ≤ 1)
end
obj = norm(A*p - ones(n_patches), Inf)
prob = minimize(obj, cons)
solve!(prob, Clarabel.Optimizer)
p_cheb = evaluate(p)
f₀_cheb = f₀(p_cheb, A)

# exact solution
p = Variable(n_lamps)
cons = []
for i = 1:n_lamps
    push!(cons, p[i] ≥ 0)
    push!(cons, p[i] ≤ 1)
end
obj_true = max([max(dot(A[k,:], p), invpos(dot(A[k,:], p))) for k = 1:n_patches]...)
prob = minimize(obj_true, cons)
solve!(prob, Clarabel.Optimizer)
p_true = evaluate(p)
f₀_true = f₀(p_true, A)