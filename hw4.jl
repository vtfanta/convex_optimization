using Clarabel
using Convex
using GLMakie; Makie.inline!(true)
using Plots

## exercise 5.1

L(x, λ) = x^2 + 1 + λ*(x-2)*(x-4)
lines(1..4, x -> x^2 +1, linewidth=4)
lines!([2,4],[0,0],color=:red)
scatter!([2],[5],color=:green)

lines!(1..4, x -> L(x, 1))
lines!(1..4, x -> L(x, 2))
lines!(1..4, x -> L(x, 3))

current_figure()

##
g(λ) = (1+9λ-λ^2)/(1+λ)
lines(0..4, g)
## sensitivity analysis
function pstar(u)
    x = Variable()
    prob = minimize(square(x)+1, square(x)-6x+8 ≤ u)
    solve!(prob, Clarabel.Optimizer; silent=true)
    return prob.optval
end

lines(-1..5, pstar; label="numeric", linestyle=:dash)
lines!(-1..5, u -> (3 - sqrt(1+u))^2 +1; color=:red, label="analytic"); current_figure()

## additional exercise 5
A = [-1 0.4 0.8;
      1 0 0;
      0 1 0]
b = [1, 0, 0.3]
x_des = [7, 2, -6]
N = 30
n = size(A, 2)

x = Variable(n, N+1)
u = Variable(N)

constrs = []
for t ∈ 1:N
    push!(constrs, x[:,t+1] == A*x[:,t] + b*u[t])
end
push!(constrs, x[:, 1] == 0)
push!(constrs, x[:, N+1] == x_des)

obj = sum([max(abs(u[t]), 2abs(u[t]) - 1) for t ∈ 1:N])

prob = minimize(obj, constrs)
solve!(prob, Clarabel.Optimizer)

## additional exercise 4.2
m, n = 30, 100
A = rand(m, n) + im*rand(m, n)
b = rand(m) + im*rand(m)
x = ComplexVariable(n)
prob1 = minimize(norm(x, 2), A*x == b)
solve!(prob1, Clarabel.Optimizer)
x1 = evaluate(x)

z = Variable(2n)
Ā = [real(A) -imag(A);
     imag(A) real(A)]
b̄ = [real(b); imag(b)]
prob2 = minimize(norm(z, 2), Ā*z == b̄)
solve!(prob2, Clarabel.Optimizer)   # check out

x = ComplexVariable(n)
prob3 = minimize(norm(x, Inf), A*x == b)
solve!(prob3, Clarabel.Optimizer)
x3 = evaluate(x)

## additional exercise 4.3
x = Variable(2)
u = Variable(2)
fix!(u, [-2, -3])
constraints = []
A = [1 2;1 -4; 5 76]
b = [u; 1]
push!(constraints, A*x ≤ b)
prob1 = minimize(quadform(x, [1 -0.5; -0.5 2]) - x[1], constraints)
solve!(prob1, Clarabel.Optimizer)
λ = getproperty.(constraints, :dual)[1]
xstar = evaluate(x)
# check KKT conditions
# 1. primal feasibility
@show xstar[1] + 2xstar[2],  evaluate(u)[1]
@show xstar[1] - 4xstar[2],  evaluate(u)[2]
@show 5xstar[1] + 76xstar[2],  1
# 2. dual feasibility
@show λ[1], λ[2], λ[3] # I assume in Convex.jl, dual variables are negative?
# 3. complementary slackness
@show λ[1]*(xstar[1] + 2xstar[2] - evaluate(u)[1])
@show λ[2]*(xstar[1] - 4xstar[2] - evaluate(u)[2])
@show λ[3]*(5xstar[1] + 76xstar[2] - 1)
# 4. gradient of Lagrangian 
2*[1 -0.5;-0.5 2]*xstar + [-1,0] - A'*λ # if I take -λ, then it is correct

## 4.3b
x = Variable(2)
u = Variable(2)
δs₁ = [0, -0.1, 0.1]
pstarpred = zeros(length(δs₁)^2)
pstarexact = zeros(length(δs₁)^2)
k = 1
for δ1 in δs₁, δ2 in δs₁
    A = [1 2;1 -4; 5 76]
    fix!(u, [-2+δ1, -3+δ2])
    b = [u; 1]
    prob = minimize(quadform(x, [1 -0.5; -0.5 2]) - x[1], A*x ≤ b)
    solve!(prob, Clarabel.Optimizer; silent=true)
    pstarexact[k] = prob.optval
    if k == 1
        λ = getproperty.(constraints, :dual)[1]
    end
    pstarpred[k] = pstarexact[1] + λ'*[δ1, δ2, 0] # formula 5.57 with flipped sign at λ
    @show δ1, δ2, pstarpred[k], pstarexact[k], pstarpred[k] ≤ pstarexact[k]
    k += 1
end