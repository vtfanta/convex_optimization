using Clarabel
using Convex
using GLMakie

# additional exercise 5
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

z = Variable(2n)
Ā = [real(A) -imag(A);
     imag(A) real(A)]
b̄ = [real(b); imag(b)]
prob2 = minimize(norm(z, 2), Ā*z == b̄)
solve!(prob2, Clarabel.Optimizer)