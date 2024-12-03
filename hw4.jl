using Clarabel
using Convex

# additional exercise 5
A = [-1 0.4 0.8;
      1 0 0;
      0 1 0]
b = [1, 0, 0.3]
x_des = [7, 2, -6]
N = 30

x = Variable(N+1)
u = Variable(N)

constrs = []
for t ∈ 1:N
    push!(constrs, x[t+1] == A*x[t] + b*u[t])
end
push!(constrs, x[1] .== 0)
push!(constrs, x[N+1] == x_des)