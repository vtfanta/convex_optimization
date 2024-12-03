#= try if 
    max. dot(c,ν)
    subject to A*ν ≥ -b

    is equivalent to
    min. dot(c,x)
    subject to A*x ≤ b
    =#
using Clarabel
using Convex
n = 3
m = 5

A = [-1 1 -3;
    -5 -2 1;
    -1 0 0;
    0 -1 0;
    0 0 -1]
b = [-10, -6, 0, 0, 0]
c = [7, 1, 5]

ν = Variable(n)
c1 = A*ν ≥ -b
obj1 = dot(c, ν)
p1 = maximize(obj1, c1)
solve!(p1, Clarabel.Optimizer)

x = Variable(n)
c2 = A*x ≤ b
obj2 = dot(c, x)
p2 = minimize(obj2, c2)
solve!(p2, Clarabel.Optimizer)