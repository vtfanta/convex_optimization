using Clarabel
using Convex
using GLMakie

## exercise 5.1

L(x, λ) = x^2 + 1 + λ*(x-2)*(x-4)
lines(-1..4, x -> x^2 +1, linewidth=4)
lines!([2,4],[0,0],color=:red)
scatter!([2],[5],color=:green)

lines!(-1..4, x -> L(x, 1))
lines!(-1..4, x -> L(x, 2))
lines!(-1..4, x -> L(x, 3))

current_figure()

g(λ) = (1+9λ-λ^2)/(1+λ)
lines(0..4, g)

## additional exercise 5
A = [-1 0.4 0.8;
      1 0 0;
      0 1 0]
b = [1, 0, 0.3]
x_des = [7, 2, -6]
N = 30

x = Variable(N+1)
u = Variable(N)