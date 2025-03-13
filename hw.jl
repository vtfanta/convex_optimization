using LinearAlgebra # don't need anything else 😼

function backtracking_line_search(A, b, c, x, ν, Δx, Δν; α = 0.1, β = 0.7)
    t = 1.0
    # assert feasibility of x + t * Δx
    while any(x + t * Δx .≤ 0.)
        t *= β
    end
    # primal-dual Armijo condition
    while norm([
            c - 1. ./ (x + t * Δx) + A' * (ν + t * Δν);
            A * (x + t * Δx) - b
        ]) > (1 - α * t) * norm([
            c - 1. ./ x + A' * ν;
            A * x - b
        ])
        t *= β
    end
    return t
end

function newton_method(A, b, c, x0; ϵ = 1e-5, α = 0.1, β = 0.7, max_iters = 1000)
    m, n = size(A)
    x = copy(x0)

    λ² = Inf
    ν = ones(m) # don't know if I can do this
    num_iter = 0

    while λ²[1] / 2 > ϵ && num_iter < max_iters
        ∇f = c - 1 ./ x # this assumes strictly feasible x
        ∇²f = Diagonal(x.^(-2))   # this too
        g = ∇f + A' * ν
        h = A * x - b
        H = ∇²f
        
        # solving the KKT system via subblock elimination
        Δν = ((A / H * A') \ (h - A / H * g))[:]
        Δx = (- H \ (g + A' * Δν))[:]
        
        λ² = (- Δx' * ∇f)[1]
        
        t = backtracking_line_search(A, b, c, x, ν, Δx, Δν; α, β)
        
        x += t * Δx
        ν += t * Δν
        
        num_iter += 1
    end
    
    return x, ν, num_iter, num_iter == max_iters ? :max_iters : :converged
end

## LP solver with strictly feasible initial point

function barrier_method(A, b, c, x0; μ = 15, ϵ = 1e-3, α = 0.1, β = 0.7, max_iters = 1000)
    t = 0.1
    x = copy(x0)
    n = length(x)
    history = Matrix{Float64}(undef, 2, 0)
    num_iter = 0
    while num_iter < max_iters
        x_opt, ν_opt, iters, _ = newton_method(A, b, t*c, x; α, β, max_iters)
        duality_gap = n / t
        history = [history [iters; duality_gap]]
        if duality_gap < ϵ
            return x_opt, ν_opt, history
        end 
        t *= μ
        num_iter += 1
    end
end

## LP solver
function lp_solver(A, b, c; ϵ = 1e-4)
    m, n = size(A)
    x0 = A \ b
    min_x0 = minimum(x0)

    if min_x0 ≤ 0
        # phase I to find strictly feasible initial point
        t0 = 2 - min_x0
        # z = x + (t - 1) * ones(n)
        # x = z - (t - 1) * ones(n)
        # t * ones(n) = z - x
        z0 = x0 + (t0 - 1) * ones(n)    
        Ā = [-A * ones(n) A]
        b̄ = b - A * ones(n)
        c̄ = [1; ones(n)]
        # @info "Starting phase I with t0 = $t0."
        t_z_opt, _, _ = barrier_method(Ā, b̄, c̄, [t0; z0])
        t_opt, z_opt = t_z_opt[1], t_z_opt[2:end]
        # @info "Phase I completed with t = $t_opt."
        if t_opt ≥ 1
            return NaN, NaN, :infeasible
        end
        x0 = z_opt + (1 - t_opt) * ones(n)
    end
    # phase II
    # @info "Starting phase II."
    x, ν, _ = barrier_method(A, b, c, x0; ϵ = ϵ)
    return x, c'*x, :optimal
end

"""
    find_optimal_charging_plan(
        a::Vector{Float64},
        c::Vector{Float64},
        m::Vector{Float64},
        r::Vector{Float64},
        d::Vector{Int64}
    )

Computes an optimal charging schedule for `N` electric vehicles over `K` hours.

# Arguments
- `a`: A `K`-element vector specifying the maximum available charging energy per hour (kWh).
- `c`: A `K`-element vector representing the cost of charging per hour (€/kWh).
- `m`: An `N`-element vector with the maximum allowed charging energy for each vehicle (kWh).
- `r`: An `N`-element vector specifying the total energy required by each vehicle (kWh).
- `d`: An `N`-element vector indicating the departure time (hour) of each vehicle.

# Returns
A tuple containing:
- An `N × K` matrix representing the optimal charging schedule (kWh allocated per vehicle per hour).
- The optimal total charging cost (€).
- A symbol indicating the type of optimization problem solved (`:LP`, `:QP`, or `:NLP`).
"""
function find_optimal_charging_plan(
    a::Vector{Float64},
    c::Vector{Float64},
    m::Vector{Float64},
    r::Vector{Float64},
    d::Vector{Int64}
)

    K = length(a) # Timespan (hours)
    N = length(m) # Number of vehicles

    num_cars = N
    num_hours = K

    deadlines = d
    wanted_energy = r
    max_power_to_car = m
    energy_cost_per_hour = c
    available_energy_per_hour = a

    # convert to standard form LP
    b₁ = copy(wanted_energy)
    b₂ = zeros(num_cars)
    b₃ = vcat([ones(num_hours) * max_power_to_car[i] for i in 1:num_cars]...)
    b₄ = copy(available_energy_per_hour)

    A₁ = fill(NaN, num_cars, 2*N*K+N+K)
    A₂ = fill(NaN, num_cars, 2*N*K+N+K)
    A₃ = fill(NaN, N*K, 2*N*K+N+K)
    A₄ = fill(NaN, K, 2*N*K+N+K)

    A₄[:, :] .= [kron(ones(1, N), I(K)) zeros(K, N) zeros(K, N*K) I(K)]

    for n = 1:num_cars
        if n == 1
            A₁[n, :] .=
                [ones(deadlines[n]); zeros(K-deadlines[n]); zeros(K*(N-1)); -1; zeros(N-1); zeros(K*(N+1))];
                
            A₂[n, :] .=
                [zeros(deadlines[n]); ones(K - deadlines[n]); zeros(K*(N-1)); zeros(N*K+N+K)]
                
            A₃[1:K, :] .=
                [I(K) zeros(K, K*(N-1)) zeros(K, N) I(K) zeros(K, K*(N-1)) zeros(K, K)]
        elseif n == num_cars
            A₁[n, :] .=
                [zeros(K*(N-1)); ones(deadlines[n]); zeros(K-deadlines[n]); zeros(N-1); -1; zeros(K*(N+1))]
                
            A₂[n, :] .=
                [zeros(K*(N-1)); zeros(deadlines[n]); ones(K-deadlines[n]); zeros(N*K+N+K)]
                
            A₃[(n-1)*K+1:n*K, :] .=
                [zeros(K, K*(N-1)) I(K) zeros(K, N) zeros(K, K*(N-1)) I(K) zeros(K, K)]
        else
            A₁[n, :] .=
                [zeros(K*(n-1)); ones(deadlines[n]); zeros(K-deadlines[n]); zeros(K*(N-n)); zeros(n-1); -1; zeros(N-n); zeros(K*(N+1))]
                
            A₂[n, :] .=
                [zeros(K*(n-1)); zeros(deadlines[n]); ones(K-deadlines[n]); zeros(K*(N-n)); zeros(N*K+N+K)]
                
            A₃[(n-1)*K+1:n*K, :] .=
                [zeros(K, K*(n-1)) I(K) zeros(K, K*(N-n)) zeros(K, N) zeros(K, K*(n-1)) I(K) zeros(K, K*(N-n)) zeros(K, K)]
        end
    end

    # Abig = [A₁; A₂; A₃; A₄]
    # bbig = vcat(b₁, b₂, b₃, b₄)
    # try omitting the zero power after deadline, perhaps improves conditioning
    Abig = [A₁; A₃; A₄]
    bbig = vcat(b₁, b₃, b₄)
    cbig = [kron(ones(N, 1), energy_cost_per_hour); zeros(N*K+N+K)]

    x_opt, opt_val, status = lp_solver(Abig, bbig, cbig)
    x_reshaped = Array(reshape(x_opt[1:N*K], (K, N))')

    return x_reshaped, opt_val[1], :LP # or :LP or :QP
end