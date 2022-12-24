# using DifferentialEquations # This package takes too much time to compile

# Define robot kinematic model
function car_ode!(du, u, p, t)
    du[1] = u[2]
    du[2] = p
end

"Class for vehicles"
mutable struct Car
    path::Int64 # Path ID
    T::Float64 # Sampling time
    v_min::Float64; v_max::Float64; # constant of bound constraint
    u_min::Float64; u_max::Float64; # constant of bound constraint
    p::Float64; v::Float64
    x::Vector{Float64}  # Current states: p, v
    d::Vector{Float64}
    u::Float64  # Control inputs, as ODE parameters
    odeprob # ODEProblem for solving the ODE

    function Car(path::Int64, T::Float64, p0::Float64, v0::Float64)
        obj = new(path, T)
        obj.p = p0; obj.v = v0
        obj.x = [p0; v0]
        obj.u = 0.0  # Control inputs
        return obj
    end
end

"Set the physical limit of the car"
function set_limit(a::Car, bounds)
    a.v_min = bounds["v_min"]; a.v_max = bounds["v_max"]
    a.u_min = bounds["u_min"]; a.u_max = bounds["u_max"]
end

"Run"
function run(a::Car, u)
    # Copy the values from u to a.u
    u = min(a.u_max, max(a.u_min, u))
    if a.v + a.T*u < a.v_min # Don't want HDV go backward
        u = (a.v_min-a.v)/a.T
    end
    a.u = u
    # odeprob = ODEProblem(car_ode!, a.x, (0.0, a.T), a.u)
    # sol = solve(odeprob)
    # Update state with the solution, and return it
    # newx = sol[end]
    # a.x .= newx
    a.x += [a.T*a.x[2] + 0.5*a.T^2*u, a.T*u]
    a.p = a.x[1]; a.v = a.x[2]
end

"Compute distance to conflict point"
function dist_to_conf(a::Car, p::Path)
    a.d = a.p .- p.conflicts["p_from_entry"]
end
