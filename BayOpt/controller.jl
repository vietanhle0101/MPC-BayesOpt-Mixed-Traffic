using LinearAlgebra, Distributions, Polynomials
using Optim, Convex, JuMP, OSQP, Ipopt
# using Gurobi, KNITRO

# GRB_ENV = Gurobi.Env()
const cutoff = 1e-3 # small cutoff in constraints to avoid numerical issues with optimization solvers

"Function to get control input for the HDVs in the simulation"
function input_for_HDV(paths::Vector{Path}, cars::Vector{Car}, my_car::Int64, your_car::Int64, W; T = 0.2, uncertain = false, solver = "Ipopt")
    my_path = cars[my_car].path
    my_p = cars[my_car].p - paths[my_path].conflicts["p_from_entry"][1]
    my_v = cars[my_car].v
    your_path = cars[your_car].path
    your_p = cars[your_car].p - paths[my_path].conflicts["p_from_entry"][1]
    your_v = cars[your_car].v

    u = IRL_CFM(my_p, my_v, your_p, your_v, W; T, uncertain)
    return u
end

"Solve optimization problem (IRL) to find control action for HDV"
function IRL_CFM(p2_0, v2_0, p1_0, v1_0, W; T = 0.2, v_ref = 12., r = 0., β = 1., uncertain = false)
    C = zeros(7)
    C[1] = W[1] + W[2]*T^2
    C[2] = 2W[2]*(v2_0 - v_ref)*T
    C[3] = W[2]*(v2_0 - v_ref)^2
    C[4] = W[3]
    C[5] = (0.5T^2)^2
    C[6] = 2*0.5T^2*(p2_0+T*v2_0)
    C[7] = ((p1_0+v1_0*T)^2+(p2_0+v2_0*T)^2-r^2)

    # The solution of this problem can be found by finding root of a polynomial (derivative of objective function)
    root = roots(Polynomial([(C[2]C[7]-C[4]C[6]), (C[2]C[6]+2C[1]C[7]-2C[4]C[5]), (C[2]C[5]+2C[1]C[6]), 2C[1]C[5]]))
    sol = root[isreal.(root)]
    u = Real(sol[1])

    if uncertain
        return u + rand(Normal(0., 0.1))
    else
        return u
    end
end

"The class to implement the MPC"
mutable struct MPC_Planner
    T::Float64
    H::Int64
    N::Int64
    params; v_min; v_max; u_min; u_max; v_ref; r
    st; input; u_nom; x_nom
    solver

    function MPC_Planner(T::Float64, H::Int64, N::Int64)
        obj = new(T, H, N)
        return obj
    end
end

"Set the control parameters"
function set_params(c::MPC_Planner, control_params)
    c.params = control_params
end

"Set the physical limits of the traffic and vehciles"
function set_limit(c::MPC_Planner, bounds, v_ref, r)
    c.v_min = bounds["v_min"]; c.v_max = bounds["v_max"]
    c.u_min = bounds["u_min"]; c.u_max = bounds["u_max"]
    c.v_ref = v_ref
    c.r = r
end

"Set state from all agent states"
function set_state(c::MPC_Planner, cars::Vector{Car})
    c.st = []
    for car in cars
        c.st = [c.st; car.x]
    end
end

"Set input from all agent inputs"
function set_input(c::MPC_Planner, cars::Vector{Car})
    c.input = []
    for car in cars
        c.input = [c.input; car.u]
    end
end

"Predict nominal trajectory over control horizon given the nominal control inputs"
function set_nominal(c::MPC_Planner, nom_inputs::AbstractMatrix)
    c.u_nom = deepcopy(nom_inputs)
    c.x_nom = zeros(2c.N, c.H+1)
    c.x_nom[:,1] = c.st
    for k in 1:c.H
        dx = []
        for i in 1:c.N
            dx = [dx; [c.T*c.x_nom[2i,k]+0.5c.T^2*c.u_nom[i,k]; c.T*c.u_nom[i,k]]]
        end
        c.x_nom[:,k+1] = c.x_nom[:,k] + dx
    end
end

"Function to solve MPC motion planning problem"
function MPC(c::MPC_Planner, node_dict; solver = "Ipopt")
    c.solver = solver
    ## Use Jump and Ipopt, or KNITRO
    if c.solver == "Ipopt"
        model = JuMP.Model(Ipopt.Optimizer)
    elseif c.solver == "KNITRO"
        model = JuMP.Model(KNITRO.Optimizer)
        set_optimizer_attribute(model, "algorithm", 5)
        set_optimizer_attribute(model, "numthreads", 4)
        set_optimizer_attribute(model, "ma_terminate", 1)
    end
    set_time_limit_sec(model, 0.2)
    set_silent(model)
    @variable(model, u[1:c.N,1:c.H])
    set_start_value.(u, c.u_nom)
    p0 = c.st[1:2:end]
    v0 = c.st[2:2:end]
    v = @expression(model, v0 .+ cumsum(u*c.T, dims=2))
    dp = hcat([k==1 ? c.T*v0 + 0.5c.T^2*u[:,k] : c.T*v[:,k-1] + 0.5c.T^2*u[:,k] for k in 1:c.H]...)
    p = @expression(model, p0 .+ cumsum(dp, dims=2))
    # Constraints
    @constraints(model, begin
        c.u_min .<= u .<= c.u_max
        c.v_min .<= v .<= c.v_max
    end)

    # Objective function
    J = sum(c.params["W1"].*u.^2) + sum(c.params["W2"].*(v.-c.v_max).^2)
    for (_, N) in node_dict
        i, j = N["Paths"] .+ 1
        p_i, p_j = N["Position_from_entry"]
        di = p_i .- p[i,:]
        dj = p_j .- p[j,:]
        for k in 1:c.H
            J = @NLexpression(model, J - c.params["W3"]*log(c.params["β"]*(di[k]^2 + dj[k]^2)))
        end
        # Safety constraints
        @constraints(model, begin
            c.r^2 + cutoff .<= di.^2 .+ dj.^2
        end)
    end
    @NLobjective(model, Min, J)

    t_comp = @elapsed JuMP.optimize!(model)
    c.u_nom = value.(u)
    return t_comp
end

"Check one step constraint"
function check_constraint(c::MPC_Planner, x::AbstractVector, u::AbstractVector, node_dict; safe_con = false)
    d = [c.u_min-u[1], u[1]-c.u_max, c.v_min-(x[2]+c.T*u[1]), (x[2]+c.T*u[1])-c.v_max]
    if safe_con == false # If no safety constraint
        return d
    else # If have safety constraint
        d = []
        for (_, N) in node_dict
            i, j = N["Paths"] .+ 1
            p_i, p_j = N["Position_from_entry"]

            if i == 1 || j == 1
                append!(d, c.r^2-(x[2i-1]+c.T*x[2i]+0.5c.T^2*u[i]-p_i)^2-(x[2j-1]+c.T*x[2j]+0.5c.T^2*u[j]-p_j)^2)
            end
        end
        return vcat([c.u_min-u[1], u[1]-c.u_max, c.v_min-(x[2]+c.T*u[1]), (x[2]+c.T*u[1])-c.v_max], d)
    end
end

"Check constraint violation of nominal trajectory over control horizon"
function check_violation(c::MPC_Planner, X::AbstractMatrix, U::AbstractMatrix, node_dict)
    violation = false
    for k in 1:c.H
        if any(check_constraint(c, X[:,k], U[:,k], node_dict) .> cutoff)
            violation = true; break
        end
    end
    return violation
end

"Function to find initial feasible solution by grid-search"
function init_nominal_sol(c::MPC_Planner, node_dict; step_size = 0.5)
    u_init = zeros(size(c.u_nom))
    u_init[1,:] = c.u_min*ones(size(u_init[1,:]))
    set_nominal(c, u_init)
    cur_idx = 1
    while check_violation(c, c.x_nom, c.u_nom, node_dict)
        u_init[1, cur_idx] += step_size
        set_nominal(c, u_init)
        cur_idx += 1
        if cur_idx == c.H
            cur_idx = 1
        end
        if u_init[1,1] == c.u_max
            return Nothing, "INF"
        end
    end
    return u_init, "FEA"
end
