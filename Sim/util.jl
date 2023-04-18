using Pickle

"Function for simulation with BayOpt"
function main_BO_GP(gp, paths, W_H; p0 = 0.0, v0 = 10.0, T = 0.2, H = 20, G = 20, L = 200,
                r = 10., v_ref = 10., solver = "Ipopt")
    # velocity and input bounds
    bound = Dict("v_min" => 0.0, "v_max" => 12.0, "u_min" => -5.0, "u_max" => 3.0)

    ## Initialize the car objects
    car1 = Car(1, T, 0., 10.)
    car2 = Car(2, T, p0, v0)
    cars = [car1, car2]
    n_cars = length(cars)

    for car in cars
        set_limit(car, bound)
        dist_to_conf(car, paths[car.path])
    end

    # W_H = [1e1, 1e-1]
    gp1, gp2 = gp
    W_AH = 1e3; W_A = 10.0 .^ [predict_y(gp1, log.(W_H)[:,:])[1][1], predict_y(gp2, log.(W_H)[:,:])[1][1] ]
    
    ## Controller object
    control = MPC_Planner(T, H, n_cars)
    control_params = Dict("W1" => [W_A[1], W_H[1]], "W2" => [W_A[2], W_H[2]], "W3" => W_AH, "β" => 1.0)
    W_irl = [W_H; W_AH]

    set_params(control, control_params)
    set_limit(control, bound, v_ref, r)

    set_state(control, cars)
    set_input(control, cars)
    set_nominal(control, zeros(n_cars, H))

    ## Warm up
    MPC(control, node_dict; solver = solver) # Available solvers: "KNITRO" "Ipopt"

    # Some array to store data
    P_hist = zeros(n_cars, 0); V_hist = zeros(n_cars, 0); U_hist = zeros(n_cars, 0)
    P_hist = hcat(P_hist, zeros(n_cars)); V_hist = hcat(V_hist, zeros(n_cars))

    for i in 1:n_cars
        P_hist[i,end] = cars[i].d[1]
        V_hist[i,end] = cars[i].v
    end

    ## Estimation object
    estimator = MHIRL(T, G)
    set_weights(estimator, [0., 0., 3.]) # Initial arbitrary weights

    ## Main loop
    comp_time = []; t_f1 = t_f2 = T*L; dist = []
    energy = energy_consumption(cars[1].u, cars[1].v, T)
    passed = false # boolean to say a car passed CZ

    for t in 1:L
        # println("Time step ", t)
        set_state(control, cars)
        set_input(control, cars)

        computation_time = 0.0 # To measure computation time (for both IRL and MPC)

        # Learn the HDV weight with MHIRL
        if t > G && ~passed
            # Extract the data first
            data = Dict("P" => P_hist[:, end-G:end], "V" => V_hist[:, end-G:end], "U" => U_hist[2, end-G+1:end])
            computation_time += @elapsed learn_weights(estimator, control, data; n_iter = 20, η = 1e-2, decay = 0.)
            # println(estimator.W)
        elseif t > 2 && ~passed
            data = Dict("P" => P_hist[:, :], "V" => V_hist[:, :], "U" => U_hist[2, :])
            computation_time += @elapsed learn_weights(estimator, control, data; n_iter = 20, η = 1e-2, decay = 0.)
            # println(estimator.W)
        end

        # Start with nominal control inputs
        nom_inputs = hcat(control.u_nom[:,2:end], zeros(n_cars,1))
        set_nominal(control, nom_inputs)

        W_A = [predict_y(gp1, estimator.W[1:2][:,:])[1][1], predict_y(gp2, estimator.W[1:2][:,:])[1][1]]

        control_params["W1"] = [3*10^W_A[1], 10^estimator.W[1]]
        control_params["W2"] = [3*10^W_A[2], 10^estimator.W[2]]
        set_params(control, control_params)

        computation_time += @elapsed MPC(control, node_dict) # "KNITRO" "Ipopt"

        # println("Takes ", computation_time)
        append!(comp_time, computation_time)

        # HDVs
        U = [control.u_nom[1,1]]
        ui = input_for_HDV(paths, cars, 2, 1, W_irl; T = T, uncertain = false)
        append!(U, ui)
        # All CAVs
        # U = control.u_nom[:,1]

        # Run the cars
        P_hist = hcat(P_hist, zeros(n_cars)); V_hist = hcat(V_hist, zeros(n_cars)); U_hist = hcat(U_hist, zeros(n_cars))
        for i in 1:n_cars
            run(cars[i], U[i])
            dist_to_conf(cars[i], paths[cars[i].path])
            P_hist[i,end] = cars[i].d[1]
            V_hist[i,end] = cars[i].v
            U_hist[i,end] = cars[i].u
        end
        append!(dist, sqrt(cars[1].d[1]^2 + cars[2].d[1]^2))
        energy += energy_consumption(cars[1].u, cars[1].v, T)

        # Find exit time for each vehcile
        if cars[1].p > paths[cars[1].path].length && t_f1 == T*L
            t_f1 = t*T - (cars[1].p - paths[cars[1].path].length)/cars[1].v; passed = true
        end
        if cars[2].p > paths[cars[2].path].length && t_f2 == T*L
            t_f2 = t*T - (cars[2].p - paths[cars[2].path].length)/cars[2].v
        end

        # When both vehicles pass the CZ
        if cars[1].p > paths[cars[1].path].length && cars[2].p > paths[cars[2].path].length
            break
        end
    end

    safe = minimum(dist .- r) >= 0.0 ? 1 : 0
    cost = t_f1 + energy + 1e3*(1-safe)

    return P_hist, V_hist, U_hist, comp_time, cost, safe
end

"Function for simulation with BayOpt"
function main_BO(itp, paths, W_H; p0 = 0.0, v0 = 10.0, T = 0.2, H = 20, G = 20, L = 200,
                r = 10., v_ref = 10., solver = "Ipopt")
    # velocity and input bounds
    bound = Dict("v_min" => 0.0, "v_max" => 12.0, "u_min" => -5.0, "u_max" => 3.0)

    ## Initialize the car objects
    car1 = Car(1, T, 0., 10.)
    car2 = Car(2, T, p0, v0)
    cars = [car1, car2]
    n_cars = length(cars)

    for car in cars
        set_limit(car, bound)
        dist_to_conf(car, paths[car.path])
    end

    # W_H = [1e1, 1e-1]
    itp_w1, itp_w2 = itp
    W_AH = 1e3; W_A = lookup_table(itp_w1, itp_w2, W_H)

    ## Controller object
    control = MPC_Planner(T, H, n_cars)
    control_params = Dict("W1" => [W_A[1], W_H[1]], "W2" => [W_A[2], W_H[2]], "W3" => W_AH, "β" => 1.0)
    W_irl = [W_H; W_AH]

    set_params(control, control_params)
    set_limit(control, bound, v_ref, r)

    set_state(control, cars)
    set_input(control, cars)
    set_nominal(control, zeros(n_cars, H))

    ## Warm up
    MPC(control, node_dict; solver = solver) # Available solvers: "KNITRO" "Ipopt"

    # Some array to store data
    P_hist = zeros(n_cars, 0); V_hist = zeros(n_cars, 0); U_hist = zeros(n_cars, 0)
    P_hist = hcat(P_hist, zeros(n_cars)); V_hist = hcat(V_hist, zeros(n_cars))

    for i in 1:n_cars
        P_hist[i,end] = cars[i].d[1]
        V_hist[i,end] = cars[i].v
    end

    ## Estimation object
    estimator = MHIRL(T, G)
    set_weights(estimator, [0., 0., 3.]) # Initial arbitrary weights

    ## Main loop
    comp_time = []; t_f1 = t_f2 = T*L; dist = []
    energy = energy_consumption(cars[1].u, cars[1].v, T)
    passed = false # boolean to say a car passed CZ

    for t in 1:L
        # println("Time step ", t)
        set_state(control, cars)
        set_input(control, cars)

        computation_time = 0.0 # To measure computation time (for both IRL and MPC)

        # Learn the HDV weight with MHIRL
        if t > G && ~passed
            # Extract the data first
            data = Dict("P" => P_hist[:, end-G:end], "V" => V_hist[:, end-G:end], "U" => U_hist[2, end-G+1:end])
            computation_time += @elapsed learn_weights(estimator, control, data; n_iter = 20, η = 1e-2, decay = 0.)
            # println(estimator.W)
        elseif t > 2 && ~passed
            data = Dict("P" => P_hist[:, :], "V" => V_hist[:, :], "U" => U_hist[2, :])
            computation_time += @elapsed learn_weights(estimator, control, data; n_iter = 20, η = 1e-2, decay = 0.)
            # println(estimator.W)
        end

        # Start with nominal control inputs
        nom_inputs = hcat(control.u_nom[:,2:end], zeros(n_cars,1))
        set_nominal(control, nom_inputs)

        W_A = lookup_table(itp_w1, itp_w2, 10 .^ estimator.W[1:2])

        control_params["W1"] = [W_A[1], 10^estimator.W[1]]
        control_params["W2"] = [W_A[2], 10^estimator.W[2]]
        set_params(control, control_params)

        computation_time += @elapsed MPC(control, node_dict) # "KNITRO" "Ipopt"

        # println("Takes ", computation_time)
        append!(comp_time, computation_time)

        # HDVs
        U = [control.u_nom[1,1]]
        ui = input_for_HDV(paths, cars, 2, 1, W_irl; T = T, uncertain = false)
        append!(U, ui)
        # All CAVs
        # U = control.u_nom[:,1]

        # Run the cars
        P_hist = hcat(P_hist, zeros(n_cars)); V_hist = hcat(V_hist, zeros(n_cars)); U_hist = hcat(U_hist, zeros(n_cars))
        for i in 1:n_cars
            run(cars[i], U[i])
            dist_to_conf(cars[i], paths[cars[i].path])
            P_hist[i,end] = cars[i].d[1]
            V_hist[i,end] = cars[i].v
            U_hist[i,end] = cars[i].u
        end
        append!(dist, sqrt(cars[1].d[1]^2 + cars[2].d[1]^2))
        energy += energy_consumption(cars[1].u, cars[1].v, T)

        # Find exit time for each vehcile
        if cars[1].p > paths[cars[1].path].length && t_f1 == T*L
            t_f1 = t*T - (cars[1].p - paths[cars[1].path].length)/cars[1].v; passed = true
        end
        if cars[2].p > paths[cars[2].path].length && t_f2 == T*L
            t_f2 = t*T - (cars[2].p - paths[cars[2].path].length)/cars[2].v
        end

        # When both vehicles pass the CZ
        if cars[1].p > paths[cars[1].path].length && cars[2].p > paths[cars[2].path].length
            break
        end
    end

    safe = minimum(dist .- r) >= 0.0 ? 1 : 0
    cost = t_f1 + energy + 1e3*(1-safe)

    return P_hist, V_hist, U_hist, comp_time, cost, safe
end

"Function for simulation with SVO"
function main_SVO(paths, W_H; p0 = 0.0, v0 = 10.0, T = 0.2, H = 20, G = 20, L = 200,
                r = 10., v_ref = 10., solver = "Ipopt")
    # velocity and input bounds
    bound = Dict("v_min" => 0.0, "v_max" => 12.0, "u_min" => -5.0, "u_max" => 3.0)

    ## Initialize the car objects
    car1 = Car(1, T, 0., 10.)
    car2 = Car(2, T, p0, v0)
    cars = [car1, car2]
    n_cars = length(cars)

    for car in cars
        set_limit(car, bound)
        dist_to_conf(car, paths[car.path])
    end

    # W_H = [1e0, 1e1]
    W_AH = 1e3; W_A = [7e-1, 7e-1] # Might be the best emperically (to have same safety level)

    ## Controller object
    control = MPC_Planner(T, H, n_cars)
    control_params = Dict("W1" => [W_A[1], W_H[1]], "W2" => [W_A[2], W_H[2]], "W3" => W_AH, "β" => 1.0)
    W_irl = [W_H; W_AH]

    set_params(control, control_params)
    set_limit(control, bound, v_ref, r)

    set_state(control, cars)
    set_input(control, cars)
    set_nominal(control, zeros(n_cars, H))

    ## Warm up
    MPC(control, node_dict; solver = solver) # Available solvers: "KNITRO" "Ipopt"

    # Some array to store data
    P_hist = zeros(n_cars, 0); V_hist = zeros(n_cars, 0); U_hist = zeros(n_cars, 0)
    P_hist = hcat(P_hist, zeros(n_cars)); V_hist = hcat(V_hist, zeros(n_cars))

    for i in 1:n_cars
        P_hist[i,end] = cars[i].d[1]
        V_hist[i,end] = cars[i].v
    end

    ## Estimation object
    estimator = MHIRL(T, G)
    set_weights(estimator, [0., 0., 3.]) # Initial arbitrary weights
    set_SVO(estimator, 0.) # Initial arbitrary weights

    ## Main loop
    comp_time = []; t_f1 = t_f2 = T*L; dist = []
    energy = energy_consumption(cars[1].u, cars[1].v, T)
    passed = false # boolean to say a car passed CZ

    for t in 1:L
        # println("Time step ", t)
        set_state(control, cars)
        set_input(control, cars)

        computation_time = 0.0 # To measure computation time (for both IRL and MPC)

        # Learn the HDV weight with MHIRL
        if t > G && ~passed
            # Extract the data first
            data = Dict("P" => P_hist[:, end-G:end], "V" => V_hist[:, end-G:end], "U" => U_hist[2, end-G+1:end])
            computation_time += @elapsed learn_SVO(estimator, control, data; n_iter = 20, η = 1e0, decay = 0.)
            # println(estimator.θ)
        elseif t > 2 && ~passed
            data = Dict("P" => P_hist[:, :], "V" => V_hist[:, :], "U" => U_hist[2, :])
            computation_time += @elapsed learn_SVO(estimator, control, data; n_iter = 20, η = 1e0, decay = 0.)
            # println(estimator.θ)
        end

        # Start with nominal control inputs
        nom_inputs = hcat(control.u_nom[:,2:end], zeros(n_cars,1))
        set_nominal(control, nom_inputs)

        control_params["W1"] = [W_A[1]*cot(π/2-estimator.θ), 10^estimator.W[1]*cot(estimator.θ)]
        control_params["W2"] = [W_A[2]*cot(π/2-estimator.θ), 10^estimator.W[2]*cot(estimator.θ)]
        set_params(control, control_params)

        computation_time += @elapsed MPC(control, node_dict) # "KNITRO" "Ipopt"

        # println("Takes ", computation_time)
        append!(comp_time, computation_time)

        # HDVs
        U = [control.u_nom[1,1]]
        ui = input_for_HDV(paths, cars, 2, 1, W_irl; T = T, uncertain = false)
        append!(U, ui)
        # All CAVs
        # U = control.u_nom[:,1]

        # Run the cars
        P_hist = hcat(P_hist, zeros(n_cars)); V_hist = hcat(V_hist, zeros(n_cars)); U_hist = hcat(U_hist, zeros(n_cars))
        for i in 1:n_cars
            run(cars[i], U[i])
            dist_to_conf(cars[i], paths[cars[i].path])
            P_hist[i,end] = cars[i].d[1]
            V_hist[i,end] = cars[i].v
            U_hist[i,end] = cars[i].u
        end
        append!(dist, sqrt(cars[1].d[1]^2 + cars[2].d[1]^2))
        energy += energy_consumption(cars[1].u, cars[1].v, T)

        # Find exit time for each vehcile
        if cars[1].p > paths[cars[1].path].length && t_f1 == T*L
            t_f1 = t*T - (cars[1].p - paths[cars[1].path].length)/cars[1].v; passed = true
        end
        if cars[2].p > paths[cars[2].path].length && t_f2 == T*L
            t_f2 = t*T - (cars[2].p - paths[cars[2].path].length)/cars[2].v
        end

        # When both vehicles pass the CZ
        if cars[1].p > paths[cars[1].path].length && cars[2].p > paths[cars[2].path].length
            break
        end
    end

    safe = minimum(dist .- r) >= 0.0 ? 1 : 0
    cost = t_f1 + energy + 1e3*(1-safe)

    return P_hist, V_hist, U_hist, comp_time, cost, safe
end

"Function to find CAV weights with look-up table generated by BayOpt"
function lookup_table(itp_w1, itp_w2, W_H::AbstractVector)
    W_H1 = max(-2.0, min(log10.(W_H)[1], 2.0))
    W_H2 = max(-2.0, min(log10.(W_H)[2], 2.0))
    W_A1 = itp_w1(W_H1, W_H2)
    W_A2 = itp_w2(W_H1, W_H2)
    return 10 .^ [W_A1, W_A2]
end

"Function to compute energy consumption (ml/s) from 0 to dt"
function energy_consumption(u_0, v_0, dt)
    energy = 0.0
    b0 = 0.1569; b1 = 2.450*1e-2
    b2 = -7.415*1e−4; b3 = 5.975*1e−5
    c0 = 0.07224; c1 = 9.681*1e−2; c2 = 1.075*1e−3
    n = 1000
    v = v_0; a = max(0, u_0)
    for i in 1:n
        v += u_0*dt/n
        energy += (b0 + b1*v + b2*v^2 + b3*v^3 + a*(c0 + c1*v + c2*v^2))*dt/n
    end
    return energy
end

"Function to load traffic data from pickle files"
function load_data()
    segment_dict = Pickle.load(open("./input/segment.pkl"))
    path_dict = Pickle.load(open("./input/path.pkl"))
    node_dict = Pickle.load(open("./input/node.pkl"))
    node_dict = Dict("N1" => node_dict["N1"])

    return segment_dict, path_dict, node_dict
end
