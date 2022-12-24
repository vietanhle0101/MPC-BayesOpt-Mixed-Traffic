using Statistics
"I will put all functions here"

σ(x) = 1 / (1 + exp(-x))

function run_single_simulation(paths, node_dict, init_cond, W_A, W_H, W_AH, cost_type; T = 0.2, H = 10, L = 300)
    control_params = Dict("W1" => 10 .^ [W_A[1], W_H[1]], "W2"=> 10 .^ [W_A[2], W_H[2]], "W3"=> 10 .^ W_AH, "β" => 1.0)
    W_irl = [10 .^ W_H; 10 .^ W_AH]
    # velocity and input bounds
    bound = Dict("v_min" => 0.0, "v_max" => 12.0, "u_min" => -5.0, "u_max" => 3.0)
    init_pos = init_cond[1:2]; init_vel = init_cond[3:4]
    ## Initialize the car objects
    car1 = Car(1, T, init_pos[1], init_vel[1])
    car2 = Car(2, T, init_pos[2], init_vel[2])
    cars = [car1, car2]
    n_cars = length(cars)

    for car in cars
        set_limit(car, bound)
        dist_to_conf(car, paths[car.path])
    end

    ## Controller object
    control = MPC_Planner(T, H, n_cars)

    r = 10.; v_ref = 10.
    set_params(control, control_params)
    set_limit(control, bound, v_ref, r)
    set_state(control, cars)
    set_nominal(control, zeros(n_cars, H))

    ## Warm up
    MPC(control, node_dict; solver = "Ipopt") # Available solvers: "KNITRO" "Ipopt"

    ## Main loop
    comp_time = []; t_f1 = t_f2 = T*L; dist = []
    energy = energy_consumption(cars[1].u, cars[1].v, T)

    for t in 1:L
        # println("Time step ", t)
        set_state(control, cars)

        # Start with nominal control inputs
        nom_inputs = hcat(control.u_nom[:,2:end], zeros(n_cars,1))
        set_nominal(control, nom_inputs)
        # if check_violation(control, control.x_nom, control.u_nom, node_dict)
        #     # println("Current nominal control inputs are not feasible!!!")
        #     nom_inputs, flag = init_nominal_sol(control, node_dict; step_size = 0.1)
        #     if flag != "INF"
        #      set_nominal(control, nom_inputs)
        #     else
        #      println("Current optimal control problem is probably not feasible, stop the simulation!!!")
        #      break
        #     end
        # end

        computation_time = MPC(control, node_dict)

        # println("Takes ", computation_time)
        # append!(comp_time, computation_time)

        # HDVs
        U = [control.u_nom[1,1]]
        ui = input_for_HDV(paths, cars, 2, 1, W_irl; uncertain = false)
        append!(U, ui)

        # Run the cars
        for i in 1:n_cars
            run(cars[i], U[i])
            dist_to_conf(cars[i], paths[cars[i].path])
        end

        append!(dist, sqrt(cars[1].d[1]^2 + cars[2].d[1]^2))
        energy += energy_consumption(cars[1].u, cars[1].v, T)

        # Find exit time for each vehcile
        if cars[1].p > paths[cars[1].path].length && t_f1 == T*L
            t_f1 = t*T - (cars[1].p - paths[cars[1].path].length)/cars[1].v
        end
        if cars[2].p > paths[cars[2].path].length && t_f2 == T*L
            t_f2 = t*T - (cars[2].p - paths[cars[2].path].length)/cars[2].v
        end

        # When both vehicles pass the CZ
        if cars[1].p > paths[cars[1].path].length && cars[2].p > paths[cars[2].path].length
            break
        end
    end

    if cost_type == "T" # Time efficiency
        true_cost = t_f1 + 1e3*σ(-100minimum(dist .- r))
        return true_cost
    elseif cost_type == "TE" # Time + Energy efficiency
        true_cost = t_f1 + energy + 1e3*σ(-100minimum(dist .- r))
        return true_cost
    elseif cost_type == "TS" # Time + Social efficiency
        true_cost = t_f1 + t_f2 + energy + 1e3*σ(-100minimum(dist .- r))
    end
end

"This function is to run the simulation, given the weights of the MPC provided"
function run_simulation(paths, node_dict, grid, W_A, W_H, W_AH, cost_type; use_multithread = false)
    println("Bayes Opt samples ", W_A)
    global all_cost = []
    n_epoch = length(grid)
    if use_multithread
        MT = Threads.SpinLock()
        Threads.@threads for epoch in 1:n_epoch
            # println("Epoch: ", epoch)
            # Initial coditions: position and velocities
            init_cond = grid[epoch]
            true_cost = run_single_simulation(paths, node_dict, init_cond, W_A, W_H, W_AH, cost_type)
            # println(true_cost)
            global all_cost
            Threads.lock(MT) do
                append!(all_cost, true_cost)
            end
        end
    else
        for epoch in 1:n_epoch
            # println("Epoch: ", epoch)
            # Initial coditions: position and velocities
            init_cond = grid[epoch]
            true_cost = run_single_simulation(paths, node_dict, init_cond, W_A, W_H, W_AH, cost_type)
            # println(true_cost)
            append!(all_cost, true_cost)
        end
    end
    # Compute the avarage cost
    average_cost = mean(all_cost)
    println("Average cost is ", average_cost)
    return average_cost
end

function run_BayOpt(paths, node_dict, grid, W_H, W_AH, cost_type; n_iters = 10, i_iters = 5, thres = 1e-3, use_multithread = false)
    f_bo(x) = run_simulation(paths, node_dict, grid, x, W_H, W_AH, cost_type; use_multithread = use_multithread)

    # Choose as a model an elastic GP with input dimensions 2.
    # The GP is called elastic, because data can be appended efficiently.
    model = ElasticGPE(
        2, # 2 input dimensions
        mean = MeanConst(0.),
        kernel = Matern(5/2,[0.0,0.0],0.0), # SEArd([0., 0.], 5.),
        logNoise = 0.)
    set_priors!(model.mean, [Normal(10., 2.)])

    # Optimize the hyperparameters of the GP using maximum a posteriori (MAP) estimates every 50 steps
    modeloptimizer = MAPGPOptimizer(every = 1, noisebounds = [-5, 5], # bounds of the logNoise
                                    kernbounds = [[-5, -5, 0], [5, 5, 5]],  # bounds of the 3 parameters GaussianProcesses.get_param_names(model.kernel)
                                    maxeval = 50)
    BayesOpt = BOpt(
        f_bo, model,
        ExpectedImprovement(), # type of acquisition
        modeloptimizer,
        [-2., -2.], [2., 2.], # lowerbounds, upperbounds
        repetitions = 1, maxiterations = i_iters+1, initializer_iterations = i_iters,
        sense = Min, # minimize the function
        acquisitionoptions = (method = :LD_LBFGS, # run optimization of acquisition function with NLopts :LD_LBFGS method
                             restarts = 10, # run the NLopt method from 5 random initial conditions each time.
                             maxtime = .5, # run the NLopt method for at most 0.1 second each time
                             maxeval = 1000), # run the NLopt methods for at most 1000 iterations (for other options see https://github.com/JuliaOpt/NLopt.jl)
        verbosity = Silent)

    solution = boptimize!(BayesOpt)
    if n_iters > i_iters
        maxiterations!(BayesOpt, 1)  # set maxiterations for the next call
        for ite in 1:n_iters
            # previous_optimizer = solution.model_optimizer
            solution = boptimize!(BayesOpt)
            # # Check the residual of new model solution and last model solution
            # # Usually that means the optimizer cannot improve further, so can stop
            # residual = max(norm(solution.model_optimizer - solution.observed_optimizer),
            #                         norm(solution.model_optimizer - previous_optimizer))
            # # println(residual)
            # # early termination in Bayesian Optimization
            # if residual < thres
            #     println("Bay Opt is early terminated at iteration ", ite)
            #     break
            # end
        end
    end
    # return the best solution so far
    W_A = Vector(solution.observed_optimizer)
    return W_A
end

"Function to compute energy consumption (ml/s) from 0 to dt"
function energy_consumption(u_0, v_0, dt)
    energy = 0.0
    b0 = 0.1569; b1 = 2.450*1e-2
    b2 = -7.415*1e-4; b3 = 5.975*1e−5
    c0 = 0.07224; c1 = 9.681*1e−2; c2 = 1.075*1e−3
    n = 1000
    v = v_0; a = max(0, u_0)
    for i in 1:n
        v += u_0*dt/n
        energy += (b0 + b1*v + b2*v^2 + b3*v^3 + a*(c0 + c1*v + c2*v^2))*dt/n
    end
    return energy
end
