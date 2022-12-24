
W_H = [-1.0, 1.0]; W_AH = 3.0
f_bo(x) = run_simulation(paths, node_dict, grid, x, W_H, W_AH, cost_type; use_multithread = use_multithread)
n_iters = 10; i_iters = 5; thres = 1e-3; use_multithread = true
# Choose as a model an elastic GP with input dimensions 2.
# The GP is called elastic, because data can be appended efficiently.
model = ElasticGPE(
    2, # 2 input dimensions
    mean = MeanConst(0.),
    kernel =  Matern(5/2,[0.0,0.0],0.0),
    logNoise = 0.)
set_priors!(model.mean, [Normal(10., 2.)])

# Optimize the hyperparameters of the GP using maximum a posteriori (MAP) estimates every 50 steps
modeloptimizer = MAPGPOptimizer(every = 1, noisebounds = [-10, 10], # bounds of the logNoise
                                kernbounds = [[-10, -10, 0], [10, 10, 10]],  # bounds of the 3 parameters GaussianProcesses.get_param_names(model.kernel)
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
maxiterations!(BayesOpt, 1)  # set maxiterations for the next call
for ite in 1:n_iters
    previous_optimizer = solution.model_optimizer
    solution = boptimize!(BayesOpt)
    # Check the residual of new model solution and last model solution
    # Usually that means the optimizer cannot improve further, so can stop
    residual = max(norm(solution.model_optimizer - solution.observed_optimizer),
                            norm(solution.model_optimizer - previous_optimizer))
    println(residual)
    # early termination in Bayesian Optimization
    if residual < thres
        println("Bay Opt is early terminated at iteration ", ite)
        break
    end
end
# return the best solution so far
W_A = Vector(solution.observed_optimizer)
