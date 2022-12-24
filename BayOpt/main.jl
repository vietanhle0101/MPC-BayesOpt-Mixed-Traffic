push!(LOAD_PATH, ".")
using Pkg; Pkg.activate("."); Pkg.instantiate()

include("path.jl")
include("car.jl")
include("controller.jl")
include("util.jl")

using Pickle, Dates, Statistics
using BayesianOptimization, GaussianProcesses, Distributions

## Now run Bayes Opt for all values of W_H
function mymain!(TABLE, grid_size, cost_type; use_multithread = true)
    ## Load map information with pickle
    segment_dict = Pickle.load(open("./input/segment.pkl"))
    path_dict = Pickle.load(open("./input/path.pkl"))
    node_dict = Pickle.load(open("./input/node.pkl"))
    node_dict = Dict("N1" => node_dict["N1"])

    ## Initialize a list of path objects
    path1 = Path(path_dict["P1"]["Segments"])
    path2 = Path(path_dict["P2"]["Segments"])
    # path3 = Path(path_dict["P3"]["Segments"])

    paths = [path1, path2]
    n_paths = length(paths)
    for (_, N) in node_dict
        i, j = N["Paths"] .+ 1
        seg1, seg2 = N["Segments"]
        d1, d2 = N["Distances"]
        add_conflicts(paths[i], segment_dict, j, seg1, d1)
        add_conflicts(paths[j], segment_dict, i, seg2, d2)
    end

    for p in paths
        CZ_length(p, segment_dict)
        sort_conflict(p)
    end

    # Create a grid of all sampled initial conditions
    flat_gridpoints(grids) = vec(collect(Iterators.product(grids...)))
    pp = LinRange(0.0, 10.0, 3)
    vv = LinRange(0.0, 10.0, 3)
    grid = flat_gridpoints((pp, pp, vv, vv))

    # Grid for weights
    n_case = grid_size[1]*grid_size[2]
    W_grid = flat_gridpoints((LinRange(-2.0, 2.0, grid_size[1]), LinRange(-2.0, 2.0, grid_size[2])))

    for ele in W_grid[1:end]
        W_H = vec(collect(ele)); W_AH = 3.
        println("Run Bayes Opt for W_H = ", W_H)
        W_A = run_BayOpt(paths, node_dict, grid, W_H, W_AH, cost_type;
                            n_iters = 40, i_iters = 10, use_multithread = use_multithread)
        println("Solution obtained is W_A = ", W_A)
        println("________________________________")
        # Store results in TABLE
        push!(TABLE, [W_A; W_H; W_AH]')
    end
end

# Run the main function
# Time + Energy efficiency
TABLE = []
grid_size = (9,9)
cost_type = "TE" # "TE", "T", "TS"
@time mymain!(TABLE, grid_size, cost_type; use_multithread = true)

## Save data for later use
using DelimitedFiles
prefix = string(Dates.monthname(today())[1:3], Dates.day(today()))
n_case = grid_size[1]*grid_size[2]
file_name = string("./results/", prefix, "_weight_table_", n_case, "_", cost_type, ".csv")
TABLE = vcat(TABLE...)
writedlm(file_name, TABLE, ',')

# ## To build GUROBI
# ENV["GUROBI_HOME"] = "/Library/gurobi950/mac64"
# import Pkg
# Pkg.add("Gurobi")
# Pkg.build("Gurobi")

## To build KNITRO
# ENV["LD_LIBRARY_PATH"] = "/Users/vietanhle/Documents/KNITRO/knitro-13.1.0-MacOS-64/lib"
# import Pkg
# Pkg.add("KNITRO")
# Pkg.build("KNITRO")
