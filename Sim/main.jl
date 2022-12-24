# push!(LOAD_PATH, ".")
using Pkg; Pkg.activate("."); Pkg.instantiate()

include("fcns.jl")
include("path.jl")
include("car.jl")
include("controller.jl")
include("estimator.jl")
include("util.jl")

using Interpolations, Random, DelimitedFiles

## Look-up table to find the optimal control weights
grid_size = (9,9)
nodes = (Vector(LinRange(-2.0, 2.0, grid_size[1])), Vector(LinRange(-2.0, 2.0, grid_size[2])))
TABLE = readdlm("./input/Time-Energy/Dec24_weight_table_81_TE.csv", ',')
itp_w1 = interpolate(nodes, reshape(TABLE[:,1], grid_size[1], grid_size[2]), Gridded(Interpolations.Linear()))
itp_w2 = interpolate(nodes, reshape(TABLE[:,2], grid_size[1], grid_size[2]), Gridded(Interpolations.Linear()))
itp = (itp_w1, itp_w2)

## If we want to use quadratic/cubic spline instead of linear interpolations
function my_f(data, x1, x2)
    l = size(data)[1]
    for i in 1:l
        if x1 == data[i,3] && x2 == data[i,4]
            return (data[i,1], data[i,2])
        end
    end
end
A1 = [my_f(TABLE, x1, x2)[1] for x1 in nodes[1], x2 in nodes[2]]
sitp_w1 = Interpolations.scale(interpolate(A1, BSpline(Cubic(Line(OnGrid())))), # Linear(), Quadratic(Line(OnGrid()))
                        -2.0:0.5:2.0, -2.0:0.5:2.0)
A2 = [my_f(TABLE, x1, x2)[2] for x1 in nodes[1], x2 in nodes[2]]
sitp_w2 = Interpolations.scale(interpolate(A2, BSpline(Cubic(Line(OnGrid())))), # Linear(), Quadratic(Line(OnGrid()))
                        -2.0:0.5:2.0, -2.0:0.5:2.0)
sitp = (sitp_w1, sitp_w2)

## Load map and initialize a list of path objects
segment_dict, path_dict, node_dict = load_data()

path1 = Path(path_dict["P1"]["Segments"])
path2 = Path(path_dict["P2"]["Segments"])
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

## Run single simulation to get the trajectory
P_hist, V_hist, U_hist, comp_time, cost, safe = main_BO(sitp, paths, 10 .^[-1.0, 0.5];
                                                    p0 = 5.0, v0 = 9.0, solver = "Ipopt", H = 10)

# Plot the results
using Plots
plot(P_hist')
plot(V_hist')
plot(U_hist[1,:])
plot(comp_time)

# Save data for later animation in python
# prefix = string(Dates.monthname(today())[1:3], Dates.day(today()))
# file_name = string("./results/", prefix, "_trajectory.csv")
# U_hist = hcat(U_hist, [NaN, NaN])
# DATA = vcat(P_hist, V_hist, U_hist)'
# writedlm(file_name, DATA)

## Comparison between BayOpt and SVO
BO_stats = Dict("SAFE" => [], "COST" => [])
SVO_stats = Dict("SAFE" => [], "COST" => [])
seed = 111; Random.seed!(seed)
n_simulation = 2000

for i in 1:n_simulation
    println("Simulation ", i)
    p0 = rand(Uniform(-10.0, 10.0))
    v0 = rand(Uniform(5.0, 12.0))
    W_A = 10 .^ [rand(Uniform(-2.0, 2.0)), rand(Uniform(-2.0, 2.0))]
    _, _, _, _, cost1, _ = main_BO(itp, paths, W_A; p0 = p0, v0 = v0, solver = "Ipopt", H = 10)
    _, _, _, _, cost2, _ = main_SVO(paths, W_A; p0 = p0, v0 = v0, solver = "Ipopt", H = 10)

    push!(BO_stats["COST"], cost1)
    push!(SVO_stats["COST"], cost2)

    if i%10 == 0
        # Number of simulations with safe guarantee
        println("Number of safe simumations: ", count(BO_stats["COST"] .< 1e3), ", ", count(SVO_stats["COST"] .< 1e3))

        idx = [i for i in 1:length(BO_stats["COST"]) if BO_stats["COST"][i] < 1e3 && SVO_stats["COST"][i] < 1e3]

        # Number of simulations with improvements
        n_improv = count(SVO_stats["COST"][idx] .> BO_stats["COST"][idx])/length(idx)
        println("Percentages of improvements: ", n_improv)

        # Average percentage of improvements
        aver_improv = mean((SVO_stats["COST"][idx] - BO_stats["COST"][idx])./SVO_stats["COST"][idx])
        println("Average improvements: ", aver_improv)
    end
end

# Number of simulations with safe guarantee
println("Number of safe simumations: ", count(BO_stats["COST"] .< 1e3), ", ", count(SVO_stats["COST"] .< 1e3))

idx = [i for i in 1:length(BO_stats["COST"]) if BO_stats["COST"][i] < 1e3 && SVO_stats["COST"][i] < 1e3]

# Number of simulations with improvements
n_improv = count(SVO_stats["COST"][idx] .> BO_stats["COST"][idx])/length(idx)
println("Percentages of improvements: ", n_improv)

# Average percentage of improvements
aver_improv = mean((SVO_stats["COST"][idx] - BO_stats["COST"][idx])./SVO_stats["COST"][idx])
println("Average improvements: ", aver_improv)

println("THAT IS THE END !!!")

## Save data for later animation in python
# using Dates
# prefix = string(Dates.monthname(today())[1:3], Dates.day(today()))
# file_name = string("./results/", prefix, "_statistics_", seed, ".csv")
# STATS = hcat(BO_stats["COST"], SVO_stats["COST"])
# writedlm(file_name, STATS)


## To build GUROBI
# ENV["GUROBI_HOME"] = "/Library/gurobi950/mac64"
# import Pkg
# Pkg.add("Gurobi")
# Pkg.build("Gurobi")


## To build KNITRO
# ENV["LD_LIBRARY_PATH"] = "/Users/vietanhle/Documents/KNITRO/knitro-13.1.0-MacOS-64/lib"
# import Pkg
# Pkg.add("KNITRO")
# Pkg.build("KNITRO")

