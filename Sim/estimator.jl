
"Class to implement moving horizon inverse reinforcement learning"
mutable struct MHIRL
    T::Float64 # Sampling time
    G::Int64 # Estimation horizon
    W # Estimated weights
    θ::Float64 # SVO angle
    ψ::Float64 # variable to parameterize SVO, θ = π/2*sigmoid(ψ)
    solver

    function MHIRL(T::Float64, G::Int64)
        obj = new(T, G)
        obj.W = [0., 0., 3.]
        return obj
    end
end

"Set the weights"
function set_weights(m::MHIRL, WW)
    m.W = WW
end

"Set the SVO, the input is ψ and θ = π/2*sigmoid(ψ)"
function set_SVO(m::MHIRL, ψ0::Float64)
    m.ψ = ψ0
    m.θ = π/2*sigmoid(m.ψ)
end

"Compute the objective"
function compute_objective(m::MHIRL, c::MPC_Planner, d1, d2, v1, v2, u2, WW)
    f_a = WW[1]*u2^2
    f_v = WW[2]*(v2-c.v_max)^2
    f_d = -WW[3]*log(c.params["β"]*(d1^2 + d2^2))
    return [f_a, f_v, f_d]
end

"Function to learn the weights with MHIRL"
function learn_weights(m::MHIRL, c::MPC_Planner, data; n_iter = 10, η = 1e1, decay = 0.1, ϵ = 1e-3)
    G = min(m.G, length(data["U"]))
    for ite in 1:n_iter
        η *= (1-decay) # decrease learning rate every iteration
        W_old = deepcopy(m.W)
        global f_all = zeros(3, G)
        global f_sol_all = zeros(3, G)
        MT = Threads.SpinLock()
        Threads.@threads for l in 1:G # Multi-thread for faster
            d1 = data["P"][1,l:l+1]; d2 = data["P"][2,l:l+1]
            v1 = data["V"][1,l:l+1]; v2 = data["V"][2,l:l+1]
            u2 =  data["U"][l]
            WW = 10 .^ m.W
            u2_sol = IRL_CFM(d2[1], v2[1], d1[1], v1[1], WW; β = 1.)
            v2_pred = v2[1] + m.T*u2_sol
            d2_pred = d2[1] + m.T*v2[1] + 0.5m.T^2*u2_sol
            global f_all, f_sol_all
            Threads.lock(MT) do
                f_all[:,l] = compute_objective(m, c, d1[2], d2[2], v1[2], v2[2], u2, WW)
                f_sol_all[:,l] = compute_objective(m, c, d1[2], d2_pred, v1[2], v2_pred, u2_sol, WW)
            end
        end
        ave_error = mean(f_sol_all, dims = 2) - mean(f_all, dims = 2)
        grad_f = ave_error
        new_W = max.(10^-5., min.(10^5., 10 .^ m.W + η*grad_f) ) # Bound constraint
        m.W = log10.(new_W)
        if norm(m.W - W_old) < ϵ
            # println("Converged")
            break
        end
    end
    m.W = m.W ./ (m.W[3]/3.0) # Normalization
end

"Function to learn the SVO with MHIRL"
function learn_SVO(m::MHIRL, c::MPC_Planner, data; n_iter = 10, η = 1e1, decay = 0.1, ϵ = 1e-3)
    G = min(m.G, length(data["U"]))
    for ite in 1:n_iter
        θ = deepcopy(m.θ)
        η *= (1-decay)
        global f_all = zeros(2, G)
        global f_sol_all = zeros(2, G)
        MT = Threads.SpinLock()
        Threads.@threads for l in 1:G # Multi-threading for faster
            d1 = data["P"][1,l:l+1]; d2 = data["P"][2,l:l+1]
            v1 = data["V"][1,l:l+1]; v2 = data["V"][2,l:l+1]
            u2 =  data["U"][l]
            WW = [10^m.W[1]*cot(m.θ), 10^m.W[2]*cot(m.θ), 10^m.W[3]]
            u2_sol = IRL_CFM(d2[1], v2[1], d1[1], v1[1], WW; β = 1.)
            v2_pred = v2[1] + m.T*u2_sol
            d2_pred = d2[1] + m.T*v2[1] + 0.5m.T^2*u2_sol
            global f_all, f_sol_all
            Threads.lock(MT) do
                ff = compute_objective(m, c, d1[2], d2[2], v1[2], v2[2], u2, WW)
                f_all[:,l] = [ff[1] + ff[2], ff[3]]
                ff = compute_objective(m, c, d1[2], d2_pred, v1[2], v2_pred, u2_sol, WW)
                f_sol_all[:,l] = [ff[1] + ff[2], ff[3]]
            end
        end
        ave_error = mean(f_sol_all, dims = 2) - mean(f_all, dims = 2)
        grad_f = dot(ave_error, [-sin(θ), cos(θ)])
        dψ = sum(grad_f)*π/2*sigmoid_der(m.ψ)
        m.ψ = max.(-5, min.(5, m.ψ + η*dψ) ) # Bound constraint
        m.θ = π/2*sigmoid(m.ψ)

        if abs(θ - m.θ) < ϵ
            # println("Converged")
            break
        end
    end
end
