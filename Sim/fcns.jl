"Some neccesary functions"
function sigmoid(x)
    return 1/(1 + exp(-x))
end

function sigmoid_der(x)
    return sigmoid(x)*(1-sigmoid(x))
end
