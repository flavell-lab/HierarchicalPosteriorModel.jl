"""
    get_variability(mu::Array{Float64}, sigma::Array{Float64})

Compute the variability of the parameters based on the given mu and sigma arrays.

# Arguments
- `mu::Array{Float64}`: An array containing the mu values of the parameters.
- `sigma::Array{Float64}`: An array containing the sigma values of the parameters.

# Returns
- `variability`: The computed variability of the parameters.
"""
function get_variability(mu::Array{Float64}, sigma::Array{Float64})
    exp_sigma = exp.(sigma)

    variability = exp_sigma[1] + exp_sigma[3] + exp_sigma[5] + exp_sigma[4] * abs(sin(mu[3])) # sigma[4] is phi, which needs to be scaled based on the mean theta value

    return variability
end

"""
    get_variability(model_params::ModelParameters)

Compute the variability metric based on the provided model parameters.

# Arguments
- `model_params`: A `ModelParameters` instance containing the mean (mu) and standard deviation (sigma) parameters.

# Returns
- `variability`: A scalar value representing the computed variability metric.
"""
function get_variability(model_params::ModelParameters)
    mu = model_params.mu
    sigma = model_params.sigma

    return get_variability(mu, sigma)
end

