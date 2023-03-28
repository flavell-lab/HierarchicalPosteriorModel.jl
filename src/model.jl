"""
struct HBParams

A structure for holding the mean (mu), standard deviation (sigma), and the spherical coordinates (x) of a model.

# Fields
- `mu::Vector{Float64}`: A vector of the global mean estimate for each parameter.
- `sigma::Vector{Float64}`: A vector of the global standard deviation estimate for each parameter.
- `x::Vector{Vector{Float64}}`: A vector of vectors representing the best parameters for each individual dataset.
"""
struct HBParams
    mu::Vector{Float64}
    sigma::Vector{Float64}
    x::Vector{Vector{Float64}}
end

"""
    joint_logprob_flat(params_flat::Vector, data::Vector{Matrix{Float64}}, mvns::Vector, idx_scaling::Vector{Int64})

Compute the joint log probability of a flat parameter vector given the data, multivariate normal distributions (mvns), and scaling indices (idx_scaling).

# Arguments
- `params_flat::Vector`: A flat vector containing the concatenated mu, sigma, and x parameters.
- `data::Vector{Matrix{Float64}}`: A vector of matrices representing the data (CePNEM fit parameters) for each dataset.
- `mvns::Vector`: A vector of multivariate normal distributions corresponding to the data.
- `idx_scaling::Vector{Int64}`: A vector of indices indicating the parameters that need to be transformed to Cartesian coordinates for comparison into `mvns`.
    Currently, it is not supported for this to be any value other than `[2,3,4]`.

# Returns
- `logprob`: The computed joint log probability of the given parameters.
"""
function joint_logprob_flat(params_flat::Vector, data::Vector{Matrix{Float64}}, mvns::Vector, idx_scaling::Vector{Int64})
    @assert(idx_scaling==[2,3,4], "Currently, spherical coordinate transform must consist of parameters 2, 3, and 4.")
    n_params = size(data[1], 2)
    mu = params_flat[1:n_params]
    sigma = params_flat[n_params + 1:2 * n_params]
    x_flat = params_flat[2 * n_params + 1:end]
    x_spher = [x_flat[(i - 1) * n_params + 1 : i * n_params] for i in 1:length(data)]

    logprob = 0.0

    # Add log probability of higher-level parameters
    logprob += sum(Distributions.logpdf.(Normal(0, 1), mu[[i for i in 1:length(mu) if !(i in idx_scaling)]]))  # Prior for c_vT and lambda
    logprob += Distributions.logpdf(Normal(0,1), mu[idx_scaling[1]]) # Prior for r; note that prior for phi and theta is uniform

    logprob += sum(Distributions.logpdf.(Normal(-3, 1), sigma[[i for i in 1:length(mu) if !(i in idx_scaling)]]))  # Prior for sigma_cvT and sigma_lambda
    logprob += Distributions.logpdf(Normal(-1, 1), sigma[idx_scaling[1]])  # Prior for sigma_r
    logprob += sum(Distributions.logpdf.(Normal(-3, 1), sigma[idx_scaling[2:end]]))  # Prior for sigma_theta and sigma_phi

    exp_sigma = exp.(sigma) .+ fill(1e-4, length(sigma))

    # Add log probability of lower-level parameters and data
    for i in 1:length(data)
        x_i_spher = x_spher[i]
        x_i_spher_scaling = [exp(x_i_spher[idx_scaling[1]]), x_i_spher[idx_scaling[2]], x_i_spher[idx_scaling[3]]]

        x_i_cart = [x_i_spher[1], spher2cart(x_i_spher_scaling)..., x_i_spher[5]]

        P_i = data[i]
        for j in 1:size(P_i,2)
            if j in idx_scaling[2:3]
                logprob += Distributions.logpdf(Normal(0, exp_sigma[j]), angle_diff(mu[j], x_i_spher[j]))
            elseif j == idx_scaling[1]
                logprob += Distributions.logpdf(Normal(mu[j], exp_sigma[j]), x_i_spher_scaling[1])
            else
                logprob += Distributions.logpdf(Normal(mu[j], exp_sigma[j]), x_i_spher[j])
            end
        end
        logprob += Distributions.logpdf(mvns[i], x_i_cart)
    end

    return logprob
end

"""
    joint_logprob(params::HBParams, data::Vector{Matrix{Float64}}, mvns::Vector; idx_scaling::Vector{Int64}=[2,3,4])

Compute the joint log probability of a HBParams instance given the data and multivariate normal distributions (mvns).

# Arguments
- `params::HBParams`: A HBParams instance containing the mu, sigma, and x parameters.
- `data::Vector{Matrix{Float64}}`: A vector of matrices representing the data (CePNEM fit parameters) for each dataset.
- `mvns::Vector`: A vector of multivariate normal distributions corresponding to the data.
- `idx_scaling::Vector{Int64}`: An optional vector of indices indicating the parameters that need to be transformed to Cartesian coordinates for comparison into `mvns`.
Currently, it is not supported for this to be any value other than its default, `[2,3,4]`.

# Returns
- `logprob`: The computed joint log probability of the given parameters.
"""
function joint_logprob(params::HBParams, data::Vector{Matrix{Float64}}, mvns::Vector; idx_scaling::Vector{Int64}=[2,3,4])
    n_params = size(data[1], 2)
    mu = params.mu
    sigma = params.sigma
    x_spher = params.x

    params_flat = [mu; sigma; vcat(x_spher...)]

    return joint_logprob_flat(params_flat, data, mvns, idx_scaling=idx_scaling)
end

"""
    joint_logprob_flat_negated(params_flat::Vector, data::Vector{Matrix{Float64}}, mvns::Vector, idx_scaling::Vector{Int64})

Compute the negated joint log probability of a flat parameter vector given the data, multivariate normal distributions (mvns), and scaling indices (idx_scaling).

# Arguments
- `params_flat::Vector`: A flat vector containing the concatenated mu, sigma, and x parameters.
- `data::Vector{Matrix{Float64}}`: A vector of matrices representing the data (CePNEM fit parameters) for each dataset.
- `mvns::Vector`: A vector of multivariate normal distributions corresponding to the data.
- `idx_scaling::Vector{Int64}`: A vector of indices indicating the parameters that need to be transformed to Cartesian coordinates for comparison into `mvns`.
Currently, it is not supported for this to be any value other than `[2,3,4]`.

# Returns
- `logprob`: The computed negated joint log probability of the given parameters.
"""
function joint_logprob_flat_negated(params_flat::Vector, data::Vector{Matrix{Float64}}, mvns::Vector, idx_scaling::Vector{Int64})
    return -joint_logprob_flat(params_flat, data, mvns, idx_scaling)
end