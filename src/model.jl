struct ModelParameters
    mu::Vector{Float64}
    sigma::Vector{Float64}
    x::Vector{Vector{Float64}}
end

function cart2spher(x_cart::Vector)
    r = norm(x_cart)
    theta = acos(x_cart[3] / r)
    phi = atan(x_cart[2], x_cart[1])
    return [r, theta, phi]
end

function spher2cart(x_spher::Vector)
    r, theta, phi = x_spher
    x = r * sin(theta) * cos(phi)
    y = r * sin(theta) * sin(phi)
    z = r * cos(theta)
    return [x, y, z]
end

function angle_diff(theta1, theta2)
    diff = mod(theta1 - theta2 + π, 2π) - π
    return diff
end

function fit_multivariate_normals(Ps::Vector{Matrix{Float64}})
    mvns = Vector{MvNormal}(undef, length(Ps))

    for i in 1:length(Ps)
        P_i = Ps[i]
        μ_i = mean(P_i, dims=1)[:]
        Σ_i = cov(P_i)
        mvn_i = MvNormal(μ_i, Σ_i)
        mvns[i] = mvn_i
    end

    return mvns
end

function bic_multivariate_normals(Ps::Vector{Matrix{Float64}}, mvns::Vector{MvNormal})
    bics = Vector{Float64}(undef, length(Ps))

    for i in 1:length(Ps)
        P_i = Ps[i]
        n = size(P_i, 1)
        k = size(P_i, 2)
        log_likelihood_i = sum(logpdf.(mvns[i], eachrow(P_i)))
        bic_i = -2 * log_likelihood_i + k * log(n)
        bics[i] = bic_i
    end

    return bics
end



function joint_logprob_flat(params_flat::Vector, data::Vector{Matrix{Float64}}, mvns::Vector, idx_scaling::Vector{Int64})
    n_params = size(data[1], 2)
    mu = params_flat[1:n_params]
    sigma = params_flat[n_params + 1:2 * n_params]
    x_flat = params_flat[2 * n_params + 1:end]
    x_spher = [x_flat[(i - 1) * n_params + 1 : i * n_params] for i in 1:length(data)]

    logprob = 0.0

    # Add log probability of higher-level parameters
    logprob += sum(Distributions.logpdf.(Normal(0, 1), mu[[i for i in 1:length(mu) if !(i in idx_scaling)]]))  # Prior for c_vT and lambda
    logprob += Distributions.logpdf(LogNormal(0,1), mu[idx_scaling[1]]) # Prior for r; note that prior for phi and theta is uniform

    logprob += sum(Distributions.logpdf.(Normal(-8, 1), sigma[[i for i in 1:length(mu) if !(i in idx_scaling)]]))  # Prior for sigma_cvT and sigma_lambda
    logprob += Distributions.logpdf(Normal(-5, 1), sigma[idx_scaling[1]])  # Prior for sigma_r
    logprob += sum(Distributions.logpdf.(Normal(-8, 1), sigma[idx_scaling[2:end]]))  # Prior for sigma_theta and sigma_phi

    # Add log probability of lower-level parameters and data
    for i in 1:length(data)
        x_i_spher = x_spher[i]

        x_i_cart = [x_i_spher[1], spher2cart(x_i_spher[idx_scaling])..., x_i_spher[5]]

        P_i = data[i]
        for j in 1:size(P_i,2)
            if j in idx_scaling[2:3]
                logprob += Distributions.logpdf(Normal(0, exp(sigma[j])), angle_diff(mu[j], x_i_spher[j]))
            elseif j == idx_scaling[1]
                logprob += Distributions.logpdf(Normal(abs(mu[j]), exp(sigma[j])), x_i_spher[j])
            else
                logprob += Distributions.logpdf(Normal(mu[j], exp(sigma[j])), x_i_spher[j])
            end
        end
        logprob += Distributions.logpdf(mvns[i], x_i_cart)
    end

    return logprob
end

function joint_logprob(params::ModelParameters, data::Vector{Matrix{Float64}}, mvns::Vector; idx_scaling::Vector{Int64}=[2,3,4])
    n_params = size(data[1], 2)
    mu = params.mu
    sigma = params.sigma
    x_spher = params.x

    params_flat = [mu; sigma; vcat(x_spher...)]

    return joint_logprob_flat(params_flat, data, mvns, idx_scaling=idx_scaling)
end

function joint_logprob_flat_negated(params_flat::Vector, data::Vector{Matrix{Float64}}, mvns::Vector, idx_scaling::Vector{Int64})
    return -joint_logprob_flat(params_flat, data, mvns, idx_scaling)
end


"""
    optimize_MAP(Ps::Vector{Matrix{Float64}}, params_init::ModelParameters)

Perform maximum a posteriori (MAP) estimation for the hierarchical model.
"""
function optimize_MAP(Ps::Vector{Matrix{Float64}}, params_init::ModelParameters; idx_scaling::Vector{Int64}=[2,3,4])
    # Initialize the KDEs
    mvns = fit_multivariate_normals(Ps)

    # Flatten the initial parameters
    mu_init = params_init.mu
    sigma_init = params_init.sigma
    x_init_flat = vcat([params_init.x[i] for i in 1:length(Ps)]...)

    params_init_flat = vcat(mu_init, sigma_init, x_init_flat)

    # Compute the gradient
    joint_logprob_grad = params -> ForwardDiff.gradient(p -> joint_logprob_flat_negated(p, Ps, mvns, [2,3,4]), params)

    # Compute the Hessian
    joint_logprob_hessian = params -> ForwardDiff.hessian(p -> joint_logprob_flat_negated(p, Ps, mvns, [2,3,4]), params)

    # println(joint_logprob_flat_negated(params_init_flat, Ps, mvns, [2,3,4]))

    # println(size(joint_logprob_grad(params_init_flat)))

    # println(size(joint_logprob_hessian(params_init_flat)))

    function g!(G, x)
        if G !== nothing
            G .= joint_logprob_grad(x)
        end
    end

    function h!(H, x)
        H .= joint_logprob_hessian(x)
    end

    # Perform Newton optimization
    result = optimize(x->joint_logprob_flat_negated(x, Ps, mvns, [2,3,4]), g!, h!, params_init_flat, Optim.Newton())

    # Unpack the optimized parameters
    params_opt = Optim.minimizer(result)
    mu_opt = params_opt[1:size(Ps[1],2)]
    sigma_opt = params_opt[size(Ps[1],2) + 1:2 * size(Ps[1],2)]
    x_opt_flat = params_opt[2 * size(Ps[1],2) + 1:end]
    x_opt = [x_opt_flat[(i - 1) * size(Ps[1],2) + 1 : i * size(Ps[1],2)] for i in 1:length(Ps)]

    # Create the optimized ModelParameters struct
    params_opt_struct = ModelParameters(mu_opt, sigma_opt, x_opt)

    return params_opt_struct, result
end


"""
    initialize_params(Ps::Vector{Matrix{Float64}}; idx_scaling::Vector{Int64}=[2,3,4])

Initialize the ModelParameters struct using the means of the corresponding
parameters in Ps.
"""
function initialize_params(Ps::Vector{Matrix{Float64}}; idx_scaling::Vector{Int64}=[2,3,4])
    n_params = size(Ps[1], 2)
    means = [mean(P, dims=1)[1, :] for P in Ps]
    mu_init = mean(means, dims=1)[1]
    sigma_init = fill(-8.0, n_params)

    sigma_init[idx_scaling[1]] = -5.0 # Prior for sigma_r

    x_init = means

    mu_init[idx_scaling] = cart2spher(mu_init[idx_scaling])
    for i in 1:length(Ps)
        x_init[i][idx_scaling] = cart2spher(x_init[i][idx_scaling])
    end

    return ModelParameters(mu_init, sigma_init, x_init)
end

