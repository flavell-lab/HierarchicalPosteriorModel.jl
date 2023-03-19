struct ModelParameters
    mu::Vector{Float64}
    sigma::Vector{Float64}
    x::Vector{Vector{Float64}}
end

function cart2spher(x_cart::Vector{Float64})
    r = norm(x_cart)
    theta = acos(x_cart[3] / r)
    phi = atan(x_cart[2], x_cart[1])
    return [r, theta, phi]
end

function spher2cart(x_spher::Vector{Float64})
    r, theta, phi = x_spher
    x = r * sin(theta) * cos(phi)
    y = r * sin(theta) * sin(phi)
    z = r * cos(theta)
    return [x, y, z]
end

function angle_diff(theta1, theta2)
    return atan(sin(theta1 - theta2), cos(theta1 - theta2))
end

"""
    initialize_kdes(Ps::Vector{Matrix{Float64}})

Initialize the KDEs for each dataset in Ps.
"""
function initialize_kdes(Ps::Vector{Matrix{Float64}})
    kdes = Vector{KDEMulti}(undef, length(Ps))

    for i in 1:length(Ps)
        P_i = Ps[i]
        kde_i = KDEMulti([ContinuousDim() for _ = 1:size(P_i,2)], nothing, [[P_i[i,param] for param=1:size(P_i,2)] for i=1:size(P_i,1)])
        kdes[i] = kde_i
    end

    return kdes
end


function joint_logprob(params::ModelParameters, data::Vector{Matrix{Float64}}, kdes::Vector; idx_scaling::Vector{Int64}=[2,3,4])
    logprob = Threads.Atomic{Float64}(0.0)

    # Add log probability of higher-level parameters
    mu = params.mu
    sigma = params.sigma
    
    logprob[] += sum(Distributions.logpdf.(Normal(0, 1), mu[[i for i in 1:length(mu) if !(i in idx_scaling)]]))  # Prior for c_vT and lambda
    logprob[] += Distributions.logpdf(LogNormal(0,1), mu[idx_scaling[1]]) # Prior for r; note that prior for phi and theta is uniform
    
    logprob[] += sum(Distributions.logpdf.(Normal(-8, 1), sigma[[i for i in 1:length(mu) if !(i in idx_scaling)]]))  # Prior for sigma_cvT and sigma_lambda
    logprob[] += Distributions.logpdf(Normal(-5, 1), sigma[idx_scaling[1]])  # Prior for sigma_r
    logprob[] += sum(Distributions.logpdf.(Normal(-8, 1), sigma[idx_scaling[2:end]]))  # Prior for sigma_theta and sigma_phi

    # Add log probability of lower-level parameters and data
    for i in 1:length(data)
        x_i_spher = params.x[i]
        x_i_cart = copy(x_i_spher)
        x_i_cart[idx_scaling] = spher2cart(x_i_spher[idx_scaling])
        
        P_i = data[i]
        for j in 1:size(P_i,2)
            if j in idx_scaling[2:3]
                Threads.atomic_add!(logprob, Distributions.logpdf(Normal(0, exp(sigma[j])), angle_diff(mu[j], x_i_spher[j])))
            else
                Threads.atomic_add!(logprob, Distributions.logpdf(Normal(mu[j], exp(sigma[j])), x_i_spher[j]))
            end
        end
        Threads.atomic_add!(logprob, log(MultiKDE.pdf(kdes[i], x_i_cart)))
    end

    return logprob[]
end



"""
    optimize_MAP(Ps::Vector{Matrix{Float64}}, params_init::ModelParameters)

Perform maximum a posteriori (MAP) estimation for the hierarchical model.
"""
function optimize_MAP(Ps::Vector{Matrix{Float64}}, params_init::ModelParameters; idx_scaling::Vector{Int64}=[2,3,4])
    # Initialize the KDEs
    kdes = initialize_kdes(Ps)
    function neg_logprob(params::Vector{Float64})
        # Unpack the parameters
        mu = params[1:size(Ps[1],2)]
        sigma = params[size(Ps[1],2) + 1:2 * size(Ps[1],2)]
        x_flat = params[2 * size(Ps[1],2) + 1:end]
        x_spher = [x_flat[(i - 1) * size(Ps[1],2) + 1 : i * size(Ps[1],2)] for i in 1:length(Ps)]

        # Create a new ModelParameters struct
        new_params = ModelParameters(mu, sigma, x_spher)

        # Calculate the negative log probability
        return -joint_logprob(new_params, Ps, kdes)
    end

    # Flatten the initial parameters
    mu_init = params_init.mu
    sigma_init = params_init.sigma
    x_init_flat = vcat([params_init.x[i] for i in 1:length(Ps)]...)

    params_init_flat = vcat(mu_init, sigma_init, x_init_flat)

    lower_bounds = fill(-5.0, length(params_init_flat))
    upper_bounds = fill(5.0, length(params_init_flat))


    lower_bounds[idx_scaling[1]] = 0.0
    lower_bounds[idx_scaling[2]] = 0.0
    lower_bounds[idx_scaling[3]] = -π

    upper_bounds[idx_scaling[1]] = 5.0
    upper_bounds[idx_scaling[2]] = π
    upper_bounds[idx_scaling[3]] = π

    for i in length(mu_init)+1:length(mu_init)+length(sigma_init)
        lower_bounds[i] = -10.0
        upper_bounds[i] = 1.0
    end

    for i in 1:length(Ps)
        lower_bounds[2 * size(Ps[1],2) + (i - 1) * size(Ps[1],2) + idx_scaling[1]] = 0.0  # lower bound for r
        lower_bounds[2 * size(Ps[1],2) + (i - 1) * size(Ps[1],2) + idx_scaling[2]] = 0.0  # lower bound for theta
        lower_bounds[2 * size(Ps[1],2) + (i - 1) * size(Ps[1],2) + idx_scaling[3]] = -π  # lower bound for phi
        upper_bounds[2 * size(Ps[1],2) + (i - 1) * size(Ps[1],2) + idx_scaling[1]] = 5.0  # upper bound for r
        upper_bounds[2 * size(Ps[1],2) + (i - 1) * size(Ps[1],2) + idx_scaling[2]] = π    # upper bound for theta
        upper_bounds[2 * size(Ps[1],2) + (i - 1) * size(Ps[1],2) + idx_scaling[3]] = π   # upper bound for phi
    end

    # Perform L-BFGS optimization
    result = optimize(neg_logprob, lower_bounds, upper_bounds, params_init_flat, Fminbox(Optim.GradientDescent()))

    # Unpack the optimized parameters
    params_opt = Optim.minimizer(result)
    mu_opt = params_opt[1:size(Ps[1],2)]
    sigma_opt = params_opt[size(Ps[1],2) + 1:2 * size(Ps[1],2)]
    x_opt_flat = params_opt[2 * size(Ps[1],2) + 1:end]
    x_opt = [x_opt_flat[(i - 1) * size(Ps[1],2) + 1 : i * size(Ps[1],2)] for i in 1:length(Ps)]

    # Create the optimized ModelParameters struct
    params_opt_struct = ModelParameters(mu_opt, sigma_opt, x_opt)

    return params_opt_struct
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

