"""
    cart2spher(x_cart::Vector)

Converts Cartesian coordinates to spherical coordinates.

# Arguments
- `x_cart::Vector`: A 3-element vector representing Cartesian coordinates (x, y, z).

# Returns
- A 3-element vector representing spherical coordinates (r, theta, phi).
"""
function cart2spher(x_cart::Vector)
    r = norm(x_cart)
    theta = acos(x_cart[3] / r)
    phi = atan(x_cart[2], x_cart[1])
    return [r, theta, phi]
end

"""
    spher2cart(x_spher::Vector)

Converts spherical coordinates to Cartesian coordinates.

# Arguments
- `x_spher::Vector`: A 3-element vector representing spherical coordinates (r, theta, phi).

# Returns
- A 3-element vector representing Cartesian coordinates (x, y, z).
"""
function spher2cart(x_spher::Vector)
    r, theta, phi = x_spher
    x = r * sin(theta) * cos(phi)
    y = r * sin(theta) * sin(phi)
    z = r * cos(theta)
    return [x, y, z]
end

"""
    angle_diff(theta1, theta2)

Calculates the signed difference between two angles.

# Arguments
- `theta1`: First angle in radians.
- `theta2`: Second angle in radians.

# Returns
- The signed difference between the two angles, in the range (-π, π].
"""
function angle_diff(theta1, theta2)
    diff = mod(theta1 - theta2 + π, 2π) - π
    return diff
end


"""
    fit_multivariate_normals(Ps::Vector{Matrix{Float64}})

Fits multivariate normal distributions to a set of input matrices.

# Arguments
- `Ps::Vector{Matrix{Float64}}`: A vector of matrices, each representing a set of data points.

# Returns
- A vector of fitted `MvNormal` objects.
"""
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

"""
    bic_multivariate_normals(Ps::Vector{Matrix{Float64}}, mvns::Vector{MvNormal})

Computes the Bayesian Information Criterion (BIC) for each fitted multivariate normal distribution.

# Arguments
- `Ps::Vector{Matrix{Float64}}`: A vector of matrices, each representing a set of data points.
- `mvns::Vector{MvNormal}`: A vector of fitted `MvNormal` objects.

# Returns
- A vector of BIC values for the corresponding fitted `MvNormal` objects.
"""
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

"""
    get_Ps(fit_results, matches, θh_pos_is_ventral; idx_use=[1,2,3,4,7], datasets_use=nothing, rngs_use=nothing, max_rng=2)

Return a vector of matrices containing the parameter sets for given `matches` and conditions.

# Arguments
- `fit_results`: A dictionary containing the CePNEM fit results for each dataset.
- `matches`: A list of tuples corresponding to a given neuron, with dataset and the corresponding index in the traces.
- `θh_pos_is_ventral`: A dictionary mapping dataset to a boolean indicating if positive θh corresponds to ventral head curvature (as opposed to dorsal).
- `idx_use`: An optional array of indices of the parameters to be used. Default is `[1,2,3,4,7]`.
- `datasets_use`: An optional list of dataset indices to be used. If not provided, all datasets in `matches` are used.
- `rngs_use`: An optional dictionary mapping dataset indices to lists of range indices to be used. If not provided, all ranges are used.
- `max_rng`: An optional integer specifying the maximum number of ranges to be used. Default is `2`.

# Returns
- `Ps`: A vector of matrices containing the parameter sets for the given `matches` and conditions.
"""
function get_Ps(fit_results, matches, θh_pos_is_ventral; idx_use=[1,2,3,4,7], datasets_use=nothing, rngs_use=nothing, max_rng=2)
    Ps = Vector{Matrix{Float64}}(undef, max_rng*length(matches))
    count = 1
    for (dataset, n) in matches
        if !isnothing(datasets_use) && !(dataset in datasets_use)
            continue
        end
        for rng=1:length(fit_results[dataset]["ranges"])
            if !isnothing(rngs_use) && !(rng in rngs_use[dataset])
                continue
            end
            Ps[count] = deepcopy(fit_results[dataset]["sampled_trace_params"][rng,n,:,:])
            Ps[count][:,3] = Ps[count][:,3] .* (1 - 2*θh_pos_is_ventral[dataset])
            Ps[count] = Ps[count][:,idx_use]
            count += 1
        end
    end
    return Ps[1:count-1]
end

"""
    get_corrected_r(analysis_dict::Dict, mu::Vector{Float64})

Compute the corrected r value based on extrapolating the given mu array to the set of extrapolated behaviors.

# Arguments
- `analysis_dict::Dict`: Dictionary containing the `extrapolated_behaviors` array.
- `mu::Vector{Float64}`: An array containing the mu values of the parameters.

# Returns
- `corrected_r`: The computed corrected r value.
"""
function get_corrected_r(analysis_dict, mu)
    mu_cart = [mu[1], spher2cart(exp_r(mu[2:4]))..., 0, 0, mu[5], 0]
    extrap = model_nl8(length(analysis_dict["extrapolated_behaviors"][:,1]), mu_cart..., analysis_dict["extrapolated_behaviors"][:,1], analysis_dict["extrapolated_behaviors"][:,2], analysis_dict["extrapolated_behaviors"][:,3])
    return std(extrap)
end

"""
    exp_r(vec)

Compute the exponential of the first element of the input vector and concatenate it with the rest of the vector.

# Arguments
- `vec`: A vector of values.

# Returns
- `exp_r_vec`: A new vector with the first element exponentiated and the rest unchanged.
"""
function exp_r(vec)
    return [exp(vec[1]), vec[2:end]...]
end


"""
    get_datasets(matches; datasets_use=nothing, rngs_use=nothing)

Retrieve datasets based on the given matches and optional filtering criteria.

# Arguments
- `matches`: A list of tuples containing the dataset identifier and the number of samples.
- `datasets_use`: An optional list of dataset identifiers to include. If not provided, all datasets in `matches` will be used.
- `rngs_use`: An optional dictionary of dataset identifiers with a list of range indices to include. If not provided, all ranges will be used.

# Returns
- `datasets`: A list of tuples containing the selected dataset identifier, range index, and number of samples.
"""
function get_datasets(matches; datasets_use=nothing, rngs_use=nothing)
    datasets = []
    for (dataset, n) in matches
        if !isnothing(datasets_use) && !(dataset in datasets_use)
            continue
        end
        for rng=1:length(fit_results[dataset]["ranges"])
            if !isnothing(rngs_use) && !(rng in rngs_use[dataset])
                continue
            end
            push!(datasets, (dataset, rng, n))
        end
    end
    return datasets
end
