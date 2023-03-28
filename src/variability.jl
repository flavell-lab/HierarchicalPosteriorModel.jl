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
    get_variability(model_params::HBParams)

Compute the variability metric based on the provided model parameters.

# Arguments
- `model_params`: A `HBParams` instance containing the mean (mu) and standard deviation (sigma) parameters.

# Returns
- `variability`: A scalar value representing the computed variability metric.
"""
function get_variability(model_params::HBParams)
    mu = model_params.mu
    sigma = model_params.sigma

    return get_variability(mu, sigma)
end

"""
    get_variability_subtypes(hierarchical_datasets::Vector, hierarchical_params::HBParams; neuron::String="")

Calculate the variability of parameters between datasets for different subtypes of variability:
inter-dataset variability, intra-dataset variability, and left-right (LR) variability.

# Arguments
- `hierarchical_datasets::Vector`: A vector containing the datasets used to fit the hierarchical model.
- `hierarchical_params::HBParams`: An instance of HBParams containing the hierarchical model parameters.
- `neuron::String`: (optional) The name of the neuron (default: empty string).

# Returns
- `inter_variability`: The calculated inter-dataset variability.
- `intra_variability`: The calculated intra-dataset variability.
- `LR_variability`: The calculated left-right variability.
- `inter_dataset_variability`: A list of values from which inter-dataset variability is computed.
- `intra_dataset_variability`: A list of values from which intra-dataset variability is computed.
- `LR_variability`: A list of values from which left-right variability is computed.
"""
function get_variability_subtypes(hierarchical_datasets::Vector, hierarchical_params::HBParams; neuron::String="")
    intra_dataset_variability = []
    inter_dataset_variability = []
    LR_variability = []

    sigma_diff = (sigma1, sigma2) -> [sigma1[1] - sigma2[1], sigma1[2] - sigma2[2], angle_diff(sigma1[3], sigma2[3]), angle_diff(sigma1[4], sigma2[4]), sigma1[5] - sigma2[5]]
    for dataset in unique([x[1] for x in hierarchical_datasets])
        all_obs = [(i,x...) for (i,x) in enumerate(hierarchical_datasets) if x[1] == dataset]
        x_dataset = zeros(length(all_obs), length(hierarchical_params.x[1]))
        for i=1:length(all_obs)
            x_dataset[i,:] .= hierarchical_params.x[all_obs[i][1]]
        end
        avg_dataset = angle_mean(x_dataset, 1) # mean over all observations in the dataset; note that this will be log(r) not r
        push!(inter_dataset_variability, avg_dataset)
        datasets_rng_1 = [x for x in all_obs if x[3] == 1]
        datasets_rng_2 = [x for x in all_obs if x[3] == 2]
        if length(datasets_rng_1) > 0 && length(datasets_rng_2) > 0
            x_dataset_rng_1 = zeros(length(datasets_rng_1), length(hierarchical_params.x[1]))
            x_dataset_rng_2 = zeros(length(datasets_rng_2), length(hierarchical_params.x[1]))
            for i=1:length(datasets_rng_1)
                x_dataset_rng_1[i,:] .= hierarchical_params.x[datasets_rng_1[i][1]]
            end
            for i=1:length(datasets_rng_2)
                x_dataset_rng_2[i,:] .= hierarchical_params.x[datasets_rng_2[i][1]]
            end
            avg_dataset_rng_1 = angle_mean(x_dataset_rng_1, 1)
            avg_dataset_rng_2 = angle_mean(x_dataset_rng_2, 1)
            push!(intra_dataset_variability, abs.(sigma_diff(avg_dataset_rng_1, avg_dataset_rng_2)) / sqrt(2))
        end

        neuron_obs = unique([x[4] for x in all_obs])

        if length(neuron_obs) > 2
            @warn("Cannot have more than 2 detections of the same neuron $neuron in a dataset $dataset")
            continue
        end
        if length(neuron_obs) == 1
            continue
        end
        obs_1 = [x for x in all_obs if x[4] == neuron_obs[1]]
        obs_2 = [x for x in all_obs if x[4] == neuron_obs[2]]

        x_dataset_obs_1 = zeros(length(obs_1), length(hierarchical_params.x[1]))
        x_dataset_obs_2 = zeros(length(obs_2), length(hierarchical_params.x[1]))
        for i=1:length(obs_1)
            x_dataset_obs_1[i,:] .= hierarchical_params.x[obs_1[i][1]]
        end
        for i=1:length(obs_2)
            x_dataset_obs_2[i,:] .= hierarchical_params.x[obs_2[i][1]]
        end
        avg_dataset_obs_1 = angle_mean(x_dataset_obs_1, 1)
        avg_dataset_obs_2 = angle_mean(x_dataset_obs_2, 1)
        push!(LR_variability, abs.(sigma_diff(avg_dataset_obs_1, avg_dataset_obs_2)) / sqrt(2))
    end

    n_params = length(hierarchical_params.x[1])

    sigma_to_std = sigma -> [std(sigma[1]), std(sigma[2]), angle_std(sigma[3]), angle_std(sigma[4]), std(sigma[5])]
    sigma_to_mean = sigma -> [mean(sigma[1]), mean(sigma[2]), angle_mean(sigma[3]), angle_mean(sigma[4]), mean(sigma[5])]

    inter_sigma = (length(inter_dataset_variability) > 1) ? sigma_to_std([[inter_dataset_variability[i][j] for i=1:length(inter_dataset_variability)] for j=1:n_params]) : fill(NaN, n_params)
    intra_sigma = (length(intra_dataset_variability) >= 1) ? sigma_to_mean([[abs(intra_dataset_variability[i][j]) for i=1:length(intra_dataset_variability)] for j=1:n_params]) : fill(NaN, n_params)
    LR_sigma = (length(LR_variability) >= 1) ? sigma_to_mean([[abs(LR_variability[i][j]) for i=1:length(LR_variability)] for j=1:n_params]) : fill(NaN, n_params)

    mu = hierarchical_params.mu

    return get_variability(mu, log.(inter_sigma)), get_variability(mu, log.(intra_sigma)), get_variability(mu, log.(LR_sigma)), inter_dataset_variability, intra_dataset_variability, LR_variability
end
