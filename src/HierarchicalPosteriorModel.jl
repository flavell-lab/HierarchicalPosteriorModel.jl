module HierarchicalPosteriorModel

using Distributions, Optim, Statistics, StatsBase, LinearAlgebra, ForwardDiff

include("model.jl")
include("util.jl")
include("fit.jl")

export
    # model.jl
    ModelParameters,
    joint_logprob_flat,
    joint_logprob,
    joint_logprob_flat_negated,
    # util.jl
    cart2spher,
    spher2cart,
    angle_diff,
    fit_multivariate_normals,
    bic_multivariate_normals,
    get_Ps,
    get_variability,
    # fit.jl
    initialize_params,
    optimize_MAP
end # module
