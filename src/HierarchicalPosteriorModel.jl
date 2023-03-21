module HierarchicalPosteriorModel

using Distributions, Optim, Statistics, StatsBase, LinearAlgebra, ForwardDiff

include("util.jl")
include("model.jl")
include("fit.jl")

export
    # util.jl
    cart2spher,
    spher2cart,
    angle_diff,
    fit_multivariate_normals,
    bic_multivariate_normals,
    get_Ps,
    get_variability,
    # model.jl
    ModelParameters,
    joint_logprob_flat,
    joint_logprob,
    joint_logprob_flat_negated,
    # fit.jl
    initialize_params,
    optimize_MAP
end # module
