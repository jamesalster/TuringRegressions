
module TuringRegressions

using Reexport
using Requires: @require
@reexport using DimensionalData
@reexport using LogExpFunctions: logit, logistic
@reexport using Distributions
@reexport using MixedModels: @formula
using StatsModels 
using Turing
using PrettyTables
using StatisticalMeasures
using ParetoSmooth
using LazyArrays
using MacroTools
using MixedModels
using Random
using MCMCDiagnosticTools

using Suppressor: @suppress
using StatsBase: mean, std
using DataFrames: DataFrame
using LinearAlgebra: I, dot
using Colors: colormap
using CategoricalArrays: categorical
using CategoricalDistributions: UnivariateFinite

include("prior.jl")
include("formula_handlers.jl")
include("turingregression.jl")
include("model.jl")
include("utils.jl")
include("parametermethods.jl")
include("predict.jl")
include("pretty.jl")
include("metrics.jl")
include("comparison.jl")

export TuringRegression,
    turing_glm,
    fit!,
    pretty,
    model_warnings,
    parameter_names,
    get_parameters,
    parameters,
    coef,
    fixef,
    internals,
    outcome,
    predictors,
    outcome_as_distribution,
    predict,
    psis_loo,
    loo_compare,
    lineribbon,
    calculate_metrics,
    default_metrics,
    default_prior,
    pseudo_r2

function __init__()
    #Makie required for band
    @require Makie="ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a" begin
        include("plots/lineribbon.jl")
        export lineribbon
        include("plots/plots.jl")
        export conditional_dependency, pp_check_dens, pp_check_dens_overlay, pp_check_hist
    end
end
end
