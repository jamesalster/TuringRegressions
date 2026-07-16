
module TuringRegressions

using Reexport

@reexport using DimensionalData
@reexport using LogExpFunctions: logit, logistic
@reexport using Distributions
@reexport using MixedModels: @formula

using StatsModels
using Turing
using PrettyTables
using MixedModels
using Random
using StatisticalMeasures
#using ParetoSmooth #Broken becayse we need more modern DynamicPPL
using MCMCDiagnosticTools

using MacroTools: prettify
using Suppressor: @suppress
using StatsBase: mean, std
using DataFrames: DataFrame
using Tables: columntable
using LinearAlgebra: I, dot, Symmetric, diagm
using CategoricalArrays: categorical
using CategoricalDistributions: UnivariateFinite
using MixedModels: _ranef_refs

include("prior.jl")
include("formula_handlers.jl")
include("turingregression.jl")
include("model.jl")
include("utils.jl")
include("parametermethods.jl")
include("predict.jl")
include("summary.jl")
include("metrics.jl")
#include("comparison.jl")
include("plots.jl")

export TuringRegression,
    turing_glm,
    fit!,
    model_warnings,
    draws,
    outcome,
    predictors,
    outcome_as_distribution,
    predict,
    #psis_loo,
    #loo_compare,
    calculate_metrics,
    default_metrics,
    default_prior,
    pseudo_r2,
    lineribbon,
    lineribbon!,
    conditional_dependency,
    pp_check_dens,
    pp_check_dens_overlay,
    pp_check_hist

end
