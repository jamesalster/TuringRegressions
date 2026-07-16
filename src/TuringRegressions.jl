
module TuringRegressions

using Reexport

@reexport using DimensionalData
@reexport using LogExpFunctions: logit, logistic
@reexport using Distributions
@reexport using MixedModels: @formula

using StatsModels
import StatsAPI: RegressionModel

# StatsAPI marks these `public`, not `export`, so name them explicitly. Must be `import`
# not `using` — extending a function in statsapi.jl requires `import`.
@reexport import StatsAPI:
    coef, coefnames, coeftable, confint, vcov, stderror, nobs, isfitted, weights,
    islinear, fitted, response, responsename, meanresponse, modelmatrix, residuals,
    predict, fit!, offset, linearpredictor, vif, gvif, score, informationmatrix,
    leverage, cooksdistance, reconstruct, reconstruct!, predict!, loglikelihood, dof,
    mss, rss, nulldeviance, nullloglikelihood, aic, aicc, bic, r2, adjr2
using Turing
using PrettyTables
using MixedModels
using Random
using StatisticalMeasures
using PosteriorStats: loo, compare
using MCMCDiagnosticTools
using MCMCChains: MCMCChains, summarize, Chains

using MacroTools: prettify
using Suppressor: @suppress
using StatsBase: mean, std, cov, CoefTable
using DataFrames: DataFrame
using Tables: columntable
using LinearAlgebra: I, dot, Symmetric, diagm, diag
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
include("comparison.jl")
include("plots.jl")
include("statsapi.jl")

export TuringRegression,
    turing_glm,
    model_warnings,
    draws,
    outcome,
    predictors,
    outcome_as_distribution,
    posterior_predict,
    psis_loo,
    loo_compare,
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
# StatsAPI RegressionModel interface (src/statsapi.jl) re-exported via @reexport above —
# includes both the implemented point-estimate methods and the ones that raise a clear
# ArgumentError (no Bayesian analogue / no MLE statistic) instead of a bare MethodError.

end
