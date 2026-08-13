
module TuringRegressions

using Reexport

@reexport using DimensionalData
@reexport using LogExpFunctions: logit, logistic
@reexport using Distributions
@reexport using MixedModels: @formula

import StatsAPI: RegressionModel
# StatsAPI marks these `public`, not `export`, so name them explicitly.
@reexport import StatsAPI:
    coef, coefnames, coeftable, confint, vcov, stderror, nobs, isfitted, weights,
    islinear, fitted, response, responsename, meanresponse, modelmatrix, residuals,
    predict, fit!, offset, linearpredictor, vif, gvif, score, informationmatrix,
    leverage, cooksdistance, reconstruct, reconstruct!, predict!, loglikelihood, dof,
    mss, rss, nulldeviance, nullloglikelihood, aic, aicc, bic, r2, adjr2
using Turing
using ReverseDiff
using MCMCDiagnosticTools
using FlexiChains: FlexiChains, VNChain
using DynamicPPL: getsym, InitFromPrior
using PosteriorStats: loo, compare

using Random
using Tables: columntable
using DataFrames: DataFrame
import StatsBase: StatsBase, mean, std, cov, CoefTable, ZScoreTransform, fit, transform
using LinearAlgebra: I, dot, Symmetric, diagm, diag, Diagonal
using StatsModels
using MixedModels

using Suppressor: @suppress
using Logging: Logging, NullLogger
using PrecompileTools: @compile_workload

using MacroTools: prettify
using PrettyTables
using Crayons: @crayon_str

include("prior.jl")
include("formula_handlers.jl")
include("standardise.jl")
include("unstandardise.jl")
include("reshape.jl")
include("turingregression.jl")
include("model.jl")
include("utils.jl")
include("parametermethods.jl")
include("predict.jl")
include("summary.jl")
include("comparison.jl")
include("plots.jl")
include("statsapi.jl")

# Warms the ~25s generic Turing compile to improve time-to-first fit
@compile_workload begin
    df = DataFrame(y=[1.0, 2.0, 1.5, 3.0, 2.5, 4.0],
        x=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        g=["a", "a", "a", "b", "b", "b"])
    Logging.with_logger(NullLogger()) do
        @suppress begin
            m1 = turing_glm(@formula(y ~ 1 + x), df, Normal)
            fit!(m1; samples=1, warmup=1, nchains=2, quiet=true)
        end
    end
end

export TuringRegression,
    turing_glm,
    model_warnings,
    model_summary,
    draws,
    outcome,
    get_fixef_predictors,
    posterior_predict,
    psis_loo,
    loo_compare,
    default_prior,
    prior_summary,
    modelcode,
    set_model_code!,
    lineribbon,
    lineribbon!,
    categorical_layout,
    pp_check_dens,
    pp_check_dens_overlay,
    pp_check_hist
# StatsAPI RegressionModel interface (src/statsapi.jl) re-exported via @reexport above

end
