
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
using ReverseDiff
using PrettyTables
using MixedModels
using Random
using StatisticalMeasures
using PosteriorStats: loo, compare
using MCMCDiagnosticTools
using FlexiChains: FlexiChains, VNChain
using DynamicPPL: getsym

using MacroTools: prettify
using Suppressor: @suppress
import StatsBase: StatsBase, mean, std, cov, CoefTable, ZScoreTransform, fit, transform
using DataFrames: DataFrame
using Tables: columntable
using LinearAlgebra: I, dot, Symmetric, diagm, diag, Diagonal
using CategoricalArrays: categorical
using CategoricalDistributions: UnivariateFinite
using MixedModels: _ranef_refs
using PrecompileTools: @compile_workload

include("prior.jl")
include("formula_handlers.jl")
include("transform.jl")
include("reshape.jl")
include("turingregression.jl")
include("model.jl")
include("model_cache.jl")
include("utils.jl")
include("parametermethods.jl")
include("predict.jl")
include("summary.jl")
include("metrics.jl")
include("comparison.jl")
include("plots.jl")
include("statsapi.jl")

# T14: warm the ~25s generic Turing/DynamicPPL/AbstractMCMC/StatsModels compile
# (T35), so the first user `turing_glm`+`fit!` doesn't pay it. `samples=1,
# warmup=1, nchains=1` — only need each code path executed once. `turing_glm`
# eval's a fresh model function at runtime (V20); calling it and `fit!` in the
# same function body hits a world-age gap (see bench/sleepstudy_bench.jl
# comment), so `fit!` goes through `Base.invokelatest`.
#
# Ranef `fit!`/`sample()` deliberately NOT precompiled: `AutoReverseDiff`
# (ranef's V25 default, either compile=true or compile=false) segfaults on
# package-image reload — a Turing/DynamicPPL serialization limitation, not
# fixable here. Ranef `turing_glm` construction-only (no `fit!`) was also
# measured and dropped — its ~1.3s runtime win was within this project's own
# noise band (BENCHLOG T26-29 precedent: <20% single-run swing = noise) while
# still costing ~1.6s more precompile time every build, a wash not worth the
# code. Ranef models still benefit substantially (~34s→~14s cold fit, see
# bench/BENCHLOG.md T14 entry) purely from the shared generic slice below.
@compile_workload begin
    df = DataFrame(y=[1.0, 2.0, 1.5, 3.0, 2.5, 4.0],
        x=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        g=["a", "a", "a", "b", "b", "b"])
    @suppress begin
        m1 = turing_glm(@formula(y ~ 1 + x), df, Normal)
        Base.invokelatest(fit!, m1; samples=1, warmup=1, nchains=1, quiet=true)
    end
end

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
    prior_summary,
    modelcode,
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
