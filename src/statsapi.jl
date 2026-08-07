"""
StatsAPI/StatsBase `RegressionModel` interface for `TuringRegression`. Three groups:
implemented point-estimate methods, no-Bayesian-analogue errors, MLE-only-statistic
errors.

Posterior-based analogues stand in for point-estimate methods: `coef` is the posterior
mean, `stderror`/`vcov` come from the posterior draws' covariance, etc. `predict` is the
StatsAPI-conformant point-estimate wrapper (mean `epred`) — for the full posterior, use
`posterior_predict`.
"""

# Every method that collapses the posterior to one number warns once (per method, not
# just once globally) that it's a point estimate, not a Bayesian summary — `_id=fn` gives
# each a distinct maxlog=1 counter despite sharing this one call site.
function _posterior_mean_warning(fn::Symbol)
    @warn "$fn(TR) reports the posterior mean — a point estimate, not the full posterior. " *
          "Use `posterior_predict`/`draws` for the full posterior." maxlog = 1 _id = fn
end

# `coef`/`fitted`/`residuals` stay correct with random effects by routing through
# `posterior_predict` (folds in Z). `modelmatrix` has no way to represent Z at all —
# returning just the fixef X would silently look like the whole design, so error instead.
function _error_if_random_effects(TR::TuringRegression, fn::Symbol)
    has_random_effects(TR) && throw(ArgumentError(
        "$fn(TR) is not meaningful for a model with random effects — it can only " *
        "describe the fixed-effects design, not the grouping/Z structure. Use " *
        "`get_fixef_predictors(TR)` for the fixed-effects matrix alone.",
    ))
end

#### (a) Point-estimate methods — implemented ####

# Fixed-effect coefficient names in the order `:fixef` draws are stored: intercept first
# (if present), then predictors. Excludes auxiliary params (σ, ν, ϕ).
function coefnames(TR::TuringRegression)
    names = String[]
    has_intercept(TR) && push!(names, "α")
    has_fixed_effects(TR) && append!(names, TR.modeldata.predictors.X_names)
    return names
end

"""
    coef(TR::TuringRegression)

Posterior mean of the fixed-effect coefficients, in `coefnames(TR)` order.
"""
function coef(TR::TuringRegression)
    _posterior_mean_warning(:coef)
    labels = Symbol.(coefnames(TR))
    fixef_draws = draws(TR, :fixef)
    return vec(mean(Array(fixef_draws[fixef=At(labels)]); dims=2))
end

"""
    vcov(TR::TuringRegression)

Covariance matrix of the fixed-effect posterior draws (not a frequentist sampling
covariance — the posterior's own covariance, no point-estimate collapse).
"""
function vcov(TR::TuringRegression)
    labels = Symbol.(coefnames(TR))
    fixef_draws = draws(TR, :fixef)
    return cov(Array(fixef_draws[fixef=At(labels)])')
end

stderror(TR::TuringRegression) = sqrt.(diag(vcov(TR)))

"""
    coeftable(TR::TuringRegression; level=0.95)

Posterior mean/std/credible-interval table for the fixed effects.
"""
function coeftable(TR::TuringRegression; level::Real=0.95)
    _posterior_mean_warning(:coeftable)
    labels = Symbol.(coefnames(TR))
    fixef_draws = Array(draws(TR, :fixef)[fixef=At(labels)])
    lo, hi = (1 - level) / 2, 1 - (1 - level) / 2
    means = vec(mean(fixef_draws; dims=2))
    stds = vec(std(fixef_draws; dims=2))
    lower = [quantile(fixef_draws[i, :], lo) for i in axes(fixef_draws, 1)]
    upper = [quantile(fixef_draws[i, :], hi) for i in axes(fixef_draws, 1)]
    pct_lo, pct_hi = round(100 * lo; digits=1), round(100 * hi; digits=1)
    return CoefTable(
        [means, stds, lower, upper],
        ["mean", "std", "$(pct_lo)%", "$(pct_hi)%"],
        coefnames(TR),
    )
end

"""
    confint(TR::TuringRegression; level=0.95)

Posterior credible interval for each fixed effect, as an `(n_coef, 2)` matrix (full
posterior quantiles, no point-estimate collapse).
"""
function confint(TR::TuringRegression; level::Real=0.95)
    labels = Symbol.(coefnames(TR))
    fixef_draws = Array(draws(TR, :fixef)[fixef=At(labels)])
    lo, hi = (1 - level) / 2, 1 - (1 - level) / 2
    lower = [quantile(fixef_draws[i, :], lo) for i in axes(fixef_draws, 1)]
    upper = [quantile(fixef_draws[i, :], hi) for i in axes(fixef_draws, 1)]
    return hcat(lower, upper)
end

nobs(TR::TuringRegression) = size(TR.modeldata.predictors.X, 1)
isfitted(TR::TuringRegression) = !isnothing(TR.samples)
weights(TR::TuringRegression) = isnothing(TR.modeldata.weights) ? ones(nobs(TR)) : TR.modeldata.weights
islinear(TR::TuringRegression{T}) where {T} = T == Normal && TR.link == identity
offset(::TuringRegression) = nothing # no offset-term support

response(TR::TuringRegression) = TR.modeldata.y
responsename(TR::TuringRegression) = string(TR.formula.lhs)
meanresponse(TR::TuringRegression) = mean(TR.modeldata.y)

function modelmatrix(TR::TuringRegression)
    _error_if_random_effects(TR, :modelmatrix)
    return TR.modeldata.predictors.X
end

# Shared by `predict`/`fitted`/`residuals`/`linearpredictor` — the point-estimate compute.
_point_pred(TR::TuringRegression, X; type, kwargs...) =
    Array(posterior_predict(mean, TR, X; type, kwargs...))

"""
    fitted(TR::TuringRegression)

Posterior mean of the fitted values on the response scale (mean `epred`).
"""
function fitted(TR::TuringRegression)
    _posterior_mean_warning(:fitted)
    return _point_pred(TR, TR.modeldata.predictors.X; type=:epred)
end

function residuals(TR::TuringRegression)
    _posterior_mean_warning(:residuals)
    return response(TR) .- _point_pred(TR, TR.modeldata.predictors.X; type=:epred)
end

"""
    linearpredictor(TR::TuringRegression, X=TR.modeldata.predictors.X; kwargs...)

Posterior mean of the linear predictor `Xβ + α [+ Zu]` (mean `linpred`).
"""
function linearpredictor(TR::TuringRegression, X=TR.modeldata.predictors.X; kwargs...)
    _posterior_mean_warning(:linearpredictor)
    return _point_pred(TR, X; type=:linpred, kwargs...)
end

"""
    predict(TR::TuringRegression, [X or new_data]; kwargs...)

StatsAPI-conformant point estimate (posterior mean `epred`), for interop with code
expecting a single point-prediction vector. For the full posterior (the package's
primary, richer API), use `posterior_predict`.
"""
function predict(TR::TuringRegression, X=TR.modeldata.predictors.X; kwargs...)
    _posterior_mean_warning(:predict)
    return _point_pred(TR, X; type=:epred, kwargs...)
end

# `vif`/`gvif` (inherited from StatsModels' generic RegressionModel) detect the intercept
# via an explicit all-ones column in `modelmatrix` — ours has none (α is fit separately),
# so the inherited method always throws the wrong reason. Override with an accurate error.
for fn in (:vif, :gvif)
    @eval function $fn(::TuringRegression, args...; kwargs...)
        throw(ArgumentError(
            "$($(QuoteNode(fn)))(TR) is not supported: the fixed-effects design matrix " *
            "has no explicit intercept column (α is fit separately), so the inherited " *
            "StatsModels method can't detect it correctly.",
        ))
    end
end

#### (b) No Bayesian analogue exists — error clearly ####

const _NO_POSTERIOR_ANALOGUE = [
    :score => "no MLE gradient exists on a NUTS posterior",
    :informationmatrix => "no Fisher information matrix exists on a NUTS posterior",
    :leverage => "no OLS hat matrix exists for a Bayesian fit",
    :cooksdistance => "no OLS hat matrix exists for a Bayesian fit",
    :reconstruct => "no design-matrix reconstruction is supported",
    Symbol("reconstruct!") => "no design-matrix reconstruction is supported",
    Symbol("predict!") => "no in-place prediction API is supported",
]
for (fn, reason) in _NO_POSTERIOR_ANALOGUE
    @eval function $fn(::TuringRegression, args...; kwargs...)
        throw(ArgumentError("$($(QuoteNode(fn)))(TR) is not supported: $($reason)."))
    end
end

#### (c) MLE-only statistics — no single well-defined value on a posterior ####

const _NO_MLE_STATISTIC = [
    :loglikelihood, :dof, :mss, :rss, :nulldeviance, :nullloglikelihood,
    :aic, :aicc, :bic, :r2, :adjr2,
]
for fn in _NO_MLE_STATISTIC
    @eval function $fn(::TuringRegression, args...; kwargs...)
        throw(ArgumentError(
            "$($(QuoteNode(fn)))(TR) is not supported: no single value is well-defined " *
            "on a Bayesian posterior (e.g. dof/aic/bic assume a fixed parameter count, " *
            "which shrinkage in random effects violates). Use `psis_loo`/`loo_compare` " *
            "for model comparison and effective-dof instead.",
        ))
    end
end
