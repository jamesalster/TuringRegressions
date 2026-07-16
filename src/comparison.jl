"""
    psis_loo(TR::TuringRegression; kwargs...)

Calculate leave-one-out cross-validation using Pareto smoothed importance
sampling. Returns a `PosteriorStats.PSISLOOResult` with predictive accuracy
measures. Higher ELPD values indicate better predictive performance.

`kwargs...` are forwarded to `PosteriorStats.loo`.
"""
function psis_loo(TR::TuringRegression; kwargs...)
    isnothing(TR.samples) && throw(ArgumentError("Model has not been fitted."))
    model_with_data = _build_model_with_data(TR)
    gq = generated_quantities(model_with_data, TR.samples)
    nobs = length(gq[1, 1].loglik)
    ll = [gq[i, j].loglik[n] for i in axes(gq, 1), j in axes(gq, 2), n in 1:nobs]
    return loo(ll; kwargs...)
end

"""
    loo_compare(models::AbstractVector{<:TuringRegression}; kwargs...)
    loo_compare(models::TuringRegression...; kwargs...)

Compare models using leave-one-out cross-validation.

# Arguments
- `models`: Vector of fitted TuringRegression objects (or passed as separate arguments)
- `kwargs...`: Additional arguments passed to `PosteriorStats.compare`
"""
function loo_compare(models::AbstractVector{<:TuringRegression}; kwargs...)
    return compare(psis_loo.(models); kwargs...)
end
loo_compare(models::TuringRegression...; kwargs...) = loo_compare(collect(models); kwargs...)
