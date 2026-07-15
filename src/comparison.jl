
#API for ParetoSmooth

"""
    psis_loo(TR::TuringRegression)

Calculate leave-one-out cross-validation using Pareto smoothed importance sampling.

Returns PSIS-LOO object with predictive accuracy measures. Lower ELPD values indicate better predictive performance.
"""
function ParetoSmooth.psis_loo(TR::TuringRegression)
    ll = loglikelihood(TR.model, TR.samples)
    ll_rshp = reshape(ll, 1, size(ll)...)
    return psis_loo(ll_rshp; source="mcmc")
end

"""
    loo_compare(models::AbstractVector{<:TuringRegression}; kwargs...)
    loo_compare(models::TuringRegression...; kwargs...)

Compare multiple models using leave-one-out cross-validation.
    Passing `model_names` as a tuple will name the outputs.

# Arguments
- `models`: Vector of fitted TuringRegression objects (or passed as separate arguments)
- `kwargs...`: Additional arguments passed to ParetoSmooth.loo_compare
"""
function ParetoSmooth.loo_compare(models::AbstractVector{<:TuringRegression}; kwargs...)
    @nospecialize models
    psis_objects = psis_loo.(models)
    return loo_compare(psis_objects; kwargs...)
end
ParetoSmooth.loo_compare(models::TuringRegression...; kwargs...) = loo_compare(collect(models); kwargs...)