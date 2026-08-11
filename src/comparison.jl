"""
    pointwise_loglik(TR::TuringRegression) -> Array{Float64,3}

Per-observation log-likelihood, `(iter, chain, obs)` — `psis_loo`'s only input.
Computed post-hoc: the `@model` uses `Turing.@addlogprob!`, so it has no observed
VarNames for DynamicPPL to track pointwise. Reuses `linpred` (already handles
fixef + random effects) then converts back to the model's internal
standardised-y scale, since that's the scale the likelihood was actually
evaluated on during sampling.

Always uses all of `TR.samples` (`drop_draws=0` internally) — LOO needs the
full kept posterior, not a user-trimmed subset. `psis_loo`'s `kwargs...` go to
`PosteriorStats.loo`, not here.

CAVEAT on weighted models: observation weights are multiplied into each pointwise
term, so a row with weight `w` stands in for `w` observations. That is not the
exchangeable one-row-one-observation object PSIS assumes — leaving such a row out
drops `w` observations at once, so the resulting ELPD and its standard error are
not comparable to an unweighted fit's. Treat weighted `psis_loo`/`loo_compare`
results as indicative only.
"""
function pointwise_loglik(TR::TuringRegression{T}) where {T}
    isnothing(TR.samples) && throw(ArgumentError("Model has not been fitted."))
    spec = family_spec(T)
    y_mean, y_scale = spec.scales_y ? (TR.tf.y.mean[1], TR.tf.y.scale[1]) : (0.0, 1.0)

    # linpred is on the original data scale; undo the y-standardisation the model's
    # likelihood was actually evaluated under (X is already standardised inside
    # linpred's stored β, so only the y-scale/centring needs inverting here).
    μ_orig = linpred(TR, TR.modeldata.predictors.X, TR.modeldata.Z; drop_draws=0, collapse=false, dropdims=false)
    μ = (μ_orig .- y_mean) ./ y_scale # (row, iter, chain)

    fixef = draws(TR, :fixef; drop_draws=0, collapse=false)
    aux(root) = reshape(Array(fixef[fixef=At([root])])[1, :, :], 1, size(μ, 2), size(μ, 3)) # (1,iter,chain)

    y = reshape(TR.modeldata.y, :, 1, 1)
    weights = reshape(something(TR.modeldata.weights, ones(length(TR.modeldata.y))), :, 1, 1)

    ll = if T == Normal
        σ = aux(:σ) ./ y_scale
        weights .* logpdf.(Normal.(μ, σ), y)
    elseif T == TDist
        σ, ν = aux(:σ) ./ y_scale, aux(:ν)
        weights .* logpdf.(μ .+ σ .* TDist.(ν), y)
    elseif T == Bernoulli
        weights .* logpdf.(BernoulliLogit.(μ), y)
    elseif T == Poisson
        weights .* logpdf.(LogPoisson.(μ), y)
    elseif T == NegativeBinomial
        ϕ_inv = 1 ./ aux(:ϕ)
        weights .* logpdf.(NegativeBinomial2.(exp.(μ), ϕ_inv), y)
    end
    return permutedims(ll, (2, 3, 1)) # (iter, chain, obs)
end

"""
    psis_loo(TR::TuringRegression; kwargs...)

Calculate leave-one-out cross-validation using Pareto smoothed importance
sampling. Returns a `PosteriorStats.PSISLOOResult` with predictive accuracy
measures. Higher ELPD values indicate better predictive performance.

`kwargs...` are forwarded to `PosteriorStats.loo`.
"""
function psis_loo(TR::TuringRegression; kwargs...)
    isnothing(TR.samples) && throw(ArgumentError("Model has not been fitted."))
    return loo(pointwise_loglik(TR); kwargs...)
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
