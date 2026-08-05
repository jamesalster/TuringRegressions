

"""
Container for regression model priors.

Stores the prior distributions for intercept, fixed effects,
random effect variance, and auxiliary parameters (like σ or ν).

All fields are on the STANDARDISED data scale (mean 0, sd 1 predictors) — see
`turing_glm` docstring. Use `prior_summary(TR)` or `show(TR)` to inspect.

`fixed_effects`/`random_effect_variance` accept either one `Distribution` (shared
across all coefficients, original behaviour) or a `Vector{<:Distribution}` (one per
column, in predictor-column order) — see `scaled_default_prior` (T21).
"""
const PriorSpec = Union{Distribution,AbstractVector{<:Distribution}}
@kwdef struct RegressionPrior
    intercept::Distribution
    fixed_effects::PriorSpec
    random_effect_variance::PriorSpec
    auxiliary::Distribution
end

"""
    default_prior(family::Type{<:Distribution}) -> RegressionPrior

Returns sensible default priors for a regression model, on the STANDARDISED
data scale (mean 0, sd 1 predictors) — see `turing_glm` docstring.

The auxiliary parameter adapts to the family:
- Normal: Exponential(1) for variance
- TDist: Gamma(2, 0.1) for degrees of freedom
- Bernoulli: Unused
```
"""
function default_prior(family::Type{<:Distribution})::RegressionPrior

    overall_defaults = (;
        intercept = Normal(0, 5),
        fixed_effects = Normal(0, 2),
        random_effect_variance = Exponential(1),
    )

    # Alter auxiliary prior
    if family ∈ [Normal, Bernoulli, Poisson, NegativeBinomial]
        return RegressionPrior(
            overall_defaults...,
            Exponential(1)
        )
    elseif family == TDist
        return RegressionPrior(
            overall_defaults...,
            Gamma(2, 0.1)
        )
    else
        error("No default prior implemented for model family: $(string(family))")
    end
end

# T21: fixef/ranef predictors no longer scaled to sd=1 (transform.jl), so a single
# std-scale default no longer fits every raw-scale column. Rescale per-column instead:
# widen/narrow the std-scale default by that column's raw sd, so it carries the same
# information (in raw units) the std-scale default carried in std units.
_rescale_prior(d::Normal, sd::Real) = Normal(d.μ, d.σ / sd)
_rescale_prior(d::Exponential, sd::Real) = Exponential(d.θ / sd)

"""
    scaled_default_prior(family, md::ModelData) -> RegressionPrior

Like `default_prior`, but rescales `fixed_effects` per fixef predictor column (and
`random_effect_variance`, when there's exactly one ranef term, per that term's slope
columns) by each column's raw sd — for use once predictors are centered-only (T21),
so the default priors stay sensible on the unscaled raw data. `intercept` and
`auxiliary` are left as-is: their scale depends on y, not on predictor scaling.
"""
function scaled_default_prior(family::Type{<:Distribution}, md::ModelData)::RegressionPrior
    base = default_prior(family)
    fixef_sds = vec(std(md.predictors.X, dims=1))
    fixed_effects = [_rescale_prior(base.fixed_effects, sd) for sd in fixef_sds]

    random_effect_variance = if length(md.Z) == 1
        re = md.Z[1]
        slope_sds = vec(std(re.predictors.X, dims=1))
        re_dists = re.predictors.has_intercept ?
            [base.random_effect_variance; [_rescale_prior(base.random_effect_variance, sd) for sd in slope_sds]] :
            [_rescale_prior(base.random_effect_variance, sd) for sd in slope_sds]
        isempty(re_dists) ? base.random_effect_variance : re_dists
    else
        base.random_effect_variance
    end

    return RegressionPrior(base.intercept, fixed_effects, random_effect_variance, base.auxiliary)
end
