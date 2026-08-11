

"""
Container for regression model priors.

Stores the prior distributions for intercept, fixed effects, random effect
variance, and auxiliary parameters (like σ or ν) — plus `lkj_eta`, the shape
param (η) of `LKJCholesky(d, η)`: η=1 flat, η>1 shrinks correlations toward
0, η<1 pushes toward ±1.

All fields are on the STANDARDISED data scale (mean 0, sd 1 predictors) — see
`turing_glm` docstring. Use `prior_summary(TR)` or `show(TR)` to inspect.
"""
@kwdef struct RegressionPrior
    intercept::Distribution
    fixed_effects::Distribution
    random_effect_variance::Distribution
    auxiliary::Distribution
    lkj_eta::Real
end

"""
    default_prior(family::Type{<:Distribution}; intercept, fixed_effects,
                  random_effect_variance, auxiliary, lkj_eta) -> RegressionPrior

Returns sensible default priors for a regression model, on the STANDARDISED
data scale (mean 0, sd 1 predictors) — see `turing_glm` docstring.

Pass any keyword to override just that one field; the rest keep their
family-appropriate default, e.g. `default_prior(Normal; lkj_eta=2.0)`.

The auxiliary parameter adapts to the family:
- Normal: Exponential(1) for variance
- TDist: Gamma(2, 0.1) for degrees of freedom
- Bernoulli: Unused

`lkj_eta` defaults to 1.0 — see `RegressionPrior` docstring.
"""
function default_prior(
    family::Type{<:Distribution};
    intercept::Union{Distribution,Nothing}=nothing,
    fixed_effects::Union{Distribution,Nothing}=nothing,
    random_effect_variance::Union{Distribution,Nothing}=nothing,
    auxiliary::Union{Distribution,Nothing}=nothing,
    lkj_eta::Union{Real,Nothing}=nothing,
)::RegressionPrior

    default_auxiliary = if family ∈ [Normal, Bernoulli, Poisson, NegativeBinomial]
        Exponential(1)
    elseif family == TDist
        Gamma(2, 0.1)
    else
        error("No default prior implemented for model family: $(string(family))")
    end

    RegressionPrior(
        intercept = something(intercept, Normal(0, 5)),
        fixed_effects = something(fixed_effects, Normal(0, 2)),
        random_effect_variance = something(random_effect_variance, Exponential(1)),
        auxiliary = something(auxiliary, default_auxiliary),
        lkj_eta = something(lkj_eta, 1.0),
    )
end
