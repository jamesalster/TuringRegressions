

"""
Container for regression model priors.

Stores the prior distributions for intercept, fixed effects,
random effect variance, and auxiliary parameters (like σ or ν).

All fields are on the STANDARDISED data scale (mean 0, sd 1 predictors) — see
`turing_glm` docstring. Use `prior_summary(TR)` or `show(TR)` to inspect.
"""
@kwdef struct RegressionPrior
    intercept::Distribution
    fixed_effects::Distribution
    random_effect_variance::Distribution
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
