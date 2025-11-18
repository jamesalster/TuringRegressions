

"""
Container for regression model priors.

Stores the prior distributions for intercept, fixed effects, 
random effects, and auxiliary parameters (like σ or ν).
"""
@kwdef struct RegressionPrior
    intercept::Distribution
    fixed_effects::Distribution
    random_effects::Distribution
    auxiliary::Distribution
end

"""
    default_prior(family::Type{<:Distribution}) -> RegressionPrior

Returns sensible default priors for a regression model.

The auxiliary parameter adapts to the family:
- Normal/Bernoulli: Exponential(1) for variance
- TDist: Gamma(2, 0.1) for degrees of freedom

# Example
```julia
prior = default_prior(Normal)  # For linear regression
prior = default_prior(TDist)   # For robust regression
```
"""
function default_prior(family::Type{<:Distribution})::RegressionPrior

    overall_defaults = (;
        intercept = Normal(0, 5),
        fixed_effects = Normal(0, 2),
        random_effects = Exponential(1),
    )

    if family ∈ [Normal, Bernoulli]
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

"""
    default_prior(TR::TuringRegression{T}) -> RegressionPrior

Convenience method that extracts the distribution family from a model.
"""