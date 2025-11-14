

@kwdef struct RegressionPrior
    intercept::Distribution
    fixed_effects::Distribution
    random_effects::Distribution
    auxiliary::Distribution
end

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
    end
end