
# Helper function for display
function clean_prior_string(x)
    replace(x, r"\{.*\}" => "", "\n" => " ")
end

# Helper function to know Turing Regression's default links
function get_link(::Type{T}) where {T<:UnivariateDistribution}
    if T ∈ [Normal, TDist]
        return identity
    elseif T == Bernoulli
        return logit
    elseif T ∈ [Poisson, NegativeBinomial]
        return log
    else
        @warn "Distribution $T unknown, assuming identity link"
        return identity
    end
end

# Inverse of get_link — used by epred to back-transform the linear predictor.
function get_invlink(::Type{T}) where {T<:UnivariateDistribution}
    if T ∈ [Normal, TDist]
        return identity
    elseif T == Bernoulli
        return logistic
    elseif T ∈ [Poisson, NegativeBinomial]
        return exp
    else
        error("Distribution $T unknown, no inverse link defined.")
    end
end

# Stan-style negative binomial parameterisation, taken from TuringGLM.jl
function NegativeBinomial2(μ::T, ϕ::T) where {T<:Real}
    # clamp both bounds: unclamped upper bound lets extreme HMC proposals (μ
    # underflowing to 0) push p exactly to 1, a non-differentiable kink in
    # max() that gives NaN gradients and crashes NUTS (see SPEC.md B2)
    p = clamp(1 / (1 + μ / ϕ), 1e-6, 1 - 1e-6)
    r = ϕ
    return NegativeBinomial(r, p)
end
