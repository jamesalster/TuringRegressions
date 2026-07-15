
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

# Stan-style negative binomial parameterisation, taken from TuringGLM.jl
function NegativeBinomial2(μ::T, ϕ::T) where {T<:Real}
    p = max(1 / (1 + μ / ϕ), 1e-6) # numerical stability
    r = ϕ
    return NegativeBinomial(r, p)
end
