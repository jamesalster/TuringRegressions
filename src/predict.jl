"""
    predict(TR::TuringRegression; type=:posterior, kwargs...)
    predict(TR::TuringRegression, X::AbstractArray; type=:posterior, kwargs...)
    predict(TR::TuringRegression, new_data::DataFrame; type=:posterior, kwargs...)
    predict(f::Function, TR::TuringRegression, args...; type=:posterior, kwargs...)

Generate predictions for new data or fitted data.
Passing a function (e.g. median) first aggregates the draws with that function.

# Arguments
- `X` / `new_data`: predictions target (optional, uses fitted data if omitted)
- `type`: Type of prediction (:posterior, :epred, :linpred)
- `drop_warmup`: Number of warmup samples to drop from each chain
- `n_draws`: Number of draws to keep (-1 for all post-warmup)
- `collapse`: Whether to collapse chains into single dimension
- `dropdims`: Whether to drop singleton dimensions (default: true)
"""
function _predict_fn(type::Symbol)
    type === :posterior && return posterior_pred
    type === :epred && return epred
    type === :linpred && return linpred
    throw(ArgumentError("type must be one of :posterior, :epred or :linpred"))
end

function predict(TR::TuringRegression, X::AbstractArray=TR.X; type::Symbol=:posterior, kwargs...)
    return _predict_fn(type)(TR, X; kwargs...)
end

function predict(f::Function, TR::TuringRegression, X::AbstractArray=TR.X; type::Symbol=:posterior, kwargs...)
    return _predict_fn(type)(f, TR, X; kwargs...)
end

function predict(TR::TuringRegression, new_data::DataFrame; kwargs...)
    return predict(TR, data_fixed_effects(TR.formula, new_data); kwargs...)
end

function predict(f::Function, TR::TuringRegression, new_data::DataFrame; kwargs...)
    return predict(f, TR, data_fixed_effects(TR.formula, new_data); kwargs...)
end

#### Internal functions ####
function linpred(
    TR::TuringRegression,
    X::AbstractArray;
    dropdims=true,
    kwargs...,
)
    # Get relevant parameters
    fixef_draws = draws(TR, :fixef; kwargs...)
    if TR.modelinfo.has_fixed_effects
        β = fixef_draws[fixef=At(Symbol.(TR.X_names))]
        ndraws = size(β, 2)
        nchains = size(β, 3)
    end
    if TR.modelinfo.has_intercept
        α = fixef_draws[fixef=At([:α])]
        ndraws = size(α, 2)
        nchains = size(α, 3)
    end

    # Initialise linear model output
    μ = zeros((Dim{:row}(size(X, 1)), Dim{:draw}(ndraws), Dim{:chain}(nchains)))

    # loop over chains for dot product vectorisation
    for i in 1:size(μ, 3)
        if TR.modelinfo.has_fixed_effects
            μ[:, :, i] .+= X * β[:, :, i]
        end
        if TR.modelinfo.has_intercept
            μ[:, :, i] .+= vec(α[:, :, i])'
        end
    end

    return dropdims ? _drop_single_dims(μ) : μ
end

function linpred(f::Function, TR::TuringRegression, X::AbstractArray; dropdims=true, kwargs...)
    μ = linpred(TR, X; dropdims=false, kwargs...)
    return _aggregate_draws(f, μ; dropdims)
end

function epred(
    TR::TuringRegression{T},
    X::AbstractArray;
    dropdims=true,
    kwargs...,
) where {T}
    μ = linpred(TR, X; kwargs...)
    invlink = let
        if TR.link == identity
            identity
        elseif TR.link == logit
            logistic
        elseif TR.link == log
            exp
        end
    end
    epreds = invlink.(μ)
    return dropdims ? _drop_single_dims(epreds) : epreds
end

function epred(f::Function, TR::TuringRegression, X::AbstractArray; dropdims=true, kwargs...)
    epreds = epred(TR, X; dropdims=false, kwargs...)
    return _aggregate_draws(f, epreds; dropdims)
end

function posterior_pred(
    TR::TuringRegression{T},
    X::AbstractArray;
    dropdims=true,
    kwargs...,
) where {T}
    epreds = epred(TR, X; kwargs...)
    ndraws = size(epreds, 2)
    fixef_draws = draws(TR, :fixef; kwargs...)
    if T == Normal
        σ = vec(fixef_draws[fixef=At([:σ])])
        posterior_preds = (rand(T(), ndraws) .* σ)' .+ epreds
    elseif T == TDist
        σ = vec(fixef_draws[fixef=At([:σ])])
        ν = vec(fixef_draws[fixef=At([:ν])])
        posterior_preds = (rand.(T.(ν)) .* σ)' .+ epreds
    elseif T == Bernoulli
        posterior_preds = rand.(Bernoulli.(epreds))
    elseif T == Poisson
        posterior_preds = rand.(Poisson.(epreds))
    elseif T == NegativeBinomial
        ϕ = vec(fixef_draws[fixef=At([:ϕ])])
        # model.jl samples ϕ as inverse-dispersion (ϕ_inv = 1/ϕ feeds NegativeBinomial2's r); invert to match
        posterior_preds = rand.(NegativeBinomial2.(epreds, (1 ./ ϕ)'))
    end
    return dropdims ? _drop_single_dims(posterior_preds) : posterior_preds
end

function posterior_pred(f::Function, TR::TuringRegression, X::AbstractArray; dropdims=true, kwargs...)
    posterior_preds = posterior_pred(TR, X; dropdims=false, kwargs...)
    return _aggregate_draws(f, posterior_preds; dropdims)
end
