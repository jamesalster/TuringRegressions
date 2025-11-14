"""
    predict(TR::TuringRegression, X::AbstractArray, fun=nothing; type=:posterior, transform=TR.standardized, std=false, kwargs...)
    predict(TR::TuringRegression, fun=nothing; kwargs...)

Generate predictions for new data or fitted data.

# Arguments
- `X`: Design matrix for predictions (optional, uses fitted data if omitted)
- `fun`: Optional function to apply across draws
- `type`: Type of prediction (:posterior, :epred, :linpred)
- `transform`: Standardize data before feeding to model? 
- `std`: Return predictions at the standardized scale? 
- `drop_warmup`: Number of warmup samples to drop from each chain
- `n_draws`: Number of draws to keep (-1 for all post-warmup)
- `collapse`: Whether to collapse chains into single dimension
- `dropdims`: Whether to drop singleton dimensions (default: true)
"""
function predict(
    TR::TuringRegression{T},
    X::AbstractArray,
    fun::Union{Nothing,Function}=nothing;
    type::Symbol=:posterior,
    kwargs...,
) where {T}

    if type === :posterior
        preds = posterior_pred(TR, X, fun; kwargs...)
    elseif type === :epred
        preds = epred(TR, X, fun; kwargs...)
    elseif type === :linpred
        preds = linpred(TR, X, fun; kwargs...)
    else
        throw(ArgumentError("type must be one of :posterior, :epred or :linpred"))
    end

    return preds
end

function predict(TR::TuringRegression, fun::Union{Nothing,Function}=nothing; kwargs...)
    return predict(TR, TR.X, fun;  kwargs...)
end

#### Internal functions ####
function linpred(
    TR::TuringRegression,
    X::AbstractArray,
    fun::Union{Nothing,Function}=nothing;
    dropdims=true,
    kwargs...,
)
    α = get_parameters(TR, [:α]; kwargs...) # vec required for NamedArray problems below
    beta_names = [Symbol("β[$i]") for i in 1:size(TR.X, 2)]
    β = get_parameters(TR, beta_names; kwargs...) # vec required for NamedArray problems below
    # handle 3d
    μ = zeros(
        eltype(α), (Dim{:row}(size(X, 1)), Dim{:draw}(size(α, 1)), Dim{:chain}(size(α, 3)))
    )
    for i in 1:size(μ, 3)
        μ[:, :, i] = vec(α[:, :, i])' .+ X * β[:, :, i]'
    end
    μ = isnothing(fun) ? μ : mapslices(fun, μ; dims=2)
    return dropdims ? drop_single_dims(μ) : μ
end

function epred(
    TR::TuringRegression{T},
    X::AbstractArray,
    fun::Union{Nothing,Function}=nothing;
    dropdims=true,
    kwargs...,
) where {T}
    μ = linpred(TR, X; kwargs...) # don't pass fun
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
    epreds = isnothing(fun) ? epreds : mapslices(fun, epreds; dims=2)
    return dropdims ? drop_single_dims(epreds) : epreds
end

function posterior_pred(
    TR::TuringRegression{T},
    X::AbstractArray,
    fun::Union{Nothing,Function}=nothing;
    dropdims=true,
    kwargs...,
) where {T}
    epreds = epred(TR, X; kwargs...) #don't pass fun
    ndraws = size(epreds, 2)
    if T == Normal
        σ = vec(get_parameters(TR, [:σ]; kwargs...))
        posterior_preds = (rand(T(), ndraws) .* σ)' .+ epreds
    elseif T == TDist
        σ = vec(get_parameters(TR, [:σ]; kwargs...))
        ν = vec(get_parameters(TR, [:ν]; kwargs...))
        posterior_preds = (rand.(T.(ν)) .* σ)' .+ epreds
    elseif T == Bernoulli
        posterior_preds = rand.(Bernoulli.(epreds))
    elseif T == Poisson
        posterior_preds = rand.(Poisson.(epreds))
    elseif T == NegativeBinomial
        ϕ⁻ = vec(get_parameters(TR, [:ϕ⁻]; kwargs...))
        posterior_preds = rand.(TuringGLM.NegativeBinomial2.(epreds, ϕ⁻'))
    end
    posterior_preds =
        isnothing(fun) ? posterior_preds : mapslices(fun, posterior_preds; dims=2)
    return dropdims ? drop_single_dims(posterior_preds) : posterior_preds
end
