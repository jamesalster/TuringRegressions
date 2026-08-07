"""
    posterior_predict(TR::TuringRegression; type=:posterior, kwargs...)
    posterior_predict(TR::TuringRegression, X::AbstractArray; type=:posterior, kwargs...)
    posterior_predict(TR::TuringRegression, new_data::DataFrame; type=:posterior, kwargs...)
    posterior_predict(f::Function, TR::TuringRegression, args...; type=:posterior, kwargs...)

Generate full-posterior predictions for new data or fitted data (primary, richest API —
see `predict` for the StatsAPI-conformant point-estimate wrapper).
Passing a function (e.g. median) first aggregates the draws with that function.

# Arguments
- `X` / `new_data`: predictions target (optional, uses fitted data if omitted)
- `type`: Type of prediction (:posterior, :epred, :linpred)
- `drop_draws`: Number of warmup samples to drop from each chain
- `n_draws`: Number of draws to keep (`Inf` for all available)
- `collapse`: Whether to collapse chains into single dimension
- `dropdims`: Whether to drop singleton dimensions (default: true)
"""
function _predict_fn(type::Symbol)
    type === :posterior && return posterior_pred
    type === :epred && return epred
    type === :linpred && return linpred
    throw(ArgumentError("type must be one of :posterior, :epred or :linpred"))
end

# Grouping info for random effects: fitted data reuses TR.modeldata.Z, a bare X matrix
# carries no grouping info to reconstruct it from.
function _resolve_z(TR::TuringRegression, X::AbstractArray)
    !has_random_effects(TR) && return nothing
    X === TR.modeldata.predictors.X && return TR.modeldata.Z
    error(
        "posterior_predict() with a raw design matrix does not support random-effects models " *
        "(no grouping information available). Use posterior_predict(TR) for fitted data or " *
        "posterior_predict(TR, new_data::DataFrame) for new data.",
    )
end

function posterior_predict(TR::TuringRegression, X::AbstractArray=TR.modeldata.predictors.X; type::Symbol=:posterior, kwargs...)
    return _predict_fn(type)(TR, X, _resolve_z(TR, X); kwargs...)
end

function posterior_predict(f::Function, TR::TuringRegression, X::AbstractArray=TR.modeldata.predictors.X; type::Symbol=:posterior, kwargs...)
    return _predict_fn(type)(f, TR, X, _resolve_z(TR, X); kwargs...)
end

function posterior_predict(
    TR::TuringRegression, new_data::DataFrame;
    type::Symbol=:posterior, allow_new_levels::Bool=false, kwargs...,
)
    md_new = extract_model_data(TR.formula, new_data)
    z = new_random_effects(TR.modeldata, new_data; allow_new_levels)
    return _predict_fn(type)(TR, md_new.predictors.X, z; kwargs...)
end

function posterior_predict(
    f::Function, TR::TuringRegression, new_data::DataFrame;
    type::Symbol=:posterior, allow_new_levels::Bool=false, kwargs...,
)
    md_new = extract_model_data(TR.formula, new_data)
    z = new_random_effects(TR.modeldata, new_data; allow_new_levels)
    return _predict_fn(type)(f, TR, md_new.predictors.X, z; kwargs...)
end

#### Internal functions ####

# Add each grouping term's random-effect contribution to μ (row, draw, chain) in place.
# A level_index of 0 (unseen new-data level, allow_new_levels=true) contributes nothing,
# i.e. the population-mean (zero) random effect.
function _add_random_effects!(μ::AbstractArray, TR::TuringRegression, z::Vector{RandomEffect}; kwargs...)
    for re in z
        layer = Array(draws(TR, re.variable; kwargs...))
        effect_names = collect(dims(TR.parameters[re.variable], :effect))
        intercept_pos = re.predictors.has_intercept ? findfirst(==(:Intercept), effect_names) : nothing
        slope_positions = has_fixed_effects(re.predictors) ? findall(!=(:Intercept), effect_names) : Int[]
        for c in 1:size(μ, 3), r in axes(μ, 1)
            g = re.level_index[r]
            g == 0 && continue
            if !isnothing(intercept_pos)
                μ[r, :, c] .+= layer[intercept_pos, g, :, c]
            end
            for (k, pos) in enumerate(slope_positions)
                μ[r, :, c] .+= re.predictors.X[r, k] .* layer[pos, g, :, c]
            end
        end
    end
    return μ
end

function linpred(
    TR::TuringRegression,
    X::AbstractArray,
    z::Union{Nothing,Vector{RandomEffect}}=nothing;
    dropdims=true,
    kwargs...,
)
    # Get relevant parameters
    fixef_draws = draws(TR, :fixef; kwargs...)
    if has_fixed_effects(TR)
        β = fixef_draws[fixef=At(Symbol.(TR.modeldata.predictors.X_names))]
        ndraws = size(β, 2)
        nchains = size(β, 3)
    end
    if has_intercept(TR)
        α = fixef_draws[fixef=At([:α])]
        ndraws = size(α, 2)
        nchains = size(α, 3)
    end

    # Initialise linear model output
    μ = zeros((Dim{:row}(size(X, 1)), Dim{:iter}(ndraws), Dim{:chain}(nchains)))

    # loop over chains for dot product vectorisation
    for i in 1:size(μ, 3)
        if has_fixed_effects(TR)
            μ[:, :, i] .+= X * β[:, :, i]
        end
        if has_intercept(TR)
            μ[:, :, i] .+= vec(α[:, :, i])'
        end
    end

    !isnothing(z) && _add_random_effects!(μ, TR, z; kwargs...)

    return dropdims ? _drop_single_dims(μ) : μ
end

function linpred(
    f::Function, TR::TuringRegression, X::AbstractArray,
    z::Union{Nothing,Vector{RandomEffect}}=nothing; dropdims=true, kwargs...,
)
    μ = linpred(TR, X, z; dropdims=false, kwargs...)
    return _aggregate_draws(f, μ; dropdims)
end

function epred(
    TR::TuringRegression{T},
    X::AbstractArray,
    z::Union{Nothing,Vector{RandomEffect}}=nothing;
    dropdims=true,
    kwargs...,
) where {T}
    μ = linpred(TR, X, z; kwargs...)
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

function epred(
    f::Function, TR::TuringRegression, X::AbstractArray,
    z::Union{Nothing,Vector{RandomEffect}}=nothing; dropdims=true, kwargs...,
)
    epreds = epred(TR, X, z; dropdims=false, kwargs...)
    return _aggregate_draws(f, epreds; dropdims)
end

function posterior_pred(
    TR::TuringRegression{T},
    X::AbstractArray,
    z::Union{Nothing,Vector{RandomEffect}}=nothing;
    dropdims=true,
    kwargs...,
) where {T}
    epreds = epred(TR, X, z; kwargs...)
    fixef_draws = draws(TR, :fixef; kwargs...)
    # σ/ν/ϕ vary per draw (iter[,chain]), not per row. Indexing with a 1-element Vector (not a
    # scalar) keeps the size-1 fixef dim in place, so Array(...) already comes out shaped
    # (1, iter[, chain]) and broadcasts across rows instead of being shared by every row
    # within a draw.
    if T == Normal
        σ = Array(fixef_draws[fixef=At([:σ])])
        posterior_preds = epreds .+ randn(size(epreds)) .* σ
    elseif T == TDist
        σ = Array(fixef_draws[fixef=At([:σ])])
        ν = Array(fixef_draws[fixef=At([:ν])])
        noise = rand.(TDist.(ν .* ones(size(epreds))))
        posterior_preds = epreds .+ noise .* σ
    elseif T == Bernoulli
        posterior_preds = rand.(Bernoulli.(epreds))
    elseif T == Poisson
        posterior_preds = rand.(Poisson.(epreds))
    elseif T == NegativeBinomial
        ϕ = Array(fixef_draws[fixef=At([:ϕ])])
        # model.jl samples ϕ as inverse-dispersion (ϕ_inv = 1/ϕ feeds NegativeBinomial2's r); invert to match
        posterior_preds = rand.(NegativeBinomial2.(epreds, 1 ./ (ϕ .* ones(size(epreds)))))
    end
    return dropdims ? _drop_single_dims(posterior_preds) : posterior_preds
end

function posterior_pred(
    f::Function, TR::TuringRegression, X::AbstractArray,
    z::Union{Nothing,Vector{RandomEffect}}=nothing; dropdims=true, kwargs...,
)
    posterior_preds = posterior_pred(TR, X, z; dropdims=false, kwargs...)
    return _aggregate_draws(f, posterior_preds; dropdims)
end
