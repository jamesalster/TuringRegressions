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

# Grouping info for random effects: fitted data reuses TR.z, a bare X matrix carries no
# grouping info to reconstruct it from.
function _resolve_z(TR::TuringRegression, X::AbstractArray)
    !TR.modelinfo.has_random_effects && return nothing
    X === TR.X && return TR.z
    error(
        "posterior_predict() with a raw design matrix does not support random-effects models " *
        "(no grouping information available). Use posterior_predict(TR) for fitted data or " *
        "posterior_predict(TR, new_data::DataFrame) for new data.",
    )
end

function posterior_predict(TR::TuringRegression, X::AbstractArray=TR.X; type::Symbol=:posterior, kwargs...)
    return _predict_fn(type)(TR, X, _resolve_z(TR, X); kwargs...)
end

function posterior_predict(f::Function, TR::TuringRegression, X::AbstractArray=TR.X; type::Symbol=:posterior, kwargs...)
    return _predict_fn(type)(f, TR, X, _resolve_z(TR, X); kwargs...)
end

function posterior_predict(
    TR::TuringRegression, new_data::DataFrame;
    type::Symbol=:posterior, allow_new_levels::Bool=false, kwargs...,
)
    _, X, _ = extract_model_data(TR.formula, new_data)
    z = new_random_effects(TR, new_data; allow_new_levels)
    return _predict_fn(type)(TR, X, z; kwargs...)
end

function posterior_predict(
    f::Function, TR::TuringRegression, new_data::DataFrame;
    type::Symbol=:posterior, allow_new_levels::Bool=false, kwargs...,
)
    _, X, _ = extract_model_data(TR.formula, new_data)
    z = new_random_effects(TR, new_data; allow_new_levels)
    return _predict_fn(type)(f, TR, X, z; kwargs...)
end

"""
    new_random_effects(TR::TuringRegression, new_data; allow_new_levels=false)

Rebuild ranef structure for `new_data` via `extract_model_data` (same path used at fit
time), then remap each grouping level onto `TR.z`'s original level order/index.

By default errors clearly if `new_data` contains a grouping level not seen during
fitting. With `allow_new_levels=true`, unseen levels get a `@warn` and are marked (level
index `0`) so `posterior_predict` uses the population-mean (zero) random effect for those rows,
instead of erroring.
"""
function new_random_effects(TR::TuringRegression, new_data; allow_new_levels::Bool=false)
    isnothing(TR.z) && return nothing
    _, _, Z_new = extract_model_data(TR.formula, new_data)
    return [
        _remap_levels(re_new, re_orig; allow_new_levels) for
        (re_new, re_orig) in zip(Z_new, TR.z)
    ]
end

# Sentinel level index 0 (never a valid 1-based level) marks an unseen level.
function _remap_levels(re_new::RandomEffect, re_orig::RandomEffect; allow_new_levels::Bool=false)
    unseen = Any[]
    level_index = map(re_new.level_index) do i
        key = re_new.levels[i]
        idx = findfirst(==(key), re_orig.levels)
        if !isnothing(idx)
            idx
        elseif allow_new_levels
            push!(unseen, key)
            0
        else
            error(
                "predict: unseen level '$key' for grouping variable :$(re_orig.variable) — " *
                "all grouping levels must have been present when the model was fitted. " *
                "Pass allow_new_levels=true to use population-mean random effects for new levels.",
            )
        end
    end
    if !isempty(unseen)
        @warn "predict: unseen level(s) for grouping variable :$(re_orig.variable); using population-mean (zero) random effect for these rows." levels =
            unique(unseen)
    end
    return RandomEffect(
        re_orig.variable, re_orig.levels, level_index, re_new.predictors,
        re_new.predictor_names, re_new.has_intercept, re_new.has_fixed_effects,
    )
end

#### Internal functions ####

# Add each grouping term's random-effect contribution to μ (row, draw, chain) in place.
# A level_index of 0 (unseen new-data level, allow_new_levels=true) contributes nothing,
# i.e. the population-mean (zero) random effect.
function _add_random_effects!(μ::AbstractArray, TR::TuringRegression, z::Vector{RandomEffect}; kwargs...)
    for re in z
        layer = Array(draws(TR, re.variable; kwargs...))
        effect_names = collect(dims(TR.parameters[re.variable], :effect))
        intercept_pos = re.has_intercept ? findfirst(==(:Intercept), effect_names) : nothing
        slope_positions = re.has_fixed_effects ? findall(!=(:Intercept), effect_names) : Int[]
        for c in 1:size(μ, 3), r in axes(μ, 1)
            g = re.level_index[r]
            g == 0 && continue
            if !isnothing(intercept_pos)
                μ[r, :, c] .+= layer[intercept_pos, g, :, c]
            end
            for (k, pos) in enumerate(slope_positions)
                μ[r, :, c] .+= re.predictors[r, k] .* layer[pos, g, :, c]
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

function posterior_pred(
    f::Function, TR::TuringRegression, X::AbstractArray,
    z::Union{Nothing,Vector{RandomEffect}}=nothing; dropdims=true, kwargs...,
)
    posterior_preds = posterior_pred(TR, X, z; dropdims=false, kwargs...)
    return _aggregate_draws(f, posterior_preds; dropdims)
end
