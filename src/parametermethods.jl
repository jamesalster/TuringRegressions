

# Internal function to get the parameter names of a model
function _get_parameter_names(TR::TuringRegression)::Vector{Symbol}
    return lookup(dims(TR.parameters)[1]).data
end

# Internal function to access parameters from samples object as DimArray
function _get_parameters(TR::TuringRegression, params::Vector{Symbol})::DimArray
    isnothing(TR.samples) && throw(ArgumentError("Model has not been fitted."))
    return TR.parameters[param=At(params)]
end

## Parameter methods
"""
    parameter_names(TR::TuringRegression, params=TR.samples.name_map[:parameters])

Get parameter names with friendly labels replacing generic β indices.
"""
function parameter_names(TR::TuringRegression, params=_get_parameter_names(TR))
    rename_dict = Dict(Symbol("β[$i]") => nm for (i, nm) in enumerate(TR.X_names))
    return [get(rename_dict, p, p) for p in params]
end

"""
    get_parameters(TR::TuringRegression, params::Vector{Symbol}; std=false, drop_warmup=200, n_draws=-1, collapse=true, kwargs...)

Extract specific parameters from fitted model as DimArray.

# Arguments
- `params`: Vector of parameter symbols to extract
- `drop_warmup`: Number of warmup samples to drop from each chain
- `n_draws`: Number of draws to keep (-1 for all post-warmup)
- `collapse`: Whether to collapse chains into single dimension
"""
function get_parameters(
    TR::TuringRegression, params::Vector{Symbol}; kwargs...
)::DimArray
    isnothing(TR.samples) && throw(ArgumentError("Model has not been fitted."))
    # access parameters
    arr = _get_parameters(TR, params) 
    # rename
    new_names = string.(parameter_names(TR, params)) # string to allow regex lookup
    arr = set(arr, Dim{:param} => new_names)
    # Filter draws
    arr = process_draws(arr; kwargs...)
    size(arr, 1) == 0 &&
        @warn "No samples returned, check kwargs and perhaps try adjusting `drop_warmup`?"
    return arr
end

"""
    parameters(TR::TuringRegression, fun=nothing; drop_warmup=200, n_draws=-1, collapse=true, dropdims=true, kwargs...)

Get all model parameters.

# Arguments
- `fun`: Optional function to apply across draws (e.g., mean, median)
- `drop_warmup`: Number of warmup samples to drop from each chain  
- `n_draws`: Number of draws to keep (-1 for all post-warmup)
- `collapse`: Whether to collapse chains into single dimension
- `dropdims`: Whether to drop singleton dimensions (default: true)
"""
function parameters(
    TR::TuringRegression, fun::Union{Nothing,Function}=nothing; dropdims=true, kwargs...
)
    params = get_parameters(TR, _get_parameter_names(TR); kwargs...)
    params = isnothing(fun) ? params : mapslices(fun, params; dims=1)
    return dropdims ? drop_single_dims(params) : params
end

"""
    fixef(TR::TuringRegression, fun=nothing; drop_warmup=200, n_draws=-1, collapse=true, dropdims=true, kwargs...)

Get fixed effect coefficients (β parameters).

# Arguments
- `fun`: Optional function to apply across draws (e.g., mean, median)
- `drop_warmup`: Number of warmup samples to drop from each chain
- `n_draws`: Number of draws to keep (-1 for all post-warmup)  
- `collapse`: Whether to collapse chains into single dimension
- `dropdims`: Whether to drop singleton dimensions (default: true)
"""
function fixef(
    TR::TuringRegression, fun::Union{Nothing,Function}=nothing; dropdims=true, kwargs...
)
    fixef_names = [:α, [Symbol("β[$i]") for i in 1:size(TR.X, 2)]...]
    params = get_parameters(TR, fixef_names; kwargs...)
    params = isnothing(fun) ? params : mapslices(fun, params; dims=1)
    return dropdims ? drop_single_dims(params) : params
end

"""
    internals(TR::TuringRegression, fun=nothing; drop_warmup=200, n_draws=-1, collapse=true, dropdims=true, kwargs...)

Get internal parameters (auxiliary parameters like σ, ν, etc).

# Arguments
- `fun`: Optional function to apply across draws (e.g., mean, median)
- `drop_warmup`: Number of warmup samples to drop from each chain
- `n_draws`: Number of draws to keep (-1 for all post-warmup)
- `collapse`: Whether to collapse chains into single dimension
- `dropdims`: Whether to drop singleton dimensions (default: true)
"""
function internals(
    TR::TuringRegression, fun::Union{Nothing,Function}=nothing; dropdims=true, kwargs...
)
    #NB different source for these
    internals_names = TR.samples.name_map[:internals]
    arr = DimArray(permutedims(TR.samples[internals_names].value, (2, 1, 3)), 
        (Dim{:param}(internals_names), Dim{:draw}, Dim{:chain}))
    # Filter draws
    arr = process_draws(arr; kwargs...)
    size(arr, 1) == 0 &&
        @warn "No samples returned, check kwargs and perhaps try adjusting `drop_warmup`?"
    return arr
    params = isnothing(fun) ? params : mapslices(fun, params; dims=1)
    return dropdims ? drop_single_dims(params) : params
end

"""
    coefs(TR::TuringRegression, fun=median)

Get coefficient point estimates using specified summary function.

# Arguments
- `fun`: Summary function to apply (default: median)
"""
function coefs(TR::TuringRegression, fun::Function=median; kwargs...)
    @info "Reducing with function: $(fun)"
    return fixef(TR, fun; kwargs...)
end

"""
    outcome(TR::TuringRegression; std=false)

Get the response variable as DimArray.
"""
function outcome(TR::TuringRegression)
    return DimArray(TR.y, (Dim{:row}))
end

"""
    predictors(TR::TuringRegression; std=false)

Get the predictor matrix variable as DimArray.
"""
function fixed_effects(TR::TuringRegression)
    return DimArray(TR.X, (Dim{:row}, Dim{:var}([TR.X_names...])))
end

"""
    outcome_as_distribution(TR::TuringRegression{Bernoulli})

Get the categorical outcome from the model as a UnivariateFinite distribution from
    `CategoricalDistributions.jl`
"""
function outcome_as_distribution(TR::TuringRegression{T}) where {T}
    if T != Bernoulli
        throw(
            ArgumentError(
                "Outcome can only be returned as a UnivariateFinite distribution from a Bernoulli model.",
            ),
        )
    end
    return Distributions.fit(UnivariateFinite, categorical(TR.y .== 1))
end
