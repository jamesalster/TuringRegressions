

# Utility function for selecting draws and collapsing chains from a samples AxisArray
function _process_draws(DA::Union{DimArray, DimStack}; drop_warmup::Int=200, n_draws::Int=-1, collapse::Bool=true)
    # Drop warmup
    arr = DA[draw=(drop_warmup + 1):size(DA, :draw)] 
    @assert n_draws <= size(arr, :draw) "$n_draws draws is too many from $(size(arr, :draw)) available. Note that n_draws is applied per chain."
    # Select draws
    arr = n_draws > 0 ? arr[draw=1:n_draws] : arr 
    # Collapse
    arr = collapse ? mergedims(arr, (:draw, :chain) => :draw) : arr
    @assert size(arr, :draw) > 0 "No samples returned, check kwargs and perhaps try adjusting `drop_warmup`?"
    return arr
end

# Drop single dimensions where possible from an array
function _drop_single_dims(DA::Union{DimArray, DimStack})
    dims_to_drop = findall(==(1), size(DA))
    return dropdims(DA; dims=Tuple(dims_to_drop))
end

# Aggregate a draws/chain-dimensioned DimArray with fun over :draw (+:chain if present)
function _aggregate_draws(f::Function, arr::DimArray; dropdims=true)
    dims_to_aggregate = hasdim(arr, :chain) ? [:draw, :chain] : [:draw]
    dimindices = ntuple(i -> dimnum(arr, dims_to_aggregate[i]), length(dims_to_aggregate))
    out = mapslices(f, arr; dims=dimindices)
    return dropdims ? _drop_single_dims(out) : out
end

"""
    draws(TR::TuringRegression, type::Symbol; drop_warmup=200, n_draws=-1, collapse=true)
    draws(f::Function, TR::TuringRegression, type::Symbol; dropdims=true, drop_warmup=200, n_draws=-1, collapse=true)
    draws(TR::TuringRegression; drop_warmup=200, n_draws=-1, collapse=true)

Extract specific draws from fitted model as `DimArray` (or `DimStack` if `type` is not passed).
Passing a function (e.g. median) aggregates the draws with that function.

# Arguments
- `type`: Symbol for the type of draw. Can be `:fixef`, :`{ranef_name}`, :{ranef_name}_sd`, `:{group_name}_corr`, `:internals`
- `drop_warmup`: Number of warmup samples to drop from each chain. (default is `200`)
- `n_draws`: Number of draws to keep (default is -1 for all post-warmup)
- `collapse`: Whether to collapse chains into single dimension (default is `true`)
- `dropdims`: Whether to drop dims over which (default is `true`)
"""
function draws(TR::TuringRegression; kwargs...)
    isnothing(TR.samples) && throw(ArgumentError("Model has not been fitted."))
    arr = _process_draws(TR.parameters; kwargs...)
    return arr
end
function draws(TR::TuringRegression, type::Symbol; kwargs...)
    isnothing(TR.samples) && throw(ArgumentError("Model has not been fitted."))
    available_types = propertynames(TR.parameters)
    type ∉ available_types && throw(ArgumentError("type $type not available, must be one of: $available_types"))
    arr = _process_draws(TR.parameters[type]; kwargs...)
    return arr
end
function draws(f::Function, TR::TuringRegression, type::Symbol; dropdims=true, kwargs...)
    arr = draws(TR, type; kwargs...)
    return _aggregate_draws(f, arr; dropdims)
end

"""
    outcome(TR::TuringRegression)

Get the response variable as DimArray.
"""
function outcome(TR::TuringRegression)
    return DimArray(TR.y, (Dim{:row}))
end

"""
    predictors(TR::TuringRegression, type::Symbol)

Get the predictor matrix variable as DimArray. `type` can be `:fixef` or `:ranef`
"""
function predictors(TR::TuringRegression, type::Symbol)
    if type === :fixef
        return DimArray(TR.X, (Dim{:row}, Dim{:var}([TR.X_names...])))
    elseif type === :ranef
        error("Not implemented")
    end
    #TODO add random_effecs version of that
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
