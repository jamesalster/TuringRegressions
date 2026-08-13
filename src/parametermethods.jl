

# Utility function for selecting draws and collapsing chains from a samples AxisArray
function _process_draws(DA::Union{DimArray, DimStack}; drop_draws::Int=0, n_draws::Real=Inf, collapse::Bool=true)
    # Drop warmup
    arr = DA[iter=(drop_draws + 1):size(DA, :iter)]
    # Select draws
    if isfinite(n_draws)
        n_draws > size(arr, :iter) && throw(ArgumentError(
            "$n_draws draws is too many from $(size(arr, :iter)) available. Note that n_draws is applied per chain."
        ))
        arr = arr[iter=1:Int(n_draws)]
    end
    # Collapse
    arr = collapse ? mergedims(arr, (:iter, :chain) => :iter) : arr
    size(arr, :iter) <= 0 && throw(ArgumentError(
        "No samples returned, check kwargs and perhaps try adjusting `drop_draws`?"
    ))
    return arr
end

# Drop single dimensions where possible from an array
function _drop_single_dims(DA::Union{DimArray, DimStack})
    dims_to_drop = findall(==(1), size(DA))
    return dropdims(DA; dims=Tuple(dims_to_drop))
end

# Aggregate a draws/chain-dimensioned DimArray with fun over :iter (+:chain if present)
function _aggregate_draws(f::Function, arr::DimArray; dropdims=true)
    dims_to_aggregate = hasdim(arr, :chain) ? [:iter, :chain] : [:iter]
    dimindices = ntuple(i -> dimnum(arr, dims_to_aggregate[i]), length(dims_to_aggregate))
    out = mapslices(f, arr; dims=dimindices)
    return dropdims ? _drop_single_dims(out) : out
end

"""
    draws(TR::TuringRegression, type::Symbol; drop_draws=0, n_draws=Inf, collapse=true)
    draws(f::Function, TR::TuringRegression, type::Symbol; dropdims=true, drop_draws=0, n_draws=Inf, collapse=true)
    draws(TR::TuringRegression; drop_draws=0, n_draws=Inf, collapse=true)

Extract specific draws from fitted model as `DimArray` (or `DimStack` if `type` is not passed).
Passing a function (e.g. median) aggregates the draws with that function.

# Arguments
- `type`: Symbol for the type of draw. Can be `:fixef`, `:{ranef_name}`, `:{ranef_name}_sd`, `:{group_name}_corr`
- `drop_draws`: Number of extra warmup samples to drop from each chain, on top of what `fit!` already discarded during Turing's adaptation phase (default is `0` — `TR.samples` holds no warmup draws already, see `fit!`'s `warmup` kwarg)
- `n_draws`: Number of draws to keep (default is `Inf` for all available draws)
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
    # Group layers are stored under a per-term unique dim name (reshape.jl) so multiple
    # ranef terms with different level counts can share one DimStack; renamed to the
    # public `:group` name here since a single extracted layer has no name collision.
    hasdim(arr, _group_dim_name(type)) && (arr = set(arr, _group_dim_name(type) => :group))
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
    return DimArray(TR.modeldata.y, (Dim{:row}))
end

"""
    get_fixef_predictors(TR::TuringRegression)

Get the fixed-effects predictor matrix as DimArray.
"""
function get_fixef_predictors(TR::TuringRegression)
    return DimArray(TR.modeldata.predictors.X, (Dim{:row}, Dim{:var}([TR.modeldata.predictors.X_names...])))
end
