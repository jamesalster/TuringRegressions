
# Utility function for selecting draws and collapsing chains from a samples AxisArray
function process_draws(
    arr::DimArray; drop_warmup::Int=200, n_draws::Int=-1, collapse::Bool=true
)
    arr = arr[:, (drop_warmup + 1):End, :] #drop warmup
    n_draws > size(arr, 2) && throw(
        ErrorException(
            "$n_draws draws is too many from $(size(arr, 2)) available. Note that n_draws is applied per chain.",
        ),
    )
    arr = n_draws > 0 ? arr[:, 1:n_draws, :] : arr #select draws
    if collapse
        return transpose(mergedims(arr, (:draw, :chain) => :draw))
    else
        return arr
    end
end

# Drop single dimensions where possible from an array
function drop_single_dims(DA::DimArray)
    dims_to_drop = findall(==(1), size(DA))
    return dropdims(DA; dims=Tuple(dims_to_drop))
end

# Helper function for display
function clean_prior_string(x)
    replace(x, r"\{.*\}" => "", "\n" => " ")
end

# Helper function to know TuringGLM's default links
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
