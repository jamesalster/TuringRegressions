
# Calcualte a single metric across draws
function _calculate_metric(metric, preds::DimArray, y::DimArray)::DimArray
    return mapslices(x -> metric(x, y), preds; dims=1)
end

# Calculate many metrics across draws, and handle the DimArray neatly
"""
    calculate_metrics(TR::TuringRegression, metrics::Vector, fun=nothing; dropdims=true, threshold=0.5, kwargs...)

Calculate multiple metrics on model predictions using expected predictions (epred).

Takes a list of metrics (like `accuracy`, `rmse`) from `StatisticalMethods.jl` and applies each one to compare 
your model's predictions against actual outcomes. Returns results in a table.

# Arguments
- `metrics`: Vector of metric functions to calculate
- `fun`: Optional function to apply across draws (e.g., mean, median, std)
- `threshold`: Class threshold for binary classification (ignored for other models)
- `drop_warmup`: Number of warmup samples to drop from each chain
- `n_draws`: Number of draws to keep (-1 for all post-warmup)  
- `collapse`: Whether to collapse chains into single dimension
- `dropdims`: Whether to drop singleton dimensions (default: true)

# Examples
```julia
calculate_metrics(my_model, [accuracy, kappa])
# collapse with function
calculate_metrics(my_model, [rmse, mae], mean, threshold=0.6)
# select draws
calculate_metrics(my_model, [rmse, mae], drop_warmup=500, collapse=false)
"""
function calculate_metrics(
    TR::TuringRegression{T},
    metrics::Vector,
    fun::Union{Nothing,Function}=nothing;
    dropdims=true,
    threshold=0.5,
    kwargs...,
)::DimArray where {T}
    preds = predict(TR; type=:epred, kwargs...)
    y = outcome(TR)

    # Special handling for bernoulli
    if T == Bernoulli
        with_auc = AreaUnderCurve() ∈ metrics

        # handle AUC
        auc_tab = with_auc ? _get_auc(preds, y) : []

        # convert to category for remaining metrics
        preds = rebuild(preds, categorical(parent(preds) .> threshold))
        y = rebuild(y, categorical(parent(y) .== 1))

        # build table
        metrics2 = filter(x -> x != auc, metrics)
        metric_table = cat(
            map(metric -> _calculate_metric(metric, preds, y), metrics2)...; dims=1
        )

        # add AUC back in if we need
        if with_auc
            metric_table =
                !isempty(metric_table) ? cat(metric_table, auc_tab; dims=1) : auc_tab
        end
    else
        # Calculate table
        metric_table = cat(
            map(metric -> _calculate_metric(metric, preds, y), metrics)...; dims=1
        )
        metrics2 = metrics
        with_auc = false
    end

    #clean names, messy with AUC
    metric_names = replace.(
        string.(metrics2), r"\(.*\)" => "", "LPLoss(p = 1)" => "MeanAbsoluteError"
    )
    metric_names = with_auc ? vcat(metric_names, "AreaUnderCurve") : metric_names

    #Broken quick method so we need to do this to set dimensions sadly
    metric_table = set(metric_table, Dim{:row} => Dim{:metric})
    metric_table = set(metric_table, Dim{:metric} => DimensionalData.Dimensions.Categorical)
    metric_table = set(metric_table, Dim{:metric} => [metric_names...])

    metric_table = isnothing(fun) ? metric_table : mapslices(fun, metric_table; dims=2)
    return dropdims ? drop_single_dims(metric_table) : metric_table
end

# Get the default metrics for a model family
function _get_default_metrics(TR::TuringRegression{T}) where {T}
    if T == Bernoulli
        return [
            accuracy,
            kappa,
            TruePositiveRate(; levels=[false, true]),
            TrueNegativeRate(; levels=[false, true]),
            auc,
        ]
    else
        return [rsq, rmse, mae]
    end
end

"""
    default_metrics(TR::TuringRegression, fun=nothing; kwargs...)

Calculate standard metrics for your model type using expected predictions (epred).

Automatically selects appropriate metrics based on your model's distribution family.
For binary classification: accuracy, kappa, true positive rate, true negative rate.
For regression: R-squared, RMSE, mean absolute error.

Other arguments as for `calculate_metrics()`.
"""
function default_metrics(
    TR::TuringRegression{T}, fun::Union{Nothing,Function}=nothing; kwargs...
) where {T}
    metrics = _get_default_metrics(TR)
    return calculate_metrics(TR, metrics, fun; kwargs...)
end

# Special function to handle AUC with distribution conversion
function _get_auc(preds::DimArray, y::DimArray)
    y_categ = categorical(parent(y) .== 1)
    mapslices(preds; dims=1) do preds_vec
        preds_as_distribution = UnivariateFinite(
            categorical([false, true]), preds_vec; augment=true
        )
        return auc(preds_as_distribution, y_categ)
    end
end
