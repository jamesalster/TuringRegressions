
# Calcualte a single metric across draws
function _calculate_metric(metric, preds::DimArray, y::DimArray)::DimArray
    return mapslices(x -> metric(x, y), preds; dims=1)
end

# Calculate many metrics across draws, and handle the DimArray neatly
"""
    calculate_metrics(TR::TuringRegression, metrics::Vector; dropdims=true, threshold=0.5, kwargs...)
    calculate_metrics(fun::Function, TR::TuringRegression, metrics::Vector; dropdims=true, threshold=0.5, kwargs...)

Calculate multiple metrics on model predictions using expected predictions (epred).

Takes a list of metrics (like `accuracy`, `rmse`) from `StatisticalMethods.jl` and applies each one to compare
your model's predictions against actual outcomes. Returns results in a table.

# Arguments
- `fun`: Optional function to apply across draws (e.g., mean, median, std), passed first matching `draws(f, TR, type)`
- `metrics`: Vector of metric functions to calculate
- `threshold`: Class threshold for binary classification (ignored for other models)
- `drop_warmup`: Number of warmup samples to drop from each chain
- `n_draws`: Number of draws to keep (-1 for all post-warmup)
- `collapse`: Whether to collapse chains into single dimension
- `dropdims`: Whether to drop singleton dimensions (default: true)

# Examples
```julia
calculate_metrics(my_model, [accuracy, kappa])
# collapse with function
calculate_metrics(mean, my_model, [rmse, mae], threshold=0.6)
# select draws
calculate_metrics(my_model, [rmse, mae], drop_warmup=500, collapse=false)
```
"""
function calculate_metrics(
    TR::TuringRegression{T},
    metrics::Vector;
    dropdims=true,
    threshold=0.5,
    kwargs...,
)::DimArray where {T}
    preds = posterior_predict(TR; type=:epred, kwargs...)
    y = outcome(TR)

    # Special handling for bernoulli
    if T == Bernoulli

        # handle metrics requiring a numeric outcome
        numeric_metric_table = zeros(0, size(preds)[2:end]...)
        if auc ∈ metrics
            numeric_metric_table = cat(numeric_metric_table, _get_auc(preds, y); dims = 1)
        end
        if pseudo_r2 ∈ metrics
            numeric_metric_table = cat(numeric_metric_table, _calculate_metric(pseudo_r2, preds, y); dims = 1)
        end

        # convert to category for remaining metrics
        preds = rebuild(preds, categorical(parent(preds) .> threshold))
        y = rebuild(y, categorical(parent(y) .== 1))

        # build table
        metrics2 = filter(x -> x ∉ [auc, pseudo_r2], metrics)
        metric_table = cat(
            map(metric -> _calculate_metric(metric, preds, y), metrics2)...; dims=1
        )

        # add AUC back in if we need
        if size(numeric_metric_table, 1) > 0
            metric_table = !isempty(metric_table) ? cat(metric_table, numeric_metric_table; dims=1) : numeric_metric_table
        end
    else
        # Calculate table
        metric_table = cat(
            map(metric -> _calculate_metric(metric, preds, y), metrics)...; dims=1
        )
        metrics2 = metrics
    end

    #clean names, messy with AUC
    metric_names = replace.(
        string.(metrics2), r"\(.*\)" => "", "LPLoss(p = 1)" => "MeanAbsoluteError"
    )
    metric_names = auc ∈ metrics ? vcat(metric_names, "AreaUnderCurve") : metric_names
    metric_names = pseudo_r2 ∈ metrics ? vcat(metric_names, "Pseudo r2") : metric_names

    if ndims(metric_table) == 2
        metric_table = DimArray(metric_table, (Dim{:metric}(metric_names), Dim{:draw}))
    elseif ndims(metric_table) == 3
        metric_table = DimArray(metric_table, (Dim{:metric}(metric_names), Dim{:draw}, Dim{:chain}))
    end

    return dropdims ? _drop_single_dims(metric_table) : metric_table
end

function calculate_metrics(
    fun::Function,
    TR::TuringRegression,
    metrics::Vector;
    dropdims=true,
    kwargs...,
)::DimArray
    metric_table = calculate_metrics(TR, metrics; dropdims=false, kwargs...)
    metric_table = mapslices(fun, metric_table; dims=2)
    return dropdims ? _drop_single_dims(metric_table) : metric_table
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
            pseudo_r2
        ]
    else
        return [rsq, rmse, mae]
    end
end

"""
    default_metrics(TR::TuringRegression; kwargs...)
    default_metrics(fun::Function, TR::TuringRegression; kwargs...)

Calculate standard metrics for your model type using expected predictions (epred).

Automatically selects appropriate metrics based on your model's distribution family.
For binary classification: accuracy, kappa, true positive rate, true negative rate.
For regression: R-squared, RMSE, mean absolute error.

Other arguments as for `calculate_metrics()`.
"""
function default_metrics(TR::TuringRegression; kwargs...)
    return calculate_metrics(TR, _get_default_metrics(TR); kwargs...)
end

function default_metrics(fun::Function, TR::TuringRegression; kwargs...)
    return calculate_metrics(fun, TR, _get_default_metrics(TR); kwargs...)
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

# Internal pseudoR2 implementation, McFadden method
function pseudo_r2(preds, y)
    y_num = convert(Vector{Float64}, y)
    # Log-likelihood of full model
    ll_full = sum(y_num .* log.(preds) .+ (1 .- y_num) .* log.(1 .- preds))
    
    # Log-likelihood of null model (intercept only)
    p_null = mean(y_num)
    ll_null = sum(y_num .* log(p_null) .+ (1 .- y_num) .* log(1 - p_null))
    
    # McFadden's R²
    return 1 - (ll_full / ll_null)
end