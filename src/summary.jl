
# build on MCMCChains.summarize
"""
    summary(io::IO, TR::TuringRegression; funs=[median, std], quantiles=[0.025, 0.975], return_table=false, standardized=false, draws_idx=nothing, kwargs...)

Display formatted summary table of model parameters.

# Arguments
- `io`: Output stream 
- `TR`: Fitted TuringRegression
- `funs`: Summary functions to apply (default: [median, std])
- `standardized`: Return standardized results? Default is false
- `quantiles`: Quantiles to compute (default: [0.025, 0.975] for 95% CI)
- `return_table`: Whether to return the summary table as NamedTuple
- `draws_idx`: Subset of draws to use (default: all draws)
- `kwargs...`: Additional arguments passed to summarize
"""
function Base.summary(
    io::IO,
    TR::TuringRegression;
    funs=[mean, std],
    quantiles=[0.025, 0.975],
    return_table=false,
    draws_idx=nothing,
    kwargs...,
)
    ##LLM in a hurry
    isnothing(TR.samples) && throw(ArgumentError("Turing Model has not yet been fit!()"))

    funs_all = vcat(funs..., [(x -> quantile(x, q)) for q in quantiles]...)
    func_names_all = vcat(
        Symbol.(funs)..., [Symbol("q$(round(q*100; digits =1))") for q in quantiles]...
    )

    draws_idx = something(draws_idx, 1:size(TR.samples, 1))

    # Summary
    summary_rows = map(_get_parameter_names(TR)) do p
        data = TR.parameters[param=At(p)]
        all_vals = vec(data)
        
        # Apply custom functions
        custom_vals = [f(all_vals) for f in funs]
        quant_vals = [quantile(all_vals, q) for q in quantiles]
        
        # Build row dynamically
        row = (parameters = p,)
        for (name, val) in zip(func_names_all, vcat(custom_vals, quant_vals))
            row = merge(row, (name => val,))
        end
        
        # Add diagnostics
        merge(row, (
            mcse = mcse(data),
            ess_bulk = ess(data),
            ess_tail = ess(data; kind=:tail),
            rhat = rhat(data)
        ))
    end 
    
    df = DataFrame(summary_rows)
    chain_info = (; (Symbol(n) => df[!, n] for n in names(df) if n != "parameters")...)
    
    ncols = length(chain_info)

    #metrics
    drop_warmup = size(TR.samples, 1) < 400 ? 0 : 200
    metric_tabs = map(
        f -> default_metrics(TR, f; drop_warmup=drop_warmup), funs_all
    )
    metric_tab = hcat(metric_tabs...)

    # show
    show(io, TR; warnings=false)
    println(io)
    pretty_table(
        io,
        chain_info;
        title="Fixed Effects",
        header=collect(keys(chain_info)),
        row_labels=parameter_names(TR),
        row_label_column_title="Parameter",
        highlighters=make_highlighters(ncols),
        formatters=(
            ft_printf("%5.2f", 1:(ncols - 5)),
            ft_printf("%5.2g", ncols-4),
            ft_printf("%5.0f", [ncols - 2, ncols - 3]),
            ft_printf("%5.3f", ncols - 1),
            ft_printf("%5.3f", ncols),
        ),
        default_options...,
    )
    pretty_table(
        io,
        Matrix(metric_tab);
        title="Prediction Metrics",
        header=func_names_all,
        row_labels=Array(dims(first(metric_tabs), 1)),
        row_label_column_title="Metric",
        formatters=(ft_printf("%5.3f")),
        default_options...,
    )
    model_warnings(chain_info)
    if return_table
        return chain_info
    else
        return nothing
    end
end

# Catch-all method for non-IO calls
function Base.summary(TR::TuringRegression, args...; kwargs...)
    summary(stdout, TR, args...; kwargs...)
end

function make_chain_info(TR::TuringRegression)
end

function model_warnings(chain_info)
    #warnings
    if any(chain_info[:rhat] .> 1.05)
        @warn "Some rhat values are > 1.05, treat parameter estimates with caution!"
    elseif any(chain_info[:rhat] .> 1.01)
        @info "Note that some rhat values are > 1.01"
    end
    if any(chain_info[:ess_bulk] .< 100)
        @warn "Some parameters have bulk ess < 100, point estimates may be unreliable!"
    elseif any(chain_info[:ess_bulk] .< 250)
        @info "Note that some parameters have bulk ess < 250"
    end
    if any(chain_info[:ess_tail] .< 100)
        @warn "Some parameters have tail ess < 100, credible intervals may be unreliable!"
    elseif any(chain_info[:ess_tail] .< 250)
        @info "Note some parameters have tail ess < 250"
    end
    if :std ∈ keys(chain_info)
        if any((chain_info[:mcse] ./ chain_info[:std]) .> 0.05)
            @warn "MCSE is > 5% of standard error for some parameters!"
        elseif any((chain_info[:mcse] ./ chain_info[:std]) .> 0.01)
            @info "MCSE is > 1% of standard error for some parameters"
        end
    end
end

"""
    model_warnings(TR::TuringRegression)

Display info and warnings about model fit. Based on `MCMCChains.summarize`
"""
function model_warnings(TR::TuringRegression)
    isnothing(TR.samples) && return nothing
    chain_info = summarize(TR.samples; sections=:parameters)
    model_warnings(chain_info.nt)
end

# Highlighters
function make_highlighters(ncols)
    return (
        #R hat
        Highlighter(
            (data, i, j) -> (j == ncols && data[j][i] > 1.05), crayon"bold magenta"
        ),
        Highlighter((data, i, j) -> (j == ncols && data[j][i] > 1.02), crayon"magenta"),
        #ESS
        Highlighter(
            (data, i, j) -> (j ∈ [ncols-1, ncols-2] && data[j][i] < 100),
            crayon"bold magenta",
        ),
        Highlighter(
            (data, i, j) -> (j ∈ [ncols-1, ncols-2] && data[j][i] < 250), crayon"magenta"
        ),
    )
end

# Default table options
default_options = (;
    tf=tf_compact,
    header_crayon=crayon"bold",
    row_label_header_crayon=crayon"bold",
    crop=:horizontal,
    show_subheader=false,
)
