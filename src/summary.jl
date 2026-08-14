
# Compute point-estimate/quantile/diagnostic columns for each label along a (label, draw, chain)
# DimArray. Used for both the :fixef table and each grouping term's SD table.
function _diagnostics_table(arr, labels, funs, func_names_all, quantiles)
    n = length(labels)
    stat_vectors = [Vector{Float64}(undef, n) for _ in func_names_all]
    mcse_vec = Vector{Float64}(undef, n)
    ess_bulk_vec = Vector{Float64}(undef, n)
    ess_tail_vec = Vector{Float64}(undef, n)
    rhat_vec = Vector{Float64}(undef, n)

    for i in 1:n
        data = arr[i, :, :]
        all_vals = vec(data)

        for (j, f) in enumerate(funs)
            stat_vectors[j][i] = f(all_vals)
        end

        quants = quantile(all_vals, quantiles)
        for (k, q) in enumerate(quants)
            stat_vectors[length(funs) + k][i] = q
        end

        mcse_vec[i] = mcse(data)
        ess_bulk_vec[i] = ess(data)
        ess_tail_vec[i] = ess(data; kind=:tail)
        rhat_vec[i] = rhat(data)
    end

    return (;
        zip(func_names_all, stat_vectors)...,
        mcse = mcse_vec, ess_bulk = ess_bulk_vec,
        ess_tail = ess_tail_vec, rhat = rhat_vec
    )
end

"""
    model_summary(io::IO, TR::TuringRegression; funs=[median, std], quantiles=[0.025, 0.975], return_table=false, drop_draws=nothing, kwargs...)

Display formatted summary table of model parameters.

# Arguments
- `io`: Output stream
- `TR`: Fitted TuringRegression
- `funs`: Summary functions to apply (default: [median, std])
- `quantiles`: Quantiles to compute (default: [0.025, 0.975] for 95% CI)
- `return_table`: Whether to return the summary table as NamedTuple
- `drop_draws`: Number of extra warmup draws to drop, on top of what `fit!` already discarded as warmup draws
- `kwargs...`: Additional arguments passed to `draws` (e.g. `n_draws`)
"""
function model_summary(
    io::IO,
    TR::TuringRegression;
    funs=[mean, std],
    quantiles=[0.025, 0.975],
    return_table=false,
    drop_draws=nothing,
    kwargs...,
)
    isnothing(TR.samples) && throw(ArgumentError("Turing Model has not yet been fit!()"))

    func_names_all = vcat(
        Symbol.(funs), [Symbol("q$(round(q*100; digits=1))") for q in quantiles]
    )

    drop_draws = something(drop_draws, 0)

    fixef_draws = draws(TR, :fixef; drop_draws=drop_draws, collapse=false, kwargs...)
    param_names = collect(dims(fixef_draws, :fixef))
    chain_info = _diagnostics_table(fixef_draws, param_names, funs, func_names_all, quantiles)

    ncols = length(chain_info)

    # header: family/formula/observations/samples, no prior — same content as the full
    # show(io, MIME"text/plain", TR) minus the prior block
    label_style = crayon"bold !underline"
    normal_style = crayon"reset"
    _print_family_formula(io, TR, label_style, normal_style)
    _print_obs_samples(io, TR, label_style, normal_style)
    println(io)
    pretty_table(
        io,
        chain_info;
        title="Fixed Effects",
        column_labels=collect(keys(chain_info)),
        row_labels=param_names,
        stubhead_label="Parameter",
        highlighters=make_highlighters(ncols),
        formatters=_stat_formatters(ncols),
        default_options...,
    )
    if has_random_effects(TR)
        # Tables are titled by the LAYER key, so two terms on one grouping variable get
        # their own headed block (`Subject_1` / `Subject_2`) rather than one ambiguous pair.
        for group in ranef_layer_keys(TR.modeldata.Z)
            level_draws = draws(TR, group; drop_draws=drop_draws, collapse=false, kwargs...)
            level_effect_names = collect(dims(level_draws, _effect_dim_name(group)))
            levels = collect(dims(level_draws, _group_dim_name(group)))
            for (ei, eff) in enumerate(level_effect_names)
                level_info = _diagnostics_table(level_draws[ei, :, :, :], levels, funs, func_names_all, quantiles)
                pretty_table(
                    io,
                    level_info;
                    title="Random Effects: $group ($eff)",
                    column_labels=collect(keys(level_info)),
                    row_labels=levels,
                    stubhead_label="Level",
                    highlighters=make_highlighters(ncols),
                    formatters=_stat_formatters(ncols),
                    default_options...,
                )
            end

            sd_draws = draws(TR, Symbol(group, "_sd"); drop_draws=drop_draws, collapse=false, kwargs...)
            effect_names = collect(dims(sd_draws, _effect_dim_name(group)))
            ranef_info = _diagnostics_table(sd_draws, effect_names, funs, func_names_all, quantiles)
            pretty_table(
                io,
                ranef_info;
                title="Random Effects: $group (SD)",
                column_labels=collect(keys(ranef_info)),
                row_labels=effect_names,
                stubhead_label="Effect",
                highlighters=make_highlighters(ncols),
                formatters=_stat_formatters(ncols),
                default_options...,
            )

            corr_sym = Symbol(group, "_corr")
            if corr_sym ∈ propertynames(TR.parameters)
                corr_point = draws(mean, TR, corr_sym; drop_draws=drop_draws, kwargs...)
                pretty_table(
                    io,
                    Matrix(corr_point);
                    title="Random Effects: $group (Correlation)",
                    column_labels=effect_names,
                    row_labels=effect_names,
                    stubhead_label="Effect",
                    formatters=[fmt__printf("%5.2f")],
                    default_options...,
                )
            end
        end
    end
    model_warnings(chain_info)
    if return_table
        return chain_info
    else
        return nothing
    end
end

# Catch-all method for non-IO calls
model_summary(TR::TuringRegression, args...; kwargs...) = model_summary(stdout, TR, args...; kwargs...)

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

# Merge every non-(iter,chain) dim of a layer into one leading axis, so each row is
# one scalar sampled parameter's (iter,chain) trace — same shape _diagnostics_table
# expects, generalised to layers that carry more than one non-`:fixef`/`:effect` dim
# (e.g. `:{group}_corr`'s `effect`×`effect2`).
function _flatten_layer(arr)
    iter_chain = (dimnum(arr, :iter), dimnum(arr, :chain))
    other = setdiff(1:ndims(arr), iter_chain)
    permuted = permutedims(Array(arr), (other..., iter_chain...))
    return reshape(permuted, :, size(permuted, ndims(permuted) - 1), size(permuted, ndims(permuted)))
end

"""
    model_warnings(TR::TuringRegression)

Display info and warnings about model fit, computed directly off `TR.parameters`
(rhat/ess/mcse work on plain `(param,iter,chain)` arrays, no chain-object API needed).
"""
function model_warnings(TR::TuringRegression)
    isnothing(TR.samples) && return nothing
    stds, mcses, ess_bulks, ess_tails, rhats = Float64[], Float64[], Float64[], Float64[], Float64[]
    for name in propertynames(TR.parameters)
        flat = _flatten_layer(TR.parameters[name])
        for i in axes(flat, 1)
            data = flat[i, :, :]
            push!(stds, std(vec(data)))
            push!(mcses, mcse(data))
            push!(ess_bulks, ess(data))
            push!(ess_tails, ess(data; kind=:tail))
            push!(rhats, rhat(data))
        end
    end
    model_warnings((; std=stds, mcse=mcses, ess_bulk=ess_bulks, ess_tail=ess_tails, rhat=rhats))
end

# Column layout: [funs..., quantiles..., mcse, ess_bulk, ess_tail, rhat]
function _stat_formatters(ncols)
    return [
        fmt__printf("%5.2f", collect(1:(ncols - 4))),
        fmt__printf("%5.2g", [ncols - 3]),
        fmt__printf("%5.0f", [ncols - 2, ncols - 1]),
        fmt__printf("%5.3f", [ncols]),
    ]
end

# Highlighters
function make_highlighters(ncols)
    return [
        #R hat
        TextHighlighter(
            (data, i, j) -> (j == ncols && data[j][i] > 1.05), crayon"bold magenta"
        ),
        TextHighlighter((data, i, j) -> (j == ncols && data[j][i] > 1.02), crayon"magenta"),
        #ESS
        TextHighlighter(
            (data, i, j) -> (j ∈ [ncols-1, ncols-2] && data[j][i] < 100),
            crayon"bold magenta",
        ),
        TextHighlighter(
            (data, i, j) -> (j ∈ [ncols-1, ncols-2] && data[j][i] < 250), crayon"magenta"
        ),
    ]
end

# Default table options
const default_options = (;
    style=TextTableStyle(; column_label=crayon"bold", stubhead_label=crayon"bold"),
    # true = crop overflowing columns with `⋯` rather than letting the terminal wrap the
    # row onto the next line, which makes a wide summary unreadable
    fit_table_in_display_horizontally=true,
)
