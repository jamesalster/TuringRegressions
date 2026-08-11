
#### Building blocks of the model

# prior arg: prior_intercept
function _intercept()
    quote
        α ~ prior_intercept
    end
end

# prior arg: prior_fixed_effects
function _fixed_effects()
    quote
        β ~ filldist(prior_fixed_effects, npredictors)
    end
end

# prior arg: prior_random_effect_variance (shared across all ranef groups)
# All branches non-centred (sample unit-scale z_raw, multiply by σ) — sampling effects
# directly makes their width depend on σ, a funnel NUTS can't pick one step size for.
# Branch = shape of the term: (1 + x | g) correlated / (0 + x | g) slopes / (1 | g).
function _random_effects(modeldata::ModelData)
    model_ranef = modeldata.Z
    body = Expr(:block)

    # Loop over ranef
    for (i, ranef) in enumerate(model_ranef)
        # Positional index, not the group's variable name — keeps generated symbols
        # identical across ranef terms that differ only by grouping name. Real group
        # name/levels reattached post-hoc in reshape.jl.

        #Name parameters
        variance_ranef = Symbol("σ_z_",i)
        ranef_matrix_raw = Symbol("r_z_",i)
        ranef_matrix = Symbol("ranef_z_",i)
        L_ranef = Symbol("L_z_",i)

        #Build varying slopes prior
        # (1 + x | g): intercept and slopes usually correlate, so joint prior — SDs and
        # a correlation matrix, split so each half takes its own prior.
        if ranef.predictors.has_intercept & has_fixed_effects(ranef.predictors)
            n_predictors = size(ranef.predictors.X, 2) + 1
            push!(body.args, quote
                $variance_ranef ~ filldist(prior_random_effect_variance, $n_predictors)
                # Correlation matrix prior, decomposed
                $L_ranef ~ LKJCholesky($n_predictors, prior_lkj_eta)
                $ranef_matrix_raw ~ filldist(MvNormal(zeros($n_predictors), I), n_groups[$i])
                # Transform: Σ^(1/2) * z_raw, where Σ^(1/2) = diag(σ_z) * L_z.
                # Transposed so rows are groups — the linear model indexes by group.
                $ranef_matrix = (diagm($variance_ranef) * $L_ranef.L * $ranef_matrix_raw)'
            end)
        # (0 + x | g): no intercept to correlate with, so no LKJ — independent SD each.
        elseif TuringRegressions.has_fixed_effects(ranef.predictors)
            n_predictors = size(ranef.predictors.X, 2)
            push!(body.args, quote
                $variance_ranef ~ filldist(prior_random_effect_variance, $n_predictors)
                $ranef_matrix_raw ~ filldist(Normal(), $n_predictors, n_groups[$i])
                $ranef_matrix = ($variance_ranef .* $ranef_matrix_raw)'
            end)
        # (1 | g): one offset per group, the plain varying-intercept case.
        elseif ranef.predictors.has_intercept
            n_predictors = 1
            push!(body.args, quote
                $variance_ranef ~ filldist(prior_random_effect_variance, $n_predictors)
                $ranef_matrix_raw ~ filldist(Normal(), n_groups[$i])
                $ranef_matrix = $ranef_matrix_raw .* $variance_ranef 
            end)
        end
    end
    return body
end

# prior arg: prior_auxiliary
function _auxiliary_parameter(family::Type{<:Distribution})
    if family == Normal
        quote
            σ ~ prior_auxiliary
        end
    elseif family == TDist
        quote
            σ ~ Exponential(1)
            ν ~ prior_auxiliary
        end
    elseif family == NegativeBinomial
        quote
            ϕ ~ prior_auxiliary
            ϕ_inv = 1 / ϕ
        end
    elseif family ∈ [Bernoulli, Poisson]
        return :() #empty quote, no code
    else
        error("Family $family has no auxiliary-parameter code generator.")
    end
end

# linear model
function _linear_model(modeldata::ModelData)
    model_ranef = modeldata.Z

    # Get terms we need
    terms = []
    if modeldata.predictors.has_intercept
        push!(terms, :α)
    end
    if has_fixed_effects(modeldata.predictors)
        push!(terms, :(X * β))
    end
    if !isempty(model_ranef)
        for (i, ranef) in enumerate(model_ranef)
            ranef_matrix = Symbol("ranef_z_", i)
            predictors = :(group_predictors[$i])

            # Columns as _random_effects built them: intercept first (if present), then
            # slopes. Row-indexing by group_idx fans each group's effect out to its obs.
            if ranef.predictors.has_intercept & has_fixed_effects(ranef.predictors)
                push!(terms, :($ranef_matrix[group_idx[:,$i], 1]))
                n_slope = size(ranef.predictors.X, 2)
                # @view avoids an alloc in the hot path — worth the extra branch.
                if n_slope == 1
                    push!(terms, :(vec($predictors) .* $ranef_matrix[group_idx[:,$i], 2]))
                else
                    push!(terms, :(sum($predictors .* @view($ranef_matrix[group_idx[:,$i], 2:end]); dims = 2)[:]))
                end
            elseif TuringRegressions.has_fixed_effects(ranef.predictors)
                n_slope = size(ranef.predictors.X, 2)
                if n_slope == 1
                    push!(terms, :(vec($predictors) .* $ranef_matrix[group_idx[:,$i], 1]))
                else
                    push!(terms, :(sum($predictors .* @view($ranef_matrix[group_idx[:,$i], :]); dims = 2)[:]))
                end
            elseif ranef.predictors.has_intercept
                push!(terms, :($ranef_matrix[group_idx[:,$i]]))
            end
        end
    end

    # Build expression
    if length(terms) == 1
        rhs = first(terms)
    else
        rhs = Expr(:call, :(.+), terms...)
    end

    # return as quote
    return quote μ = $rhs end
end

# per-observation logpdf expr, shared weighted/unweighted (loop-based families only)
function _obs_logpdf(family::Type{<:Distribution})
    if family == Bernoulli
        :(logpdf(BernoulliLogit(μ[n]), y[n])) #Not scaled
    elseif family == Poisson
        :(logpdf(LogPoisson(μ[n]), y[n])) #Not scaled
    elseif family == NegativeBinomial
        :(logpdf(NegativeBinomial2(exp(μ[n]), ϕ_inv), y[n])) #Not scaled
    else
        error("Family $family has no per-observation logpdf code generator.")
    end
end

# likelihood. weighted picked at construct-time, not inside NUTS hot path.
function _likelihood(family::Type{<:Distribution}, weighted::Bool)
    if !weighted && family == Normal
        return quote
            Turing.@addlogprob! logpdf(MvNormal(μ, σ), y)
        end
    elseif !weighted && family == TDist
        return quote
            Turing.@addlogprob! logpdf(arraydist((μ) .+ σ .* TDist.(ν)), y)
        end
    elseif weighted && family == Normal
        return quote
            for n in 1:nobs
                Turing.@addlogprob! weights[n] * logpdf(Normal(μ[n], σ), y[n])
            end
        end
    elseif weighted && family == TDist
        return quote
            for n in 1:nobs
                Turing.@addlogprob! weights[n] * logpdf(μ[n] + σ * TDist(ν), y[n])
            end
        end
    end

    obs = _obs_logpdf(family)
    term = weighted ? :(weights[n] * $obs) : obs
    quote
        for n in 1:nobs
            Turing.@addlogprob! $term
        end
    end
end

#### Main function to assemble the model code
function build_model_body(family::Type{<:Distribution}, modeldata::ModelData)
    model_ranef = modeldata.Z
    weighted = is_weighted(modeldata)

    # Empty quote
    body = Expr(:block)
    labels = String[] # section label per body.args entry, display-only (show_code)

    # Prior
    if has_intercept(modeldata)
        push!(body.args, _intercept())
        push!(labels, "Intercept prior")
    end
    if has_fixed_effects(modeldata)
        push!(body.args, _fixed_effects())
        push!(labels, "Fixed-effects prior")
    end
    if has_random_effects(modeldata)
        push!(body.args, _random_effects(modeldata))
        push!(labels, "Random-effects prior")
    end

    if family ∉ [Bernoulli, Poisson] #Bernoulli and Poisson have no auxiliary parameter
        push!(body.args, _auxiliary_parameter(family))
        push!(labels, "Auxiliary parameter prior")
    end

    # Linear Model
    push!(body.args, _linear_model(modeldata))
    push!(labels, "Linear model")

    # Likelihood
    push!(body.args, _likelihood(family, weighted))
    push!(labels, weighted ? "Likelihood (weighted)" : "Likelihood")

    # No custom return: extraction reads sampled VarNames straight off the chain
    # (`DimArray(TR.samples)`) in fit!/reshape.jl.

    return body, labels
end

#### Build the raw @model Expr (unevaluated) — shared by construct_model and modelcode
# so the printed/returned code is exactly what actually runs, not a re-derived copy.
function build_model_expr(family::Type{<:Distribution}, modeldata::ModelData)
    body, _ = build_model_body(family, modeldata)

    # Unique name per generated model: DynamicPPL dispatches model evaluation on
    # typeof(f), so reusing "turing_regression" for every model let Turing's
    # internal AD/dual-number caches (keyed on that shared type) leak between
    # models with different parameter counts, causing BoundsErrors during sampling.
    # PID suffix because gensym's counter restarts each session: the precompile workload's
    # model name regenerates verbatim at runtime and overwrites the precompiled method.
    fname = Symbol(gensym(:turing_regression), :_, getpid())
    # Priors passed as separate runtime args, not baked into the Expr or bundled
    # into a struct — keeps model shape prior-independent for a (currently unused)
    # future cache keyed on that shape.
    return quote
        @model function $(fname)(y, X, n_groups, group_idx, group_predictors, weights,
            prior_intercept, prior_fixed_effects, prior_random_effect_variance, prior_auxiliary,
            prior_lkj_eta)
            nobs, npredictors = size(X)
            $body
        end
    end
end

#### Wrapper function for the above, to handle some additional logic
function construct_model(family::Type{<:Distribution}, modeldata::ModelData)
    return eval(build_model_expr(family, modeldata))
end

"""
    modelcode(TR::TuringRegression) -> Expr

Print the generated Turing model code for `TR`, annotated with a comment
header per section (priors / linear model / likelihood), and RETURN the
underlying `Expr` — the same one `construct_model` would `eval`. Rebuilds on
demand from `TR.modeldata` — this is always the STANDARD generated model, not
any edited version previously passed to `set_model_code!`; no effect on the
already-fitted model.

Edit the returned `Expr` and pass it to `set_model_code!(TR, expr)` to swap in
a hand-modified model. Read `set_model_code!`'s docstring before using it.
"""
function modelcode(TR::TuringRegression{T}) where {T}
    body, labels = build_model_body(T, TR.modeldata)
    sections = ["    # $label\n" * string(prettify(frag)) for (label, frag) in zip(labels, body.args)]
    println("""
    @model function turing_model(y, X, n_groups, group_idx, group_predictors, weights,
        prior_intercept, prior_fixed_effects, prior_random_effect_variance, prior_auxiliary,
        prior_lkj_eta)
        nobs, npredictors = size(X)
    $(join(sections, "\n\n"))
    end
    """)
    return build_model_expr(T, TR.modeldata)
end

"""
    set_model_code!(TR::TuringRegression, expr::Expr) -> TuringRegression

Override `TR`'s generated model with a hand-edited `Expr` (start from
`modelcode(TR)`, edit, pass back in). Evals `expr` and replaces `TR.model`.

Before using:
- `expr` must eval to a callable taking exactly `(y, X, n_groups, group_idx,
  group_predictors, weights, prior_intercept, prior_fixed_effects,
  prior_random_effect_variance, prior_auxiliary, prior_lkj_eta)` in that
  order — `fit!` calls it positionally. Wrong signature → `MethodError` at
  `fit!` time, not here.
- Sampled VarNames in the body must keep their roots (`:α`, `:β`, family aux,
  per-ranef `σ_z_<i>`/`r_z_<i>`/`L_z_<i>`) — rename/drop one and post-fit
  reshaping fails loudly rather than silently mislabeling draws.
- `expr` is `eval`'d once; only the resulting `Function` is kept (`TR.model`) —
  the edited source itself isn't stored anywhere on `TR`.
- Resets `TR.samples`/`TR.parameters` to `nothing` — they're draws from the
  OLD model, keeping them would silently pass off stale results as current.
  Refit after calling this.
"""
function set_model_code!(TR::TuringRegression, expr::Expr)
    model_fn = eval(expr)
    model_fn isa Function || throw(ArgumentError(
        "expr must eval to a Function, got $(typeof(model_fn))"
    ))
    TR.model = model_fn
    TR.samples = nothing
    TR.parameters = nothing
    return TR
end
