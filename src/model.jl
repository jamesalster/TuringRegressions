
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
function _random_effects(modeldata::ModelData)
    model_ranef = modeldata.Z
    body = Expr(:block)

    # Loop over ranef
    for (i, ranef) in enumerate(model_ranef)
        # Positional index, not the group's variable name (e.g. Subject) — keeps
        # generated symbols (and thus the compiled model) identical across ranef terms
        # that share structural shape but differ only in grouping-variable name, so the
        # model cache (model_cache.jl) doesn't have to treat them as different models.
        # Real group name/levels are reattached post-hoc from TR.modeldata.Z[i] when
        # splitting the flat sampled-VarName array into named layers (fit!/reshape.jl).

        #Name parameters
        variance_ranef = Symbol("σ_z_",i)
        ranef_matrix_raw = Symbol("r_z_",i)
        ranef_matrix = Symbol("ranef_z_",i)
        L_ranef = Symbol("L_z_",i)

        # No free mean parameter here: the population-level α/β already model the
        # mean effect. A free ranef mean would be additively confounded with β
        # (only their sum is identified), producing a slow/degenerate NUTS ridge
        # and biased marginals (T9). Ranef components are mean-zero by construction.
        #Build varying slopes prior
        if ranef.predictors.has_intercept & has_fixed_effects(ranef.predictors)
            n_predictors = size(ranef.predictors.X, 2) + 1
            push!(body.args, quote
                $variance_ranef ~ filldist(prior_random_effect_variance, $n_predictors)
                $L_ranef ~ LKJCholesky($n_predictors, 2.0)
                $ranef_matrix_raw ~ filldist(MvNormal(zeros($n_predictors), I), n_groups[$i])
                # Transform: Σ^(1/2) * z_raw, where Σ^(1/2) = diag(σ_z) * L_z
                $ranef_matrix = (diagm($variance_ranef) * $L_ranef.L * $ranef_matrix_raw)'
            end)
        elseif TuringRegressions.has_fixed_effects(ranef.predictors)
            n_predictors = size(ranef.predictors.X, 2)
            push!(body.args, quote
                $variance_ranef ~ filldist(prior_random_effect_variance, $n_predictors)
                $ranef_matrix_raw ~ filldist(Normal(), $n_predictors, n_groups[$i])
                $ranef_matrix = ($variance_ranef .* $ranef_matrix_raw)'
            end)
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

            if ranef.predictors.has_intercept & has_fixed_effects(ranef.predictors)
                push!(terms, :($ranef_matrix[group_idx[:,$i], 1]))
                push!(terms, :(sum($predictors .* $ranef_matrix[group_idx[:,$i], 2:end]; dims = 2)[:]))
            elseif TuringRegressions.has_fixed_effects(ranef.predictors)
                push!(terms, :(sum($predictors .* $ranef_matrix[group_idx[:,$i], :]; dims = 2)[:]))
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

# likelihood
function _likelihood(family::Type{<:Distribution})
    if family == Normal
        quote
            Turing.@addlogprob! logpdf(MvNormal(μ, σ), y)
        end
    elseif family == TDist
        quote
            Turing.@addlogprob! logpdf(arraydist((μ) .+ σ .* TDist.(ν)), y)
        end
    elseif family == Bernoulli
        quote
            for n in 1:nobs
                Turing.@addlogprob! logpdf(BernoulliLogit(μ[n]), y[n]) #Not scaled
            end
        end
    elseif family == Poisson
        quote
            for n in 1:nobs
                Turing.@addlogprob! logpdf(LogPoisson(μ[n]), y[n]) #Not scaled
            end
        end
    elseif family == NegativeBinomial
        quote
            for n in 1:nobs
                Turing.@addlogprob! logpdf(NegativeBinomial2(exp(μ[n]), ϕ_inv), y[n]) #Not scaled
            end
        end
    end
end

# weighted likelihood
function _weighted_likelihood(family::Type{<:Distribution})
    if family == Normal
        quote
            for n in 1:nobs
                Turing.@addlogprob! weights[n] * logpdf(Normal(μ[n], σ), y[n])
            end
        end
    elseif family == TDist
        quote
            for n in 1:nobs
                Turing.@addlogprob! weights[n] * logpdf(μ[n] + σ * TDist(ν), y[n])
            end
        end
    elseif family == Bernoulli
        quote
            for n in 1:nobs
                Turing.@addlogprob! weights[n] * logpdf(BernoulliLogit(μ[n]), y[n])
            end
        end
    elseif family == Poisson
        quote
            for n in 1:nobs
                Turing.@addlogprob! weights[n] * logpdf(LogPoisson(μ[n]), y[n])
            end
        end
    elseif family == NegativeBinomial
        quote
            for n in 1:nobs
                Turing.@addlogprob! weights[n] * logpdf(NegativeBinomial2(exp(μ[n]), ϕ_inv), y[n]) #Not scaled
            end
        end
    end
end

#### Main function to assemble the model code
function build_model_body(family::Type{<:Distribution}, modeldata::ModelData)
    model_ranef = modeldata.Z
    weighted = is_weighted(modeldata)

    # Empty quote
    body = Expr(:block)

    # Prior
    if has_intercept(modeldata)
        push!(body.args, _intercept())
    end
    if has_fixed_effects(modeldata)
        push!(body.args, _fixed_effects())
    end
    if has_random_effects(modeldata)
        push!(body.args, _random_effects(modeldata))
    end

    if family ∉ [Bernoulli, Poisson] #Bernoulli and Poisson have no auxiliary parameter
        push!(body.args, _auxiliary_parameter(family))
    end

    # Linear Model
    push!(body.args, _linear_model(modeldata))

    # Likelihood
    if weighted
        push!(body.args, _weighted_likelihood(family))
    else
        push!(body.args, _likelihood(family))
    end

    # No custom return: extraction reads sampled VarNames straight off the chain
    # (`DimArray(TR.samples)`, R10) in fit!/reshape.jl — see note above _pointwise_loglik.

    return body
end

#### Wrapper function for the above, to handle some additional logic
function construct_model(family::Type{<:Distribution}, modeldata::ModelData, show_code::Bool=false)

    #handle logic here
    body = build_model_body(family, modeldata)

    # build model code
    # Unique name per generated model: DynamicPPL dispatches model evaluation on
    # typeof(f), so reusing "turing_regression" for every model let Turing's
    # internal AD/dual-number caches (keyed on that shared type) leak between
    # models with different parameter counts, causing BoundsErrors during sampling.
    fname = gensym(:turing_regression)
    model_code = quote
        @model function $(fname)(y, X, n_groups, group_idx, group_predictors, weights,
            prior_intercept, prior_fixed_effects, prior_random_effect_variance, prior_auxiliary)
            nobs, npredictors = size(X)
            $body
        end
    end

    model_code_str = prettify(model_code)
    if show_code
        println("Generated model:\n $(model_code_str)")
    end

    return eval(model_code), model_code_str
end