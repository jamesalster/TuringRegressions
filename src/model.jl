
#### Building blocks of the model

# prior
function _intercept(prior::Distribution)
    quote
        α ~ $prior
    end
end

# prior
function _fixed_effects(prior::Distribution)
    quote
        β ~ filldist($prior, npredictors)
    end
end

# prior
function _random_effects(prior::Distribution, prior_fixef::Distribution, model_ranef::Vector{RandomEffect})
    body = Expr(:block)

    # Loop over ranef
    for (i, ranef) in enumerate(model_ranef)
        ranef_name = string(ranef.variable)

        #Name parameters
        variance_ranef = Symbol("σ_ranef",ranef_name)
        intercept_ranef = Symbol("β_z_",ranef_name)
        ranef_matrix_raw = Symbol("r_z_",ranef_name)
        ranef_matrix = Symbol("ranef_z_",ranef_name)
        R_ranef = Symbol("R_z_",ranef_name)

        #Build varying slopes prior
        n_predictors = 0
        if ranef.has_intercept
            n_predictors += 1
        end
        if ranef.has_fixed_effects
            n_predictors += size(ranef.predictors, 2)
        end

        push!(body_args, quote
            $variance_ranef ~ filldist($prior, $n_predictors)
            $intercept_ranef ~ filldist($prior_fixef, $n_predictors) #ideally with 0 included?
        end)

        if n_predictors > 1
            push!(body_args, quote
                $R_ranef ~ LKJ($n_predictors, 2.0)
                $ranef_matrix_raw ~ filldist(MvNormal(zeros(n_predictors), Symmetric(diagm($variance_ranef)* $R_ranef * diagm($variance_ranef))), n_groups)
                $ranef_matrix = $ranef_matrix_raw .+ $intercept_ranef
            end)
        else
            push!(body_args, quote
                $ranef_matrix_raw ~ filldist(Normal(), n_predictors)
                $ranef_matrix = ranef_matrix_raw .* $variance_ranef .+ intercept_ranef
            end)
        end
    end
    return body
end

# prior
function _auxiliary_parameter(prior::Distribution, family::Type{<:Distribution})
    if family == Normal
        quote
            σ ~ $prior
        end
    elseif family == TDist
        quote 
            σ ~ Exponential(1)
            ν ~ $prior
        end
    elseif family == NegativeBinomial
        quote 
            ϕ ~ $prior
            ϕ_inv = 1 / ϕ
        end
    elseif family ∈ [Bernoulli, Poisson]
        return :() #empty quote, no code
    end
end

# linear model
function _linear_model(has_intercept::Bool, has_fixed_effects::Bool, has_random_effects::Bool, model_ranef::Union{Vector{RandomEffect}, Nothing})

    # Get terms we need
    terms = []
    if has_intercept 
        push!(terms, :α)
    end
    if has_fixed_effects 
        push!(terms, :(X_scaled * β))
    end
    if has_random_effects 
        for ranef in model_ranef
            ranef_name = string(ranef.variable)
            ranef_matrix = Symbol("ranef_z_", ranef_name)
            intercept_slopes = Symbol("β_z_", ranef_name)
            predictors_scaled = Symbol("Xscaled_z_", ranef_name)
            if ranef.has_intercept & ranef.has_fixed_effects
                push!(terms, :($ranef_matrix[1, group_idx]))
                push!(terms, :(($intercept_slopes .+ $ranef_matrix[2:end, group_idx])) .* $predictors_scaled)
            elseif ranef.has_fixed_effects
                push!(terms, :(($intercept_slopes .+ $ranef_matrix[:,group_idx])) .* $predictors_scaled)
            elseif ranef.has_intercept
                push!(terms, :($ranef_matrix[group_idx]))
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
            Turing.@addlogprob! logpdf(MvNormal(μ, σ^2 * I), y_scaled)
        end
    elseif family == TDist
        quote
            Turing.@addlogprob! logpdf(arraydist((μ) .+ σ .* TDist.(ν)), y_scaled)
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
                Turing.@addlogprob! weights[n] * logpdf(Normal(μ[n], σ), y_scaled[n])
            end
        end
    elseif family == TDist
        quote
            for n in 1:nobs
                Turing.@addlogprob! weights[n] * logpdf(μ[n] + σ * TDist(ν), y_scaled[n])
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

# data standardisation
function _standardise_data(family::Type, has_fixed_effects::Bool, has_random_effects::Bool, model_ranef::Union{Vector{RandomEffect}, Nothing})
    # Empty quote
    body = Expr(:block)

    # Do X if there are fixed effects
    if has_fixed_effects
        push!(body.args, quote
            X_means = mean(X, dims=1)[:]
            X_stds = std(X, dims=1)[:]
            X_scaled = (X .- X_means') ./ X_stds'
        end)
    end

    if has_random_effects
        for (i, ranef) in enumerate(model_ranef)
            if ranef.has_fixed_effects
                ranef_name = string(ranef.variable)
                predictors_mn = Symbol("Xmn_z_", ranef_name)
                predictors_sd = Symbol("Xstd_z_", ranef_name)
                predictors_scaled = Symbol("Xscaled_z_", ranef_name)
                push!(body.args, quote
                    $predictors_mn = mean(X_ranef[$i], dims=1)[:]
                    $predictors_sd = std(X_ranef[$i], dims=1)[:]
                    $predictors_scaled = (X_ranef[$i] .- $predictors_mn') ./ $predictors_sd'
                end)
            end
        end
    end

    # Do y if model family requires it
    if family ∈ [Normal, TDist]
        push!(body.args, quote
            y_mean = mean(y)
            y_std = std(y)
            y_scaled = (y .- y_mean) / y_std
        end)
    end

    return body
end

# parameter scaling
function _generated_quantities(family::Type{<:Distribution}, has_fixed_effects::Bool, has_intercept::Bool, has_random_effects::Bool, model_ranef::Union{Vector{RandomEffect}, Nothing})
    # Empty quote
    body = Expr(:block) 
    return_list = Expr[]

    # Calculations and objects to return
    if has_fixed_effects
        if family ∈ [Bernoulli, Poisson, NegativeBinomial] # Not standardised
            push!(body.args, :(β_original = β ./ X_stds))
        else
            push!(body.args, :(β_original = (y_std ./ X_stds) .* β))
        end
        push!(return_list, :(β=β_original))
    end
    if has_intercept
        if family ∈ [Bernoulli, Poisson, NegativeBinomial] # Not standardised
            push!(body.args, :(α_original = α - dot(X_means, β_original)))
        else
            push!(body.args, :(α_original = y_mean - dot(X_means, β_original) + y_std * α))
        end
        push!(return_list, :(α=α_original))
    end
    if family ∈ [Normal, TDist]
        push!(body.args, :(σ_original = y_std * σ))
        push!(return_list, :(σ=σ_original))
    end
    if family == TDist
        push!(return_list, :(ν=ν))
    end
    if family == NegativeBinomial
        push!(return_list, :(ϕ=ϕ))
    end

    if has_random_effects
        for (i, ranef) in enumerate(model_ranef)
            predictors_mn = Symbol("Xmn_z_", ranef_name)
            predictors_sd = Symbol("Xstd_z_", ranef_name)
            predictors_scaled = Symbol("Xscaled_z_", ranef_name)
            ranef_matrix = Symbol("ranef_z_", ranef_name)
            intercept_slopes = Symbol("β_z_", ranef_name)
            predictors_scaled = Symbol("Xscaled_z_", ranef_name)
            predictors_orig = Symbol("β_orig_z_", ranef_name)
            if ranef.has_fixed_effects & ranef.has_intercept
                if family ∈ [Bernoulli, Poisson, NegativeBinomial] # Not standardised
                    push!(body.args, :($predictors_orig = ($intercept_slopes .+ $ranef_matrix[2:end, :]) ./ $predictors_sd))
                else
                    push!(body.args, :($predictors_orig = (y_std ./ $predictors_sd) .* ($intercept_slopes .+ $ranef_matrix[2:end, :])))
                end
                push!(return_list, :($intercept_slopes=$predictors_orig)) #cheeky double name
            if ranef.has_fixed_effects
                if family ∈ [Bernoulli, Poisson, NegativeBinomial] # Not standardised
                    push!(body.args, :(β_original = β ./ $predictors_sd))
                else
                    push!(body.args, :(β_original = (y_std ./ $predictors_sd) .* β))
                end
            if ranef.has_intercept
                if family ∈ [Bernoulli, Poisson, NegativeBinomial] # Not standardised
            push!(body.args, :(α_original = y_mean - dot(X_means, β_original) + y_std * α))
            push!(return_list, :(α=α_original))
            push!(body.args, :(β_original = β ./ X_stds))
            push!(body.args, :(β_original = (y_std ./ X_stds) .* β))
            push!(return_list, :(β=β_original))
        end
    end

    # add return line to body as a named tuple
    return_tuple = Expr(:tuple, return_list...)
    return_stmt = Expr(:return, return_tuple)
    push!(body.args, return_stmt)

    return body
end

#### Main function to assemble the model code
function build_model_body(family::Type{<:Distribution}, model_info::ModelInfo, model_ranef::Vector{RandomEffect}, prior::RegressionPrior)

    # Empty quote
    body = Expr(:block) 

    # Data transformation
    push!(body.args, _standardise_data(family, model_info.has_fixed_effects, model_info.has_random_effects, model_ranef))

    # Prior
    if model_info.has_intercept
        push!(body.args, _intercept(prior.intercept))
    end
    if model_info.has_fixed_effects
        push!(body.args, _fixed_effects(prior.fixed_effects))
    end
    if model_info.has_random_effects
        push!(body.args, _random_effects(prior.random_effects, prior.fixed_effects, model_ranef))
    end

    if family ∉ [Bernoulli, Poisson] #Bernoulli and Poisson have no auxiliary parameter
        push!(body.args, _auxiliary_parameter(prior.auxiliary, family))
    end

    # Linear Model
    push!(body.args, _linear_model(model_info.has_intercept, model_info.has_fixed_effects, model_info.has_random_effects, model_ranef))

    # Likelihood
    if model_info.weighted
        push!(body.args, _weighted_likelihood(family))
    else
        push!(body.args, _likelihood(family))
    end

    # Generated Quantitites
    push!(body.args, _generated_quantities(family, model_info.has_fixed_effects, model_info.has_intercept, model_info.has_random_effects))


    return body
end

#### Wrapper function for the above, to handle some additional logic
function construct_model(family::Type{<:Distribution}, model_info::ModelInfo, model_ranef::Vector{RandomEffect}, prior::RegressionPrior, show_code::Bool=false)

    #handle logic here
    body = build_model_body(family, model_info, model_ranef, prior)

    # argument names
    args = [:y, :X]
    model_info.has_random_effects && push!(args, [:n_gr, :group_idx, :group_predictors])
    model_info.weighted && push!(args, :weights)
    
    # build model code
    model_code = quote
        @model function turing_regression($(args...))
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