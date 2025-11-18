
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
function _random_effects(prior::Distribution)
    quote
        τ ~ $prior
        zⱼ ~ filldist(Normal(), n_gr)
    end
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
    elseif family ∈ [Bernoulli, Poisson]
        return :() #empty quote, no code
    end
end

# linear model
function _linear_model(has_intercept::Bool, has_fixed_effects::Bool, has_random_effects::Bool)

    # Get terms we need
    terms = []
    if has_intercept 
        push!(terms, :α)
    end
    if has_fixed_effects 
        push!(terms, :(X_scaled * β))
    end
    #if has_random_effects 
    #    push!(terms, :(τ .* getindex.((zⱼ,), idxs)))
    #end

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
    end
end

# data standardisation
function _standardise_data(family::Type, has_fixed_effects::Bool)
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
function _generated_quantities(family::Type{<:Distribution}, has_fixed_effects::Bool, has_intercept::Bool)
    # Empty quote
    body = Expr(:block) 
    return_list = Expr[]

    # Calculations and objects to return
    if has_fixed_effects
        if family ∈ [Bernoulli, Poisson] # Not standardised
            push!(body.args, :(β_original = β ./ X_stds))
        else
            push!(body.args, :(β_original = (y_std ./ X_stds) .* β))
        end
        push!(return_list, :(β=β_original))
    end
    if has_intercept
        if family ∈ [Bernoulli, Poisson] # Not standardised
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

    # add return line to body as a named tuple
    return_tuple = Expr(:tuple, return_list...)
    return_stmt = Expr(:return, return_tuple)
    push!(body.args, return_stmt)

    return body
end

#### Main function to assemble the model code
function build_model_body(family::Type{<:Distribution}, model_info::ModelInfo, prior::RegressionPrior)

    # Empty quote
    body = Expr(:block) 

    # Data transformation
    push!(body.args, _standardise_data(family, model_info.has_fixed_effects))

    # Prior
    if model_info.has_intercept
        push!(body.args, _intercept(prior.intercept))
    end
    if model_info.has_fixed_effects
        push!(body.args, _fixed_effects(prior.fixed_effects))
    end
    if model_info.has_random_effects
        push!(body.args, _random_effects(prior.random_effects))
    end

    if family ∉ [Bernoulli, Poisson] #Bernoulli and Poisson have no auxiliary parameter
        push!(body.args, _auxiliary_parameter(prior.auxiliary, family))
    end

    # Linear Model
    push!(body.args, _linear_model(model_info.has_intercept, model_info.has_fixed_effects, model_info.has_random_effects))

    # Likelihood
    if model_info.weighted
        push!(body.args, _weighted_likelihood(family))
    else
        push!(body.args, _likelihood(family))
    end

    # Generated Quantitites
    push!(body.args, _generated_quantities(family, model_info.has_fixed_effects, model_info.has_intercept))

    return body
end

#### Wrapper function for the above, to handle some additional logic
function construct_model(family::Type{<:Distribution}, model_info::ModelInfo, prior::RegressionPrior, show_code::Bool=false)

    #handle logic here
    body = build_model_body(family, model_info, prior)

    # argument names
    args = [:y, :X]
    model_info.has_random_effects && push!(args, :z)
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