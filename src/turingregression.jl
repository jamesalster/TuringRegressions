
struct ModelInfo
    has_intercept::Bool
    has_fixed_effects::Bool
    has_random_effects::Bool
    weighted::Bool
end

mutable struct TuringRegression{T<:Distribution}
    formula::FormulaTerm
    model::Function
    prior::RegressionPrior
    link::Function
    y::AbstractVector
    X::AbstractMatrix
    z::Union{Nothing,AbstractMatrix}
    weights::Union{Nothing,Vector{Float64}}
    X_names::Union{Nothing,Vector{String}}
    z_names::Union{Nothing,Vector{String}}
    modelinfo::ModelInfo
    samples::Union{Nothing,Chains}
    parameters::Union{Nothing,DimArray}
end

## Main model fit function
function turing_glm(formula::FormulaTerm, 
    data, 
    family::Type{<:Distribution}, 
    priors::RegressionPrior=default_prior(family), 
    weights::Union{Nothing, Vector{Float64}}=nothing,
    show_code::Bool=false) 

    # Get data arrays
    y = data_response(formula, data)
    X = data_fixed_effects(formula, data)
    #z = data_random_effects(formula, data)

    # Make model info
    model_info = ModelInfo(
        has_intercept(formula),
        size(X, 2) > 0,
        has_ranef(formula),
        !isnothing(weights)
    )

    model_obj = construct_model(family, model_info, priors, show_code)

    return TuringRegression{family}(
        formula,
        model_obj,
        priors,
        get_link(family),
        y,
        X,
        nothing,
        #z,
        weights,
        get_fixef_names(formula, data),
        nothing,
        model_info,
        nothing,
        nothing
    )
end

## Method for y and X
function turing_glm(
    y::AbstractVector,
    X::AbstractArray,
    ::Type{T};
    names::Vector{Symbol}=Symbol[],
    kwargs...,
) where {T<:UnivariateDistribution}
    if isempty(names)
        X_names = ntuple(i -> Symbol("X$i"), size(X, 2))
    else
        X_names = ntuple(i -> Symbol(names[i]), length(names))
    end
    table = (; NamedTuple{X_names}(eachcol(X))..., NamedTuple{(:y,)}([y])...)
    #TODO improve
    formula = "y ~ " * join([string.(term) for term in X_names], " + ")
    formula_obj = eval(Meta.parse("@formula($formula)"))
    return turing_glm(formula_obj, table, T; kwargs...)
end


function Base.show(io::IO, TR::TuringRegression{T}; warnings=true) where {T}
    # Define styles once at the top
    header_style = crayon"bold underline"
    label_style = crayon"bold !underline"
    normal_style = crayon"reset"  # or just use no crayon

    println(io, header_style, "TuringRegression Model")

    # Family
    print(io, label_style, "Family: ")
    family_string = "$T (link: $(string(TR.link)))"
    println(io, normal_style, family_string)

    # Formula  
    print(io, label_style, "Formula: ")
    println(io, normal_style, string(TR.formula))

    # Prior
    println(io, label_style, "Prior:")
    pr = TR.prior
    print(io, normal_style, "  Intercept: ")
    println(io, normal_style, clean_prior_string(string(pr.intercept)))
    print(io, normal_style, "  Fixed Effects: ")
    println(io, normal_style, clean_prior_string(string(pr.fixed_effects)))
    if TR.modelinfo.has_random_effects
        print(io, normal_style, "  Random Effects: ")
        println(io, normal_style, clean_prior_string(string(pr.random_effects)))
    end

    if T ∈ [Normal, TDist]
        print(io, normal_style, "  Auxiliary: ")
        println(io, normal_style, clean_prior_string(string(pr.auxiliary)))
    end

    # Observations
    print(io, label_style, "Observations: ")
    println(io, normal_style, size(TR.X, 1))

    # Samples
    print(io, label_style, "Samples: ")
    if isnothing(TR.samples)
        println(io, normal_style, "empty")
    else
        sz = size(TR.samples)
        println(io, normal_style, "$(sz[1] * sz[3]) samples across $(sz[3]) chains")
    end

    if warnings
        println(io)
        model_warnings(TR)
    end
end


#### Methods ####

"""
    fit!(TR::TuringRegression; sampler=NUTS(), parallel=MCMCThreads(), N=2000, nchains=4, quiet=true, kwargs...)

Fit the model using MCMC sampling. Updates the model in-place with results.
    Kwargs are passed to Turing's `sample()` function.

# Arguments
- `sampler`: MCMC sampler (default: NUTS())
- `parallel`: Parallelization method (default: MCMCThreads())
- `N`: Number of samples per chain
- `nchains`: Number of parallel chains
- `quiet`: Suppress sampling output
"""
function fit!(
    TR::TuringRegression;
    sampler=NUTS(),
    parallel=MCMCThreads(),
    N=2000,
    nchains=4,
    quiet=true,
    kwargs...,
)
    if TR.modelinfo.has_random_effects & TR.modelinfo.weighted
        model_with_data = TR.model(TR.y, TR.X, TR.z, TR.weights)
    elseif TR.modelinfo.has_random_effects 
        model_with_data = TR.model(TR.y, TR.X, TR.z)
    elseif TR.modelinfo.weighted 
        model_with_data = TR.model(TR.y, TR.X, TR.weights)
    else
        model_with_data = TR.model(TR.y, TR.X)
    end

    if quiet
        TR.samples = @suppress sample(model_with_data, sampler, parallel, N, nchains; kwargs...)
    else
        TR.samples = sample(model_with_data, sampler, parallel, N, nchains; kwargs...)
    end

    # Recover the standardised parametes from generated quantities
    # bit messy
    gq = generated_quantities(model_with_data, TR.samples)
    param_names = collect(keys(first(gq)))
    arrays = Vector{Array{Float64, 3}}()
    param_keys = Vector{Symbol}()
    for param in param_names
        if param === :β
            gq_array = stack([x[param] for x in gq])
            push!(arrays, gq_array)
            push!(param_keys, Symbol.(["β[$i]" for i in 1:size(gq_array, 1)])...)
        else  
            push!(arrays, stack([[x[param]] for x in gq]))
            push!(param_keys, param)
        end
    end
    param_array = vcat(arrays...) #along the first dimension
    TR.parameters = DimArray(param_array, (Dim{:param}(param_keys), Dim{:draw}, Dim{:chain}))
end

