"""
Flags describing model structure.

Tracks which components are present: intercept, fixed effects, 
random effects, and whether sampling weights are used.
"""
struct ModelInfo
    has_intercept::Bool
    has_fixed_effects::Bool
    has_random_effects::Bool
    weighted::Bool
end

"""
A Bayesian regression model fitted with Turing.jl.

Stores the formula, data, priors, and MCMC samples. The type parameter 
`T` is the response distribution (Normal, Bernoulli, TDist, etc).

# Fields
- `formula`: The regression formula
- `model`: Compiled Turing model function
- `prior`: Prior distributions for parameters
- `samples`: MCMC chains (nothing until `fit!` is called)
- `parameters`: Standardized parameter draws
"""
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
    modelcode::Expr
    samples::Union{Nothing,Chains}
    parameters::Union{Nothing,DimArray}
end

"""
    turing_glm(formula, data, family; priors, weights, show_code)

Fit a Bayesian regression model.

Data is automatically standardized. Supply priors scaled for 
standardized predictors (mean=0, sd=1).

# Arguments
- `formula`: Regression formula (e.g., `@formula(y ~ x1 + x2)`)
- `data`: DataFrame with response and predictors
- `family`: Response distribution (Normal, Bernoulli, TDist, etc.)
- `priors`: Prior distributions (defaults provided if omitted)
- `weights`: Optional sampling weights
- `show_code`: Print the generated Turing model code

# Example
```julia
model = turing_glm(@formula(mpg ~ hp + wt), mtcars, Normal)
fit!(model)
```
"""
function turing_glm(formula::FormulaTerm, 
    data::DataFrame, 
    family::Type{<:Distribution}, 
    priors::RegressionPrior=default_prior(family), 
    weights::Union{Nothing, Vector{Float64}}=nothing,
    show_code::Bool=false) 

    if family ∉ [Normal, TDist, Bernoulli, Poisson, NegativeBinomial]
        error("Family: $(string(family)) not supported.")
    end

    # Get data arrays
    y = data_response(formula, data)
    X = data_fixed_effects(formula, data)

    # Make model info
    model_info = ModelInfo(
        has_intercept(formula),
        size(X, 2) > 0,
        has_ranef(formula),
        !isnothing(weights)
    )

    model_obj, model_code = construct_model(family, model_info, priors, show_code)

    return TuringRegression{family}(
        formula,
        model_obj,
        priors,
        get_link(family),
        y,
        X,
        nothing,
        weights,
        get_fixef_names(formula, data),
        nothing,
        model_info,
        model_code,
        nothing,
        nothing
    )
end

"""
    turing_glm(y, X, family; names, kwargs...)

Fit a model using raw arrays instead of a formula.

# Arguments
- `y`: Response vector
- `X`: Predictor matrix
- `family`: Response distribution
- `names`: Variable names (auto-generated if omitted)

# Example
```julia
model = turing_glm(y, X, Normal, names=[:age, :income])
```
"""
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
    formula = "y ~ " * join([string.(term) for term in X_names], " + ")
    formula_obj = eval(Meta.parse("@formula($formula)"))
    return turing_glm(formula_obj, table, T; kwargs...)
end

"""
    show(io, TR::TuringRegression)

Print a summary of the model: family, formula, priors, and sample status.
"""
function Base.show(io::IO, TR::TuringRegression{T}; warnings=true) where {T}
    header_style = crayon"bold underline"
    label_style = crayon"bold !underline"
    normal_style = crayon"reset"

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

    if T == TDist
        print(io, normal_style, "  Error Variance: ")
        println(io, normal_style, "Exponential(θ=1.0)")
        print(io, normal_style, "  Auxiliary (ν): ")
        println(io, normal_style, clean_prior_string(string(pr.auxiliary)))
    elseif T == Normal
        print(io, normal_style, "  Auxiliary (σ): ")
        println(io, normal_style, clean_prior_string(string(pr.auxiliary)))
    elseif T == NegativeBinomial2
        print(io, normal_style, "  Auxiliary (1/ϕ): ")
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
    fit!(TR::TuringRegression; sampler, parallel, N, nchains, quiet, kwargs...)

Run MCMC sampling to fit the model. Updates the model in-place.

# Arguments
- `sampler`: MCMC algorithm (default: NUTS())
- `parallel`: How to parallelize chains (default: MCMCThreads())
- `N`: Samples per chain (default: 2000)
- `nchains`: Number of chains (default: 4)
- `quiet`: Hide sampling progress (default: true)

# Example
```julia
fit!(model, N=1000, nchains=2)
```
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

    # Recover standardised parameters from generated quantities - a bit of help from claude
    gq = generated_quantities(model_with_data, TR.samples)
    param_names = collect(keys(first(gq)))
    param_names = :α ∈ param_names ? [:α; filter(!=(:α), param_names)] : param_names

    
    # Extract all parameters in one pass, thanks to claude for help
    param_dict = Dict(p => [gq[i, j][p] for i in axes(gq, 1), j in axes(gq, 2)] 
                    for p in param_names)

    arrays = []
    labels = Symbol[]
    for param in param_names
        if param === :β
            arr = stack(param_dict[param])  # (params, draws, chains)
            push!(arrays, arr)
            append!(labels, [Symbol("β[$i]") for i in 1:size(arr, 1)])
        else
            arr = param_dict[param]  # (draws, chains)
            push!(arrays, reshape(arr, 1, size(arr)...))  # (params, draws, chains)
            push!(labels, param)
        end
    end

    TR.parameters = DimArray(vcat(arrays...), (Dim{:param}(labels), Dim{:draw}, Dim{:chain}))

    return TR
end