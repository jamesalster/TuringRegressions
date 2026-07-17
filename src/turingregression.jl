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
mutable struct TuringRegression{T<:Distribution} <: RegressionModel
    formula::FormulaTerm
    model::Function
    prior::RegressionPrior
    link::Function
    modeldata::ModelData
    tf::Transform
    modelcode::Expr
    samples::Union{Nothing,VNChain}
    parameters::Union{Nothing,DimStack}
end

# Derived flags — mirror the ModelData accessors (formula_handlers.jl) for the fitted TR.
has_intercept(TR::TuringRegression) = has_intercept(TR.modeldata)
has_fixed_effects(TR::TuringRegression) = has_fixed_effects(TR.modeldata)
has_random_effects(TR::TuringRegression) = has_random_effects(TR.modeldata)
is_weighted(TR::TuringRegression) = is_weighted(TR.modeldata)

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
    family::Type{<:Distribution};
    priors::RegressionPrior=default_prior(family),
    weights::Union{Nothing, Vector{Float64}}=nothing,
    show_code::Bool=false)

    if family ∉ [Normal, TDist, Bernoulli, Poisson, NegativeBinomial]
        error("Family: $(string(family)) not supported.")
    end

    # Get data arrays. `modeldata.f` is `formula` with schema/contrasts baked in —
    # stored on TR and reused by posterior_predict(TR, new_data::DataFrame) so grouping levels /
    # categorical contrasts are never re-derived from (possibly small/partial) new data.
    modeldata = extract_model_data(formula, data, weights)
    _, tf = standardise(modeldata, family)

    model_obj, model_code = cached_construct_model(family, modeldata, show_code)

    return TuringRegression{family}(
        modeldata.f,
        model_obj,
        priors,
        get_link(family),
        modeldata,
        tf,
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
    formula = term(:y) ~ sum(term.(X_names))
    return turing_glm(formula, table, T; kwargs...)
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
    if has_random_effects(TR)
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
    elseif T == NegativeBinomial
        print(io, normal_style, "  Auxiliary (1/ϕ): ")
        println(io, normal_style, clean_prior_string(string(pr.auxiliary)))
    end

    # Observations
    print(io, label_style, "Observations: ")
    println(io, normal_style, size(TR.modeldata.predictors.X, 1))

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

Run MCMC sampling to fit the model. Updates the model in-place. Kwargs are passed to Turing's `sample()`.

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
# Slim helper (§5.3): derives the grouping arrays (n_groups/group_idx/group_predictors)
# from ModelData.Z once, then calls the model with its unpacked-argument signature.
# Shared with psis_loo (comparison.jl), which needs the same conditioned model.
function _build_model_with_data(TR::TuringRegression)
    md = apply_transform(TR.tf, TR.modeldata)
    Z = md.Z
    n_groups = [length(re.levels) for re in Z]
    group_idx = isempty(Z) ? Matrix{Int}(undef, length(md.y), 0) : reduce(hcat, (re.level_index for re in Z))
    group_predictors = [re.predictors.X for re in Z]
    weights = something(md.weights, ones(length(md.y)))
    pr = TR.prior
    return TR.model(md.y, md.predictors.X, n_groups, group_idx, group_predictors, weights,
        pr.intercept, pr.fixed_effects, pr.random_effects, pr.auxiliary)
end

function fit!(
    TR::TuringRegression{T};
    sampler=NUTS(),
    parallel=MCMCThreads(),
    N=2000,
    nchains=4,
    quiet=true,
    kwargs...,
) where {T}
    model_with_data = _build_model_with_data(TR)

    if quiet
        TR.samples = @suppress sample(model_with_data, sampler, parallel, N, nchains; chain_type=VNChain, kwargs...)
    else
        TR.samples = sample(model_with_data, sampler, parallel, N, nchains; chain_type=VNChain, kwargs...)
    end

    # Raw standardised-scale sampled params straight off the chain, stacked into
    # (iter,chain,param) with vector/matrix VarNames split into indices (`β[1]`,
    # `L.L[2,1]`, ...) — R10, no model return statement needed. `reshape_params` splits
    # this flat array into named layers (still standardised scale); `unstandardise`
    # then back-transforms those layers to the original data scale (V10).
    raw = DimArray(TR.samples)
    std_params = reshape_params(raw, TR.modeldata, T)
    TR.parameters = unstandardise(std_params, TR.tf, TR.modeldata, T)
    return TR
end

"""
    default_prior(TR::TuringRegression{T})

Convenience method that extracts the distribution family from a model.
"""
default_prior(TR::TuringRegression{T}) where {T} = default_prior(T)