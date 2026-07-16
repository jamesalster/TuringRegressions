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
mutable struct TuringRegression{T<:Distribution} <: RegressionModel
    formula::FormulaTerm
    model::Function
    prior::RegressionPrior
    link::Function
    y::AbstractVector
    X::AbstractMatrix
    z::Union{Nothing,Vector{RandomEffect}}
    weights::Union{Nothing,Vector{Float64}}
    X_names::Union{Nothing,Vector{String}}
    modelinfo::ModelInfo
    modelcode::Expr
    samples::Union{Nothing,Chains}
    parameters::Union{Nothing,DimStack}
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
    family::Type{<:Distribution};
    priors::RegressionPrior=default_prior(family),
    weights::Union{Nothing, Vector{Float64}}=nothing,
    show_code::Bool=false)

    if family ∉ [Normal, TDist, Bernoulli, Poisson, NegativeBinomial]
        error("Family: $(string(family)) not supported.")
    end

    # Get data arrays. `schema_formula` is `formula` with schema/contrasts baked in —
    # stored on TR and reused by posterior_predict(TR, new_data::DataFrame) so grouping levels /
    # categorical contrasts are never re-derived from (possibly small/partial) new data.
    y, X, Z, formula_with_schema = extract_model_data(formula, data)

    # Make model info
    model_info = ModelInfo(
        has_intercept(formula),
        size(X, 2) > 0,
        !isnothing(Z),
        !isnothing(weights)
    )

    model_obj, model_code = construct_model(family, model_info, Z, priors, show_code)

    return TuringRegression{family}(
        formula_with_schema,
        model_obj,
        priors,
        get_link(family),
        y,
        X,
        Z,
        weights,
        get_fixef_names(formula, data),
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
    elseif T == NegativeBinomial
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
    _build_model_with_data(TR::TuringRegression)

Build the DynamicPPL model conditioned on TR's data (y, X, random-effect
grouping structures, weights as applicable). Return the conditioned model.
"""
function _build_model_with_data(TR::TuringRegression)
    # Prepare random effect data structures
    if TR.modelinfo.has_random_effects
        n_gr = zeros(Int, length(TR.z))
        group_idx = zeros(Int, size(first(TR.z).predictors, 1), length(TR.z))
        group_predictors = Vector{Matrix{Float64}}(undef, length(TR.z))

        for (i, ranef) in enumerate(TR.z)
            n_gr[i] = length(ranef.levels)
            group_idx[:,i] = ranef.level_index
            group_predictors[i] = ranef.predictors #this is an empty matrix if no fixed effects for the ranef
        end
    end

    # Call model function
    if TR.modelinfo.has_random_effects & TR.modelinfo.weighted
        return TR.model(TR.y, TR.X, n_gr, group_idx, group_predictors, TR.weights)
    elseif TR.modelinfo.has_random_effects
        return TR.model(TR.y, TR.X, n_gr, group_idx, group_predictors)
    elseif TR.modelinfo.weighted
        return TR.model(TR.y, TR.X, TR.weights)
    else
        return TR.model(TR.y, TR.X)
    end
end

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
function fit!(
    TR::TuringRegression;
    sampler=NUTS(),
    parallel=MCMCThreads(),
    N=2000,
    nchains=4,
    quiet=true,
    kwargs...,
)
    model_with_data = _build_model_with_data(TR)

    # Sample. chain_type forced to MCMCChains.Chains: newer Turing defaults to
    # FlexiChains.FlexiChain, whose internals (._data/._metadata/._structures) are
    # incompatible with every .samples access site elsewhere in this package
    # (name_map, indexing, etc). Forcing Chains keeps the rest of the package working
    # without a rewrite.
    if quiet
        TR.samples = @suppress sample(model_with_data, sampler, parallel, N, nchains; chain_type=MCMCChains.Chains, kwargs...)
    else
        TR.samples = sample(model_with_data, sampler, parallel, N, nchains; chain_type=MCMCChains.Chains, kwargs...)
    end

    # Recover standardised parameters from generated quantities, thanks to claude
    gq = generated_quantities(model_with_data, TR.samples)
    param_names = filter(!=(:loglik), collect(keys(first(gq))))
    param_names = :α ∈ param_names ? [:α; filter(!=(:α), param_names)] : param_names

    # Extract all parameters in one pass
    param_dict = Dict(p => [gq[i, j][p] for i in axes(gq, 1), j in axes(gq, 2)] 
                    for p in param_names)

    draw_dim = Dim{:draw}(axes(gq, 1))
    chain_dim = Dim{:chain}(axes(gq, 2))

    # :fixef layer — α, β, aux params (unchanged content/shape from before T1)
    fixef_arrays = []
    fixef_labels = Symbol[]
    for param in filter(p -> !occursin("_z_", string(p)), param_names)
        if param === :β
            arr = stack(param_dict[param])
            push!(fixef_arrays, arr)
            append!(fixef_labels, Symbol.(TR.X_names))
        else
            push!(fixef_arrays, reshape(param_dict[param], 1, size(param_dict[param])...))
            push!(fixef_labels, param)
        end
    end
    fixef_arr = DimArray(vcat(fixef_arrays...), (Dim{:fixef}(fixef_labels), draw_dim, chain_dim))

    layers = Dict{Symbol,Any}(:fixef => fixef_arr)

    # One layer per random-effect grouping term
    for re in (TR.modelinfo.has_random_effects ? TR.z : RandomEffect[])
        group = re.variable
        intercept_sym = Symbol("α_z_", group)
        beta_sym = Symbol("β_z_", group)
        sd_sym = Symbol("σ_z_", group)
        R_sym = Symbol("R_z_", group)
        offset_sym = Symbol("offset_z_", group)

        effect_names = Symbol[]
        re.has_intercept && push!(effect_names, :Intercept)
        re.has_fixed_effects && append!(effect_names, Symbol.(re.predictor_names))

        # Main layer: (effect, group, draw, chain)
        if re.has_intercept & re.has_fixed_effects
            combined = [vcat(reshape(param_dict[intercept_sym][i, j], 1, :), param_dict[beta_sym][i, j])
                        for i in axes(gq, 1), j in axes(gq, 2)]
            main_arr = stack(combined)
        elseif re.has_fixed_effects
            main_arr = stack(param_dict[beta_sym])
        else # re.has_intercept only
            combined = [reshape(param_dict[intercept_sym][i, j], 1, :) for i in axes(gq, 1), j in axes(gq, 2)]
            main_arr = stack(combined)
        end
        layers[group] = DimArray(main_arr, (Dim{:effect}(effect_names), Dim{:group}(re.levels), draw_dim, chain_dim))

        # :<group>_sd layer — back-transformed group-level SDs, one per effect
        sd_arr = stack(param_dict[sd_sym])
        layers[Symbol(group, "_sd")] = DimArray(sd_arr, (Dim{:effect}(effect_names), draw_dim, chain_dim))

        # :<group>_corr layer — only when correlated (full L*L' per draw)
        if R_sym ∈ param_names
            corr_arr = stack(param_dict[R_sym])
            layers[Symbol(group, "_corr")] = DimArray(
                corr_arr, (Dim{:effect}(effect_names), Dim{:effect2}(effect_names), draw_dim, chain_dim)
            )
        end

        # :<group>_offset layer — only for slope-only-no-intercept terms, hidden from user-facing accessors
        if offset_sym ∈ param_names
            offset_arr = stack(param_dict[offset_sym])
            layers[Symbol(group, "_offset")] = DimArray(offset_arr, (Dim{:group}(re.levels), draw_dim, chain_dim))
        end
    end

    # Internals
    internals_names = TR.samples.name_map[:internals]
    layers[:internals] = DimArray(permutedims(TR.samples[internals_names].value, (2, 1, 3)),
        (Dim{:internal}(internals_names), Dim{:draw}, Dim{:chain}))

    TR.parameters = DimStack(NamedTuple(layers))
    return TR
end

"""
    default_prior(TR::TuringRegression{T})

Convenience method that extracts the distribution family from a model.
"""
default_prior(TR::TuringRegression{T}) where {T} = default_prior(T)