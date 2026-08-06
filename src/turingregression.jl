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
    turing_glm(formula, data, family; priors, weights)

Fit a Bayesian regression model.

Data is automatically standardized internally (mean 0, sd 1) for sampling
efficiency. **Priors are specified on this STANDARDISED scale, not the
original data's units** — `Normal(0, 2)` for `fixed_effects` means 2 std
devs of the (standardised) predictor, not 2 units of the raw data. Inspect
the priors in force with `TR.prior`, `prior_summary(TR)`, or `show(TR)`.
Inspect the generated Turing model code with `modelcode(TR)`.

# Arguments
- `formula`: Regression formula (e.g., `@formula(y ~ x1 + x2)`)
- `data`: DataFrame with response and predictors
- `family`: Response distribution (Normal, Bernoulli, TDist, etc.)
- `priors`: Prior distributions, standardised scale (defaults from `default_prior(family)` if omitted)
- `weights`: Optional sampling weights

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
    weights::Union{Nothing, Vector{Float64}}=nothing)

    if family ∉ [Normal, TDist, Bernoulli, Poisson, NegativeBinomial]
        error("Family: $(string(family)) not supported.")
    end

    # Get data arrays. `modeldata.f` is `formula` with schema/contrasts baked in —
    # stored on TR and reused by posterior_predict(TR, new_data::DataFrame) so grouping levels /
    # categorical contrasts are never re-derived from (possibly small/partial) new data.
    modeldata = extract_model_data(formula, data, weights)
    _, tf = standardise(modeldata, family)

    model_obj, model_code = cached_construct_model(family, modeldata)

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
    df = DataFrame(X, collect(X_names))
    df.y = y
    formula = term(:y) ~ sum(term.(X_names))
    return turing_glm(formula, df, T; kwargs...)
end

# Shared by Base.show(io,TR) and prior_summary(TR) — one place that knows how to
# print a RegressionPrior against a given family (label_style/normal_style: crayons).
function _print_prior(io::IO, pr::RegressionPrior, family::Type{<:Distribution}, label_style, normal_style; has_ranef::Bool=true)
    println(io, label_style, "Prior (standardised scale):")
    print(io, normal_style, "  Intercept: ")
    println(io, normal_style, clean_prior_string(string(pr.intercept)))
    print(io, normal_style, "  Fixed Effects: ")
    println(io, normal_style, clean_prior_string(string(pr.fixed_effects)))
    if has_ranef
        print(io, normal_style, "  Random Effect Variance: ")
        println(io, normal_style, clean_prior_string(string(pr.random_effect_variance)))
    end

    if family == TDist
        print(io, normal_style, "  Error Variance: ")
        println(io, normal_style, "Exponential(θ=1.0)")
        print(io, normal_style, "  Auxiliary (ν): ")
        println(io, normal_style, clean_prior_string(string(pr.auxiliary)))
    elseif family == Normal
        print(io, normal_style, "  Auxiliary (σ): ")
        println(io, normal_style, clean_prior_string(string(pr.auxiliary)))
    elseif family == NegativeBinomial
        print(io, normal_style, "  Auxiliary (1/ϕ): ")
        println(io, normal_style, clean_prior_string(string(pr.auxiliary)))
    end
end

"""
    prior_summary(TR::TuringRegression)
    prior_summary(io::IO, TR::TuringRegression)

Print `TR.prior` on its own — same block `show(TR)` prints, without formula/samples/
warnings. Priors are always on the STANDARDISED scale (mean 0, sd 1 predictors),
regardless of the original data's units — see `turing_glm` docstring.
"""
function prior_summary(io::IO, TR::TuringRegression{T}) where {T}
    _print_prior(io, TR.prior, T, crayon"bold !underline", crayon"reset"; has_ranef=has_random_effects(TR))
    return nothing
end
prior_summary(TR::TuringRegression) = prior_summary(stdout, TR)

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

    _print_prior(io, TR.prior, T, label_style, normal_style; has_ranef=has_random_effects(TR))

    # Observations
    print(io, label_style, "Observations: ")
    println(io, normal_style, size(TR.modeldata.predictors.X, 1))

    # Samples
    print(io, label_style, "Samples: ")
    if isnothing(TR.samples)
        println(io, normal_style, "empty")
    else
        sz = size(TR.samples)  # FlexiChain: (iter, chain)
        println(io, normal_style, "$(sz[1] * sz[2]) samples across $(sz[2]) chains")
    end

    if warnings
        println(io)
        model_warnings(TR)
    end
end


#### Methods ####

"""
    fit!(TR::TuringRegression; sampler, parallel, samples, nchains, warmup, quiet, kwargs...)

Run MCMC sampling to fit the model. Updates the model in-place. Kwargs are passed to Turing's `sample()`.

Budget: both `samples` (kept draws) and `warmup` (adaptation draws, discarded) are
TOTALS across all chains, split evenly over `nchains`. Division rounds UP (`cld`), so
the actual per-chain count — and thus the total — is never less than requested: e.g.
`samples=101, nchains=4` keeps 26/chain = 104 total. Warmup is sampled IN ADDITION to
`samples` (not carved out of it): each chain runs `warmup/nchains` adaptation draws that
are discarded, then `samples/nchains` kept draws. Warmup is never returned; `warmup=0`
disables it. This is independent from `drop_warmup` in `draws`/`summary`, which trims
already-kept draws at extraction time.

# Arguments
- `sampler`: MCMC algorithm (default: `NUTS()` w/ adtype auto-picked from `TR.modeldata` — `AutoReverseDiff(compile=true)` if random effects present, else `AutoForwardDiff()`; Pass `sampler=NUTS(;adtype=...)` to override.)
- `parallel`: How to parallelize chains (default: MCMCThreads())
- `samples`: Total kept draws across all chains, split over `nchains`, rounded up (default: 2000)
- `nchains`: Number of chains (default: 4)
- `warmup`: Total adaptation draws across all chains (IN ADDITION to `samples`), split over `nchains`, rounded up, discarded (default: equal to `samples`)
- `quiet`: Suppress all sampler output — hides both the progress bar and any warnings (default: true). Set `false` for a live progress bar (a single aggregate bar across threaded chains); override with `progress=false` via kwargs.

# Example
```julia
fit!(model, samples=4000, nchains=4)  # 1000 kept per chain
```
"""
# Derives the grouping arrays (n_groups/group_idx/group_predictors) from ModelData.Z
# once, then calls the model with its unpacked-argument signature. Shared with psis_loo
# (comparison.jl), which needs the same conditioned model.
function _build_model_with_data(TR::TuringRegression)
    md = apply_transform(TR.tf, TR.modeldata)
    Z = md.Z
    n_groups = [length(re.levels) for re in Z]
    group_idx = isempty(Z) ? Matrix{Int}(undef, length(md.y), 0) : reduce(hcat, (re.level_index for re in Z))
    group_predictors = [re.predictors.X for re in Z]
    weights = something(md.weights, ones(length(md.y)))
    pr = TR.prior
    return TR.model(md.y, md.predictors.X, n_groups, group_idx, group_predictors, weights,
        pr.intercept, pr.fixed_effects, pr.random_effect_variance, pr.auxiliary)
end

function fit!(
    TR::TuringRegression{T};
    sampler=NUTS(; adtype=has_random_effects(TR.modeldata) ? AutoReverseDiff(; compile=true) : AutoForwardDiff()),
    parallel=MCMCThreads(),
    samples=2000,
    nchains=4,
    warmup=samples,
    quiet=true,
    kwargs...,
) where {T}
    model_with_data = _build_model_with_data(TR)

    # `samples` and `warmup` are totals across chains; split with ceil division so the
    # realised count is never below what was asked (round up = add, not subtract).
    per_chain = cld(samples, nchains)
    warmup_per_chain = cld(warmup, nchains)
    # AbstractMCMC's `N` already means kept draws; `discard_initial` adds
    # `warmup_per_chain` steps on top (total steps sampled = per_chain + warmup_per_chain).
    if quiet
        TR.samples = @suppress sample(model_with_data, sampler, parallel, per_chain, nchains; nadapts=warmup_per_chain, discard_initial=warmup_per_chain, chain_type=VNChain, kwargs...)
    else
        TR.samples = sample(model_with_data, sampler, parallel, per_chain, nchains; nadapts=warmup_per_chain, discard_initial=warmup_per_chain, chain_type=VNChain, progress=true, kwargs...)
    end

    # Raw standardised-scale sampled params straight off the chain, stacked into
    # (iter,chain,param) with vector/matrix VarNames split into indices (`β[1]`,
    # `L.L[2,1]`, ...), no model return statement needed. `reshape_params` splits this
    # flat array into named layers (still standardised scale); `unstandardise` then
    # back-transforms those layers to the original data scale.
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