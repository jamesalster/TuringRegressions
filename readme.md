
# TuringRegressions.jl

An alternative and more fully featured version of [TuringGLM.jl](https://turinglang.org/TuringGLM.jl/stable/) for Bayesian regression.

Uses DimArrays from `DimensionalData` for outputs, allowing easy indexing.

## Installation

```julia
using Pkg
Pkg.add("TuringRegressions")
```

## Usage

```julia
using TuringRegressions, RDatasets, Statistics

# Load car data
mtcars = dataset("datasets", "mtcars")

# Create a model
# NB priors are on the standardised scale for now (A TODO is to fix that)
mod = turing_glm(
    @formula(MPG ~ Cyl + Disp),
    mtcars,
    Normal
)

# Fit the model
fit!(mod, N=1000, nchains=2)

# View results
summary(mod)

# Get coefficients
fixed_effects = draws(mod, :fixef) # With uncertainty
draws(median, mod, :fixef) # Pass function to reduce
draws(x -> quantile(x, [0.05, 0.95]), mod, :fixef) # Reduce with custom function

# Use the power of DimensionalData's orderless indexing
fixed_effects[param=At("Cyl")]
draws(mod, :fixef; collapse=false)[chain=2:3, param=Where(x -> occursin(r"yl", x))]

# Extract parameters with options controlling output
draws(mod, :fixef; drop_warmup=100, n_draws=500, collapse=false)
draws(mod, :internals) # Sampling information

# Random effects — correlated intercept + slope per group
sleepstudy = dataset("lme4", "sleepstudy")
re_mod = turing_glm(
    @formula(Reaction ~ 1 + Days + (1 + Days | Subject)),
    sleepstudy,
    Normal
)
fit!(re_mod, N=1000, nchains=2)

propertynames(draws(re_mod)) # (:fixef, :Subject, :Subject_sd, :Subject_corr, :internals)

draws(re_mod, :Subject) # per-Subject offsets, dims (effect, group, draw)
draws(mean, re_mod, :Subject)[effect=At(:Days), group=At(308)] # one subject's slope offset
draws(re_mod, :Subject_sd) # group-level SDs, dims (effect, draw)
draws(re_mod, :Subject_corr) # intercept/slope correlation matrix, dims (effect, effect2, draw)

# Make predictions
predict(mod)  # For original data
predict(mod, type=:epred)  # Expected values
predict(mod, type=:linpred)  # Linear predictor

# Predict on new data
new_data = [6.0 200.0; 8.0 350.0]
predict(mod, new_data) # Uses fitted posterior draws
predict(mean, mod, new_data) # Optionally pass function to reduce

# Metrics
using StatisticalMeasures # to be able to pass metrics, otherwise defaults only
calculate_metrics(mod, [rsq, rmse]) # All draws
calculate_metrics(median, mod, [rsq, rmse]) # Pass function to reduce
default_metrics(mean, mod) # Models have defaults defined

# Compare models
robust_mod = turing_glm(@formula(MPG ~ Cyl + Disp), mtcars, TDist)
fit!(robust_mod, N=4000, nchains=3)

# Plots - using DimensionalData integration
using GLMakie

# Coefficients
coefs = draws(median, mod, :fixef)
violin(coefs; scale=:width, show_median=:true, side=:left)
rainclouds(coefs)
boxplot(coefs)

# Trace plot
coefs2 = draws(mod, :fixef; collapse=false)[param=At([:α])]
Makie.series(coefs2'; linewidth = 0.3) # NB the transpose

# Scatter for sampling
pair = draws(mod, :fixef)[param=At([:α, :σ])]
scatter(pair)
triple = draws(mod, :fixef)[param=At([:α, :σ, :Cyl])]
scatter(triple)

# Conditional dependency provided as a function
conditional_dependency(mod, :Disp)

#PP check provided as a function
pp_check_hist(mod; bins=30)
pp_check_dens(mod; type=:linpred)
pp_check_dens_overlay(mod)
```

## API

### Model Creation
* `turing_glm(formula, data, family)` - Create from formula and data

### Fitting
* `fit!(model; sampler=NUTS(), parallel=MCMCThreads(), N=2000, nchains=4, kwargs...)` - Run MCMC sampling, mutates model

### Parameter Extraction
* `draws(model; drop_warmup, n_draws, collapse)` - Whole parameter `DimStack` (all layers)
* `draws(model, type; drop_warmup, n_draws, collapse)` - Single layer `DimArray`. `type` one of `propertynames(model.parameters)`, e.g. `:fixef`, `:{group}`, `:{group}_sd`, `:{group}_corr`, `:{group}_offset`, `:internals`
* `draws(f, model, type; dropdims, kwargs...)` - Apply reducer `f` (e.g. `median`) over draw/chain dims
* `outcome(model)` - Response variable
* `predictors(model, type)` - Predictor table
* `outcome_as_distribution(model)` - Response variable as CategoricalDistributions.jl object (Bernoulli only)

### Predictions
* `predict(model, X=model.X; type=:posterior, kwargs...)` - Generate predictions (`type` one of `:posterior`, `:epred`, `:linpred`)
* `predict(f, model, X=model.X; type, kwargs...)` - Reduce draws with `f` first
* `predict(model, new_data::DataFrame; kwargs...)` - Predict on new data, remaps random-effect levels

### Model Comparison
* `psis_loo(model; kwargs...)` - Leave-one-out cross-validation via Pareto-smoothed importance sampling (`PosteriorStats.loo`). `kwargs...` forwarded to `PosteriorStats.loo`.
* `loo_compare(models::AbstractVector{<:TuringRegression}; kwargs...)` / `loo_compare(models::TuringRegression...; kwargs...)` - Compare fitted models by ELPD (`PosteriorStats.compare`). `kwargs...` forwarded to `PosteriorStats.compare`.

### Utilities
* `summary(model)` - Formatted summary with diagnostics (rhat, ess, mcse)
* `model_warnings(model)` - Report rhat/ess/mcse warnings
* `calculate_metrics(model, [metrics]; threshold=0.5, kwargs...)` - Model metrics (from StatisticalMeasures.jl)
* `calculate_metrics(fun, model, [metrics]; kwargs...)` - Reduce draws with `fun` first, matching `draws`
* `default_metrics(model)` / `default_metrics(fun, model)` - Default model metrics

### Plots
* `lineribbon!()` - Makie recipe for banded intervalsm, used in `conditional_dependency()`
* `conditional_dependency(model, var)` - Show dependency of outcome on one variable
* `pp_check_hist(model)` as well as `pp_check_dens()` and `pp_check_dens_overlay()` - Posterior predictive checks
* See also the examples below for more quick plots

### Common Arguments

Parameter extraction functions accept:

* `drop_warmup=200` - Warmup samples to drop
* `n_draws=-1` - Number of draws (-1 for all)
* `collapse=true` - Collapse chains into single dimension

## Notes

Priors are passed in at the standardised variable scale.

Accepted model families are `Normal`, `TDist`, `Bernoulli`, `Poisson`, and `NegativeBinomial`.

## Thanks

This pacakge was heavily inspired by and uses small snippets of code from TuringJL
It also uses the power of [DimensionalData.jl](https://rafaqz.github.io/DimensionalData.jl/stable/) for its outputs.

## TODO

* Priors currently expressed on standardised scale — move to original data scale
* Investigate slow NUTS sampling for correlated random-effects models

See `SPEC.md` for the full task list.

