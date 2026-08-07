
# TuringRegressions.jl

An alternative and more fully featured version of [TuringGLM.jl](https://turinglang.org/TuringGLM.jl/stable/) for Bayesian regression fits, modelled on `brms` in `R`. Handles both fixed and random effects, including correlated intercepts and slopes.

It is deliberately fully featured and heavy, providing a convenient front-end to a large number of packages.

Design Features:

* Handles Normal, TDist, Binomial, Poisson, NegativeBinomial families
* Fixed and Random effects supported, including varying slopes
* Models run on standardised scale for performance to ensure correct outputs vs GLM/lme4 (three fits are exactly benchmarked in tests)
* Priors are somewhat customisable, and specified on standardised scale
* Model code dynamically constructed, can be viewed and exported
* Sampling performance in NUTS optimised as far as possible; ReverseDiff used for random effect models to improve performance
* Outputs use `DimensionalData` for easy indexing
* `PrettyTable` model summaries, with prediction metrics
* Prediction on same or new data
* `StatsAPI.RegressionModel` interface implemented as far as possible
* Model comparison with Psis and Loo
* `Makie` Plot recipes
* Precompiles a single fixed effects fit, expensive but saving ~10s+ on time-to-first-fit across all fit types.

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
# NB priors are on the standardised scale (mean 0, sd 1 predictors), not the
# original data's units — see `prior_summary`/Notes below
mod = turing_glm(
    @formula(MPG ~ Cyl + Disp),
    mtcars,
    Normal
)

# Fit the model
fit!(mod, N=1000, nchains=2)

# View results
model_summary(mod)
model_summary(mod; show_metrics=true) # Just taken from posterior draws
prior_summary(mod) # Just the prior block — same as show(mod), no formula/samples/warnings

# Get coefficients
fixed_effects = draws(mod, :fixef) # With uncertainty
draws(median, mod, :fixef) # Pass function to reduce
draws(x -> quantile(x, [0.05, 0.95]), mod, :fixef) # Reduce with custom function

# Use the power of DimensionalData's orderless indexing
fixed_effects[param=At("Cyl")]
draws(mod, :fixef; collapse=false)[chain=2:3, param=Where(x -> occursin(r"yl", x))]

# Extract parameters with options controlling output
draws(mod, :fixef; drop_draws=100, n_draws=500, collapse=false)
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

# Make predictions (full posterior — the package's primary predict API)
posterior_predict(mod)  # For original data
posterior_predict(mod, type=:epred)  # Expected values
posterior_predict(mod, type=:linpred)  # Linear predictor

# Predict on new data
new_data = [6.0 200.0; 8.0 350.0]
posterior_predict(mod, new_data) # Uses fitted posterior draws
posterior_predict(mean, mod, new_data) # Optionally pass function to reduce

# StatsAPI point-estimate predict (interop only, see StatsAPI section below)
predict(mod, new_data) # posterior-mean epred, plain Vector

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
coefs = draws(mod, :fixef)
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

# Ribbon plot: median line + interval bands over draws at each x
x = 1:0.1:5
y = randn(1000, length(x))  # rows = draws, one column per x
lineribbon(x, y)

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
* `draws(model; drop_draws, n_draws, collapse)` - Whole parameter `DimStack` (all layers)
* `draws(model, type; drop_draws, n_draws, collapse)` - Single layer `DimArray`. `type` one of `propertynames(model.parameters)`, e.g. `:fixef`, `:{group}`, `:{group}_sd`, `:{group}_corr`, `:{group}_offset`, `:internals`
* `draws(f, model, type; dropdims, kwargs...)` - Apply reducer `f` (e.g. `median`) over draw/chain dims
* `outcome(model)` - Response variable
* `predictors(model, type)` - Predictor table
* `outcome_as_distribution(model)` - Response variable as CategoricalDistributions.jl object (Bernoulli only)

### Predictions
* `posterior_predict(model, X=model.X; type=:posterior, kwargs...)` - Generate full-posterior predictions (`type` one of `:posterior`, `:epred`, `:linpred`) — the package's primary predict API
* `posterior_predict(f, model, X=model.X; type, kwargs...)` - Reduce draws with `f` first
* `posterior_predict(model, new_data::DataFrame; kwargs...)` - Predict on new data, remaps random-effect levels

### Model Comparison
* `psis_loo(model; kwargs...)` - Leave-one-out cross-validation via Pareto-smoothed importance sampling (`PosteriorStats.loo`). `kwargs...` forwarded to `PosteriorStats.loo`.
* `loo_compare(models::AbstractVector{<:TuringRegression}; kwargs...)` / `loo_compare(models::TuringRegression...; kwargs...)` - Compare fitted models by ELPD (`PosteriorStats.compare`). `kwargs...` forwarded to `PosteriorStats.compare`.

### StatsAPI

`TuringRegression <: StatsAPI.RegressionModel`. Point estimates are posterior-based (e.g. `coef` = posterior mean) and cover fixed effects only.

* `coef`, `coefnames`, `coeftable`, `confint`, `vcov`, `stderror` - Fixed-effect estimates and credible intervals
* `nobs`, `isfitted`, `weights`, `islinear`, `response`, `responsename`, `meanresponse`, `modelmatrix` - Metadata and data accessors (`modelmatrix` errors on random-effects models, use `predictors(model, :fixef)` instead)
* `fitted`, `residuals`, `predict` - Point-estimate predictions (posterior-mean `epred`); use `posterior_predict` above for the full posterior
* MLE-only stats with no Bayesian analogue (`dof`, `aic`/`bic`, `r2`, `leverage`, ...) raise a clear error instead of a number — see `psis_loo`/`loo_compare` for model comparison

### Utilities
* `model_summary(model)` - Formatted summary with diagnostics (rhat, ess, mcse)
* `model_warnings(model)` - Report rhat/ess/mcse warnings
* `calculate_metrics(model, [metrics]; threshold=0.5, kwargs...)` - Model metrics (from StatisticalMeasures.jl)
* `calculate_metrics(fun, model, [metrics]; kwargs...)` - Reduce draws with `fun` first, matching `draws`
* `default_metrics(model)` / `default_metrics(fun, model)` - Default model metrics

### Plots
* `lineribbon(x, y)`/`lineribbon!()` - Makie recipe: median line + interval ribbons, y = draws (rows) x x-positions (cols)
* `pp_check_hist(model)` as well as `pp_check_dens()` and `pp_check_dens_overlay()` - Posterior predictive checks
* See also the examples above for more quick plots

### Common Arguments

Parameter extraction functions accept:

* `drop_draws=200` - Warmup samples to drop
* `n_draws=-1` - Number of draws (-1 for all)
* `collapse=true` - Collapse chains into single dimension

## Thanks

This pacakge was heavily inspired by and uses small snippets of code from TuringGLM. 

This is intended as a front-end package to a large number of others, and depends notably on the Turing.jl and DimensionalData.jl ecosystems.

The original was hand-written but the random-effect implementation, tidying up and docs were co-written with Claude.

## TODO

Consider more prior customisability. In particular, the LKJ prior on correlated slopes is fixed at 1.0 which leads to the sleepstudy benchmark being slightly off the lme4/brms defaults.
