
# TuringRegressions.jl

An alternative and more fully featured version of [TuringGLM.jl](https://turinglang.org/TuringGLM.jl/stable/) for Bayesian regression fits, modelled on `brms` in `R`. Handles both fixed and random effects, including correlated intercepts and slopes.

It is deliberately fully featured and heavy, providing a convenient front-end to a large number of packages.

Design Features:

* Handles Normal, TDist, Bernoulli, Poisson, NegativeBinomial families
* Fixed and Random effects supported, including varying slopes
* Models run on standardised scale for performance to ensure correct outputs vs GLM/lme4 (six fits are benchmarked against GLM/lme4 in the test suite, coefficients to within 0.02)
* Priors are somewhat customisable, and specified on standardised scale
* Model code dynamically constructed, can be viewed and exported and also modified manually
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
using TuringRegressions, RDatasets, Statistics, DataFrames

# Load car data
mtcars = dataset("datasets", "mtcars")

# Create a model
# NB priors are on the standardised scale (mean 0, sd 1 predictors), not the
# original data's units — see `prior_summary`/Notes below
mod = turing_glm(
    @formula(MPG ~ Cyl + Disp),
    mtcars,
    Normal;
    priors=default_prior(Normal),          # optional, override any field
    weights=nothing,                       # optional per-observation weights
)

# Fit the model. `samples` and `warmup` are TOTALS across chains, and warmup is
# discarded in addition to `samples` — this draws 500 kept draws per chain.
fit!(mod; samples=1000, warmup=1000, nchains=2)

# View results
model_summary(mod)
model_summary(mod; show_metrics=true) # Just taken from posterior draws
prior_summary(mod) # Just the prior block — same as show(mod), no formula/samples/warnings

# Get coefficients
fixed_effects = draws(mod, :fixef) # With uncertainty
draws(median, mod, :fixef) # Pass function to reduce
draws(x -> quantile(x, [0.05, 0.95]), mod, :fixef) # Reduce with custom function

# Use the power of DimensionalData's orderless indexing.
# NB the dimension is named `:fixef`, and its values are Symbols.
fixed_effects[fixef=At(:Cyl)]
draws(mod, :fixef; collapse=false)[chain=1:2, fixef=Where(x -> occursin(r"yl", string(x)))]

# Extract parameters with options controlling output
draws(mod, :fixef; drop_draws=100, n_draws=300, collapse=false)

# Random effects — correlated intercept + slope per group
sleepstudy = dataset("lme4", "sleepstudy")
re_mod = turing_glm(
    @formula(Reaction ~ 1 + Days + (1 + Days | Subject)),
    sleepstudy,
    Normal;
    # η > 1 shrinks the intercept/slope correlation toward 0, η = 1 (default) is flat
    priors=default_prior(Normal; lkj_eta=2.0),
)
fit!(re_mod; samples=1000, warmup=1000, nchains=2)

propertynames(draws(re_mod)) # (:fixef, :Subject, :Subject_sd, :Subject_corr)

draws(re_mod, :Subject) # per-Subject offsets, dims (effect, group, draw)
draws(mean, re_mod, :Subject)[effect=At(:Days), group=At("308")] # one subject's slope offset (levels are Strings)
draws(re_mod, :Subject_sd) # group-level SDs, dims (effect, draw)
draws(re_mod, :Subject_corr) # intercept/slope correlation matrix, dims (effect, effect2, draw)

# MixedModels-style nested (`a/b`) and interaction (`a&b`) grouping formulas are supported.
turing_glm(@formula(Reaction ~ 1 + Days + (1 | Batch / Subject)), sleepstudy, Normal)   # -> :Batch, :Batch__Subject
turing_glm(@formula(Reaction ~ 1 + Days + (1 | Batch & Subject)), sleepstudy, Normal)   # -> :Batch__Subject

# Make predictions (full posterior — the package's primary predict API)
posterior_predict(mod)  # For original data
posterior_predict(mod, type=:epred)  # Expected values
posterior_predict(mod, type=:linpred)  # Linear predictor

# Predict on new data. A raw matrix works for fixed-effect models; random-effect
# models need a DataFrame so grouping levels can be remapped.
new_data = [6.0 200.0; 8.0 350.0]
posterior_predict(mod, new_data) # Uses fitted posterior draws
posterior_predict(mean, mod, new_data) # Optionally pass function to reduce

# Unseen grouping levels error by default; opt in to treating them as the
# population mean (zero random effect)
new_subjects = DataFrame(Days=[1.0, 2.0], Subject=["308", "999"], Reaction=[0.0, 0.0])
posterior_predict(re_mod, new_subjects; allow_new_levels=true)

# StatsAPI point-estimate predict (interop only, see StatsAPI section below)
predict(mod, new_data) # posterior-mean epred, plain Vector

# Metrics
using StatisticalMeasures # to be able to pass metrics, otherwise defaults only
calculate_metrics(mod, [rsq, rmse]) # All draws
calculate_metrics(median, mod, [rsq, rmse]) # Pass function to reduce
default_metrics(mean, mod) # Models have defaults defined

# Compare models
robust_mod = turing_glm(@formula(MPG ~ Cyl + Disp), mtcars, TDist)
fit!(robust_mod; samples=4000, nchains=3)

psis_loo(mod)                      # PSIS-LOO for one model
loo_compare(mod, robust_mod)       # rank fitted models by ELPD

# Plots - using DimensionalData integration
using GLMakie

# Coefficients. `categorical_layout` flattens the draws into the (positions,
# values) pair these plots want, with the parameter names as tick labels —
# passing the DimArray straight in is broken on DimensionalData 0.30 (see Notes).
coefs = draws(mod, :fixef)
pos, vals, axis = categorical_layout(coefs)

violin(pos, vals; axis, scale=:width, show_median=true, side=:left)
rainclouds(pos, vals; axis)
boxplot(pos, vals; axis)

# Trace plot
coefs2 = draws(mod, :fixef; collapse=false)[fixef=At(:α)]  # one param: (iter, chain)
Makie.series(coefs2'; linewidth = 0.3) # NB the transpose

# Scatter for sampling
pair = draws(mod, :fixef)[fixef=At([:α, :σ])]
scatter(pair)
triple = draws(mod, :fixef)[fixef=At([:α, :σ, :Cyl])]
scatter(parent(triple)) # NB `parent` — DimensionalData draws a 2D axis and drops the 3rd param

# Ribbon plot: median line + interval bands over draws at each x.
# Here, the linear predictor over a grid of Disp, holding Cyl at its mean.
disp_grid = range(extrema(mtcars.Disp)...; length=50)
grid = [fill(mean(mtcars.Cyl), 50) collect(disp_grid)]
lp = posterior_predict(mod, grid; type=:linpred) # (row, iter)
lineribbon(disp_grid, parent(lp)') # NB the transpose: rows = draws, one column per x

#PP check provided as a function
pp_check_hist(mod; bins=30)
pp_check_dens(mod; type=:linpred)
pp_check_dens_overlay(mod)
```

## API

### Model Creation
* `turing_glm(formula, data, family; priors=default_prior(family), weights=nothing)` - Create from formula and data
* `turing_glm(y, X, family; names=Symbol[], kwargs...)` - Create from arrays instead of a formula
* `default_prior(family; intercept, fixed_effects, random_effect_variance, auxiliary, lkj_eta)` - Build a `RegressionPrior`, overriding only the fields you name. All priors are on the STANDARDISED scale; `lkj_eta` is the LKJ shape for random-effect correlations (1.0 = flat)

### Fitting
* `fit!(model; sampler=NUTS(), parallel=MCMCThreads(), samples=2000, warmup=samples, nchains=4, quiet=true, kwargs...)` - Run MCMC sampling, mutates model. `samples` and `warmup` are TOTALS split across chains; warmup is discarded and is *in addition to* `samples`. `kwargs...` go to `sample`, so passing `N`/`discard_initial` directly is an error.

### Parameter Extraction
* `draws(model; drop_draws, n_draws, collapse)` - Whole parameter `DimStack` (all layers)
* `draws(model, type; drop_draws, n_draws, collapse)` - Single layer `DimArray`. `type` one of `propertynames(model.parameters)`, e.g. `:fixef`, `:{group}`, `:{group}_sd`, `:{group}_corr`
* `draws(f, model, type; dropdims, kwargs...)` - Apply reducer `f` (e.g. `median`) over draw/chain dims
* `outcome(model)` - Response variable
* `get_fixef_predictors(model)` - Fixed-effect predictor table
* `outcome_as_distribution(model)` - Response variable as CategoricalDistributions.jl object (Bernoulli only)

### Predictions
* `posterior_predict(model, X=model.modeldata.predictors.X; type=:posterior, kwargs...)` - Generate full-posterior predictions (`type` one of `:posterior`, `:epred`, `:linpred`) — the package's primary predict API
* `posterior_predict(f, model, X=model.modeldata.predictors.X; type, kwargs...)` - Reduce draws with `f` first
* `posterior_predict(model, new_data::DataFrame; allow_new_levels=false, kwargs...)` - Predict on new data, remaps random-effect levels. Grouping levels unseen during fitting error unless `allow_new_levels=true`, which gives them the population-mean (zero) random effect. A raw matrix is rejected for random-effect models, since it carries no grouping information

### Model Comparison
* `psis_loo(model; kwargs...)` - Leave-one-out cross-validation via Pareto-smoothed importance sampling (`PosteriorStats.loo`). `kwargs...` forwarded to `PosteriorStats.loo`.
* `loo_compare(models::AbstractVector{<:TuringRegression}; kwargs...)` / `loo_compare(models::TuringRegression...; kwargs...)` - Compare fitted models by ELPD (`PosteriorStats.compare`). `kwargs...` forwarded to `PosteriorStats.compare`.

### StatsAPI

`TuringRegression <: StatsAPI.RegressionModel`. Point estimates are posterior-based (e.g. `coef` = posterior mean) and cover fixed effects only.

* `coef`, `coefnames`, `coeftable`, `confint`, `vcov`, `stderror` - Fixed-effect estimates and credible intervals
* `nobs`, `isfitted`, `weights`, `islinear`, `response`, `responsename`, `meanresponse`, `modelmatrix` - Metadata and data accessors (`modelmatrix` errors on random-effects models, use `get_fixef_predictors(model)` instead)
* `fitted`, `residuals`, `predict` - Point-estimate predictions (posterior-mean `epred`); use `posterior_predict` above for the full posterior
* MLE-only stats with no Bayesian analogue (`dof`, `aic`/`bic`, `r2`, `leverage`, ...) raise a clear error instead of a number — see `psis_loo`/`loo_compare` for model comparison

### Utilities
* `model_summary(model)` - Formatted summary with diagnostics (rhat, ess, mcse)
* `model_warnings(model)` - Report rhat/ess/mcse warnings
* `prior_summary(model)` - Print the prior block alone
* `calculate_metrics(model, [metrics]; threshold=0.5, kwargs...)` - Model metrics (from StatisticalMeasures.jl)
* `calculate_metrics(fun, model, [metrics]; kwargs...)` - Reduce draws with `fun` first, matching `draws`
* `default_metrics(model)` / `default_metrics(fun, model)` - Default model metrics (regression: rsq/rmse/mae; Bernoulli: accuracy/kappa/TPR/TNR/auc/pseudo_r2)
* `pseudo_r2(preds, y)` - McFadden's pseudo-R² for binary outcomes. A metric to pass to `calculate_metrics`, not a model accessor
* `modelcode(model)` - Print + return the generated Turing `@model` code as an `Expr`
* `set_model_code!(model, expr)` - Override the model with a hand-edited `Expr` (from `modelcode`); read docstring before using

### Plots
* `lineribbon(x, y)`/`lineribbon!()` - Makie recipe: median line + interval ribbons, y = draws (rows) x x-positions (cols)
* `pp_check_hist(model)` as well as `pp_check_dens()` and `pp_check_dens_overlay()` - Posterior predictive checks
* `categorical_layout(dimarray)` - Flatten draws into `(positions, values, axis)` for `violin`/`boxplot`/`rainclouds`
* See also the examples above for more quick plots

### Common Arguments

Parameter extraction functions accept:

* `drop_draws=0` - EXTRA draws to drop per chain, on top of the warmup `fit!` already discarded
* `n_draws=Inf` - Number of draws to keep, applied per chain (`Inf` for all)
* `collapse=true` - Collapse chains into single dimension

## Thanks

This package was heavily inspired by and uses small snippets of code from TuringGLM. 

This is intended as a front-end package to a large number of others, and depends notably on the Turing.jl and DimensionalData.jl ecosystems.

The original was hand-written but the random-effect implementation, tidying up and docs were co-written with Claude.

## TODO

Consider more prior customisability — currently priors are one-per-role (`RegressionPrior`) rather than `brms`-style per-term/per-coefficient.

Plotting a `DimArray` of draws directly with `violin`/`boxplot`/`rainclouds` is broken on DimensionalData 0.30: each category is drawn from an interleaved mixture of every other one, and categories are positioned by the sum of their label's character codes, so parameter names land at arbitrary spacings. Both are reported upstream; `categorical_layout` works around them. `scatter` over three parameters needs `parent()` for the same reason — DimensionalData builds a 2D axis and silently drops the third (fixed on its `main`, unreleased at the time of writing).

Note that the random-effect SDs sit above lme4/brms on the sleepstudy benchmark (intercept SD ≈ 29 vs lme4's 24.7). The LKJ shape is now settable via `default_prior(family; lkj_eta=...)`, but the default of 1.0 is flat on the standardised-scale correlation, which is the suspected cause.
