
# TuringRegressions.jl

An alternative and more fully featured version of [TuringGLM.jl](https://turinglang.org/TuringGLM.jl/stable/) for Bayesian regression.

Uses DimArrays for outputs, allowing easy indexing.

## Installation

```julia
using Pkg
Pkg.add("TuringRegressions")
```

## Usage

```julia
using TuringGLModels, RDatasets, Statistics

# Load car data
mtcars = dataset("datasets", "mtcars")

# Create a model
mod = turing_glm(
    @formula(MPG ~ Cyl + Disp),
    mtcars,
    Normal
)

# Fit the model
fit!(mod, N=1000, nchains=2)

# View results
pretty(mod)

# Get coefficients
fixed_effects = fixef(mod)  # With uncertainty
coefs(mod)  # Point estimates, equivalent to fixef(mod,  median)
fixef(mod, std) # Reduce to point estimate with passed function
fixef(mod, x -> quantile(x, [0.05, 0.95])) # Reduce with custom function

# Use the power of DimensionalData's orderless indexing
fixed_effects[param=At("Cyl")]
parameters(mod, collapse=false)[chain=2:3, param=Where(x -> occursin(r"yl", x))]

# Extract parameters with options
parameters(mod, drop_warmup=100, n_draws=500, collapse=false)
internals(mod, median)  # Get variance parameters

# Make predictions
predict(mod)  # For original data
predict(mod, type=:epred)  # Expected values
predict(mod, type=:linpred)  # Linear predictor

# Predict on new data
new_data = [6.0 200.0; 8.0 350.0]
predict(mod, new_data, mean) # Optionally pass fucntion

# Metrics
using StatisticalMeasures # to be able to pass metrics, otherwise defaults only
calculate_metrics(mod, [rsq, rmse]) # All draws
calculate_metrics(mod, [rsq, rmse], median) # Pass function to reduce
default_metrics(mod, mean) # Models have defaults defined

# Compare models
robust_mod = turing_glm(@formula(MPG ~ Cyl + Disp), mtcars, TDist)
fit!(robust_mod, N=4000, nchains=3)

loo_compare(mod, robust_mod)

# Plots - using DimensionalData integration
using GLMakie

# Coefficients
coefs = coefs(mod)
violin(coefs; scale=:width, show_median=:true, side=:left)
rainclouds(coefs)
boxplot(coefs)

# Trace plot
coefs2 = dropdims(get_parameters(mod, [:α], collapse=false); dims = 2)
Makie.series(coefs2'; linewidth = 0.3) # NB the transpose

# Scatter for sampling
pair = get_parameters(mod, [:α, :σ])
scatter(pair)
triple = hcat(get_parameters(mod, [:α, :σ, ]), internals(mod)[param=At("lp")])
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
* `turing_glm(y, X, family)` - Create from matrices

### Fitting
* `fit!(model)` - Run MCMC sampling

### Parameter Extraction
* `parameters(model, fun)` - All parameters
* `fixef(model, fun)` - Fixed effects  
* `internals(model, fun)` - Sampling information 
* `coef(model, fun)` - Point estimates
* `get_parameters(model, params)` - Specific parameters

### Predictions
* `predict(model, X)` - Generate predictions
* `linpred(model, X)` - Linear predictor
* `epred(model, X)` - Expected values
* `posterior_pred(model, X)` - Posterior predictive samples

### Model Comparison
* `psis_loo(model)` - Leave-one-out cross-validation
* `loo_compare(models...)` - Compare multiple models

### Utilities
* `pretty(model)` - Formatted summary
* `parameter_names(model)` - Parameter names
* `outcome(model)` - Response variable
* `outcome_as_distribution(model)` - Response variable as CategoricalDistributions.jl object (Bernoulli only)
* `predictors(model)` - Predictor table
* `calculate_metrics(model, [metrics])` - Model metrics (from StatisticalMeasures.jl)
* `default_metrics(model)` - Default model metrics

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

## Thanks

This pacakge was heavily inspired by and uses small snippets of code from TuringJL
It also uses the power of [DimensionalData.jl](https://rafaqz.github.io/DimensionalData.jl/stable/) for its outputs.

## TODO

* Add full support for random effects
* redo docs to reflect changes
* redo all the tests

