# T21 — drop predictor standardisation, results

Branch: `remove-predictor-standardisation`. Not merged — recommend discard.

## What changed

- `src/transform.jl`: `compute_transform` no longer scales fixef/ranef predictors
  to sd=1 — keeps centering, drops the divide. Implemented via fitting a normal
  `ZScoreTransform` then `.scale .= 1.0` in place (NOT `scale=false`, which
  returns an empty `.scale` vector and breaks the downstream division in
  `_unstandardise_fixef`/`_unstandardise_ranef`).
- `src/prior.jl`: `RegressionPrior.fixed_effects`/`.random_effect_variance`
  loosened from `Distribution` to `PriorSpec = Union{Distribution,
  AbstractVector{<:Distribution}}`. New `scaled_default_prior(family, md)`
  rescales the plain `default_prior` per predictor column by that column's raw
  sd (`Normal(μ, σ/sd)`, `Exponential(θ/sd)`), so unscaled (raw, centered-only)
  predictors still get sensibly-widthed priors. Ranef-variance rescaling only
  applied when the model has exactly one ranef term (matches this test's scope);
  falls back to the plain scalar default otherwise.
- `src/model.jl`: new `_dist_for(d, n)` dispatch — `filldist(d, n)` for a scalar
  `Distribution` (old behaviour), `arraydist(d)` for a `Vector{<:Distribution}`
  (new, per-column). Swapped into `_fixed_effects`/`_random_effects` in place of
  the raw `filldist` calls.
- `src/turingregression.jl`: `turing_glm`'s `priors` kwarg default changed from
  `default_prior(family)` (evaluated before `modeldata` exists) to `nothing`,
  resolved to `scaled_default_prior(family, modeldata)` post-`modeldata` when the
  caller doesn't supply their own priors.
- `src/TuringRegressions.jl`: include-order fix, `formula_handlers.jl` (defines
  `ModelData`) now loads before `prior.jl` (now references it).

`unstandardise`/back-transform code itself is untouched — verified algebraically
that with X centered-only (scale=1) and y still standardised, the existing
division-based back-transform reduces correctly to raw-scale coefficients.

## Results — T20 dev-subset, full budget (samples=2000, warmup=2000, nchains=4)

All 15/15 tests pass on both branches.

### Accuracy — posterior mean vs canonical (GLM MLE / lme4 REML)

| param | main (std) | T21 (unstd) | canonical |
|---|---|---|---|
| Normal α | 34.693 | 34.613 | 34.661 |
| Normal Cyl | -1.600 | -1.577 | -1.587 |
| Normal Disp | -0.020 | -0.021 | -0.021 |
| Bernoulli α | 2.046 | 2.055 | 2.044 |
| Bernoulli Class:2nd | -1.014 | -1.034 | -1.018 |
| Bernoulli Class:3rd | -1.779 | -1.792 | -1.778 |
| Bernoulli Class:Crew | -0.855 | -0.866 | -0.858 |
| Bernoulli Sex:Male | -2.427 | -2.423 | -2.420 |
| Bernoulli Age:Child | 1.070 | 1.074 | 1.062 |
| RE Intercept | 251.574 | 251.849 | 251.400 |
| RE Days | 10.489 | 10.514 | 10.500 |
| RE Subject_sd[Intercept] | 38.465 | 37.996 | 24.700 |
| RE Subject_sd[Days] | 6.085 | 6.075 | 5.900 |

No accuracy regression — T21 numbers track main within noise, both track
canonical within noise. Note `Subject_sd[Intercept]` runs high vs canonical
(~24.7) on BOTH branches (~38) — pre-existing model bias, not caused by T21.

### Fit time — wall-clock seconds, full budget

| model | main (std) | T21 (unstd) | delta |
|---|---|---|---|
| Normal (mtcars) | 21.7s | 36.8s | +70% |
| Bernoulli (titanic) | 5.3s | 6.4s | +21% |
| sleepstudy RE (1+Days\|Subject) | 75.5s | 106.0s | +40% |

Real slowdown across all 3 fits, not noise-sized. Likely cause: the
`arraydist`-over-per-column-`Distribution`-vector path (used for the rescaled
priors) samples slower through NUTS than `filldist` over one shared scalar
dist — different AD/broadcast shape.

## Conclusion

Drop this branch. Removing predictor standardisation buys nothing (no accuracy
gain) and costs a real, consistent 20-70% slowdown — worse on the model that
matters most for T22 (the RE benchmark, +40%). The per-column prior machinery
needed to keep priors sensible on raw predictor scale (`PriorSpec`,
`_dist_for`/`arraydist`) is itself the likely source of the slowdown, and adds
meaningful complexity (struct type widened, model codegen branches on prior
shape, `turing_glm`'s prior-resolution logic changed) for a design goal
(unscaled predictors, sensible raw-scale priors) that the timing numbers say
isn't worth it. Standardisation (C2) stays as-is on `main`.
