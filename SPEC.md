# SPEC — TuringRegressions.jl

Distilled code 2026-07-15. WIP package. `?` = unconfirmed, user verify.

## §G GOAL

Bayesian GLM package Julia. Alternative TuringGLM.jl, more features.
Fit regression via Turing.jl NUTS. Output = `DimArray` (DimensionalData) →
orderless named indexing params/draws/chains. User writes `@formula`, picks
family, calls `fit!`, extracts coefs / predicts / metrics / compares / plots.

Core loop: `turing_glm(formula, data, family)` → `fit!` → `summary`/`draws`/`predict`.

## §C CONSTRAINTS

- C1. Julia. Turing.jl 0.36.3 MCMC. Sampler default NUTS, parallel MCMCThreads, N=2000, nchains=4.
- C2. Data auto-standardised inside model. User priors currently scaled to std predictors (mean 0, sd 1) — **T4 flips this to original scale.**
- C3. Params stored/returned `TR.parameters::DimStack`. One layer per group (`:fixef`, per-ranef `:{group}`, `:{group}_sd`, `:{group}_corr`, `:{group}_offset`, `:internals`), layer's own dims incl `:draw, :chain`. Chains collapsible to one `:draw` dim via `draws(...)`.
- C4. Families supported: `Normal, TDist, Bernoulli, Poisson, NegativeBinomial`. Others → `error`.
- C5. Links fixed per family (util `get_link`): Normal/TDist→identity, Bernoulli→logit, Poisson/NegBin→log.
- C6. Model code generated Julia `Expr` at runtime, `eval`'d into `@model turing_regression`. Not hand-written. `show_code=true` prints generated source — **T5 re-verifies this path.**
- C7. Makie plots via native package extension (`ext/TuringRegressionsMakieExt.jl`, `[weakdeps]`/`[extensions]` in Project.toml). No `Requires`, no `Colors` dep.
- C8. NegBin canonical form: local `NegativeBinomial2(μ, mean/dispersion)`, `max(1/(1+μ/ϕ), 1e-6)`. No TuringGLM dependency (removed).
- C9. Touch scaling logic (T3/T4) → re-verify all families against GLM (V3, V4, V13).

## §I INTERFACES (public surface)

Model creation:
- `turing_glm(formula::FormulaTerm, data::DataFrame, family, priors=default_prior(family), weights=nothing, show_code=false)` → `TuringRegression{family}`
- `turing_glm(y::Vector, X::Array, ::Type{T}; names=Symbol[], kwargs...)` array form, synthesises formula, forwards.
- `default_prior(family)` / `default_prior(TR)` → `RegressionPrior` (intercept N(0,5), fixef N(0,2), ranef Exp(1), aux family-dep).

Fitting:
- `fit!(TR; sampler=NUTS(), parallel=MCMCThreads(), N=2000, nchains=4, quiet=true, kwargs...)` mutates TR. Recovers unstandardised params via `generated_quantities` into `TR.parameters` (DimStack).

Param extraction (`src/parametermethods.jl`):
- `draws(TR; drop_warmup=200, n_draws=-1, collapse=true)` whole `TR.parameters` DimStack (all layers), warmup dropped/chains collapsed per kwargs.
- `draws(TR, type::Symbol; drop_warmup=200, n_draws=-1, collapse=true)` single layer DimArray. `type` must be one of `propertynames(TR.parameters)` else `ArgumentError`. Valid types: `:fixef`, `:{group}` (ranef effect×group), `:{group}_sd`, `:{group}_corr` (only if correlated ranef), `:{group}_offset` (only slope-only-no-intercept ranef).
- `draws(f::Function, TR, type::Symbol; dropdims=true, kwargs...)` apply reducer `f` over draw/chain dims.
- `outcome(TR)` → DimArray of y.
- `predictors(TR, type)` → DimArray of X (name split from old `fixed_effects`).
- `outcome_as_distribution(TR)` → `UnivariateFinite` (classification metrics interop).

Predict (`src/predict.jl`, real working API):
- `predict(TR, X=TR.X; type=:posterior, kwargs...)` / `predict(f::Function, TR, X=TR.X; type, kwargs...)`
- `predict(TR, new_data::DataFrame, ...)` rebuilds X from formula, remaps random-effect levels (`new_random_effects`, `allow_new_levels` kwarg).
- `type ∈ (:posterior, :epred, :linpred)`.
- Internal: `linpred` (Xβ+α [+Zu]), `epred` (invlink∘linpred), `posterior_pred` (adds family noise).

Metrics (StatisticalMeasures.jl):
- `calculate_metrics(TR, metrics::Vector, fun=nothing; threshold=0.5, kwargs...)` epred-based. Bernoulli branch: AUC + pseudo_r2 special-cased, rest categorical.
- `default_metrics(TR, fun=nothing)` regression → `[rsq, rmse, mae]`; Bernoulli → `[accuracy, kappa, TPR, TNR, auc, pseudo_r2]`.
- `pseudo_r2(preds, y)` McFadden, exported.

Comparison (ParetoSmooth.jl, `src/comparison.jl`):
- `psis_loo(TR)` → PsisLoo.
- `loo_compare(models::TuringRegression...)` / `loo_compare(models::AbstractVector{<:TuringRegression})`, kwargs forwarded.

Display (`src/summary.jl`, `src/turingregression.jl`):
- `Base.show(io, TR; warnings=true)` family/formula/prior/obs/samples + warnings.
- `Base.summary(io, TR; funs=[mean,std], quantiles=[0.025,0.975], return_table=false, drop_warmup=nothing, show_metrics=false, kwargs...)`. Prints fixef table + per-grouping-term Random Effects tables (SD table always; Correlation matrix only if `{group}_corr` layer present). Metrics table opt-in via `show_metrics=true`.
- `model_warnings(TR)` — rhat/ess/mcse based `@warn`/`@info`.

Plots (Makie ext, `ext/TuringRegressionsMakieExt.jl`):
- `lineribbon`/`lineribbon!`, `conditional_dependency`, `pp_check_dens`, `pp_check_dens_overlay`, `pp_check_hist`. Stubs exported from main package, methods added by extension when Makie loaded.

Types:
- `TuringRegression{T<:Distribution}` — holds formula, X/X_names, y, family, prior, parameters, model info.
- `ModelInfo` — `has_intercept`/`has_fixed_effects`/`has_random_effects`/`weighted`.
- `RegressionPrior` — `intercept`/`fixed_effects`/`random_effects`/`auxiliary`.

## §V INVARIANTS

- V1. Family restricted to `{Normal, TDist, Bernoulli, Poisson, NegativeBinomial}`; others → `error` at `turing_glm` construction.
- V2. `draws(TR, type)` with invalid `type` → `ArgumentError`.
- V3. Fixef point estimates (posterior mean) vs GLM MLE within tolerance — Normal/Poisson/NegBin atol 0.025 (runtests), Bernoulli looser (see V13).
- V4. `epred` posterior mean vs `GLM.predict` within tolerance (family-dependent, see runtests) for Normal/Poisson/NegBin/Bernoulli.
- V5. Link relations: `linpred == log(epred)` for log-link families; `linpred == epred` for identity-link families.
- V6. `var(posterior_pred) > var(epred)` — posterior predictive adds observation noise (also holds for count families).
- V7. `draws(TR, type)` shape `(n_effect_dims..., (N-drop_warmup)*nchains)` when collapsed; extra trailing `:chain` dim if `collapse=false`.
- V8. `n_draws` requested beyond available post-warmup draws → `ErrorException`.
- V9. Returned param labels: `[:α, :<X_names...>, aux...]`; `:α` always present for models with intercept.
- V10. Standardised→original param recovery in `_generated_quantities`: `β_orig = (y_std/X_stds).*β` (Gaussian family) or `β./X_stds` (count/binary); `α` likewise via `dot(X_means, β_orig)`. **Will move out of generated model under T3.**
- V11. `summary` auto `drop_warmup`: 0 if `N<400` else 200.
- V12. `rhat>1.05` / `ess<100` / `mcse>5%std` → `@warn`; softer thresholds → `@info`.
- V13. Bernoulli param recovery vs GLM looser tolerance: coef atol 0.05, epred atol 0.05.
- V14. Weighted fit with `weights ≡ 1` == unweighted fit (params equal within tolerance). Currently ZERO tests exercising `_weighted_likelihood` (model.jl) — add under T1.
- V15. `predict(TR, new_data::DataFrame)` uses raw new X, original-scale stored β (no re-standardisation of new X). `epred` on new data ≈ `GLM.predict` on same new data. Guards against re-introducing standardisation in the predict path.
- V16. `extract_random_effect` predictor slicing conditions on `has_intercept(term.lhs)` — `(0+x...|g)` (no-intercept ranef terms) sliced differently from intercept-bearing ranef terms. Fixed 2026-07-15, guarded by dedicated tests.

## §T TASKS

T1|.|Confirm full new test suite passes AND benchmarks fit well|V2,V14,I.plots

T2|.|README rewrite — API surface has drifted (old `fixef`/`parameters`/`get_parameters` references, TuringGLM mentions). Bring in line with current `draws`/`predict`/`summary` interface|I

T3|.|BIG JOB: redo scaling w.r.t. model. Move standardise/back-transform OUT of the generated model — compute scaling stats ONCE outside (NamedTuple `X_means`/`X_stds`/`y_mean`/`y_std` + per-ranef equivalents), pass scaled data into the model, back-transform (fixef + ranef + Σ) in Julia inside `fit!`. Delete `_standardise_data`/`_generated_quantities` in favour of this. Reworks in-model ranef scaling. Add round-trip test `unstandardise∘standardise == id`|C2,C6,V10,V3,V4,V14

T4|.|BIG JOB: flip prior scaling (depends on T3's centralised affine map). Today user specifies priors on standardised (mean 0, sd 1) scale. Change so priors are given on ORIGINAL data scale — e.g. `Normal(10,20)` on a predictor with mean 10, sd 20 → transformed to `Normal(0,1)` internally for the standardised fit — reported back on original scale in `summary`/`show`/prior display. Use `Distributions.AffineDistribution` (`shift + scale*d`) for the forward transform (prior) and its inverse for the param back-transform — don't hand-write new per-family algebra. Store user's original prior for display; fit on scaled. Re-verify V-invariants under Turing/NUTS with `AffineDistribution` priors|C2,I

T5|.|Re-verify model code-gen (`show_code`, model.jl generated `Expr`) after T3/T4 land, since both touch generated-model internals. FOLD IN: collapse `_likelihood` + `_weighted_likelihood` (currently ~90% duplicated) into one family-dispatched function, with `weights` defaulting to `ones(...)` so unweighted fit is just weighted-fit-with-1s (makes V14 a true structural guarantee, not a coincidence)|C6,V14

T6|.|BIG JOB: `TuringRegression` as StatsAPI/StatsBase `RegressionModel`. Posterior-based methods where a point-estimate API expects one; skip/error clearly where no sane mapping exists (e.g. `StatsModels.TableRegressionModel`-only methods). Add formula-schema tests|I

## §NOTE

- P8: `TR.link` field (predict.jl) — used internally, V1-bounded to the 5 families, not user-facing.
