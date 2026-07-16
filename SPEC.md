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
- V17. Test suite runs via `Pkg.test()`, not direct `julia --project=. test/runtests.jl` — `Pkg.test()`'s isolated temp env is the only one that both (a) resolves deps fresh from `[extras]`/`[compat]` and (b) exercises the package's real symbol table end-to-end. Requires `[compat]` pinned on every Turing-stack package whose version drift can silently break internals (`Turing`, `DynamicPPL`, `FlexiChains`, `MCMCChains` — currently 0.46/0.42/0.6/7), so the temp env's independent resolve can't drift to an incompatible combo.

## §B BUGS

id|date|cause|fix
B1|2026-07-16|`src/turingregression.jl` used `MCMCChains.Chains` as a qualifier but `src/TuringRegressions.jl` only did `using MCMCChains: summarize, Chains` — never bound the module name itself, so any `MCMCChains.X` reference was always an `UndefVarError`, in every env. Masked because direct-script test runs never got far enough to hit `fit!` (GLM missing from load path via legacy `[extras]`/`[targets]`, only visible to `Pkg.test()`) — only surfaced once `Pkg.test()` actually ran.|`using MCMCChains: MCMCChains, summarize, Chains`; V17

## §T TASKS

T1|~|Two-pass test run: (1) all fits incl benchmark ones cheap (`BENCH_N`/`BENCH_NCHAINS` env-driven, default 300/2) — suite must run error-free, accuracy asserts may fail; (2) re-up benchmark fits to N=2000, nchains via `TR_BENCH_NCHAINS` env override maxed to `Threads.nthreads()` (was fixed 4, too slow) — fix only small/obvious benchmark-tolerance misses, no rebuild|V2,V14,I.plots

T2|.|README rewrite — API surface has drifted (old `fixef`/`parameters`/`get_parameters` references, TuringGLM mentions). Bring in line with current `draws`/`predict`/`summary` interface|I

T3|.|BIG JOB: redo scaling w.r.t. model. Move standardise/back-transform OUT of the generated model — compute scaling stats ONCE outside (NamedTuple `X_means`/`X_stds`/`y_mean`/`y_std` + per-ranef equivalents), pass scaled data into the model, back-transform (fixef + ranef + Σ) in Julia inside `fit!`. Delete `_standardise_data`/`_generated_quantities` in favour of this. Reworks in-model ranef scaling. Add round-trip test `unstandardise∘standardise == id`|C2,C6,V10,V3,V4,V14

T4|.|BIG JOB: flip prior scaling (depends on T3's centralised affine map). Today user specifies priors on standardised (mean 0, sd 1) scale. Change so priors are given on ORIGINAL data scale — e.g. `Normal(10,20)` on a predictor with mean 10, sd 20 → transformed to `Normal(0,1)` internally for the standardised fit — reported back on original scale in `summary`/`show`/prior display. Use `Distributions.AffineDistribution` (`shift + scale*d`) for the forward transform (prior) and its inverse for the param back-transform — don't hand-write new per-family algebra. Store user's original prior for display; fit on scaled. Re-verify V-invariants under Turing/NUTS with `AffineDistribution` priors|C2,I

T5|.|Re-verify model code-gen (`show_code`, model.jl generated `Expr`) after T3/T4 land, since both touch generated-model internals. FOLD IN: collapse `_likelihood` + `_weighted_likelihood` (currently ~90% duplicated) into one family-dispatched function, with `weights` defaulting to `ones(...)` so unweighted fit is just weighted-fit-with-1s (makes V14 a true structural guarantee, not a coincidence)|C6,V14

T6|.|BIG JOB: `TuringRegression` as StatsAPI/StatsBase `RegressionModel`. Posterior-based methods where a point-estimate API expects one; skip/error clearly where no sane mapping exists (e.g. `StatsModels.TableRegressionModel`-only methods). Add formula-schema tests|I

T7|.|Swap `src/comparison.jl` off ParetoSmooth (broken, already stripped from Project.toml) onto PSIS.jl + PosteriorStats.jl. `psis_loo` → PSIS.jl `psis` on reshaped loglik array; `loo_compare` → PosteriorStats.jl `compare`/`loo` API. Re-check return-type shape/fields callers rely on (`elpd`, etc), update tests|C7

T8|.|Investigate: drop MCMCChains for FlexiChains in param extraction/`summary`. Currently `fit!` forces `chain_type=MCMCChains.Chains` (Turing 0.46 default is `FlexiChains.FlexiChain`) purely to keep every `.samples` access site (`name_map`, indexing) working w/o rewrite. Check whether native FlexiChains gives cleaner/faster param extraction (parametermethods.jl `draws`) + `summary`/`show` — its `._data`/`._metadata`/`._structures` layout may map onto `DimStack` output more directly than MCMCChains' AxisArray does. Scope: survey FlexiChains API, prototype one extraction path, compare against current before committing to a rewrite|C1,I

T9|.|Investigate: full-budget NUTS on sleepstudy RE model (`Reaction ~ 1 + Days + (1+Days\|Subject)`, N=2000, nchains maxed to `Threads.nthreads()`) much slower than equivalent brms/rstan fit, same model/data. Check before trusting T1 pass-2 timing as acceptable: (a) centered vs non-centered ranef parameterisation in generated `Expr` — centered is the classic NUTS-slow culprit for hierarchical models, (b) redundant recomputation in generated model code, (c) AD backend choice, (d) whether `MCMCThreads` actually parallelises across available threads on this machine. Don't treat T1 pass-2 as done if slowness is masking a perf bug rather than genuine sampling cost|T1,C6

## §NOTE

- P8: `TR.link` field (predict.jl) — used internally, V1-bounded to the 5 families, not user-facing.
