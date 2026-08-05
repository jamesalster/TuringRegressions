# SPEC — TuringRegressions.jl

WIP package. `?` = unconfirmed, user verify. Code is oracle where noted.

## §G GOAL

Bayesian GLM package for Julia. Alternative to TuringGLM.jl, more features.
Fit regression via Turing.jl NUTS. Output = `DimStack` (DimensionalData) for
orderless named indexing of params/draws/chains. User writes `@formula`, picks
family, calls `fit!`, then extracts coefs / predicts / metrics / compares / plots.

Core loop: `turing_glm(formula, data, family)` → `fit!` → `summary`/`draws`/`predict`.

Fit speed inside model = top priority. Generated `@model` code stays minimal —
no unnecessary work in the NUTS hot path. Anything doable outside the model
(standardise/unstandardise, reshape, back-transform, loglik) lives OUTSIDE it.

## §C CONSTRAINTS

- C1. Julia. Turing.jl 0.36.3 MCMC. Default sampler NUTS, parallel MCMCThreads, samples=2000, nchains=4, warmup=samples (defaults to 2000). samples/warmup are totals across chains; warmup is extra, on top of samples.
- C2. Data standardised OUTSIDE model. User priors specified on std-scale predictors (mean 0, sd 1) — permanent. No original-scale prior input or display; only clear std-scale labeling + `prior_summary(TR)`.
- C3. Params stored `TR.parameters::DimStack`. One layer per group (`:fixef`, per-ranef `:{group}`, `:{group}_sd`, `:{group}_corr`, `:{group}_offset`, `:internals`); each layer has dims incl `:iter, :chain` (FlexiChains names). Chains collapsible to one `:iter` via `draws(...)`.
- C4. Families: `Normal, TDist, Bernoulli, Poisson, NegativeBinomial`. Others → `error`.
- C5. Links fixed per family (`get_link`): Normal/TDist→identity, Bernoulli→logit, Poisson/NegBin→log.
- C6. Model code generated as Julia `Expr` at runtime, `eval`'d into `@model turing_regression`. Not hand-written. `show_code=true` prints generated source.
- C7. Makie plots via native package extension (`ext/TuringRegressionsMakieExt.jl`, `[weakdeps]`/`[extensions]` in Project.toml). No `Requires`, no `Colors` dep.
- C9. Touch scaling logic → re-verify all families vs GLM (V3, V4, V13).
- C10. Full `Pkg.test()` is heavy (20-30+ min NUTS). Developing/testing a feature → write a small targeted script for it, not the full suite. Run full `Pkg.test()` only when user explicitly asks.

## §I INTERFACES (public surface)

Model creation:
- `turing_glm(formula::FormulaTerm, data::DataFrame, family, priors=default_prior(family), weights=nothing, show_code=false)` → `TuringRegression{family}`
- `turing_glm(y::Vector, X::Array, ::Type{T}; names=Symbol[], kwargs...)` array form, synthesises formula, forwards.
- `default_prior(family)` / `default_prior(TR)` → `RegressionPrior` (intercept N(0,5), fixef N(0,2), random_effect_variance Exp(1), aux family-dep). All fields std-scale.
- `scaled_default_prior(family, md::ModelData)` → `RegressionPrior` (T21, branch `remove-predictor-standardisation`). Used as `turing_glm`'s actual default when caller omits `priors` (kwarg default is `nothing`, resolved post-`modeldata`). Per-column-rescaled `fixed_effects`/`random_effect_variance` for centered-only (unscaled) predictors — see V25.

Fitting:
- `fit!(TR; sampler=NUTS(), parallel=MCMCThreads(), samples=2000, nchains=4, warmup=samples, quiet=true, kwargs...)` mutates TR. Budget: `samples` (kept draws) and `warmup` (adaptation, discarded) are TOTALS across chains, each split over `nchains` via ceil (`per_chain=cld(samples,nchains)`, `warmup_per_chain=cld(warmup,nchains)`) — rounds UP so realised count never below request. `warmup` is IN ADDITION to `samples` (each chain runs warmup_per_chain discarded + per_chain kept), default = `samples`. Discarded upstream by Turing (`nadapts`/`discard_initial`), never reaches `TR.samples`; `warmup=0` disables. Passes `per_chain` as `N` to Turing `sample`. Recovers unstandardised params OUTSIDE the model: `DimArray(TR.samples)` → `reshape_params` → `unstandardise` (src/reshape.jl) → `TR.parameters`. No in-model back-transform.

Param extraction (`src/parametermethods.jl`):
- `draws(TR; drop_warmup=0, n_draws=Inf, collapse=true)` whole `TR.parameters` DimStack, warmup dropped / chains collapsed per kwargs.
- `draws(TR, type::Symbol; ...)` single layer DimArray. `type` ∈ `propertynames(TR.parameters)` else `ArgumentError`: `:fixef`, `:{group}`, `:{group}_sd`, `:{group}_corr` (correlated ranef only), `:{group}_offset` (slope-only-no-intercept ranef only).
- `draws(f::Function, TR, type::Symbol; dropdims=true, kwargs...)` apply reducer `f` over draw/chain dims.
- `outcome(TR)` → DimArray of y. `predictors(TR, type)` → DimArray of X. `outcome_as_distribution(TR)` → `UnivariateFinite`.

Predict (`src/predict.jl`, primary API):
- `posterior_predict(TR, X=TR.modeldata.predictors.X; type=:posterior, kwargs...)` / `posterior_predict(f::Function, TR, X; type, kwargs...)`
- `posterior_predict(TR, new_data::DataFrame, ...)` rebuilds X from formula, remaps ranef levels (`new_random_effects`, `allow_new_levels` kwarg).
- `type ∈ (:posterior, :epred, :linpred)`. Internal: `linpred` (Xβ+α [+Zu]), `epred` (invlink∘linpred), `posterior_pred` (adds family noise).
- `predict(TR, X; kwargs...)` (`src/statsapi.jl`): StatsAPI point estimate (posterior mean `epred`) — interop only, `posterior_predict` preferred.

Metrics (StatisticalMeasures.jl):
- `calculate_metrics(TR, metrics::Vector, fun=nothing; threshold=0.5, kwargs...)` epred-based. Bernoulli: AUC + pseudo_r2 special-cased, rest categorical.
- `default_metrics(TR, fun=nothing)` regression → `[rsq, rmse, mae]`; Bernoulli → `[accuracy, kappa, TPR, TNR, auc, pseudo_r2]`.
- `pseudo_r2(preds, y)` McFadden, exported.

Comparison (PSIS.jl + PosteriorStats.jl, `src/comparison.jl`):
- `psis_loo(TR; kwargs...)` → `PSISLOOResult`, kwargs forwarded to `PosteriorStats.loo`.
- `loo_compare(models::TuringRegression...; kwargs...)` / `loo_compare(models::AbstractVector{<:TuringRegression}; kwargs...)` → `ModelComparisonResult`, forwarded to `PosteriorStats.compare`.

Display (`src/summary.jl`, `src/turingregression.jl`):
- `Base.show(io, TR; warnings=true)` family/formula/prior/obs/samples + warnings. Prior block labeled "Prior (standardised scale):".
- `prior_summary(io, TR)` / `prior_summary(TR)` — just the prior block. Shares `_print_prior` helper with `show`.
- `Base.summary(io, TR; funs=[mean,std], quantiles=[0.025,0.975], return_table=false, drop_warmup=nothing, show_metrics=false, kwargs...)`. Fixef table + per-grouping-term Random Effects tables (SD always; Correlation matrix only if `{group}_corr` present). Metrics table opt-in via `show_metrics=true`.
- `model_warnings(TR)` — rhat/ess/mcse based `@warn`/`@info`.

Plots (Makie ext):
- `lineribbon`/`lineribbon!`, `conditional_dependency`, `pp_check_dens`, `pp_check_dens_overlay`, `pp_check_hist`. Stubs exported from main package, methods added by extension when Makie loaded. Call `posterior_predict` internally.

StatsAPI (`src/statsapi.jl`): `TuringRegression <: StatsAPI.RegressionModel`. Point estimates are posterior mean — `coef`/`coeftable`/`fitted`/`residuals`/`linearpredictor`/`predict` `@warn maxlog=1` that they collapse the posterior. `vcov`/`confint`/`stderror` use full posterior (no warning). `modelmatrix`/`vif`/`gvif` error on ranef models. No-Bayesian-analogue and no-single-MLE methods raise `ArgumentError` pointing at `psis_loo`/`loo_compare`. Names imported+re-exported via `@reexport import StatsAPI: ...` — must be `import` not `using` (adding a method needs `import`; `using`-only silently defines a fresh local generic).

Types:
- `Predictors` — `has_intercept::Bool`, `X::AbstractMatrix`, `X_names::Union{Nothing,Vector{String}}`. Shared shape for fixef and each ranef term. `has_fixed_effects(p) = size(p.X,2)>0` (function, not field).
- `RandomEffect` — `variable::Symbol`, `levels::Vector`, `level_index::Vector{Int}`, `predictors::Predictors`.
- `ModelData` — `f::FormulaTerm`, `y::AbstractVector`, `predictors::Predictors` (fixef), `Z::Vector{RandomEffect}` (empty ⇒ no ranef), `weights::Union{Nothing,Vector{Float64}}`. Always RAW/unstandardised; reused for predict's new-data path. Accessors dispatch over it and `TuringRegression`: `has_intercept`/`has_fixed_effects` (from `.predictors`), `has_random_effects(md)=!isempty(md.Z)`, `is_weighted(md)=!isnothing(md.weights)`.
- `LinearTransform` (`src/transform.jl`) — `means::Vector{Float64}`, `stds::Vector{Float64}`. One per `Predictors`; empty vectors when predictor set empty.
- `Transform` — affine-map constants, computed once per fit, stored on `TR.tf`. Holds `fixef::LinearTransform`, `ranef::Vector{LinearTransform}` (aligned with `ModelData.Z`), and `y` as a `LinearTransform` (`tf.y.mean[1]`/`tf.y.scale[1]`); y-scaling gated on `spec.scales_y`. `?` exact field layout — code is oracle.
  - `compute_transform(md::ModelData, family)::Transform` — pure, means/stds from raw data.
  - `apply_transform(tf::Transform, md::ModelData)::ModelData` — applies constants → scaled `ModelData`.
  - `standardise(md, family) = (apply_transform(tf,md), tf) where tf = compute_transform(md,family)`.
  - `unstandardise_data(md_std, tf)::ModelData` — inverse, used by round-trip guard (V21).
  - Back-transform of drawn params lives in `unstandardise` (src/reshape.jl), reading `tf.*`, OUTSIDE the model, post-fit.
- `TuringRegression{T<:Distribution}` — `formula, model, prior, link, modeldata::ModelData, tf::Transform, modelcode, samples, parameters`. `modeldata` always RAW (V21).
- `RegressionPrior` — `intercept`/`fixed_effects`/`random_effects`/`auxiliary`. Passed as runtime model args, not baked into the generated `Expr` — keeps `cached_construct_model`'s cache key purely structural (V20). `fixed_effects`/`random_effect_variance` typed `PriorSpec = Union{Distribution, AbstractVector{<:Distribution}}` (T21) — scalar (old, one dist for all cols) or per-col vector (new).

## §V INVARIANTS

- V1. Family restricted to `{Normal, TDist, Bernoulli, Poisson, NegativeBinomial}`; others → `error` at `turing_glm`.
- V2. `draws(TR, type)` with invalid `type` → `ArgumentError`.
- V3. Fixef point estimates (posterior mean) vs GLM MLE within tolerance — Normal/Poisson/NegBin atol 0.025, Bernoulli looser (V13).
- V4. `epred` posterior mean vs `GLM.predict` within tolerance (family-dependent) for Normal/Poisson/NegBin/Bernoulli.
- V5. Link relations: `linpred == log(epred)` for log-link families; `linpred == epred` for identity-link.
- V6. `var(posterior_pred) > var(epred)` — posterior predictive adds observation noise (also count families).
- V7. `draws(TR, type)` shape `(n_effect_dims..., (N-drop_warmup)*nchains)` when collapsed; extra trailing `:chain` dim if `collapse=false`.
- V8. `n_draws` beyond available post-warmup draws → `AssertionError`.
- V9. Returned param labels `[:α, :<X_names...>, aux...]`; `:α` always present for intercept models.
- V10. Standardised→original param recovery lives in `unstandardise` (src/reshape.jl) via `_scale_effects`/`_center`, reading `tf::Transform` (nested `tf.fixef`, `tf.y`, `tf.ranef`). Runs OUTSIDE the model, post-fit. `?` exact algebra + field names — code is oracle.
- V11. `drop_warmup` default 0 (`draws()`/`summary()`) — `fit!` discards warmup upstream via `nadapts`/`discard_initial`, so `TR.samples` never contains warmup. Downstream drop is user-override only.
- V12. `rhat>1.05` / `ess<100` / `mcse>5%std` → `@warn`; softer → `@info`.
- V13. Bernoulli param recovery vs GLM: coef atol 0.05, epred atol 0.05.
- V14. Weighted fit with `weights ≡ 1` == unweighted fit (params equal within tolerance, SD-normalized — different numeric NUTS path, not bit-identical).
- V15. `posterior_predict(TR, new_data::DataFrame)` uses raw new X, original-scale stored β (no re-standardisation). `epred` on new data ≈ `GLM.predict` on same new data.
- V16. `extract_random_effect` predictor slicing conditions on `has_intercept(term.lhs)` — `(0+x...|g)` sliced differently from intercept-bearing ranef terms. Guarded by dedicated tests.
- V17. Test suite runs via `Pkg.test()`, not direct `julia --project=. test/runtests.jl` — its isolated temp env both resolves deps fresh from `[extras]`/`[compat]` and exercises the real symbol table end-to-end. Requires `[compat]` pinned on every Turing-stack package that can drift (`Turing`, `DynamicPPL`, `FlexiChains`, `MCMCChains` — currently 0.46/0.42/0.6/7).
- V19. Per-observation log-likelihood computed by standalone `pointwise_loglik(TR)` (`src/comparison.jl`), NOT inside the model — likelihood uses `@addlogprob!` (not `y[n] ~ Dist`) so no observed VarNames exist for DynamicPPL. Re-evaluates per-obs logpdf post-fit (reuses `linpred`, undoes y-standardisation). `psis_loo`/`loo_compare` call it. Returns `Array{Float64,3}` shape `(iter, chain, obs)`.
- V20. `turing_glm` builds via `cached_construct_model` (`src/model_cache.jl`). `MODEL_CACHE::Dict{Any,Tuple{Function,Expr}}` + `ReentrantLock`, keyed PURELY STRUCTURALLY: `family` + `ModelData`'s 4 accessor bools + per-ranef `(variable, has_intercept, has_fixed_effects, n_predictors)`. Priors NOT in key (runtime args). Identical spec refit reuses same model function (skips ~25s recompile).
- V21. Round-trip: `unstandardise_data(apply_transform(tf,md), tf) ≈ md` on `.predictors.X`, each `.Z[i].predictors.X`, and `.y`, across all 5 families × 3 ranef shapes.
- V22. `posterior_predict`/`predict` never re-standardise: `TR.modeldata` stays RAW; transient `md_std` during `fit!` is local. New-data predict feeds raw X straight through.
- V23. Ranef components (`ranef_matrix` in `_random_effects`, model.jl) must be mean-zero by construction — no free param acts as extra mean shift (would be confounded with population fixed effect over same predictor → NUTS ridge, biased marginals). Population `α`/`β` are the only mean-carrying params; ranef branches add zero-mean deviations only (`diagm(σ)*L*z_raw` or `σ.*z_raw`).
- V24. Post-`fit!`, `size(TR.samples, 1) == cld(samples, nchains)` exactly, regardless of `warmup`. Inexact division rounds up (realised total ≥ requested `samples`).
- V25. (T21, branch `remove-predictor-standardisation`, not yet on main) `compute_transform` sets `fixef.scale`/`ranef[i].scale` to all-ones in place (fit normal `ZScoreTransform` then `.scale .= 1.0` — NOT `scale=false`, that gives empty vector and breaks `_unstandardise_fixef`/`_unstandardise_ranef` division). Centering kept, scaling dropped, both fixef+ranef. Model code's `_dist_for(prior, n)` dispatches `filldist` (scalar `Distribution`) vs `arraydist` (`Vector{<:Distribution}`) so `_fixed_effects`/`_random_effects` (model.jl) transparently take either. `scaled_default_prior` rescales per-column by raw sd (`Normal(μ,σ/sd)`, `Exponential(θ/sd)`); ranef variance rescaling only when `length(md.Z)==1` (matches T20 scope), else falls back to plain scalar default (doc'd limitation, not silent-wrong). `unstandardise` (V10) needs no change — algebra verified: `β_model = β_raw/sd_y` (X centered-only), back-transform `β_orig = β_model * y_scale / 1.0` still correct.

## §B BUGS

id|date|cause|fix

## §T TASKS

T14|.|First `using TuringRegressions` / `Pkg.test()` pays full TTFX (Turing/DynamicPPL/model-macro compile) — separate latency source from the fixed T13 gensym issue (V20). Investigate `PrecompileTools.@compile_workload` block (small `turing_glm` fit, minimal `N`/`nchains`) in `src/TuringRegressions.jl`. Open question: does it help given `construct_model`'s model type is `eval`'d at runtime (gensym'd name), not load time — workload warms builder machinery (Turing/DynamicPPL/AD) but not a reusable compiled model type. Needs a think before implementing|V20
T17|.|`TR.link` field (predict.jl) — used internally, V1-bounded to the 5 families, not user-facing. Confirm it can stay internal / no action needed|V1
T19|.|`fit!(quiet=false)` live progress bar not showing in VSCode Julia REPL. Likely needs a progress-capable logger (TerminalLoggers) or VSCode's ProgressLogging integration. Low priority — sort later|
T20|x|Full Pkg.test() too slow for dev iteration. Add small subset: one Normal fit, one Bernoulli fit, mixedmodels benchmark fit only — reusable runner for T21/T22 dev loop|C10
T21|.|Branch `remove-predictor-standardisation`: drop predictor standardisation (C2) inside model, keep centering only, for BOTH fixef+ranef. Rescaled priors via `scaled_default_prior` (V25), not plain `default_prior` reuse — implemented, parse-checked, not yet run/tested. Still to do: T20 subset run vs main baseline (fit time + accuracy, V3/V4/V13 tolerances)|C2,V3,V4,V13,V25,T20
T22|.|Post-compile NUTS perf on mixedmodels-benchmark fit v slow vs Stan/lme4 — profile + improve. Use T20 subset to iterate|T20
