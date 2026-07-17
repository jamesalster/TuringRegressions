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
- C2. Data standardised OUTSIDE model now (`Transform`/`standardise`, moved out under T3 step 3 — see §I Types). User priors still scaled to std predictors (mean 0, sd 1) — **T4 flips this to original scale** (not started).
- C3. Params stored/returned `TR.parameters::DimStack`. One layer per group (`:fixef`, per-ranef `:{group}`, `:{group}_sd`, `:{group}_corr`, `:{group}_offset`, `:internals`), layer's own dims incl `:draw, :chain`. Chains collapsible to one `:draw` dim via `draws(...)`.
- C4. Families supported: `Normal, TDist, Bernoulli, Poisson, NegativeBinomial`. Others → `error`.
- C5. Links fixed per family (util `get_link`): Normal/TDist→identity, Bernoulli→logit, Poisson/NegBin→log.
- C6. Model code generated Julia `Expr` at runtime, `eval`'d into `@model turing_regression`. Not hand-written. `show_code=true` prints generated source — **T5 re-verifies this path.**
- C7. Makie plots via native package extension (`ext/TuringRegressionsMakieExt.jl`, `[weakdeps]`/`[extensions]` in Project.toml). No `Requires`, no `Colors` dep.
- C8. NegBin canonical form: local `NegativeBinomial2(μ, mean/dispersion)`, `clamp(1/(1+μ/ϕ), 1e-6, 1-1e-6)` (both bounds — see V18/B2). No TuringGLM dependency (removed).
- C9. Touch scaling logic (T3/T4) → re-verify all families against GLM (V3, V4, V13).
- C10. `Pkg.test()` full suite is heavy — full-budget NUTS runs can take 20-30+ min. When developing/testing a new feature, write a small targeted test/script exercising only that feature; don't run the full suite. Only run full `Pkg.test()` when the user explicitly asks for it.

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

Predict (`src/predict.jl`, real working API, primary/richest — renamed from `predict` under T6):
- `posterior_predict(TR, X=TR.modeldata.predictors.X; type=:posterior, kwargs...)` / `posterior_predict(f::Function, TR, X=TR.modeldata.predictors.X; type, kwargs...)`
- `posterior_predict(TR, new_data::DataFrame, ...)` rebuilds X from formula, remaps random-effect levels (`new_random_effects`, `allow_new_levels` kwarg).
- `type ∈ (:posterior, :epred, :linpred)`.
- Internal: `linpred` (Xβ+α [+Zu]), `epred` (invlink∘linpred), `posterior_pred` (adds family noise).
- `predict(TR, X=TR.modeldata.predictors.X; kwargs...)` (`src/statsapi.jl`, T6): StatsAPI-conformant point estimate (posterior mean `epred`) — interop only, `posterior_predict` is preferred.

Metrics (StatisticalMeasures.jl):
- `calculate_metrics(TR, metrics::Vector, fun=nothing; threshold=0.5, kwargs...)` epred-based. Bernoulli branch: AUC + pseudo_r2 special-cased, rest categorical.
- `default_metrics(TR, fun=nothing)` regression → `[rsq, rmse, mae]`; Bernoulli → `[accuracy, kappa, TPR, TNR, auc, pseudo_r2]`.
- `pseudo_r2(preds, y)` McFadden, exported.

Comparison (PSIS.jl + PosteriorStats.jl, `src/comparison.jl`, T7):
- `psis_loo(TR; kwargs...)` → `PosteriorStats.PSISLOOResult`, kwargs forwarded to `PosteriorStats.loo`.
- `loo_compare(models::TuringRegression...; kwargs...)` / `loo_compare(models::AbstractVector{<:TuringRegression}; kwargs...)` → `PosteriorStats.ModelComparisonResult`, kwargs forwarded to `PosteriorStats.compare`.

Display (`src/summary.jl`, `src/turingregression.jl`):
- `Base.show(io, TR; warnings=true)` family/formula/prior/obs/samples + warnings.
- `Base.summary(io, TR; funs=[mean,std], quantiles=[0.025,0.975], return_table=false, drop_warmup=nothing, show_metrics=false, kwargs...)`. Prints fixef table + per-grouping-term Random Effects tables (SD table always; Correlation matrix only if `{group}_corr` layer present). Metrics table opt-in via `show_metrics=true`.
- `model_warnings(TR)` — rhat/ess/mcse based `@warn`/`@info`.

Plots (Makie ext, `ext/TuringRegressionsMakieExt.jl`):
- `lineribbon`/`lineribbon!`, `conditional_dependency`, `pp_check_dens`, `pp_check_dens_overlay`, `pp_check_hist`. Stubs exported from main package, methods added by extension when Makie loaded. Calls `posterior_predict` internally (renamed under T6 — a rename here is easy to miss since this file lives in `ext/`, not `src/`).

StatsAPI (`src/statsapi.jl`, T6): `TuringRegression <: StatsAPI.RegressionModel`. Point estimates are posterior mean — `coef`/`coeftable`/`fitted`/`residuals`/`linearpredictor`/`predict` `@warn maxlog=1` once each that they collapse the posterior. `vcov`/`confint`/`stderror` use the full posterior (no warning). `modelmatrix`/`vif`/`gvif` error on random-effects models (no way to represent Z structure). No-Bayesian-analogue (`score`, `informationmatrix`, `leverage`, `cooksdistance`, `reconstruct`/`reconstruct!`, `predict!`) and no-single-MLE-value (`loglikelihood`, `dof`, `mss`, `rss`, `nulldeviance`, `nullloglikelihood`, `aic`, `aicc`, `bic`, `r2`, `adjr2`) methods raise `ArgumentError` pointing at `psis_loo`/`loo_compare`. All names imported+re-exported via `@reexport import StatsAPI: ...` in `TuringRegressions.jl` — must be `import` not `using` (StatsAPI marks these `public` not `export`, and `using Mod: f` alone doesn't let you add a method to `f`, only `import Mod: f` does — `using`-only silently defines a fresh unrelated local `f` instead of extending the real generic, which is a `MethodError`-producing footgun, not a load error).

Types:
- `Predictors` — `has_intercept::Bool`, `X::AbstractMatrix`, `X_names::Union{Nothing,Vector{String}}`. Shared shape for fixef and each ranef term. `has_fixed_effects(p) = size(p.X,2)>0` — function, not field (multiple-dispatch accessor, see below).
- `RandomEffect` — `variable::Symbol`, `levels::Vector`, `level_index::Vector{Int}`, `predictors::Predictors`.
- `ModelData` — `f::FormulaTerm`, `y::AbstractVector`, `predictors::Predictors` (fixef), `Z::Vector{RandomEffect}` (empty ⇒ no ranef), `weights::Union{Nothing,Vector{Float64}}`. Always RAW/unstandardised; same struct reused for `predict`'s new-data path. 4 structural accessors dispatch over it (and over `TuringRegression`, so callers don't care which they hold): `has_intercept`/`has_fixed_effects(md)` (from `md.predictors`), `has_random_effects(md) = !isempty(md.Z)`, `is_weighted(md) = !isnothing(md.weights)`.
- `LinearTransform` (`src/transform.jl`) — `means::Vector{Float64}`, `stds::Vector{Float64}`. One per `Predictors` (fixef + each ranef); empty vectors when that predictor set is empty.
- `Transform` — `fixef::LinearTransform`, `y_mean::Float64`, `y_std::Float64`, `scale_y::Bool` (`family ∈ {Normal,TDist}`), `ranef::Vector{LinearTransform}` (aligned with `ModelData.Z`). The affine map's constants, computed once per fit, stored on `TR.tf`.
  - `compute_transform(md::ModelData, family)::Transform` — pure, means/stds from raw data.
  - `apply_transform(tf::Transform, md::ModelData)::ModelData` — applies given constants → scaled `ModelData`.
  - `standardise(md, family) = (apply_transform(tf,md), tf) where tf = compute_transform(md,family)`.
  - `unstandardise_data(md_std, tf)::ModelData` — inverse, used by round-trip guard (V21).
  - Back-transform of drawn params (not just data) lives inside `_generated_quantities` (model.jl), reading `tf.*`.
- `TuringRegression{T<:Distribution}` — `formula, model, prior, link, modeldata::ModelData, tf::Transform, modelcode, samples, parameters`. `modeldata` always RAW (V21).
- `RegressionPrior` — `intercept`/`fixed_effects`/`random_effects`/`auxiliary`. Passed as runtime model args (`prior_intercept` etc), not baked into the generated `Expr` — keeps `cached_construct_model`'s cache key purely structural (V20).

## §V INVARIANTS

- V1. Family restricted to `{Normal, TDist, Bernoulli, Poisson, NegativeBinomial}`; others → `error` at `turing_glm` construction.
- V2. `draws(TR, type)` with invalid `type` → `ArgumentError`.
- V3. Fixef point estimates (posterior mean) vs GLM MLE within tolerance — Normal/Poisson/NegBin atol 0.025 (runtests), Bernoulli looser (see V13).
- V4. `epred` posterior mean vs `GLM.predict` within tolerance (family-dependent, see runtests) for Normal/Poisson/NegBin/Bernoulli.
- V5. Link relations: `linpred == log(epred)` for log-link families; `linpred == epred` for identity-link families.
- V6. `var(posterior_pred) > var(epred)` — posterior predictive adds observation noise (also holds for count families).
- V7. `draws(TR, type)` shape `(n_effect_dims..., (N-drop_warmup)*nchains)` when collapsed; extra trailing `:chain` dim if `collapse=false`.
- V8. `n_draws` requested beyond available post-warmup draws → `AssertionError`.
- V9. Returned param labels: `[:α, :<X_names...>, aux...]`; `:α` always present for models with intercept.
- V10. Standardised→original param recovery in `_generated_quantities`, reading `tf::Transform`: `β_orig = (tf.y_std./tf.fixef.stds).*β` (Gaussian family) or `β./tf.fixef.stds` (count/binary); `α` likewise via `dot(tf.fixef.means, β_orig)`. Scaling constants come from `TR.tf`, not recomputed inside the model. Back-transform still runs inside the generated `@model` (extraction fully out to Julia is `T3` step 4, not started).
- V11. `summary` auto `drop_warmup`: 0 if `N<400` else 200.
- V12. `rhat>1.05` / `ess<100` / `mcse>5%std` → `@warn`; softer thresholds → `@info`.
- V13. Bernoulli param recovery vs GLM looser tolerance: coef atol 0.05, epred atol 0.05.
- V14. Weighted fit with `weights ≡ 1` == unweighted fit (params equal within tolerance, SD-normalized comparison — `_weighted_likelihood`'s per-obs `@addlogprob!` loop takes different numeric NUTS path than `_likelihood`'s vectorized logpdf, not bit-identical).
- V15. `posterior_predict(TR, new_data::DataFrame)` uses raw new X, original-scale stored β (no re-standardisation of new X). `epred` on new data ≈ `GLM.predict` on same new data. Guards against re-introducing standardisation in the predict path.
- V16. `extract_random_effect` predictor slicing conditions on `has_intercept(term.lhs)` — `(0+x...|g)` (no-intercept ranef terms) sliced differently from intercept-bearing ranef terms. Fixed 2026-07-15, guarded by dedicated tests.
- V17. Test suite runs via `Pkg.test()`, not direct `julia --project=. test/runtests.jl` — `Pkg.test()`'s isolated temp env is the only one that both (a) resolves deps fresh from `[extras]`/`[compat]` and (b) exercises the package's real symbol table end-to-end. Requires `[compat]` pinned on every Turing-stack package whose version drift can silently break internals (`Turing`, `DynamicPPL`, `FlexiChains`, `MCMCChains` — currently 0.46/0.42/0.6/7), so the temp env's independent resolve can't drift to an incompatible combo.
- V19. `_generated_quantities` (`src/model.jl`) return tuple always includes a `:loglik` key — per-observation log-likelihood vector, added under T7 for `psis_loo`/`loo_compare`. Computed post-hoc inside `_generated_quantities` rather than via DynamicPPL's normal VarName-based pointwise-loglik tracking, because `model.jl`'s likelihood is written with `Turing.@addlogprob!` (not `y[n] ~ Dist(...)`) — no observed VarNames exist for DynamicPPL to see. Any code calling `generated_quantities(model_with_data, TR.samples)` directly must filter/exclude `:loglik` before treating the returned keys as fitted parameter names — `src/turingregression.jl`'s fixef-layer param loop does this via `filter(!=(:loglik), ...)`.
- V20. `turing_glm` builds its model via `cached_construct_model` (`src/model_cache.jl`), not `construct_model` directly. `MODEL_CACHE::Dict{Any,Tuple{Function,Expr}}` + `ReentrantLock`, keyed PURELY STRUCTURALLY: `family` + `ModelData`'s 4 accessor bools + per-ranef `(variable, has_intercept, has_fixed_effects, n_predictors)`. Priors are NOT part of the key — they're runtime model args (`prior_intercept` etc), so distinct priors on an otherwise-identical shape share one cache entry/gensym'd model. Identical spec fit repeatedly reuses the same model function (skips ~25s recompile per T13 finding).
- V21. Round-trip: `unstandardise_data(apply_transform(tf,md), tf) ≈ md` holds on `.predictors.X`, each `.Z[i].predictors.X`, and `.y`, across all 5 families × 3 ranef shapes (fixef-only, ranef intercept+slope, ranef slope-only).
- V22. `posterior_predict`/`predict` never re-standardise: `TR.modeldata` stays RAW, and any transient `md_std` built during `fit!` is local to that call. New-data predict paths must feed raw X straight through — this is a live guard against re-introducing standardisation into predict (see V15).
- V23. Random-effect components (`ranef_matrix` in `_random_effects`, model.jl) must be mean-zero by construction — no free parameter may act as an extra mean shift, since that would be additively confounded with the population-level fixed effect covering the same predictor (only the sum is identified, producing a NUTS ridge and biased marginals — B4). Population `α`/`β` are the only mean-carrying params; ranef branches only add zero-mean deviations (`diagm(σ)*L*z_raw` or `σ.*z_raw`).

## §R RESEARCH

id|topic|finding|source
R1|FlexiChains|already Turing 0.46 default return, `chain_type=VNChain` explicit alt to MCMCChains.Chains|github.com/penelopeysm/FlexiChains.jl
R2|FlexiChains|indexing `chain[@varname(x)]` returns `DimMatrix` (DimensionalData) natively — no AxisArray/reshape step needed|github.com/penelopeysm/FlexiChains.jl/blob/main/README.md
R3|FlexiChains|no `name_map`; keyed by `VarName` not `Symbol` — structured names (`z[1]`) built into key, no separate lookup table|github.com/TuringLang/MCMCChains.jl/issues/335, FlexiChains README
R4|FlexiChains|CONFIRMED via scratch: `summarystats(chain)` returns `FlexiSummary`, no `.nt`. Index `ss[Parameter(@varname(x))]` → vector ordered per `ss._stat_indices == [:mean,:std,:mcse,:ess_bulk,:ess_tail,:rhat,:q5,:q50,:q95]`. `FlexiChains.rhat(chain)` also exists as standalone dispatch (src/summary.jl:484 in FlexiChains pkg). `:internals`-equivalent (MCMCChains `name_map[:internals]`): `keys(chain)` returns `Parameter{VarName}`/`Extra` union — sampler stats are `Extra(:n_steps)`, `Extra(:is_accept)`, `Extra(:acceptance_rate)`, `Extra(:log_density)`, `Extra(:hamiltonian_energy)`, `Extra(:tree_depth)`, etc, indexed via `chain[FlexiChains.Extra(:n_steps)]`|local scratch (flexichains_probe.jl/probe2.jl), FlexiChains 0.6, Turing 0.46
R5|FlexiChains|already compat-pinned `FlexiChains = "0.6"` in Project.toml, but only in `[extras]`/test target, not `[deps]` — real dep add needed for non-test use|Project.toml:39,48,55
R6|FlexiChains|ships `FlexiChainsMCMCChainsExt` conversion ext + `FlexiChainsDynamicPPLExt` + `FlexiChainsPosteriorStatsExt` — interop path exists both ways if partial switch wanted|Manifest.toml:826-839
R7|FlexiChains|`DynamicPPL.returned(model, chain::FlexiChain{<:VarName})` re-evaluates model per draw, returns `DimMatrix`/`DimArray` — if model's return value is itself a `DimArray`, dims auto-stack into result (draw/chain dims appended)|github.com/penelopeysm/FlexiChains.jl ext/FlexiChainsDynamicPPLExt.jl:438-451
R8|current code|extraction (src/turingregression.jl:280-310) manually loops `generated_quantities` NamedTuple output, stacks/vcats into per-layer DimArrays by hand — exactly what R7's auto-stack could replace, if model.jl's `@model` return stmt changes to return a `DimArray` instead of plain NamedTuple|local, src/turingregression.jl:280-310, src/model.jl (`_generated_quantities`)
R9|FlexiChains|CONFIRMED via scratch run (dims(r1) printed): single-`DimArray` model return -> `DynamicPPL.returned(model, chain)` auto-stacks into `DimArray` with dims literally named `:iter` (Sampled Int64 range), `:chain` (Sampled Int64 range), `:param` (Categorical Symbol) -- NOT `:draw`. `NamedTuple`-of-`DimArray` (needed for multi-layer `:fixef`/`:{group}`/`:{group}_sd`) does NOT auto-stack -- stays `DimMatrix` of NamedTuples, still needs manual per-layer split, just over smaller pre-labeled DimVectors. Installed FlexiChains 0.6 `VNChain` (= `FlexiChain{VarName}`): `returned(model, chain::VNChain)` does NOT accept a `stack` kwarg (MethodError if passed) -- auto-stacks implicitly, only warns "implicit stacking will be removed in a future release, default will change to false". No action needed now; re-check kwarg support on FlexiChains version bump|local scratch (flexichains_probe.jl), `~/.julia/packages/FlexiChains/Rb8fl/src/chain.jl:782` warning
R10|FlexiChains|`DimArray(chain)` (no `returned`/reevaluation, no `stack=true`) unwraps VNChain's raw *sampled* VarNames directly into stacked `(iter,chain,param)` DimArray. Only exposes raw sampled params (standardised-scale priors), not the model's derived/back-transformed return value — doesn't replace `returned()` for actual extraction need, but a free cheap path if raw param access ever wanted separately|local scratch run
R11|T13 gensym|confirmed NOT a mistaken fix, by removing it in a scratch repro (fixed model name instead of `gensym`): 8 sequential fits varying family/predictor-count/weighted all fine, but refitting a ranef model (intercept+slope) after two *other* ranef shapes (intercept-only, slope-only) reused the name → `KeyError: :α_z_Subject not found`. Same class of bug as the BoundsError the original comment describes (shared `typeof(model)` across structurally different generated code corrupts Turing/DynamicPPL/AD internal caches), different symptom. Confirms per-shape uniqueness is required — the fix to chase is caching by structural key, not removing gensym|local scratch (`test_gensym.jl`, `test_gensym2.jl`), commit 9e14183 (`fname = gensym(...)` introduced same commit as an unrelated `Union{...,Nothing}` type-annotation fix, hence suspicion it was a coincidental band-aid — it isn't)
R12|T13 cache-key scope|traced every construct-time-only (not runtime-arg) input baked into the generated `Expr` in src/model.jl: family (branch dispatch), `ModelInfo`'s 4 bools (branch structure), and per-ranef (in `model_ranef` order) — `variable` name (embedded in generated symbols e.g. `α_z_Subject`), `has_intercept`/`has_fixed_effects` (branch choice), and when `has_fixed_effects`, the literal `size(ranef.predictors,2)` interpolated directly into code (e.g. `LKJCholesky($n_predictors,2.0)`, model.jl:38-41) — two ranefs with identical flags but different slope counts are NOT cache-equivalent|local, src/model.jl:19-120
R13|T13 cache-key scope, priors|priors are NOT passed as runtime args to the compiled model — `_intercept`/`_fixed_effects`/`_random_effects`/`_auxiliary_parameter` all do `$prior` interpolation (model.jl:7,14,40 etc), baking the actual `Distribution` object into the generated code as a literal. Confirmed `_build_model_with_data` (turingregression.jl:215-238) never passes `TR.prior` to `TR.model(...)`. So a structural-only cache key would silently reuse stale prior values across calls with different priors — cache key MUST also cover the 4 `RegressionPrior` distributions (e.g. `(typeof(d), Distributions.params(d))` per distribution). Considered refactoring priors to runtime args instead (would let the key drop priors, and lines up with T4's plan to transform priors outside the model) but rejected for T13 — bigger diff than a memoization layer, and T4 already owns that rewrite|local, src/model.jl, src/turingregression.jl:215-238

## §B BUGS

id|date|cause|fix

## §T TASKS

T3|.|BIG JOB, merged w/ old T8 (co-dependent — both rewrite `_generated_quantities`/extraction, doing separately means touching same code twice). Data-flow structs (`ModelData`/`Predictors`/`RandomEffect`/`Transform`, standardise-outside-model, priors-as-args) are DONE — remaining scope is the FlexiChains + unscaling step, not started:
  drop MCMCChains for FlexiChains — model's `@model` return stmt becomes single flat `DimArray` (not NamedTuple), pulled via `DynamicPPL.returned(model, chain; stack=true)` which auto-stacks into one `(iter,chain,param)` DimArray (R9; NamedTuple-of-DimArray does NOT auto-stack, so don't return a layered NamedTuple). Replace today's manual per-layer Dict+stack+vcat loop over `generated_quantities` with this; back-transform (`unstandardise`, building a `DimStack`) applied AFTER unstacking, not inside the model — deletes `_generated_quantities` entirely. Promote FlexiChains `[extras]`→`[deps]` (R5). Swap `chain_type=MCMCChains.Chains`→`VNChain`, `summarize`→`summarystats` (confirm fields, R4), `.name_map`/`.value` indexing→`VarName`-keyed access (R2,R3).
  Once that lands, rest of API (`parametermethods.jl` `draws`, `summary.jl`, etc) should adopt FlexiChains' own conventions rather than forcing into today's names — e.g. `:iter` not `:draw`, FlexiChains' own dim ordering. Touches V7 (draws shape), V9 (param labels), C3 (layer/dim naming) — re-verify those invariants under new naming. `pointwise_loglik(TR)` becomes a standalone post-hoc helper (V19) called by `psis_loo`/`turingregression.jl`, no longer riding inside `_generated_quantities`'s NamedTuple.
  Re-verify V-invariants under Turing/NUTS after. Then T4 (prior flip) and T5 (likelihood merge, code-gen re-verify) proceed|C2,C6,C9,V3,V4,V7,V9,V10,V14,V19,R1-R10

T4|.|BIG JOB: flip prior scaling (depends on T3's centralised affine map). Today user specifies priors on standardised (mean 0, sd 1) scale. Change so priors are given on ORIGINAL data scale — e.g. `Normal(10,20)` on a predictor with mean 10, sd 20 → transformed to `Normal(0,1)` internally for the standardised fit — reported back on original scale in `summary`/`show`/prior display. Use `Distributions.AffineDistribution` (`shift + scale*d`) for the forward transform (prior) and its inverse for the param back-transform — don't hand-write new per-family algebra. Store user's original prior for display; fit on scaled. Re-verify V-invariants under Turing/NUTS with `AffineDistribution` priors|C2,I

T5|.|Re-verify model code-gen (`show_code`, model.jl generated `Expr`) after T3/T4 land, since both touch generated-model internals. FOLD IN: collapse `_likelihood` + `_weighted_likelihood` (currently ~90% duplicated) into one family-dispatched function, with `weights` defaulting to `ones(...)` so unweighted fit is just weighted-fit-with-1s (makes V14 a true structural guarantee, not a coincidence)|C6,V14

T14|.|First `using TuringRegressions` / first `Pkg.test()` pays full TTFX (Turing/DynamicPPL/model-macro compile) — separate latency source from the now-fixed T13 gensym-per-call issue (V20). Investigate `PrecompileTools.@compile_workload` block (small `turing_glm` fit, `N`/`nchains` minimal) in `src/TuringRegressions.jl` to precompile the hot path ahead of time. Open question: does `PrecompileTools` even help here given `construct_model`'s `eval`'d model type is built at runtime (gensym'd name) not load time — precompile workload would warm the *builder* machinery (Turing/DynamicPPL internals, AD dispatch) but not produce a reusable compiled model type itself. Needs a think before implementing|V20

- P8: `TR.link` field (predict.jl) — used internally, V1-bounded to the 5 families, not user-facing.
- P9: revisit whether flat `DimArray` model return (T3 step 4) is even needed — check if NamedTuple-of-DimArray (no auto-stack, R9) would've been simpler/good enough before committing further code to the flat-label scheme.
