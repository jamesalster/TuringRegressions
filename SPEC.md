# SPEC — TuringRegressions.jl

Distilled code 2026-07-15. WIP package. `?` = unconfirmed, user verify.

## §G GOAL

Bayesian GLM package Julia. Alternative TuringGLM.jl, more features.
Fit regression via Turing.jl NUTS. Output = `DimArray` (DimensionalData) →
orderless named indexing params/draws/chains. User writes `@formula`, picks
family, calls `fit!`, extracts coefs / predicts / metrics / compares / plots.

Core loop: `turing_glm(formula, data, family)` → `fit!` → `summary`/`draws`/`predict`.

**Fit speed inside model = top priority.** Generated `@model` code stays minimal —
no unnecessary work in the NUTS hot path. Anything doable outside the model
(standardise/unstandardise, reshape, back-transform, loglik) lives OUTSIDE it.

## §C CONSTRAINTS

- C1. Julia. Turing.jl 0.36.3 MCMC. Sampler default NUTS, parallel MCMCThreads, samples_per_chain=500, nchains=4, warmup=500.
- C2. Data standardised OUTSIDE model now (`Transform`/`standardise`, moved out under T3 step 3 — see §I Types). User priors specified on std predictors (mean 0, sd 1) — stays this way, permanently, see R14. No original-scale prior input, no original-scale prior display either (both rejected, R14). T4 shipped clear std-scale labeling + `prior_summary(TR)` only.
- C3. Params stored/returned `TR.parameters::DimStack`. One layer per group (`:fixef`, per-ranef `:{group}`, `:{group}_sd`, `:{group}_corr`, `:{group}_offset`, `:internals`), layer's own dims incl `:iter, :chain` (FlexiChains names, `:iter` not `:draw` since T3). Chains collapsible to one `:iter` dim via `draws(...)`.
- C4. Families supported: `Normal, TDist, Bernoulli, Poisson, NegativeBinomial`. Others → `error`.
- C5. Links fixed per family (util `get_link`): Normal/TDist→identity, Bernoulli→logit, Poisson/NegBin→log.
- C6. Model code generated Julia `Expr` at runtime, `eval`'d into `@model turing_regression`. Not hand-written. `show_code=true` prints generated source 
- C7. Makie plots via native package extension (`ext/TuringRegressionsMakieExt.jl`, `[weakdeps]`/`[extensions]` in Project.toml). No `Requires`, no `Colors` dep.
- C9. Touch scaling logic (T3/T4) → re-verify all families against GLM (V3, V4, V13).
- C10. `Pkg.test()` full suite is heavy — full-budget NUTS runs can take 20-30+ min. When developing/testing a new feature, write a small targeted test/script exercising only that feature; don't run the full suite. Only run full `Pkg.test()` when the user explicitly asks for it.

## §I INTERFACES (public surface)

Model creation:
- `turing_glm(formula::FormulaTerm, data::DataFrame, family, priors=default_prior(family), weights=nothing, show_code=false)` → `TuringRegression{family}`
- `turing_glm(y::Vector, X::Array, ::Type{T}; names=Symbol[], kwargs...)` array form, synthesises formula, forwards.
- `default_prior(family)` / `default_prior(TR)` → `RegressionPrior` (intercept N(0,5), fixef N(0,2), random_effect_variance Exp(1), aux family-dep). All fields std-scale, permanently — T4/R14.

Fitting:
- `fit!(TR; sampler=NUTS(), parallel=MCMCThreads(), samples_per_chain=nothing, samples=nothing, nchains=4, warmup=500, quiet=true, kwargs...)` mutates TR (T16). Budget: give EITHER `samples_per_chain` (kept draws/chain) OR `samples` (total kept draws, split `samples ÷ nchains` per chain, must divide evenly) — both set → `ArgumentError`. Neither set → 500/chain. `warmup` (default 500) = extra draws sampled per chain on top of `per_chain`, discarded upstream by Turing (`nadapts=warmup`, `discard_initial=warmup`) — never reach `TR.samples`; `warmup=0` disables. Passes `per_chain` as `N` (AbstractMCMC's `N` already means kept draws; `discard_initial` adds `warmup` extra steps on top, total steps sampled = `per_chain + warmup`) to Turing `sample`. Recovers unstandardised params OUTSIDE the model: `DimArray(TR.samples)` → `reshape_params` → `unstandardise` (src/reshape.jl) → `TR.parameters` (DimStack). No in-model back-transform (T3).

Param extraction (`src/parametermethods.jl`):
- `draws(TR; drop_warmup=0, n_draws=Inf, collapse=true)` whole `TR.parameters` DimStack (all layers), warmup dropped/chains collapsed per kwargs.
- `draws(TR, type::Symbol; drop_warmup=0, n_draws=Inf, collapse=true)` single layer DimArray. `type` must be one of `propertynames(TR.parameters)` else `ArgumentError`. Valid types: `:fixef`, `:{group}` (ranef effect×group), `:{group}_sd`, `:{group}_corr` (only if correlated ranef), `:{group}_offset` (only slope-only-no-intercept ranef).
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
- `Base.show(io, TR; warnings=true)` family/formula/prior/obs/samples + warnings. Prior block labeled "Prior (standardised scale):" (T4).
- `prior_summary(io, TR)` / `prior_summary(TR)` — just the prior block `show` prints, no formula/obs/samples/warnings. Shares `_print_prior` helper with `show` (turingregression.jl, T4).
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
- `Transform` — affine-map constants, computed once per fit, stored on `TR.tf`. `?` field layout stale: code now nests `tf.y` as a `LinearTransform` (indexed `tf.y.mean[1]`/`tf.y.scale[1]`, comparison.jl) and gates y-scaling on `spec.scales_y` (family_spec), NOT flat `y_mean`/`y_std`/`scale_y`. Also holds `fixef::LinearTransform`, `ranef::Vector{LinearTransform}` (aligned with `ModelData.Z`). Code is oracle — T15 reconciles.
  - `compute_transform(md::ModelData, family)::Transform` — pure, means/stds from raw data.
  - `apply_transform(tf::Transform, md::ModelData)::ModelData` — applies given constants → scaled `ModelData`.
  - `standardise(md, family) = (apply_transform(tf,md), tf) where tf = compute_transform(md,family)`.
  - `unstandardise_data(md_std, tf)::ModelData` — inverse, used by round-trip guard (V21).
  - Back-transform of drawn params (not just data) lives in `unstandardise` (src/reshape.jl), reading `tf.*` — runs OUTSIDE the model, post-fit (T3).
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
- V10. Standardised→original param recovery lives in `unstandardise` (src/reshape.jl) via `_scale_effects`/`_center` helpers, reading `tf::Transform` (nested `tf.fixef`, `tf.y`, `tf.ranef` LinearTransforms). Runs OUTSIDE the model, post-fit (T3 done — no back-transform in generated `@model`). `?` exact algebra + Transform field names — code is oracle, this wording is stale; T15 reconciles.
- V11. `drop_warmup` default 0 (`draws()` and `summary()`) — `fit!` (T16) discards warmup upstream via `nadapts=warmup`/`discard_initial=warmup` (B1), so `TR.samples` never contains warmup draws. No auto-heuristic; downstream drop is user-override only.
- V12. `rhat>1.05` / `ess<100` / `mcse>5%std` → `@warn`; softer thresholds → `@info`.
- V13. Bernoulli param recovery vs GLM looser tolerance: coef atol 0.05, epred atol 0.05.
- V14. Weighted fit with `weights ≡ 1` == unweighted fit (params equal within tolerance, SD-normalized comparison — `_weighted_likelihood`'s per-obs `@addlogprob!` loop takes different numeric NUTS path than `_likelihood`'s vectorized logpdf, not bit-identical).
- V15. `posterior_predict(TR, new_data::DataFrame)` uses raw new X, original-scale stored β (no re-standardisation of new X). `epred` on new data ≈ `GLM.predict` on same new data. Guards against re-introducing standardisation in the predict path.
- V16. `extract_random_effect` predictor slicing conditions on `has_intercept(term.lhs)` — `(0+x...|g)` (no-intercept ranef terms) sliced differently from intercept-bearing ranef terms. Fixed 2026-07-15, guarded by dedicated tests.
- V17. Test suite runs via `Pkg.test()`, not direct `julia --project=. test/runtests.jl` — `Pkg.test()`'s isolated temp env is the only one that both (a) resolves deps fresh from `[extras]`/`[compat]` and (b) exercises the package's real symbol table end-to-end. Requires `[compat]` pinned on every Turing-stack package whose version drift can silently break internals (`Turing`, `DynamicPPL`, `FlexiChains`, `MCMCChains` — currently 0.46/0.42/0.6/7), so the temp env's independent resolve can't drift to an incompatible combo.
- V19. Per-observation log-likelihood is computed by the standalone `pointwise_loglik(TR)` (`src/comparison.jl`), NOT inside the model — model.jl's likelihood uses `Turing.@addlogprob!` (not `y[n] ~ Dist(...)`) so no observed VarNames exist for DynamicPPL to track. `pointwise_loglik` re-evaluates per-obs logpdf post-fit (reuses `linpred`, undoes y-standardisation to the scale sampling ran on); `psis_loo`/`loo_compare` call it. Returns `Array{Float64,3}` shape `(iter, chain, obs)`. (T3 moved this out of the deleted `_generated_quantities`.)
- V20. `turing_glm` builds its model via `cached_construct_model` (`src/model_cache.jl`), not `construct_model` directly. `MODEL_CACHE::Dict{Any,Tuple{Function,Expr}}` + `ReentrantLock`, keyed PURELY STRUCTURALLY: `family` + `ModelData`'s 4 accessor bools + per-ranef `(variable, has_intercept, has_fixed_effects, n_predictors)`. Priors are NOT part of the key — they're runtime model args (`prior_intercept` etc), so distinct priors on an otherwise-identical shape share one cache entry/gensym'd model. Identical spec fit repeatedly reuses the same model function (skips ~25s recompile per T13 finding).
- V21. Round-trip: `unstandardise_data(apply_transform(tf,md), tf) ≈ md` holds on `.predictors.X`, each `.Z[i].predictors.X`, and `.y`, across all 5 families × 3 ranef shapes (fixef-only, ranef intercept+slope, ranef slope-only).
- V22. `posterior_predict`/`predict` never re-standardise: `TR.modeldata` stays RAW, and any transient `md_std` built during `fit!` is local to that call. New-data predict paths must feed raw X straight through — this is a live guard against re-introducing standardisation into predict (see V15).
- V23. Random-effect components (`ranef_matrix` in `_random_effects`, model.jl) must be mean-zero by construction — no free parameter may act as an extra mean shift, since that would be additively confounded with the population-level fixed effect covering the same predictor (only the sum is identified, producing a NUTS ridge and biased marginals — B4). Population `α`/`β` are the only mean-carrying params; ranef branches only add zero-mean deviations (`diagm(σ)*L*z_raw` or `σ.*z_raw`).
- V24. Post-`fit!`, `size(TR.samples, 1) == per_chain` exactly (the requested `samples_per_chain`, or `samples ÷ nchains`) regardless of `warmup` — warmup draws never land in `TR.samples` (B1).

## §R RESEARCH

id|topic|finding|source
R11|T13 gensym|confirmed NOT a mistaken fix, by removing it in a scratch repro (fixed model name instead of `gensym`): 8 sequential fits varying family/predictor-count/weighted all fine, but refitting a ranef model (intercept+slope) after two *other* ranef shapes (intercept-only, slope-only) reused the name → `KeyError: :α_z_Subject not found`. Same class of bug as the BoundsError the original comment describes (shared `typeof(model)` across structurally different generated code corrupts Turing/DynamicPPL/AD internal caches), different symptom. Confirms per-shape uniqueness is required — the fix to chase is caching by structural key, not removing gensym|local scratch (`test_gensym.jl`, `test_gensym2.jl`), commit 9e14183 (`fname = gensym(...)` introduced same commit as an unrelated `Union{...,Nothing}` type-annotation fix, hence suspicion it was a coincidental band-aid — it isn't)
R12|T13 cache-key scope|traced every construct-time-only (not runtime-arg) input baked into the generated `Expr` in src/model.jl: family (branch dispatch), `ModelInfo`'s 4 bools (branch structure), and per-ranef (in `model_ranef` order) — `variable` name (embedded in generated symbols e.g. `α_z_Subject`), `has_intercept`/`has_fixed_effects` (branch choice), and when `has_fixed_effects`, the literal `size(ranef.predictors,2)` interpolated directly into code (e.g. `LKJCholesky($n_predictors,2.0)`, model.jl:38-41) — two ranefs with identical flags but different slope counts are NOT cache-equivalent|local, src/model.jl:19-120
R14|T4, prior scale|Investigated forward-transforming user-given original-scale priors into std-scale via `Distributions.AffineDistribution`/`shift+scale*d` (collapses to native type for location-scale families — e.g. `c+s*Normal`→`Normal`, `s*Exponential`→`Exponential` — zero NUTS-hot-path cost, collapse happens outside model at construction). Blocked: `fixed_effects`/`random_effects` priors are ONE shared `Distribution` replicated via `filldist(prior,n)` across predictors (model.jl); each predictor has a different original-scale sd, so one original-scale prior maps to a DIFFERENT std-scale dist per predictor — `filldist` can't express that, needs `arraydist(vector-of-n-distinct-dists)` instead, a model.jl structural change. Also intercept's true std-scale value depends on fixed-effect coefficients too (X-centering cross-term) — pure affine transform of intercept alone ignores that cross-term, exact only when fixed-effect prior means are 0. Compared to brms: brms does NOT auto-standardise predictors, so never faces this bridging problem — priors apply directly to whatever scale raw data is in; user standardises data themselves first if they want std-scale coefs. brms's `decomp="QR"` reparametrises internally for sampling efficiency but that's hidden algebra, priors still stated in original coefficient units throughout — not a user-facing scale split like ours. DECISION: keep priors std-scale as input (matches brms's implicit assumption — priors specified on the scale coefficients actually get sampled at); T4 narrowed to original-scale DISPLAY only (read-only, reuses existing `unstandardise` back-transform factors). Revisit arraydist approach only if real user need for original-scale prior INPUT emerges. FOLLOW-UP (same day): display compromise cut too — decided clear std-scale labeling in `show`/`summary` + a dedicated `prior_summary(TR)` print function was enough; no computed original-scale numbers shipped at all. `implied_original_prior` (transform.jl) and its formatter helper were written then deleted as dead code once the display call site was removed|conversation 2026-07-17
R13|T13 cache-key scope, priors|priors are NOT passed as runtime args to the compiled model — `_intercept`/`_fixed_effects`/`_random_effects`/`_auxiliary_parameter` all do `$prior` interpolation (model.jl:7,14,40 etc), baking the actual `Distribution` object into the generated code as a literal. Confirmed `_build_model_with_data` (turingregression.jl:215-238) never passes `TR.prior` to `TR.model(...)`. So a structural-only cache key would silently reuse stale prior values across calls with different priors — cache key MUST also cover the 4 `RegressionPrior` distributions (e.g. `(typeof(d), Distributions.params(d))` per distribution). Considered refactoring priors to runtime args instead (would let the key drop priors, and lines up with T4's plan to transform priors outside the model) but rejected for T13 — bigger diff than a memoization layer, and T4 already owns that rewrite|local, src/model.jl, src/turingregression.jl:215-238

## §B BUGS

id|date|cause|fix
B1|2026-07-17|`fit!` passed `N=per_chain+warmup` to Turing `sample()`, assuming `N` meant total draws with warmup subtracted via `discard_initial`/`nadapts`. AbstractMCMC's `N` already means KEPT draws — `discard_initial` adds warmup on top, not subtracted from `N`. Caught by scratch test asserting `size(mod.samples,1)==samples_per_chain` after fit, which returned per_chain+warmup instead|V11 (pass `per_chain` as `N`, `discard_initial=warmup`)

## §T TASKS

T16|x|`fit!` budget API (independent of T4/T5, src/turingregression.jl:219-233 + docstring 188-201). Rename `N`→`samples_per_chain` (default `nothing`, resolves 500); add `samples=nothing` (total kept, `samples ÷ nchains` per chain when set, error if not evenly divisible; `ArgumentError` if both `samples_per_chain` and `samples` set) and `warmup=500`. Pass `per_chain` as `N` (already means kept draws to AbstractMCMC), `discard_initial=warmup` adds warmup steps on top — B1 caught `N=per_chain+warmup` double-counting warmup, fixed. Docstring: state warmup discarded upstream via Turing's `nadapts`/`discard_initial`, `warmup=0` to disable. Downstream `drop_warmup` default now 0 (draws()/summary()) — warmup discarded upstream by Turing, not kept in chain|I,V11

T5|.|Re-verify model code-gen (`show_code`, model.jl generated `Expr`) after T3/T4 land, since both touch generated-model internals. FOLD IN: collapse `_likelihood` + `_weighted_likelihood` (currently ~90% duplicated) into one family-dispatched function, with `weights` defaulting to `ones(...)` so unweighted fit is just weighted-fit-with-1s (makes V14 a true structural guarantee, not a coincidence)|C6,V14

T14|.|First `using TuringRegressions` / first `Pkg.test()` pays full TTFX (Turing/DynamicPPL/model-macro compile) — separate latency source from the now-fixed T13 gensym-per-call issue (V20). Investigate `PrecompileTools.@compile_workload` block (small `turing_glm` fit, `N`/`nchains` minimal) in `src/TuringRegressions.jl` to precompile the hot path ahead of time. Open question: does `PrecompileTools` even help here given `construct_model`'s `eval`'d model type is built at runtime (gensym'd name) not load time — precompile workload would warm the *builder* machinery (Turing/DynamicPPL internals, AD dispatch) but not produce a reusable compiled model type itself. Needs a think before implementing|V20

- P8: `TR.link` field (predict.jl) — used internally, V1-bounded to the 5 families, not user-facing.
- P9: revisit whether flat `DimArray` model return (T3 step 4) is even needed — check if NamedTuple-of-DimArray (no auto-stack, R9) would've been simpler/good enough before committing further code to the flat-label scheme.
