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

Predict (`src/predict.jl`, real working API):
- `predict(TR, X=TR.X; type=:posterior, kwargs...)` / `predict(f::Function, TR, X=TR.X; type, kwargs...)`
- `predict(TR, new_data::DataFrame, ...)` rebuilds X from formula, remaps random-effect levels (`new_random_effects`, `allow_new_levels` kwarg).
- `type ∈ (:posterior, :epred, :linpred)`.
- Internal: `linpred` (Xβ+α [+Zu]), `epred` (invlink∘linpred), `posterior_pred` (adds family noise).

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
- V8. `n_draws` requested beyond available post-warmup draws → `AssertionError`.
- V9. Returned param labels: `[:α, :<X_names...>, aux...]`; `:α` always present for models with intercept.
- V10. Standardised→original param recovery in `_generated_quantities`: `β_orig = (y_std/X_stds).*β` (Gaussian family) or `β./X_stds` (count/binary); `α` likewise via `dot(X_means, β_orig)`. **Will move out of generated model under T3.**
- V11. `summary` auto `drop_warmup`: 0 if `N<400` else 200.
- V12. `rhat>1.05` / `ess<100` / `mcse>5%std` → `@warn`; softer thresholds → `@info`.
- V13. Bernoulli param recovery vs GLM looser tolerance: coef atol 0.05, epred atol 0.05.
- V14. Weighted fit with `weights ≡ 1` == unweighted fit (params equal within tolerance, SD-normalized comparison — `_weighted_likelihood`'s per-obs `@addlogprob!` loop takes different numeric NUTS path than `_likelihood`'s vectorized logpdf, not bit-identical).
- V15. `predict(TR, new_data::DataFrame)` uses raw new X, original-scale stored β (no re-standardisation of new X). `epred` on new data ≈ `GLM.predict` on same new data. Guards against re-introducing standardisation in the predict path.
- V16. `extract_random_effect` predictor slicing conditions on `has_intercept(term.lhs)` — `(0+x...|g)` (no-intercept ranef terms) sliced differently from intercept-bearing ranef terms. Fixed 2026-07-15, guarded by dedicated tests.
- V17. Test suite runs via `Pkg.test()`, not direct `julia --project=. test/runtests.jl` — `Pkg.test()`'s isolated temp env is the only one that both (a) resolves deps fresh from `[extras]`/`[compat]` and (b) exercises the package's real symbol table end-to-end. Requires `[compat]` pinned on every Turing-stack package whose version drift can silently break internals (`Turing`, `DynamicPPL`, `FlexiChains`, `MCMCChains` — currently 0.46/0.42/0.6/7), so the temp env's independent resolve can't drift to an incompatible combo.
- V19. `_generated_quantities` (`src/model.jl`) return tuple always includes a `:loglik` key — per-observation log-likelihood vector, added under T7 for `psis_loo`/`loo_compare`. Computed post-hoc inside `_generated_quantities` rather than via DynamicPPL's normal VarName-based pointwise-loglik tracking, because `model.jl`'s likelihood is written with `Turing.@addlogprob!` (not `y[n] ~ Dist(...)`) — no observed VarNames exist for DynamicPPL to see. Any code calling `generated_quantities(model_with_data, TR.samples)` directly must filter/exclude `:loglik` before treating the returned keys as fitted parameter names — `src/turingregression.jl`'s fixef-layer param loop does this via `filter(!=(:loglik), ...)`.
- V20. Random-effect components (`ranef_matrix` in `_random_effects`, model.jl) must be mean-zero by construction — no free parameter may act as an extra mean shift, since that would be additively confounded with the population-level fixed effect covering the same predictor (only the sum is identified, producing a NUTS ridge and biased marginals — B4). Population `α`/`β` are the only mean-carrying params; ranef branches only add zero-mean deviations (`diagm(σ)*L*z_raw` or `σ.*z_raw`).

## §R RESEARCH

id|topic|finding|source
R1|FlexiChains|already Turing 0.46 default return, `chain_type=VNChain` explicit alt to MCMCChains.Chains|github.com/penelopeysm/FlexiChains.jl
R2|FlexiChains|indexing `chain[@varname(x)]` returns `DimMatrix` (DimensionalData) natively — no AxisArray/reshape step needed|github.com/penelopeysm/FlexiChains.jl/blob/main/README.md
R3|FlexiChains|no `name_map`; keyed by `VarName` not `Symbol` — structured names (`z[1]`) built into key, no separate lookup table|github.com/TuringLang/MCMCChains.jl/issues/335, FlexiChains README
R4|FlexiChains|`summarystats(chain)` gives mean/std/mcse/ess/rhat — likely drop-in for `summarize(...).nt` (src/summary.jl:230) but exact field names unverified `?`|turinglang.org/docs/core-functionality/
R5|FlexiChains|already compat-pinned `FlexiChains = "0.6"` in Project.toml, but only in `[extras]`/test target, not `[deps]` — real dep add needed for non-test use|Project.toml:39,48,55
R6|FlexiChains|ships `FlexiChainsMCMCChainsExt` conversion ext + `FlexiChainsDynamicPPLExt` + `FlexiChainsPosteriorStatsExt` — interop path exists both ways if partial switch wanted|Manifest.toml:826-839
R7|FlexiChains|`DynamicPPL.returned(model, chain::FlexiChain{<:VarName})` re-evaluates model per draw, returns `DimMatrix`/`DimArray` — if model's return value is itself a `DimArray`, dims auto-stack into result (draw/chain dims appended)|github.com/penelopeysm/FlexiChains.jl ext/FlexiChainsDynamicPPLExt.jl:438-451
R8|current code|extraction (src/turingregression.jl:280-310) manually loops `generated_quantities` NamedTuple output, stacks/vcats into per-layer DimArrays by hand — exactly what R7's auto-stack could replace, if model.jl's `@model` return stmt changes to return a `DimArray` instead of plain NamedTuple|local, src/turingregression.jl:280-310, src/model.jl (`_generated_quantities`)
R9|FlexiChains|verified via scratch model (playground.jl): single-`DimArray` model return → `returned()` DOES auto-stack into clean `(iter,chain,param)` DimArray. `NamedTuple`-of-`DimArray` (needed for multi-layer `:fixef`/`:{group}`/`:{group}_sd`) does NOT auto-stack — stays `DimMatrix` of NamedTuples, still needs manual per-layer split, just over smaller pre-labeled DimVectors. Also: FlexiChains deprecating implicit stacking, needs explicit `stack=true` going forward|local scratch run, `~/.julia/packages/FlexiChains/Rb8fl/src/chain.jl:782` warning
R10|FlexiChains|`DimArray(chain)` (no `returned`/reevaluation, no `stack=true`) unwraps VNChain's raw *sampled* VarNames directly into stacked `(iter,chain,param)` DimArray. Only exposes raw sampled params (standardised-scale priors), not the model's derived/back-transformed return value — doesn't replace `returned()` for actual extraction need, but a free cheap path if raw param access ever wanted separately|local scratch run
R11|T13 gensym|confirmed NOT a mistaken fix, by removing it in a scratch repro (fixed model name instead of `gensym`): 8 sequential fits varying family/predictor-count/weighted all fine, but refitting a ranef model (intercept+slope) after two *other* ranef shapes (intercept-only, slope-only) reused the name → `KeyError: :α_z_Subject not found`. Same class of bug as the BoundsError the original comment describes (shared `typeof(model)` across structurally different generated code corrupts Turing/DynamicPPL/AD internal caches), different symptom. Confirms per-shape uniqueness is required — the fix to chase is caching by structural key, not removing gensym|local scratch (`test_gensym.jl`, `test_gensym2.jl`), commit 9e14183 (`fname = gensym(...)` introduced same commit as an unrelated `Union{...,Nothing}` type-annotation fix, hence suspicion it was a coincidental band-aid — it isn't)
R12|T13 cache-key scope|traced every construct-time-only (not runtime-arg) input baked into the generated `Expr` in src/model.jl: family (branch dispatch), `ModelInfo`'s 4 bools (branch structure), and per-ranef (in `model_ranef` order) — `variable` name (embedded in generated symbols e.g. `α_z_Subject`), `has_intercept`/`has_fixed_effects` (branch choice), and when `has_fixed_effects`, the literal `size(ranef.predictors,2)` interpolated directly into code (e.g. `LKJCholesky($n_predictors,2.0)`, model.jl:38-41) — two ranefs with identical flags but different slope counts are NOT cache-equivalent|local, src/model.jl:19-120
R13|T13 cache-key scope, priors|priors are NOT passed as runtime args to the compiled model — `_intercept`/`_fixed_effects`/`_random_effects`/`_auxiliary_parameter` all do `$prior` interpolation (model.jl:7,14,40 etc), baking the actual `Distribution` object into the generated code as a literal. Confirmed `_build_model_with_data` (turingregression.jl:215-238) never passes `TR.prior` to `TR.model(...)`. So a structural-only cache key would silently reuse stale prior values across calls with different priors — cache key MUST also cover the 4 `RegressionPrior` distributions (e.g. `(typeof(d), Distributions.params(d))` per distribution). Considered refactoring priors to runtime args instead (would let the key drop priors, and lines up with T4's plan to transform priors outside the model) but rejected for T13 — bigger diff than a memoization layer, and T4 already owns that rewrite|local, src/model.jl, src/turingregression.jl:215-238

## §B BUGS

id|date|cause|fix

## §T TASKS

T3|.|BIG JOB, merged w/ old T8 (co-dependent — both rewrite `_generated_quantities`/extraction, doing separately means touching same code twice). **NEEDS A BIG THINK before starting** — three parts sketched below, not fully speced, sequencing/interfaces between them unresolved:
  (a) move standardise/back-transform OUT of generated model — compute scaling stats ONCE outside (NamedTuple `X_means`/`X_stds`/`y_mean`/`y_std` + per-ranef equivalents), pass scaled data into model, back-transform (fixef + ranef + Σ) in Julia inside `fit!`. Delete `_standardise_data` in favour of this. Add round-trip test `unstandardise∘standardise == id`.
  (b) drop MCMCChains for FlexiChains — model's `@model` return stmt becomes single flat `DimArray` (not NamedTuple), pulled via `DynamicPPL.returned(model, chain; stack=true)` which auto-stacks into one `(iter,chain,param)` DimArray (verified R9; NamedTuple-of-DimArray does NOT auto-stack, so don't return a layered NamedTuple). Get raw stacked params THIS way instead of today's manual per-layer Dict+stack+vcat loop over `generated_quantities`; standardise/back-transform (a) applied AFTER unstacking, not inside the model. Promote FlexiChains `[extras]`→`[deps]` (R5). Swap `chain_type=MCMCChains.Chains`→`VNChain`, `summarize`→`summarystats` (confirm fields, R4), `.name_map`/`.value` indexing→`VarName`-keyed access (R2,R3).
  (c) once (b) lands, rest of API (parametermethods.jl `draws`, summary.jl, etc) should adopt FlexiChains' own conventions rather than forcing into today's names — e.g. `:iter` not `:draw`, FlexiChains' own dim ordering — instead of relabeling/reordering into current DimStack scheme. Touches V7 (draws shape), V9 (param labels), C3 (layer/dim naming) — re-verify those invariants under new naming.
  Delete `_generated_quantities` (src/model.jl) in favour of (a)+(b) combined. Re-verify V-invariants under Turing/NUTS. Split (a)/(b)/(c) into separate PRs once the big-think resolves ordering — don't attempt as one diff|C2,C6,C9,V3,V4,V7,V9,V10,V14,R1-R10

T4|.|BIG JOB: flip prior scaling (depends on T3's centralised affine map). Today user specifies priors on standardised (mean 0, sd 1) scale. Change so priors are given on ORIGINAL data scale — e.g. `Normal(10,20)` on a predictor with mean 10, sd 20 → transformed to `Normal(0,1)` internally for the standardised fit — reported back on original scale in `summary`/`show`/prior display. Use `Distributions.AffineDistribution` (`shift + scale*d`) for the forward transform (prior) and its inverse for the param back-transform — don't hand-write new per-family algebra. Store user's original prior for display; fit on scaled. Re-verify V-invariants under Turing/NUTS with `AffineDistribution` priors|C2,I

T5|.|Re-verify model code-gen (`show_code`, model.jl generated `Expr`) after T3/T4 land, since both touch generated-model internals. FOLD IN: collapse `_likelihood` + `_weighted_likelihood` (currently ~90% duplicated) into one family-dispatched function, with `weights` defaulting to `ones(...)` so unweighted fit is just weighted-fit-with-1s (makes V14 a true structural guarantee, not a coincidence)|C6,V14

T6|.|`TuringRegression` <: StatsAPI `RegressionModel`. Researched 2026-07-16 (StatsAPI `statisticalmodel.jl`/`regressionmodel.jl` read in full). New file `src/statsapi.jl` holds all of this (not scattered into existing files).

Three groups:

(a) EASY — implement, point estimate = posterior mean, reuses existing `draws`/`predict`/`outcome` machinery:
`coefnames` (`:α`+X_names), `coef` (mean fixef), `coeftable`/`confint` (level kwarg), `vcov`/`stderror` (cov of fixef draws), `nobs`, `isfitted`, `weights`, `islinear` (true only Normal+identity link), `fitted`/`response`/`responsename`/`meanresponse`/`modelmatrix`/`residuals`, `linearpredictor`, `offset` (→ `nothing`, unsupported). `vif`/`gvif` (StatsModels-owned generic) likely free once `modelmatrix`+`coefnames` exist — verify, don't assume.
Every method that collapses the posterior to one number (`coef`, `coeftable`, `fitted`, `residuals`, `linearpredictor`, new `predict` — see below) `@warn maxlog=1` that it's reporting the posterior mean, not a Bayesian summary — full posterior available via `posterior_predict`/`draws`. `vcov`/`confint`/`stderror` use the full posterior (covariance/quantiles, not a mean plug-in) — no warning needed there.

(b) SKIP, error clearly — no MLE exists on a NUTS posterior, concept doesn't map: `score` (grad @ MLE), `informationmatrix` (Fisher info), `leverage`, `cooksdistance` (OLS hat matrix), `reconstruct`/`reconstruct!`, `predict!` (no in-place design).

(c) SKIP, error clearly, pointing to `psis_loo`/`loo_compare` — decided 2026-07-16 after discussion: `loglikelihood`, `dof`, `mss`, `rss`, `nulldeviance`, `nullloglikelihood`, `aic`, `aicc`, `bic`, `r2`, `adjr2`. Reason: no single "the" likelihood on a Bayesian posterior (mean-plug-in vs mean-of-per-draw are both defensible, neither canonical); `nulldeviance`/`nullloglikelihood` need an actual second NUTS fit of an intercept-only model every call (no closed-form null unlike frequentist GLM, since the null model has its own priors/posterior) — plus an unresolved design question of whether a ranef model's "null" keeps grouping structure; `dof` as raw param count badly miscalibrates `aic`/`bic` for hierarchical/ranef models (shrinkage means a ranef level isn't really "1 free param") where this package's value-add concentrates. `psis_loo`'s `p_loo` is the honest effective-dof answer and already exists (T7) at no extra fit cost — steer users there instead of a misleading frequentist-flavored number.

Naming, decided 2026-07-16: extend `StatsAPI.fit!` directly (shapes already match — `fit!(TR; kwargs...)` — zero cost, avoids two unrelated `fit!` generics coexisting badly). Extend `StatsAPI.predict` too, but semantics differ from today's `predict`: StatsAPI convention wants a point vector, ours returns full posterior draws by default. Resolution: **rename current rich draws-returning function `predict` → `posterior_predict`** (same signatures/kwargs, unchanged behaviour); new `predict(TR, [newX])` becomes the StatsAPI-conformant point estimate (mean `epred`), `@warn`ed per (a) above. README states `posterior_predict` is the preferred/primary API — `predict` exists for StatsAPI interop only. Breaking rename — update all internal call sites (`predict.jl`, `metrics.jl`, tests, docs) from `predict`(rich) → `posterior_predict`.

T13|.|**PARKED 2026-07-16 — bigger than expected, needs its own PR.** First `using TuringRegressions` / first `Pkg.test()` pays full TTFX (Turing/DynamicPPL/model-macro compile). Add a `PrecompileTools.@compile_workload` block (small `turing_glm` fit, `N`/`nchains` minimal) in `src/TuringRegressions.jl` to precompile the hot path ahead of time, cutting first-run latency. **Separate, bigger latency source found during T9 (2026-07-16)**: `construct_model` (model.jl:433) `gensym`s a brand-new model type on *every* `turing_glm()` call. Measured on sleepstudy correlated ranef: 1st `fit!` (compile+sample) 38.1s vs 2nd `fit!` on the *same* compiled type (sample only) 13.1s — ~25s of every single fit is recompilation, not sampling, and repeat fits of the same formula/family/priors/ranef shape never get to reuse it. `PrecompileTools` can't help since the gensym'd type doesn't exist until runtime.

Investigated 2026-07-16 (see R11-R13): confirmed gensym itself is NOT a mistaken fix — removing it reproduces real cache corruption (`KeyError`) across differing ranef shapes fit in sequence, same bug class the original comment warns about. Fix is to memoize the compiled model function by a structural key so identical specs reuse the same eval'd type (and its already-JIT'd AD/sampler methods), and only genuinely new shapes pay `gensym`+`eval`.

Design (full writeup: `/home/james/.claude/plans/serialized-doodling-hanrahan.md`, may be pruned — key points here): new `src/model_cache.jl` holding `MODEL_CACHE::Dict{Any,Tuple{Function,String}}` + `ReentrantLock`, a `_model_cache_key(family, model_info, model_ranef, prior)` builder, and the cache-check/insert wrapper. Key must cover (R12) family + `ModelInfo`'s 4 bools + per-ranef `(variable, has_intercept, has_fixed_effects, n_predictors-if-any)` in order, AND (R13) all 4 `RegressionPrior` distributions as `(typeof(d), Distributions.params(d))` — priors are baked into generated code as literals, not passed at runtime, so a structural-only key would silently serve stale priors. `construct_model` (model.jl:418-447) becomes: build key → cache hit returns `(model_fn, model_code_str)` as-is → cache miss builds body/gensym/eval as today, then stores. No changes needed to `turing_glm`/`fit!`/`_build_model_with_data`/the `_intercept` family of body-builders — pure memoization layer over existing codegen. Considered passing priors as runtime args instead (would drop them from the key, and lines up with T4's planned prior rewrite) — rejected for T13 scope, bigger diff, leave for T4.

Verification when resumed: rerun scratch repro (varied families/predictor-counts/3 ranef shapes interleaved) — no `KeyError`; assert two identical-spec `turing_glm` calls return `===` model_fn (cache hit), a differing-prior call returns a different one, a differing-ranef-shape call returns a different one; `@elapsed`-time first vs second identical-spec `turing_glm`+`sample()` to confirm TTFX collapses; full `Pkg.test()` regression pass|

- P8: `TR.link` field (predict.jl) — used internally, V1-bounded to the 5 families, not user-facing.
