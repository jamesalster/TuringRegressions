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
- C3. Params stored `TR.parameters::DimStack`. One layer per group (`:fixef`, per-ranef `:{group}`, `:{group}_sd`, `:{group}_corr`, `:internals`); each layer has dims incl `:iter, :chain` (FlexiChains names). Chains collapsible to one `:iter` via `draws(...)`.
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

Fitting:
- `fit!(TR; sampler=NUTS(;adtype=default_adtype(TR.modeldata)), parallel=MCMCThreads(), samples=2000, nchains=4, warmup=samples, quiet=true, kwargs...)` mutates TR. `adtype` defaults `AutoReverseDiff(compile=true)` if `has_random_effects(TR.modeldata)` else `AutoForwardDiff()` (V25) — override by passing `sampler=NUTS(;adtype=...)` explicitly. Budget: `samples` (kept draws) and `warmup` (adaptation, discarded) are TOTALS across chains, each split over `nchains` via ceil (`per_chain=cld(samples,nchains)`, `warmup_per_chain=cld(warmup,nchains)`) — rounds UP so realised count never below request. `warmup` is IN ADDITION to `samples` (each chain runs warmup_per_chain discarded + per_chain kept), default = `samples`. Discarded upstream by Turing (`nadapts`/`discard_initial`), never reaches `TR.samples`; `warmup=0` disables. Passes `per_chain` as `N` to Turing `sample`. Recovers unstandardised params OUTSIDE the model: `DimArray(TR.samples)` → `reshape_params` → `unstandardise` (src/reshape.jl) → `TR.parameters`. No in-model back-transform.

Param extraction (`src/parametermethods.jl`):
- `draws(TR; drop_draws=0, n_draws=Inf, collapse=true)` whole `TR.parameters` DimStack, warmup dropped / chains collapsed per kwargs.
- `draws(TR, type::Symbol; ...)` single layer DimArray. `type` ∈ `propertynames(TR.parameters)` else `ArgumentError`: `:fixef`, `:{group}`, `:{group}_sd`, `:{group}_corr` (correlated ranef only).
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
- `Base.summary(io, TR; funs=[mean,std], quantiles=[0.025,0.975], return_table=false, drop_draws=nothing, show_metrics=false, kwargs...)`. Fixef table + per-grouping-term Random Effects tables (SD always; Correlation matrix only if `{group}_corr` present). Metrics table opt-in via `show_metrics=true`.
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
- `RegressionPrior` — `intercept`/`fixed_effects`/`random_effects`/`auxiliary`. Passed as runtime model args, not baked into the generated `Expr` — keeps `cached_construct_model`'s cache key purely structural (V20).

## §V INVARIANTS

- V1. Family restricted to `{Normal, TDist, Bernoulli, Poisson, NegativeBinomial}`; others → `error` at `turing_glm`.
- V2. `draws(TR, type)` with invalid `type` → `ArgumentError`.
- V3. Fixef point estimates (posterior mean) vs GLM MLE within tolerance — Normal/Poisson/NegBin atol 0.025, Bernoulli looser (V13).
- V4. `epred` posterior mean vs `GLM.predict` within tolerance (family-dependent) for Normal/Poisson/NegBin/Bernoulli.
- V5. Link relations: `linpred == log(epred)` for log-link families; `linpred == epred` for identity-link.
- V6. `var(posterior_pred) > var(epred)` — posterior predictive adds observation noise (also count families).
- V7. `draws(TR, type)` shape `(n_effect_dims..., (N-drop_draws)*nchains)` when collapsed; extra trailing `:chain` dim if `collapse=false`.
- V8. `n_draws` beyond available post-warmup draws → `ArgumentError`.
- V9. Returned param labels `[:α, :<X_names...>, aux...]`; `:α` always present for intercept models.
- V10. Standardised→original param recovery lives in `unstandardise` (src/unstandardise.jl) via `coef_map`, reading `tf::Transform` (nested `tf.fixef`, `tf.y`, `tf.ranef`). Runs OUTSIDE the model, post-fit. `?` exact algebra + field names — code is oracle.
- V11. `drop_draws` default 0 (`draws()`/`summary()`) — `fit!` discards warmup upstream via `nadapts`/`discard_initial`, so `TR.samples` never contains warmup. Downstream drop is user-override only.
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
- V25. `fit!` default `adtype` picked from `has_random_effects(TR.modeldata)`: ranef present → `AutoReverseDiff(compile=true)`, else → `AutoForwardDiff()`. Basis: T24 sweep showed ranef presence (not param count) determines which backend wins — see `bench/BENCHLOG.md`. User-supplied `sampler=NUTS(;adtype=...)` always overrides.
- V26. KNOWN, not a bug: LKJ(η=1) corr prior (model.jl:47) is flat on std-scale ranef correlation, not raw-scale (C2 standardises outside model) — biases ranef SD/corr point estimates vs raw-scale tools (lme4/brms). Documented in readme.md Notes. See T34.
- V27. Comments: why not what, short. No spec-tag refs (T/V/C/G ids) in code — code comments stay self-contained, spec is separate doc.

## §B BUGS

id|date|cause|fix

## §T TASKS

T14,T20,T21,T23-T36|x|closed benchmark/perf work (adtype default V25, ranef sd/corr fix, TTFX precompile, type-stability). Summary + full run log: `bench/BENCHLOG.md` (branch `benchmark-improvements`)
T17|.|`TR.link` field (predict.jl) — used internally, V1-bounded to the 5 families, not user-facing. Confirm it can stay internal / no action needed|V1
T19|.|`fit!(quiet=false)` live progress bar not showing in VSCode Julia REPL. Likely needs a progress-capable logger (TerminalLoggers) or VSCode's ProgressLogging integration. Low priority — sort later|
T34|.|Think about how to pass ranef corr prior (LKJ η) in — currently hardcoded 1.0 (model.jl:47), known std-scale-vs-raw-scale flatness issue (V26). Options TBD: expose η / whole corr prior as `RegressionPrior` field, reparameterise, or leave as documented divergence. Also more widely: think how to let user pass more specific priors in generally (brms-style — per-term/per-coef, not just per-role) instead of current one-prior-per-role `RegressionPrior`|C2,V10,V26
T40|x|Rationalise `src/transform.jl` — `Transform`/`LinearTransform`/`compute_transform`/`apply_transform`/`standardise`/`unstandardise_data` layout is a pain to work in (T33 touched it). Think about a simpler structure once T34 settles (touches same file). Done: split into `standardise.jl` (forward) + `unstandardise.jl` (back-transform), shared `coef_map` for fixef/ranef point-estimate + sd/corr|T33,T34

T41-T58 from full code review (main @ fa9a6b1 + uncommitted InitFromPrior work). Do IN ORDER. T41-T43 = wrong numbers, silent. T56-T58 = release blockers, last by user request.

T41|x|BUG posterior predictive noise wrong, `posterior_pred` (src/predict.jl:164-191). Two symptoms, one line. (a) Normal: `(rand(T(), ndraws) .* σ)' .+ epreds` draws ONE standard normal per draw index then broadcasts it across all N rows → every obs within a draw shares identical noise; joint predictive is a rigid translation of the epred vector. Marginals stay correct so nothing currently fails. Same for TDist `(rand.(T.(ν)) .* σ)'`. NegBin/Bernoulli/Poisson are correct (elementwise `rand.` over epreds). Visible in `pp_check_dens_overlay` — every curve same shape, only shifted. (b) `collapse=false`: epreds is (row,iter,chain) so `ndraws=size(epreds,2)` is per-chain, but `σ=vec(fixef_draws[...])` has length iter*chain → DimensionMismatch; NegBin `(1 ./ ϕ)'` fails the same way. Fix both at once: `epreds .+ randn(size(epreds)) .* reshape(σ, 1, size(epreds)[2:end]...)`, reshape ϕ for NegBin likewise. Tests: variance ACROSS ROWS within one draw ≈ σ² (catches a); `posterior_predict(mod; type=:posterior, collapse=false)` runs for all 5 families (catches b)|V6

T42|x|BUG slope-only ranef `{group}_offset` is computed but never applied. src/unstandardise.jl:100-101 emits the layer; `_add_random_effects!` (src/predict.jl:69-87) adds only `X[r,k]*b_g` and ignores it. Model fits on x_std=(x-m)/s, so the raw-scale ranef contribution is `Σx·b_raw − Σm·b_raw`; that second term is a per-group constant. An intercept-bearing ranef absorbs it via `coef_map`'s cross terms (src/unstandardise.jl:14-27); `(0+x|g)` has nowhere to put it → predictions off by a per-group constant. Existing test runtests.jl:497-511 is `@test_nowarn` only, no numbers, so it passes. PREFERRED fix: stop centring a ranef term's predictor columns when that term has no intercept (`compute_transform`, src/standardise.jl:34) — the offset then never arises and the `{group}_offset` layer plus `_center`/`_scale_effects` (src/reshape.jl:48-49) all delete. Alt fix: consume the offset inside `_add_random_effects!`. Either way update C3 (drops `:{group}_offset` from the layer list). Test: `(0+Days|Subject)` sleepstudy epred vs an lme4/MixedModels fit, numeric tolerance|V16,C3

T43|~|BUG no-intercept formula + y-scaling family is silently biased. `compute_transform` (src/standardise.jl:32-39) centred y whenever `family_spec(family).scales_y`, regardless of `has_intercept(md)`. For `@formula(y ~ 0 + x)` with Normal the fit ran on centred y and centred X, but `_unstandardise_fixef` (src/unstandardise.jl:32-55) with `has_int=false` only rescales β — there was nowhere to put `y_mean` or the `Σ mean_x·β` cross term. FIXED the transform math: `compute_transform` now skips centring both fixef X and y when `!has_intercept(md)`, same pattern T42 used for slope-only ranef. BUT this exposes a second, distinct problem, found while verifying against `GLM.lm(@formula(MPG ~ 0 + Cyl + Disp), mtcars)`: skipping centring leaves std-scale X with a non-zero mean (`mean(x)/sd(x)`, e.g. ≈3.4 for Cyl), which inflates the std-scale coefficient magnitude needed to match GLM (≈2.09 for Cyl here) to right where the fixed-width `fixed_effects ~ Normal(0,2)` prior (src/prior.jl:35) actively shrinks it — coef came back 6.72 vs GLM 7.05 (diff 0.33, exceeds V3's 0.3 atol) and stayed biased at 4x samples/chains, so it's prior-induced shrinkage, not MCMC noise. T42's identical skip-centring fix didn't hit this because ranef slopes use an adaptive hierarchical variance prior (τ estimated from data), not a fixed-width one — fixef has no such adaptivity. Interim mitigation: `turing_glm` now `@warn`s on any no-intercept formula with fixed effects. Real fix still open — options: (a) `error` on the combination instead of warning, (b) widen/adapt the fixed_effects prior for no-intercept fits, (c) fit no-intercept models on raw (unstandardised) scale. Needs a decision before this can close. Test: `y ~ 0 + x` Normal vs `GLM.lm(@formula(y ~ 0 + x))`, coef atol per V3 — not yet added, blocked on picking a real fix since skip-centring alone doesn't pass it|V3,V9

T44|x|BUG `fit!` docstring attached to the wrong function. The `"""..."""` block at src/turingregression.jl:208-234 is followed by a comment and then `_build_model_with_data` (line 238), so Julia binds the doc to the private helper and `?fit!` returns nothing — the package's most important user-facing docstring is invisible. Fix: move the block to immediately above `function fit!` (line 250). Trivial, do it first if warming up|

T45|x|BUG summary table column formatters off by one, copy-pasted 3×. src/summary.jl:97-103 (fixef), :123-129 (ranef levels), :145-151 (ranef SD). Columns are `[funs..., quantiles..., mcse, ess_bulk, ess_tail, rhat]` and `ncols=length(chain_info)`. With defaults ncols=8: `1:(ncols-5)`=1:3 leaves col 4 (q97.5) formatted `%5.2g`; `[ncols-4]`=4 hits q97.5 instead of mcse; `[ncols-2,ncols-3]`=6,5 formats mcse `%5.0f` so it PRINTS AS A WHOLE NUMBER; `[ncols-1]`=7 gives ess_tail `%5.3f`. `make_highlighters` (:257-273) indexes correctly (ncols-1/ncols-2 = ess columns) — that mismatch is the proof the formatters are the wrong ones. Fix: shift all four groups +1 (stats `1:(ncols-4)`, mcse `ncols-3`, ess `ncols-2,ncols-1`, rhat `ncols`) AND extract one `_stat_formatters(ncols)` helper for all 3 call sites — the triplication is why this survived|

T46|.|BUG `conditional_dependency` always throws on ranef models (ext/TuringRegressionsMakieExt.jl:81-119). It `@warn`s "random effects are held at their fitted values" (:85), then calls `posterior_predict(TR, predgrid)` with a raw Matrix (:105), which `_resolve_z` (src/predict.jl:28-36) errors on unconditionally for ranef models. The warning promises behaviour the next line makes impossible. Fix: either build the prediction grid as a DataFrame carrying the grouping columns (routes through the new_data path, warning then becomes true), or replace the `@warn` with an `ArgumentError` naming the limitation. Test: sleepstudy ranef model, assert whichever behaviour is chosen|

T47|x|BUG `dropdims=false` not forwarded inward. `epred` (src/predict.jl:135-154) calls `linpred(TR, X, z; kwargs...)` without passing `dropdims`, so linpred applies its own default `true`; `posterior_pred` (:171) inherits the same via `epred`. Result: `posterior_predict(mod; type=:epred, dropdims=false)` still drops singleton dims. `linpred(f, ...)` (:127-133) does it right — copy that pattern: `dropdims=false` on every inner call, one `_drop_single_dims` at the outermost layer only|

T48|x|`Base.summary` / `Base.show` contract violations. (a) `Base.summary(io, TR)` (src/summary.jl:53) prints tables and returns `nothing`; the Base contract is a ONE-LINE `String`, so any generic code calling `summary(obj)` misbehaves. Rename the public API to `model_summary` (or `summarize`), export it, update readme + runtests.jl:268-284. (b) `Base.show(io, TR)` (src/turingregression.jl:168) is the COMPACT 2-arg method Julia uses inside containers, but it prints a multi-line block AND fires `@warn` through `model_warnings` (`warnings=true` default) — displaying a `Vector{TuringRegression}` spews tables and warnings. Fix: move the long form to `show(io, ::MIME"text/plain", TR)`, make 2-arg show one line (family, formula, fitted or not), and never warn from `show`|

T49|x|`@assert` used for user-input validation in `_process_draws` (src/parametermethods.jl:9,14). Assertions can be elided under some optimisation/`--check-bounds` settings, and `AssertionError` is the wrong type — every other validation in the package throws `ArgumentError`. Fix: convert both to `throw(ArgumentError(...))`, then update V8 and runtests.jl:192 (`@test_throws AssertionError` → `ArgumentError`)|V8

T50|x|Missing validation at function boundaries — all three currently surface as bare MethodErrors or silent wrong behaviour. (a) `turing_glm`'s `weights::Union{Nothing,Vector{Float64}}` (src/turingregression.jl:61) rejects `ones(Int,n)` and Float32 vectors with an opaque MethodError; widen to `AbstractVector{<:Real}`, convert internally, and check `length(weights) == nrow(data)` plus `all(≥(0), weights)`. (b) array-form `turing_glm` (:104-120) never checks `length(names) == size(X,2)` (silently truncates via `ntuple`), and `df.y = y` silently clobbers a user column already named `y`. (c) `posterior_predict(TR, X::AbstractArray)` never checks `size(X,2)` against the fitted predictor count. Each should fail loudly naming what went wrong and where|

T51|x|Silent `nothing` fallthroughs for unhandled families/links. (a) `epred`'s `invlink` `let` block (src/predict.jl:143-151) evaluates to `nothing` when the link is not identity/logit/log, producing a `nothing.(μ)` mystery error. Fix: add `get_invlink(family)` beside `get_link` (src/utils.jl:8-19) with an explicit `error` on fallthrough, and call it. (b) `_auxiliary_parameter` (src/model.jl:70-88) and `_obs_logpdf` (:141-149) both fall off the end returning `nothing` for unlisted families — currently unreachable thanks to V1's whitelist, but adding a family to `turing_glm` and forgetting these two fails at `eval` time with an unreadable error. Add `else error(...)` to both|V1,C5

T52|x|Dead code + the missing round-trip test. (a) `TuringRegression.modelcode::Expr` field (src/turingregression.jl:21) is written at construction and never read — `modelcode(TR)` (src/model.jl:255) rebuilds from `TR.modeldata`. Drop the field and `construct_model`'s Expr return, or make `modelcode(TR)` return the stored one; do not keep both. (b) `unstandardise_data` (src/standardise.jl:65) is called nowhere, and its docstring claims it is "used by the round-trip test guarding the standardisation constants" — that test does not exist. Write it (V21: 5 families × 3 ranef shapes) or delete the function; the test is worth more than the deletion. (c) `using MixedModels: _ranef_refs` (src/TuringRegressions.jl:41) is unused AND a private-API import — src/formula_handlers.jl:79-84 explains why it is deliberately avoided; delete the import. (d) `_center`/`_scale_effects` (src/reshape.jl:48-49) are used only by src/unstandardise.jl, contradicting reshape.jl's own "pure structural reshuffle" header — move them, or delete them along with T42. (e) `predictors(TR, :ranef)` → `error("Not implemented")` (src/parametermethods.jl:82): implement it or stop advertising it in the docstring|V21,T42

T53|.|Document (or remove) the world-age trap. `turing_glm` `eval`s a freshly gensym'd model function, so `f() = fit!(turing_glm(...))` throws a world-age error on the FIRST fit of any new model shape. Already hit internally: the precompile workload calls `fit!` through `Base.invokelatest` for exactly this reason (src/TuringRegressions.jl:61-79). Users who wrap both calls in one function will hit it, and the cache (V20) makes it intermittent — fine on the second shape-identical call, which is the worst kind of bug report. Fix: either route `fit!`'s `sample` call through `Base.invokelatest` (costs one dispatch per fit, not per NUTS step — measure before rejecting), or add a prominent "Gotchas" section to readme + the `turing_glm` docstring. Decide which|C6,V20

T54|.|Doc corrections, README + docstrings. readme.md: `fit!(mod, N=1000, nchains=2)` — the kwarg is `samples`, not `N` (3 occurrences; today `N` is silently forwarded to `sample()`); `:internals` does NOT exist (lines 67, 78, 146 and src/parametermethods.jl:41 — `TR.parameters` holds only `:fixef` + ranef layers, so `draws(mod, :internals)` throws ArgumentError) — either drop it from the docs or actually add the layer per C3; `[param=At(...)]` should be `[fixef=At(...)]` (lines 62, 63, 118, 122); `n_draws=-1` (line 188) is `Inf`; line 10 says "Binomial", the family is `Bernoulli`; line 104 sets up `robust_mod` then never calls `loo_compare`; line 193 "pacakge" typo. Docstrings: `Base.summary` (src/summary.jl:39) advertises `funs=[median, std]`, actual default is `[mean, std]`; the `TuringRegression` struct docstring (src/turingregression.jl:12) calls `parameters` "Standardized parameter draws" — they are UNstandardised, original scale, i.e. exactly backwards; `default_prior` (src/prior.jl:29) has a stray unmatched triple-backtick. ALSO this spec: §I Fitting, the §I Types `Transform` bullet, and V10 still cite `unstandardise` as living in `src/reshape.jl` — T40 split it out to `src/unstandardise.jl`. (The `drop_warmup`→`drop_draws` rename is already applied to §I/§V.)|C3,V11,T40

T55|.|Test gaps, each mapping to a bug above. Add: (1) numeric prediction check for slope-only ranef (T42); (2) `collapse=false` prediction across all 5 families (T41); (3) any no-intercept formula at all (T43); (4) posterior-predictive dispersion — variance across ROWS within a single draw (T41); (5) the `allow_new_levels=true` path and the unseen-level error (src/formula_handlers.jl:108-151 — a documented feature with zero coverage); (6) standardisation round-trip per V21 (T52b); (7) weights × random effects together (each is tested alone, never combined); (8) nested grouping `(1|a/b)` — untested, and it is unknown whether MixedModels' `apply_schema` expands it into two RandomEffectsTerms or something else. Also document the weighted-LOO caveat: `pointwise_loglik` (src/comparison.jl:29-45) multiplies weights into the pointwise log-likelihood, which is not the exchangeable one-row-per-observation object PSIS assumes|T41,T42,T43,T52,V21

T56|.|Packaging blockers — registry auto-merge fails without these. Add `LICENSE` (none in repo). Add a `julia` compat entry. Add `[compat]` for the ~17 deps lacking one (only 9 of 26 have bounds; the `Makie` weakdep needs one too). Add `Crayons` to `[deps]`: `crayon"..."` is used 10× across src/summary.jl and src/turingregression.jl but currently resolves only through PrettyTables' transitive re-export — it breaks the day PrettyTables stops re-exporting. Remove `MCMCChains` and `PSIS` from `[deps]`: referenced nowhere in src/, ext/ or test/, and both pull heavy trees — NB V17 names MCMCChains as a pinned Turing-stack dep, so re-read V17 before removing. Also: comments at src/TuringRegressions.jl:64,70 cite `bench/sleepstudy_bench.jl` and `bench/BENCHLOG.md`, neither of which is in the repo (bench/ holds only a Manifest.toml) — commit them or drop the references. Also consider do we need Tables for columntable?|V17

T57|.|CI. No `.github/workflows` at all. Add a test workflow (Julia version × OS matrix, `julia-actions/setup-julia` + `julia-actions/julia-runtest`) plus coverage upload. NB a full `Pkg.test()` is 20-30+ min of NUTS (C10), so run the `TR_DEV_SUBSET=true` path on PRs and the full suite on main pushes / nightly only. Tests must run via `Pkg.test()`, not direct `julia test/runtests.jl` (V17)|C10,V17

T58|.|Documenter.jl docs site. Add `docs/` (Project.toml, make.jl, src/index.md), build from the existing docstrings, publish to GitHub Pages via `julia-actions/julia-docdeploy`. Pages: getting started; the priors-are-on-the-standardised-scale explainer (C2 — the single most surprising thing about this package, and the thing most likely to produce silently wrong models in user hands); random effects; prediction; model comparison; StatsAPI interop; full API reference. Do LAST: docstrings must be correct first (T54) and it needs a CI workflow to deploy from (T57)|T54,T57,C2
