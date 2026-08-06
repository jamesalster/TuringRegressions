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

Fitting:
- `fit!(TR; sampler=NUTS(;adtype=default_adtype(TR.modeldata)), parallel=MCMCThreads(), samples=2000, nchains=4, warmup=samples, quiet=true, kwargs...)` mutates TR. `adtype` defaults `AutoReverseDiff(compile=true)` if `has_random_effects(TR.modeldata)` else `AutoForwardDiff()` (V25) — override by passing `sampler=NUTS(;adtype=...)` explicitly. Budget: `samples` (kept draws) and `warmup` (adaptation, discarded) are TOTALS across chains, each split over `nchains` via ceil (`per_chain=cld(samples,nchains)`, `warmup_per_chain=cld(warmup,nchains)`) — rounds UP so realised count never below request. `warmup` is IN ADDITION to `samples` (each chain runs warmup_per_chain discarded + per_chain kept), default = `samples`. Discarded upstream by Turing (`nadapts`/`discard_initial`), never reaches `TR.samples`; `warmup=0` disables. Passes `per_chain` as `N` to Turing `sample`. Recovers unstandardised params OUTSIDE the model: `DimArray(TR.samples)` → `reshape_params` → `unstandardise` (src/reshape.jl) → `TR.parameters`. No in-model back-transform.

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
- `RegressionPrior` — `intercept`/`fixed_effects`/`random_effects`/`auxiliary`. Passed as runtime model args, not baked into the generated `Expr` — keeps `cached_construct_model`'s cache key purely structural (V20).

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
- V25. `fit!` default `adtype` picked from `has_random_effects(TR.modeldata)`: ranef present → `AutoReverseDiff(compile=true)`, else → `AutoForwardDiff()`. Basis: T24 sweep showed ranef presence (not param count) determines which backend wins — see `bench/BENCHLOG.md`. User-supplied `sampler=NUTS(;adtype=...)` always overrides.

## §B BUGS

id|date|cause|fix

## §T TASKS

T14|.|First `using TuringRegressions` / `Pkg.test()` pays full TTFX (Turing/DynamicPPL/model-macro compile) — separate latency source from the fixed T13 gensym issue (V20). Investigate `PrecompileTools.@compile_workload` block (small `turing_glm` fit, minimal `N`/`nchains`) in `src/TuringRegressions.jl`. Open question: does it help given `construct_model`'s model type is `eval`'d at runtime (gensym'd name), not load time — workload warms builder machinery (Turing/DynamicPPL/AD) but not a reusable compiled model type. Needs a think before implementing|V20
T17|.|`TR.link` field (predict.jl) — used internally, V1-bounded to the 5 families, not user-facing. Confirm it can stay internal / no action needed|V1
T19|.|`fit!(quiet=false)` live progress bar not showing in VSCode Julia REPL. Likely needs a progress-capable logger (TerminalLoggers) or VSCode's ProgressLogging integration. Low priority — sort later|
T20|x|Full Pkg.test() too slow for dev iteration. Add small subset: one Normal fit, one Bernoulli fit, mixedmodels benchmark fit only — reusable runner for T21/T22 dev loop|C10
T21|x|Branch: drop predictor standardisation (C2) inside model, keep centering only. Compare fit time + accuracy vs current std approach (V3/V4/V13 tolerances) using T20 subset. CLOSED — tried on branch, decided not to proceed; C2 std stays|C2,V3,V4,V13,T20
T22|.|Post-compile NUTS perf on mixedmodels-benchmark fit v slow vs Stan/lme4 — profile + improve. Umbrella for T23-T32. Use T23 harness to iterate; running results log in `bench/BENCHLOG.md`. GATE: work T24-T32 one at a time; after each, report benchmark delta vs baseline (cold time, warm time, ESS/sec, rhat, param recovery vs lme4 gold — see BENCHLOG.md) to user and WAIT for explicit approve/disapprove before keeping the change or moving to next|T20,T23
T23|x|Built `bench/sleepstudy_bench.jl` — single fixed benchmark fit: `Reaction ~ 1 + Days + (1 + Days | Subject)`, Normal, ~42 params, 180 obs. Reports (a) cold-run wall time incl compile (b) warm-run wall time (c) ESS/sec, rhat per fixef/ranef-sd param (d) posterior mean vs lme4 REML gold. Appends each run to `bench/BENCHLOG.md` (running log, not overwritten) instead of per-gradient BenchmarkTools/DaemonMode profiling — simpler, two full fit! calls in one process cover cold vs warm. `Random.seed!` fixed. lme4 ! valid target for wall time (REML optimiser, not MCMC) — used only as param-recovery gold standard. Every T24-T32 change measured against this ONE script, one change at a time, vs recorded BENCHLOG baseline|T22,C10
T24|x|AD backend. Swept `AutoReverseDiff(compile=true)`, `AutoMooncake()`, `AutoEnzyme()` vs baseline ForwardDiff — see `bench/BENCHLOG.md`. Result: heuristic is RANEF PRESENCE not param count (original guess wrong) — ReverseDiff(compile=true) wins ~2-3.5x on any ranef model (tested ~20 params intercept-only, ~42 params full slope+corr); ForwardDiff wins on fixef-only models regardless of N (3 params/180 obs sleepstudy, ~5-6 params/2201 obs Titanic). Mooncake: 117s cold vs ~30s baseline, no warm win — rejected. Enzyme: `EnzymeRuntimeActivityError` on generated model code, not pursued — rejected. Landed: `fit!` defaults `adtype = has_random_effects(TR.modeldata) ? AutoReverseDiff(compile=true) : AutoForwardDiff()`, overridable via new `adtype` kwarg. `ReverseDiff` promoted from bench-only to main `[deps]`|T23,V25
T25|.|Codegen: hoist ranef container indexing out of hot path. `model.jl:110-115` does `group_idx[:,$i]` (fresh `Vector{Int}` alloc EVERY logdensity eval) + `group_predictors[$i]` (Vector{Matrix} element access). Model code generated per structure → splat ranef data into separate positional args (`group_idx_1::Vector{Int}`, `group_pred_1::Matrix{Float64}`). Concrete types, no indexing, no alloc. Touches `_linear_model`, `_build_model_with_data` (turingregression.jl:237), `modelcode` signature|T23,V20
T26|.|Hand-roll Normal likelihood. `model.jl:146` `logpdf(MvNormal(μ, σ), y)` constructs ScalMat/PDMat wrapper each eval over duals. Replace: `-nobs*log(σ) - sum(abs2, y .- μ)/(2σ^2) - nobs*log(2π)/2`. Same math, no distribution construction. Biggest per-eval saving on Normal path|T23,V3
T27|.|Drop `diagm`. `model.jl:50` `diagm(σ_z)*L.L*z_raw` ≡ `(σ_z .* L.L)*z_raw` (row scaling). Kills p×p alloc + one matmul. Must stay mean-zero (V23)|T23,V23
T28|.|`model.jl:48` `filldist(MvNormal(zeros(p), I), n_groups)` → `filldist(Normal(), p, n_groups)`. Identical distribution, skips MvNormal/PDMat machinery. The no-intercept branch (model.jl:56) already does this — inconsistent|T23
T29|.|Slice allocs in `_linear_model`. `model.jl:111` `sum(pred .* ranef[idx, 2:end]; dims=2)[:]` = 4 allocs (`2:end` slice, `.*`, `sum`, `[:]`). `n_predictors` known at codegen → special-case p==1 to `pred_vec .* ranef[idx,2]`; views for general case|T23
T30|.|Check DynamicPPL `LKJCholesky` bijector cost. Non-centered parameterisation already correct (matches Stan), so remaining gap on the corr prior is transform overhead — Stan's `cholesky_factor_corr` is hand-tuned. Measure before touching|T23,V23
T31|.|Warmup budget. Defaults give 500 warmup/chain (2000 total ÷ 4 via cld); Stan default 1000/chain. Undercooked adaptation → bad step size → more leapfrog steps/draw → slow AND low ESS. Test warmup=4000. May mean changing default `warmup=samples` (C1, I.fit!)|T23,C1,V24
T32|.|`target_acceptance`. `NUTS()` default 0.65; brms/Stan use 0.8 for ranef models. Higher accept = smaller steps, fewer divergences/U-turns. Measure ESS/sec both ways, consider changing default sampler|T23,C1
T33|x|BENCHLOG `Subject_sd Intercept` 38.81 vs lme4 REML gold 24.7. Cause: `_unstandardise_ranef` recentered point estimates from std-scale (mean-Days) back to raw Days=0 but back-transformed `sd`/`corr` by elementwise scaling — inconsistent quantities. FIXED: `_ranef_transform_matrix` builds the affine map `A` (Intercept row carries `-mean_x*y/sd_x`); `M`, `sd`, `corr` all go through one per-draw `Σ_orig = A·(D·R·D)·A'`, SDs read off its diagonal. Correlated terms MUST include `R` — a diagonal-only `A·D·D·A'` drops ρ and gave 47.73. Landed 31.09; then LKJ η 2.0→1.0 (model.jl:47, flat corr prior, matches brms) → 29.14, raw corr −0.019 (gold 0.07), max rhat 1.021→1.008. Residual 29.14 vs 24.7 is not a bug — see T34. C9 re-verify of V3/V4/V13 still outstanding for both edits|V10,V21,C9,T34
T34|.|LOW PRIORITY, think first. C2 standardises outside the model, so the LKJ prior sits on the CENTERED parameterisation's correlation (intercept-at-mean-X vs slope), not the raw-scale one lme4/brms use. LKJ(1) is flat in `ρ_std` but NOT flat in `ρ_raw` — sleepstudy data want `ρ_std ≈ 0.75`, we recover 0.677, and that shortfall alone explains the residual `Subject_sd Intercept` 29.14 vs gold 24.7 (`Var(v₀) = y²(Var(u₀) + (m/s)²Var(u₁) − 2(m/s)·ρ_std·sd(u₀)sd(u₁))` — very sensitive to ρ_std). Not fixable by η tuning; inherent to C2. Options: (a) expose η / the whole corr prior as a `RegressionPrior` field so users can counteract it — cheapest, likely right, (b) reparameterise so the prior lands on raw-scale corr, (c) document as known divergence from lme4 and leave. η currently hardcoded 1.0 at model.jl:47|C2,T33,V10
