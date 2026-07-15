# SPEC — TuringRegressions.jl

Distilled from code 2026-07-13. WIP package. `?` = unconfirmed, user verify.

## §G GOAL

Bayesian GLM package for Julia. Alternative to TuringGLM.jl, more features.
Fit regression via Turing.jl NUTS. Output = `DimArray` (DimensionalData) →
orderless named indexing of params/draws/chains. User writes `@formula`, picks
family, calls `fit!`, then extracts coefs / predicts / metrics / compares / plots.

Core loop: `turing_glm(formula, data, family)` → `fit!` → `summary`/`fixef`/`predict`.

## §C CONSTRAINTS

- C1. Julia. Turing.jl 0.36.3 for MCMC. Sampler default NUTS, parallel MCMCThreads, N=2000, nchains=4.
- C2. Data auto-standardised inside model. User priors must be scaled for std predictors (mean 0, sd 1).
- C3. Params stored/returned as `TR.parameters::DimStack` (post-T1a2, was single `DimArray`). One layer per param group (`:fixef`, per-ranef `:{group}`, `:{group}_sd`, `:{group}_corr`, `:{group}_offset`, `:internals`), each layer's own dims incl `:draw, :chain`. Chains collapsible to one `:draw` dim via `draws(...)`.
- C4. Families supported: `Normal, TDist, Bernoulli, Poisson, NegativeBinomial`. Others → `error`.
- C5. Links fixed per family (util `get_link`): Normal/TDist→identity, Bernoulli→logit, Poisson/NegBin→log.
- C6. Model code generated as Julia `Expr` at runtime, `eval`'d into `@model turing_regression`. Not hand-written.
- C7. Makie plots optional — loaded lazily via `Requires.@require` in `__init__`, NOT native pkg extension. No `ext/` dir, no `[weakdeps]`. `?` migrate to native ext later.
- C8. Standardisation asymmetric: X always scaled if fixed effects; y scaled ONLY for Normal/TDist (count/binary families keep raw y, likelihood on link scale).
- C9. Reexports: DimensionalData, Distributions, `logit`/`logistic`, `@formula` (from MixedModels).
- C10. `@formula` sourced from MixedModels (not StatsModels) → allows `(1|g)` ranef syntax.
- C11. NegBin uses Stan-style `NegativeBinomial2(μ, ϕ)` reparam (mean/dispersion), copied from TuringGLM. `p = max(1/(1+μ/ϕ), 1e-6)` for stability.
- C12. Likelihood via `@addlogprob!` not `y ~ dist`, to allow weights + scaling. `BernoulliLogit`/`LogPoisson` used (log-scale, numerically stable).
- C13. Scaling in→out is fragile. Model fits on standardised X (+y for gaussian); params recovered to original scale in `_generated_quantities` (V10). Every family branches differently (C8). Adding random effects makes this MUCH harder — τ/zⱼ scale must compose with X_stds/y_std correctly. Touch scaling logic → re-verify all families against GLM (V3,V4).

## §I INTERFACES (public surface)

Model creation:
- `turing_glm(formula::FormulaTerm, data::DataFrame, family, priors=default_prior(family), weights=nothing, show_code=false)` → `TuringRegression{family}`
- `turing_glm(y::Vector, X::Array, ::Type{T}; names=Symbol[], kwargs...)` — array form, synth formula, forwards.
- `default_prior(family)` / `default_prior(TR)` → `RegressionPrior` (intercept N(0,5), fixef N(0,2), ranef Exp(1), aux family-dep).

Fitting:
- `fit!(TR; sampler=NUTS(), parallel=MCMCThreads(), N=2000, nchains=4, quiet=true, kwargs...)` → mutates TR. Recovers unstd params via `generated_quantities` → `TR.parameters` DimArray.

Param extraction — REWRITTEN T1a2 (src/parametermethods.jl). Old `parameters`/`fixef`/`internals`/`coef`/`get_parameters`/`parameter_names` all REMOVED, replaced by `draws`:
- `draws(TR; drop_warmup=200, n_draws=-1, collapse=true)` → whole `TR.parameters` `DimStack` (all layers), warmup dropped/chains collapsed per kwargs.
- `draws(TR, type::Symbol; drop_warmup=200, n_draws=-1, collapse=true)` → single layer as `DimArray`. `type` must be one of `propertynames(TR.parameters)` (else `ArgumentError` listing valid types): `:fixef`, `:{group}` (ranef effect×group), `:{group}_sd`, `:{group}_corr` (only if correlated ranef), `:{group}_offset` (only slope-only-no-intercept ranef, internal use), `:internals`.
- `draws(fun::Function, TR, type::Symbol; dropdims=true, kwargs...)` → aggregates draws(+chains) with `fun` (e.g. `median`), drops resulting singleton dims by default.
- `outcome(TR)` → y as `DimArray` (`Dim{:row}`).
- `predictors(TR, type::Symbol)` → `:fixef` implemented (X `DimArray`, dims `Dim{:row}, Dim{:var}`); `:ranef` → `error("Not implemented")` (TODO).
- `outcome_as_distribution(TR)` — Bernoulli only → `UnivariateFinite`.

`TR.parameters::DimStack` layers (built `fit!`, turingregression.jl:265-344):
- `:fixef` — α, β (renamed to `TR.X_names`), aux params (σ/ν/ϕ). Dims `Dim{:fixef}, :draw, :chain`.
- per ranef grouping term, layer named `group = re.variable` — Dims `Dim{:effect}` (Intercept+slopes), `Dim{:group}` (levels), `:draw`, `:chain`.
- `{group}_sd` — group-level SDs, `Dim{:effect}, :draw, :chain`.
- `{group}_corr` — only if correlated intercept+slopes (full `L*L'`), `Dim{:effect}, Dim{:effect2}, :draw, :chain`.
- `{group}_offset` — only slope-only-no-intercept ranef terms (centering offset absorbing X-centering, no ranef intercept to absorb it); NOT user-facing, consumed internally by predict.jl (T1b) only. `Dim{:group}, :draw, :chain`.
- `:internals` — sampler diagnostics (lp, tree_depth etc), `Dim{:internal}, :draw, :chain`.

NOTE: predict.jl NOT yet migrated to `draws` API — still calls removed `get_parameters` (predict.jl:57,62,116,119,127). Currently broken/stale; blocks T1b completion.

Prediction (`type ∈ :posterior|:epred|:linpred`):
- `predict(TR, X::Matrix, fun=nothing; type=:posterior, kwargs...)`
- `predict(TR, fun=nothing)` — fitted data.
- `predict(TR, new_data::DataFrame, fun=nothing)` — rebuild X from formula.
- internal: `linpred` (Xβ+α), `epred` (invlink∘linpred), `posterior_pred` (add family noise).

Metrics (StatisticalMeasures.jl):
- `calculate_metrics(TR, metrics::Vector, fun=nothing; threshold=0.5, kwargs...)` — epred-based. Bernoulli branch: AUC + pseudo_r2 special-cased, rest categorical.
- `default_metrics(TR, fun=nothing)` — regression [rsq,rmse,mae]; Bernoulli [accuracy,kappa,TPR,TNR,auc,pseudo_r2].
- `pseudo_r2(preds, y)` — McFadden, exported.

Comparison (ParetoSmooth.jl):
- `psis_loo(TR)` → PsisLoo. `loo_compare(models...)` / `loo_compare([models])` — kwargs `model_names`.

Display:
- `Base.show(io, TR; warnings=true)` — family/formula/prior/obs/samples + warnings.
- `Base.summary(io, TR; funs=[mean,std], quantiles=[0.025,0.975], return_table=false, draws_idx, kwargs...)` — pretty table + metrics + warnings.
- `model_warnings(TR)` / `model_warnings(chain_info)` — rhat/ess/mcse thresholds.
- `pretty` — EXPORTED + tested (`sprint(pretty, model)`) but NO DEFINITION in src. `?` missing/removed. Likely alias of `summary`. BUG B1.

Plots (Requires/Makie, exported inside `__init__`):
- `lineribbon` — Makie `@recipe`, median line + quantile bands (widths [0.66,0.95], greys).
- `conditional_dependency(TR, var::Symbol; type=:posterior)` — vary one var, hold rest at mean.
- `pp_check_hist` / `pp_check_dens` / `pp_check_dens_overlay(TR; ...)` — posterior predictive checks.

Types:
- `TuringRegression{T<:Distribution}` mutable — formula, model fn, prior, link, y, X, z, weights, X_names, z_names, modelinfo, modelcode, samples, parameters.
- `ModelInfo` — has_intercept/has_fixed_effects/has_random_effects/weighted flags.
- `RegressionPrior` `@kwdef` — intercept/fixed_effects/random_effects/auxiliary Distributions.

## §V INVARIANTS

- V1. Family ∉ {Normal,TDist,Bernoulli,Poisson,NegativeBinomial} → `error` at `turing_glm`.
- V2. `fixef`/`predict`/etc on unfitted model (`samples===nothing`) → `ArgumentError "not been fitted"`.
- V3. `draws(median, TR, :fixef)` ≈ Bayesian point ests ≈ GLM MLE coefs, atol 0.025 on standardised recovery (test Vs. GLM). Holds Normal/Poisson/NegBin (runtests:31,42,56). Bernoulli looser → V13. `?` runtests not yet updated to new `draws` API — re-verify.
- V4. `predict(type=:epred)` ≈ `GLM.predict` (test atol 0.1 Normal, 1 Poisson, 2 NegBin, 0.05 Bernoulli).
- V5. Link relations: `linpred == log(epred)` for log-link; `linpred == epred` for identity.
- V6. `var(posterior) > var(epred)` — posterior pred adds noise. `var(posterior) > var(epred)` also count.
- V7. `draws(TR, type)` shape = (n_effect_dims..., (N-drop_warmup)*nchains) collapsed; extra trailing `:chain` dim if `collapse=false`.
- V8. `n_draws > available post-warmup` → `ErrorException`.
- V9. Returned param labels: `[:α, :<X_names...>, aux...]`; α first when present.
- V10. Standardised→original param recovery in `_generated_quantities`: β_orig = (y_std/X_stds).*β (gaussian) or β./X_stds (count/binary); α likewise with `dot(X_means, β_orig)`.
- V11. summary auto drop_warmup: 0 if N<400 else 200.
- V12. rhat>1.05 / ess<100 / mcse>5%std → `@warn`; softer thresholds → `@info`.
- V13. Bernoulli param recovery vs GLM looser: coef atol 0.05, epred atol 0.05 (runtests:68,71). Split from V3.
- V14. Weighted fit with `weights ≡ 1` == unweighted fit (params equal, tolerance). Guards `_weighted_likelihood` (model.jl:105) — currently ZERO tests. Untested until written.
- V15. `predict(TR, new_data::DataFrame)` uses RAW new X (formula_handlers.jl:9) with ORIGINAL-scale stored β (model.jl:174 β_original). No re-standardisation of new X. epred on new data ≈ GLM.predict on same new data. Guards against re-introducing standardise in predict path.
- V16. `extract_random_effect` predictor slicing must condition on `has_intercept(term.lhs)`: drop col 1 only when true, else use all cols. Guards `(0+x...|g)` ranef terms getting correct predictor count/identity.
- V17. Non-ranef formulas must skip the `MixedModels.MixedModel` construction path entirely in `extract_model_data` — use plain `StatsModels.modelmatrix`/`ModelFrame` when `Z` is empty, only route through `MixedModels` when ranef terms present. Guards `turing_glm` working for ordinary (no `|`) formulas.

## §T TASKS

id|st|task|cites
T1a1|x|: finish mixed model on `random_effects` git branch - done, details in git.
T1a2|x|OUTPUT SHAPE - turingregression output changed to DimStack, parameter methods redone as `draws`/`outcome`/`predictors` (src/parametermethods.jl). Old `parameters`/`fixef`/`internals`/`coef`/`get_parameters`/`parameter_names` removed. predict.jl NOT yet migrated (T1b) — still calls removed `get_parameters`, currently broken.|C3,I.param
T1b|.|predict.jl ranef support, depends T1. Today `linpred` (src/predict.jl) zero random-effects awareness — sums only α+X*β, ignores TR.z entirely. Add: accept new-data grouping-factor levels (unseen level → clear error, no silent NA/missing); look up per-group ranef draws via new `ranef(TR,group_var)` accessor (intercept dev + slope devs + `:<group>_offset` layer when present), add group contribution to μ for `:linpred`/`:epred`/`:posterior`. Handle `predict(TR)` (fitted TR.X, groups known) and `predict(TR,new_data::DataFrame)` (map new_data grouping col values onto TR.z levels). Files: src/predict.jl. TESTS (test/runtests.jl "Random Effects Prediction" testset, written now, fails until T1b lands): predict on fitted data reproduces in-sample fit reasonably; predict on new data w/ known levels matches manual reconstruction from TR.parameters; predict w/ unseen grouping level errors clearly|T1,I.pred,V4
T1c|.|MINOR, depends T1: grouping-var collision guard. 2 ranef terms sharing a grouping var (e.g. `(x1|g)+(x2|g)`) collide on DimStack layer name `:g` — 2nd term's layer silently overwrites 1st, pre-existing risk carried forward (same collision existed in old string-label scheme, not a regression). Add `@warn` in `extract_model_data`/`turing_glm` (src/formula_handlers.jl or src/turingregression.jl) when `length(unique(variable for term in Z))<length(Z)`, naming colliding variable. Warning only — real fix (per-term layer namespacing) deferred|T1
T2|.|DROP `pretty` — unexport (TuringRegressions.jl:44) + delete test line (runtests:282). No alias; `summary` already does it|B1,I.display
T3|.|fix `posterior_pred` NegBin: reads param `:ϕ⁻`, but `_generated_quantities` returns `:ϕ`|B2,I.pred
T4|.|fix `show` NegBin branch: `T == NegativeBinomial2` never true (T is NegativeBinomial)|B3,I.display
T5|.|Canonical NegBin = LOCAL `NegativeBinomial2` (utils.jl:46). Fix predict.jl:128 to use local not `TuringGLM.NegativeBinomial2`. Then REMOVE TuringGLM from Project.toml — it's the only use (P1); heavy dep gone|C11,B2
T6|.|repair commented-out testsets (Model Creation, Model Fit) — ref stale fields (`unstd_params`, `standardized`, `Z_names`) not on struct|V2
T7|.|fix test var-name mismatches: `mod`/`model_count`/`mod_empty` vs defined `model`/`mod_count`/`model_empty`|
T8|x|resolved by T1a2: fn is now `predictors(TR, type)`, `fixed_effects` name dropped entirely. Readme/docstring now consistent — verify readme text updated too.|I.param
T9|.|readme API lists `linpred`/`epred`/`posterior_pred` as public, not exported. Export or relabel internal.|I.pred
T10|.|`_weighted_likelihood` exists but no test + no exposed `weights` path in readme; verify weighted fit works|C12
T11|.|readme usage block corrupted (compressed `[271 items...]`, typos `TuringGLModels`, `fucntion`). Rewrite.|I
T12|.|P5: inline `data_response` (formula_handlers.jl:4–6) at turingregression.jl:78, delete the wrapper fn (one line round `response()`)|
T13|.|P6: collapse two `loo_compare` bodies → vararg forwards to vector: `loo_compare(m::TuringRegression...; kw...) = loo_compare(collect(m); kw...)` (comparison.jl:27,43)|I.comp
T14|.|P7: array-form `turing_glm` builds formula via `eval(Meta.parse("@formula(...)"))` (turingregression.jl:138–139). Replace with programmatic `term(:y) ~ sum(term.(xnames))` — no eval, no parse|I.model
T15|.|P4: `const default_options` (summary.jl:183) — untyped module global, violates no-untyped-global rule|
T16|.|write plot tests — runtests warns "No tests yet implemented for plots"; add headless CairoMakie target|I.plots
T17|.|BIG JOB: swap generated model `Expr` → strings-with-comments so `show_code`/printed model carries explanatory comments (Expr strips them). Rewrites model.jl code-gen (V3/V4 guard) — re-verify all families vs GLM after. FOLD IN P2: collapse `_likelihood`+`_weighted_likelihood` (90% dup) into one family dispatch, weights default `ones` → makes V14 true by construction. FOLD IN scaling redesign: move standardise/back-transform OUT of generated model — compute scaling stats ONCE outside (NamedTuple `X_means/X_stds/y_mean/y_std` + per-ranef), pass scaled data in, back-transform (fixef+ranef+Σ) in Julia in `fit!`, DELETE `_standardise_data`+`_generated_quantities`. Reworks the in-model ranef scaling T1 wrote. Round-trip test `unstandardise∘standardise==id`|C2,C6,C8,I,V3,V4,V14
T18|.|BIG JOB: move Makie plots out of `Requires.@require`/`__init__` into native pkg extension (`ext/`, `[weakdeps]`, `[extensions]`) per new Julia usage. Also drop Colors dep (P3): only use is `colormap("Grays",125)` lineribbon.jl:45 — use Makie's `cgrad`/`to_colormap` in the ext instead|C7
T19|.|BIG JOB: prior center+scale wiring (FLIPS C2). Today user must specify priors on the standardised (mean 0, sd 1) scale. Change so priors are given on the ORIGINAL data scale — e.g. `Normal(10,20)` on a coef whose predictor has mean 10, sd 20 → transformed to `Normal(0,1)` for the internal standardised fit — and reported back on original scale in `summary`/`show`/prior display. REUSE the centralised affine map from T17 — prior forward-transform is the INVERSE of the param back-transform; don't hand-write new per-family algebra. Transform via `Distributions.AffineDistribution` (`shift + scale*d`) — works for any univariate prior, no param rewriting; store user's original prior for display, fit with scaled. Guard: aux σ = scale-only (shift 0). Verify Turing/NUTS samples AffineDistribution + filldist cleanly. Update C2, V-invariants + tests|C2,I|NEW
T20|.|BIG JOB: make `TuringRegression` implement the StatsAPI/StatsBase `RegressionModel` interface (coef, coefnames, nobs, dof, dof_residual, vcov, stderror, loglikelihood, deviance, residuals, fitted, predict, response, modelmatrix, formula, confint, ...) sensibly for a Bayesian fit — posterior-based analogues (point est = median, vcov = posterior cov, confint = credible interval), error/skip methods with no Bayesian meaning. Consider wrapping in `StatsModels.TableRegressionModel` so formula-schema machinery + `@formula` term handling come for free. New invariant + tests|I|NEW

NOTE|.|P8: `epred` picks invlink by function `==` on TR.link (predict.jl:92–100). Fragile but V1-bounded to 3 links. No action unless a 4th link appears|V1


## §B BUGS

id|date|cause|fix
B1|2026-07-13|`pretty` exported (TuringRegressions.jl:44) + tested (runtests:282) but no definition|T2
B2|2026-07-13|`posterior_pred` NegBin gets `:ϕ⁻` (predict.jl:127); generated_quantities returns `:ϕ` (model.jl:197). Also calls `TuringGLM.NegativeBinomial2` not local. MethodError/KeyError on NegBin posterior predict|T3,T5
B3|2026-07-13|`show` (turingregression.jl:184) tests `T == NegativeBinomial2` — wrong; family type is `NegativeBinomial`. NegBin aux prior line never prints|T4
B4|2026-07-14|`extract_random_effect` (formula_handlers.jl:106) slices `modelcols(term.lhs,d)[:, 2:end]` unconditionally, assuming leading intercept col. Slope-only-no-intercept ranef terms e.g. `(0+x1+x2|g)` have NO intercept col there — slice wrongly drops first real predictor (x1) instead. Discovered verifying T1 Step 4 (model.jl generic, bug is upstream)|V16
B5|2026-07-14|`extract_model_data` (formula_handlers.jl:76) builds X via `MixedModels.modelmatrix(MixedModel(formula,data))` unconditionally — MixedModels.jl throws `ArgumentError "Formula contains no random effects"` for formulas with zero ranef terms. Blocks ALL non-ranef `turing_glm` calls, contradicting V3's claim `Vs. GLM` testset currently passes — needs re-verification. Confirmed pre-existing (unchanged since HEAD~1), not introduced this session|V17
