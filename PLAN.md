# PLAN — T3/T4/T5 big think

Draft. Data-flow map + refactor design. Discuss before coding.

## §1 CURRENT DATA FLOW (as-built)

```
turing_glm(formula, data, family)
  │
  ├─ extract_model_data(formula, data)          [formula_handlers.jl]
  │    → y   ::Vector           RAW (unstandardised)
  │    → X   ::Matrix           RAW, intercept col dropped
  │    → Z   ::Vector{RandomEffect} | nothing   RAW ranef predictors + grouping
  │    → formula_with_schema   (schema/contrasts baked in — reused at predict)
  │
  ├─ ModelInfo(has_intercept, has_fixef, has_ranef, weighted)
  │
  ├─ cached_construct_model(family, modelinfo, Z, priors, show_code)
  │    key = structural (family, modelinfo bools, per-ranef shape, prior params)
  │    build_model_body → @model fn, gensym'd name, eval'd. Body contains:
  │       _standardise_data   ← SCALING COMPUTED HERE, INSIDE MODEL (per NUTS eval!)
  │           X_means/X_stds/X_scaled, per-ranef Xmn/Xstd/Xscaled,
  │           y_mean/y_std/y_scaled  (y scaled ONLY Normal/TDist)
  │       priors               ← interpolated as $prior literals (STANDARDISED scale)
  │       _linear_model        μ = α + X_scaled·β + ranef      (standardised)
  │       _likelihood          on y_scaled (Normal/TDist) | y (count/binary)
  │       _generated_quantities ← BACK-TRANSFORM HERE, INSIDE MODEL
  │           α/β/σ/ranef → original scale, returned as NamedTuple per draw
  │
  └─ TuringRegression{family}  stores RAW y,X,Z,weights + model fn + modelcode

fit!(TR)
  │
  ├─ _build_model_with_data(TR)   condition model on RAW y,X, n_gr/group_idx/
  │                               group_predictors, weights
  ├─ sample(...) → MCMCChains.Chains   (raw STANDARDISED-scale params + internals)
  ├─ generated_quantities(model, samples)  → back-transformed ORIGINAL-scale
  │                                          params, NamedTuple per (draw,chain)
  └─ manual per-layer Dict + stack + vcat → DimStack → TR.parameters
         layers: :fixef, :{group}, :{group}_sd, :{group}_corr,
                 :{group}_offset, :internals              (ORIGINAL scale)

posterior_predict(TR, X | new_data)          [predict.jl]
  │   uses TR.parameters (ORIGINAL scale) + RAW X   → NO standardisation (V15)
  ├─ linpred:  μ = X·β + α + ranef            (original params, raw X)
  ├─ epred:    link⁻¹(μ)
  └─ posterior_pred: sample observation dist
```

## §2 THE AFFINE MAP (the one true idea)

Standardisation is a **pure affine map** fixed by 4 constants + link.
No draw-dependence except the params themselves. Runs THREE directions:

Constants (computed once from fit data):
- `X_means, X_stds`  (length p)
- `y_mean, y_std`    (scalars; =0,1 when y NOT scaled)
- per-ranef `Xmn_g, Xstd_g`
- `scale_y::Bool` = family ∈ {Normal, TDist}

### 2a. FORWARD — scale data (feed model)
```
X_scaled = (X - X_means) / X_stds
y_scaled = (y - y_mean) / y_std          (only if scale_y)
```

### 2b. BACKWARD — unscale params (T3, today in _generated_quantities)
identity link (Normal/TDist):
```
β_orig = (y_std / X_stds) .* β
α_orig = y_mean - dot(X_means, β_orig) + y_std·α
σ_orig = y_std · σ
```
log/logit link (Bernoulli/Poisson/NegBin — y NOT scaled):
```
β_orig = β ./ X_stds
α_orig = α - dot(X_means, β_orig)
```
ranef: same shape per grouping term (see model.jl:317-345), SD + corr too.

### 2c. BACKWARD-ON-DISTRIBUTIONS — map priors (T4)
User gives prior on ORIGINAL β_orig ~ Normal(m, s). Model samples β_std.
β_std = (X_stds / y_std)·β_orig  ⇒  β_std ~ Normal(m·X_stds/y_std, s·X_stds/y_std).
This is exactly `Distributions.AffineDistribution` — same constants, inverse of 2b.

**2b and 2c are inverses of ONE map. Single source of truth. This is why T3+T4
must be built together — one `ScalingInfo` + one affine module serves both.**

### 2d. RANEF back-transform (model.jl:306-364 — the hairy part)
Ranef predictors scaled by their OWN per-group Xmn_g/Xstd_g (NOT the fixef ones).
y_std factor applies only when scale_y. Three shapes, each different:

**intercept + slope** (correlated):
```
β_orig  = (y_std ./ Xstd_g) .* r'              # slopes, per group col
α_orig  = y_std·r_intercept - [dot(Xmn_g, β_orig[:,g]) for g]   # intercept absorbs centering
σ_orig  = vcat(y_std·σ[1], (y_std ./ Xstd_g).·σ[2:end])         # SD per effect
R_orig  = L·L'                                 # CORRELATION UNSCALED (scale-invariant)
```
**slope only (no intercept)** — the tricky one:
```
β_orig    = (y_std ./ Xstd_g) .* r'
σ_orig    = (y_std ./ Xstd_g) .* σ
offset_g  = [-dot(Xmn_g, β_orig[:,g]) for g]   # HIDDEN centering offset
```
  No ranef intercept exists to absorb the `-dot(Xmn,β)` shift from centering →
  parked in a private `:{group}_offset` layer, consumed ONLY by predict.jl
  (V16 fixed this 2026-07-15). Refactor MUST preserve offset or slope-only ranef
  predictions break silently.

**intercept only**:  `α_orig = y_std·r`,  `σ_orig = y_std·σ`.

### 2e. RANEF constraints that MUST survive
- **V20**: ranef components mean-zero in construction (`diagm(σ)·L·z_raw`). Only
  α/β carry mean. Back-transform must not inject a mean. The offset (2d) is NOT a
  mean on the ranef — it's a fixed centering correction folded into prediction.
- Correlation R is scale-invariant → never touched by the affine map.
- Ranef affine map is per-grouping-term, keyed on the SAME 3-shape branch as
  extract_random_effect (V16). Forward/backward/prior-map all branch identically.
- T4 prior mapping for ranef: `random_effects` prior is on the SD (Exponential(1)).
  Original-scale SD prior → standardised via `Xstd_g / y_std` factor, same as 2c.

## §3 TARGET ARCHITECTURE — 2 structs + 3 functions (user's design)

The data representation is now unified. `ModelData`/`LinearModelData` (already
committed on `scaling_refactor`) is the "nice struct" holding data for ANY model
shape — fixef, ranef, weighted. `Transform` is its affine companion. Three
functions, everything broadcast over the whole DimArray in one go:

```
1. apply_formula(formula, data [; reference, allow_new_levels])  → ModelData
      formula + raw table → ModelData (RAW, original scale).
      Works for all shapes. With `reference` (a fitted ModelData) it remaps ranef
      levels + warns on unseen levels.  COALESCES: extract_model_data +
      extract_random_effect + new_random_effects + _remap_levels.

2. standardise(md::ModelData, family)  → (md_std::ModelData, tf::Transform)
      compute §2 constants from md, return standardised ModelData + the Transform.
      Round-trip guard: apply(tf, md) ≈ md_std, and unstandardise reverses it.
      COALESCES / DELETES: _standardise_data (model.jl).

3. unstandardise(raw_params::DimArray, tf::Transform)  → DimStack
      inverse affine (§2b/§2d) over ALL draws×chains by broadcast, emits the
      layered DimStack (:fixef, :{group}, :{group}_sd/_corr/_offset).
      COALESCES / DELETES: _generated_quantities (model.jl) + the manual
      per-draw extraction loop (turingregression.jl:279-358).
```

Flow:
```
turing_glm(formula, data, family)
  ├─ md = apply_formula(formula, data)                   RAW ModelData
  ├─ md_std, tf = standardise(md, family)                §2a  (or defer to fit!, §5.2)
  ├─ priors_std = tf(user_priors)                        T4, §2c — inverse-on-dists
  ├─ model = cached_construct_model(family, md)          structural key only; NO
  │     scaling, NO _generated_quantities, NO prior literals. Pure std-space model.
  └─ TR stores md (raw) + tf + model    (+ priors)

fit!(TR)
  ├─ condition model on md_std   (grouping arrays derived from md_std.Z — see §5.3)
  ├─ sample(...; chain_type = VNChain)                   T3b FlexiChains
  ├─ raw = DynamicPPL.returned(model, chain; stack=true) → ONE (iter,chain,param) DimArray
  │        (R9: single DimArray auto-stacks; NamedTuple does NOT)
  └─ TR.parameters = unstandardise(raw, TR.tf)           DimStack, original scale

predict(TR, new_data)
  ├─ md_new = apply_formula(TR.modeldata.f, new_data; reference=TR.modeldata,
  │                         allow_new_levels)             remap + warn, ONE path
  └─ original-scale params + RAW md_new.X → μ            UNCHANGED math (V15)
```

## §4 THE STRUCTS

Committed already (`scaling_refactor`), renamed `LinearModelData`→`Predictors` /
field `linearmodeldata`→`predictors` (2026-07-16, ranef predictors are predictors
too so the fixef-flavoured name didn't fit):
```julia
struct Predictors                # reused for fixef AND each ranef
    has_intercept::Bool
    has_fixed_effects::Bool
    X::AbstractMatrix
    X_names::Union{Nothing,Vector{String}}
end
struct RandomEffect
    variable::Symbol; levels::Vector; level_index::Vector{Int}
    predictors::Predictors
end
struct ModelData
    f::FormulaTerm; y::AbstractVector
    predictors::Predictors                # fixef
    Z::Vector{RandomEffect}               # empty ⇒ no ranef
    weights::Union{Nothing,Vector{Float64}}
end
```
`ModelInfo` is DELETED — its 4 bools now derive via multiple-dispatch accessors in
`formula_handlers.jl`: `has_intercept(md)`/`has_fixed_effects(md)` from `md.predictors`,
`has_random_effects(md) = !isempty(md.Z)`, `is_weighted(md) = !isnothing(md.weights)`.
Same 4 accessors overloaded for `TuringRegression` (`turingregression.jl`) so call
sites don't care whether they hold a `ModelData` or a fitted `TR`.

NEW — the affine companion (mirrors ModelData's shape so ops broadcast cleanly):
```julia
struct LinearTransform          # per LinearModelData (fixef + each ranef)
    means::Vector{Float64}
    stds::Vector{Float64}
end
struct Transform
    fixef::LinearTransform
    y_mean::Float64; y_std::Float64; scale_y::Bool
    ranef::Vector{LinearTransform}        # aligned with ModelData.Z
    link                                  # picks identity vs log/logit branch (§2b)
end
```
`tf` stored on TR. `standardise` builds it; `unstandardise`/`tf(prior)` apply it.

## §5 OPEN QUESTIONS / DECISIONS

**5.1 Priors: literals vs model args?**  Today priors baked into Expr → part of
cache key. Under T4 priors become data-dependent (scaled) → baking them explodes
cache (every dataset = new entry). Fix: pass prior DISTRIBUTIONS as model arguments,
generated code says `α ~ prior_α`. Then cache key drops prior entirely → purely
structural (family + shapes). Cleaner, changes model signature. **Recommend: yes,
args.** Confirm.

**5.2 Where compute scaling — turing_glm or fit!?**  Data known at turing_glm.
T4 needs `tf` to map priors at construct time. ⇒ `standardise` in turing_glm, store
`tf` on TR. Model stays cached structurally (data-independent) — `tf` never enters
the model, only the standardised ModelData fed to it + prior args. Consistent w/ 5.1.

**5.3 Model signature — DECIDED: unpacked args (perf).**  Model keeps
`(y, X, n_groups, group_idx, group_predictors, weights)` unpacked. Two reasons:
(1) `ModelData` has abstract fields (`y::AbstractVector`, `X::AbstractMatrix`) →
type-unstable inside `@model`, slow AD; (2) grouping arrays must be built ONCE, not
per logdensity eval. Keep storage as `ModelData` on TR; a slim helper (old
`_build_model_with_data` role) takes `md_std` → derives grouping arrays once →
calls `TR.model(y, X, n_groups, ...)`. Revert WIP `TR.model(TR.modeldata)` sites.

**5.4 T5 likelihood merge.**  Collapse `_likelihood`+`_weighted_likelihood` →
one family-dispatched fn, `weights` defaulting `ones(nobs)`. Makes V14 structural.
NOTE current WIP `_pointwise_loglik` inverted the branch: `if !isnothing(weights)`
returns the UNWEIGHTED expr — bug to fix during the merge.

**5.5 T3b sequencing.**  standardise-out and FlexiChains-swap both land in the new
`unstandardise`. Build `unstandardise` FlexiChains-native from the start ⇒ do both
together, one function, no double rewrite. **Lean: together.**

**5.6 V-invariant re-verification.**  V5,V10,V13,V14,V15,V20 all touch scaling/
back-transform math. Round-trip test `unstandardise∘standardise == id` is the new
guard. Re-run full NUTS param-recovery vs GLM after move (V5/V13).

## §6 CURRENT WIP STATE (scaling_refactor) — RESOLVED 2026-07-16

All of §7 steps 1+2 done, package compiles clean (`using TuringRegressions`).
Fixed, in order:
- `formula_handlers.jl`: `has_intercept = has_intercept(formula)` local-shadows-fn
  bug → renamed local to `formula_has_intercept` in `extract_model_data`.
- `LinearModelData`→`Predictors` rename (see §4) across all files.
- `model.jl` `_random_effects`/`_linear_model`/`_standardise_data`/
  `_generated_quantities`: every `ranef.has_intercept`/`.has_fixed_effects`/`.X`
  → `ranef.predictors.{has_intercept,has_fixed_effects,X}` (double-dot typo +
  wrong-var-name class of bug, was throughout).
- `model.jl` `_pointwise_loglik`: signature was `(family, modeldata::ModelData)`
  but called with a `weighted::Bool` — changed signature to `(family, weighted)`
  to match call site, and flipped the inverted branch (§5.4 note) since it was
  already blocking compilation, not just a latent bug.
- `model.jl` `build_model_body`/`construct_model`: now take `modeldata::ModelData`
  directly (not `model_info::ModelInfo` + separate `model_ranef`), deriving bools
  via the §4 accessors. Model signature resolved per §5.3:
  `(y, X, n_groups, group_idx, group_predictors, weights)`.
- `model_cache.jl`: `_model_cache_key`/`cached_construct_model` updated to the
  `ModelData`-based signature; cache key's structural tuple built from the §4
  accessors instead of the deleted `ModelInfo`.
- `turingregression.jl`: `turing_glm` constructor was missing the `formula` field
  entirely (arg-count mismatch) — now passes `modeldata.f`. Added
  `_build_model_with_data(TR)` (the resurrected slim helper from §5.3) — computes
  `n_groups`/`group_idx`/`group_predictors` once from `TR.modeldata.Z` and calls
  `TR.model(...)` with the unpacked signature. `fit!` and `comparison.jl`'s
  `psis_loo` both call it now instead of the stale `TR.model(TR.modeldata)`.
- `predict.jl`/`statsapi.jl`/`summary.jl`/`parametermethods.jl`: every stale
  `TR.X`/`TR.X_names`/`TR.z`/`TR.y`/`TR.weights`/`TR.modelinfo.*` reference
  rewritten to `TR.modeldata.predictors.*` / `TR.modeldata.Z` / `TR.modeldata.y` /
  `TR.modeldata.weights` / the §4 accessor functions.
- `extract_model_data`: `weights` param given a `=nothing` default — predict.jl's
  `posterior_predict(TR, new_data::DataFrame)` path calls it with 2 args.

**Verified 2026-07-16 (later same day)**: scratch `fit!()` run for fixef-only,
ranef intercept+slope, and ranef slope-only (Normal family) — all fit clean, no
errors. §6 fully resolved. Round-trip/V-invariant re-verification (§5.6) still
pending, comes with T3 (§3 steps 3-4, not started).

## §7 SEQUENCE (user-ordered)

Build + scratch-test at every step. NO `Pkg.test()` until the very end.

1. **Simplify formula_handlers** — DONE 2026-07-16. See §7a below.
2. **Fix fit!/model.jl grouping** — resurrect the slim helper that computes
   n_groups/group_idx/group_predictors once from the (std) ModelData and calls the
   unpacked model (§5.3). Fix §6 model.jl bugs. Branch compiles + scratch fit works.
   DONE — see §6 verification above; `_build_model_with_data` already resurrected
   as part of the §6 fixes, confirmed working by the scratch fits.
3. **Scaling extraction** — `Transform` + `standardise`; move scaling OUT of
   model.jl into Julia; feed std data to model. Round-trip test. Priors as args (5.1).
   NOT STARTED.
4. **FlexiChains + unscaling** — swap chain type; `returned(...; stack=true)`;
   `unstandardise` builds the DimStack (deletes _generated_quantities + extraction
   loop). Then T4 prior-map, T5 likelihood merge, T3c dim renames.

Then, and only then, `Pkg.test()` full re-verify (V5/V13/V14/V15/V20).

## §7a STEP 1 DONE — formula_handlers simplified (2026-07-16)

`extract_predictors(term::MatrixTerm, d)` is now the ONE reusable extraction fn
(§7 step 1's ask). Key insight, confirmed by scratch probe: after
`apply_schema(formula, schema(formula, data), MixedModel)`, the fixed-effect part
of `f.rhs` is **always exactly one `MatrixTerm`** — whether or not any
random-effects terms are present, even when it's intercept-only or has zero
predictors (`0 + (1|g)` still yields a bare `MatrixTerm{Tuple{InterceptTerm{false}}}`).
Each `RandomEffectsTerm.lhs` is the same kind of `MatrixTerm`. So one function,
using `StatsModels.hasintercept`/`coefnames` on the term, builds `Predictors` for
both fixef and every ranef — replacing the old duplicated hand-rolled logic.

Deleted entirely (no longer needed):
- `has_intercept(formula)` — custom `ConstantTerm`-scanning function.
- `get_fixef_names` — the `ModelFrame(formula, data)` + `coefnames` + string-filter
  workaround. `coefnames(term)` on the MatrixTerm gives the same expanded
  per-column names directly (verified: correctly expands multi-level categoricals,
  e.g. `["(Intercept)", "HP", "gp: 6", "gp: 8"]`).

Also (from a follow-up in-session request): **`Predictors.has_fixed_effects` field
removed**, replaced by a function `has_fixed_effects(p::Predictors) = size(p.X, 2)
> 0`, with dispatch overloads for `RandomEffect`/`ModelData`/`TuringRegression`
mirroring the existing accessor pattern. All `.predictors.has_fixed_effects` field
accesses across `model.jl`/`model_cache.jl`/`predict.jl`/`turingregression.jl`
rewritten to the function call. Caught a shadowing bug this introduced: `model.jl`'s
`_standardise_data`/`_generated_quantities` each have a local `Bool` parameter
literally named `has_fixed_effects` — inside those functions the new function name
was shadowed, so calls at the ranef-loop sites needed `TuringRegressions.has_fixed_effects(...)`
to reach the real generic instead of erroring "objects of type Bool are not callable".

Also moved (from `predict.jl`, at user's request — these are formula/`ModelData`
concerns, not predict concerns): `new_random_effects`/`_remap_levels`. Signature
changed from `new_random_effects(TR::TuringRegression, new_data; ...)` to
`new_random_effects(reference::ModelData, new_data; ...)` — decouples
`formula_handlers.jl` from the `TuringRegression` type (defined later in the
include order; needed to avoid a forward-reference) and lines up with §3's
`apply_formula(...; reference=...)` design. `predict.jl` call sites updated to
pass `TR.modeldata`.

**Bug found + fixed during verification** (pre-existing, not introduced by this
session, but caught while scratch-testing the moved functions):
`extract_random_effect` used `MixedModels._ranef_refs`, which looks grouping
values up in the term's fitted contrasts dict and throws `KeyError` on any
level unseen at fit time. This broke `posterior_predict(TR, new_data::DataFrame)`
on data with a new grouping level — it errored inside `extract_model_data`
*before* `_remap_levels`'s `allow_new_levels` handling ever got a chance to run,
regardless of the flag. Fixed by replacing with an own `_ranef_group_values`
(handles plain `CategoricalTerm` and `InteractionTerm` grouping, e.g.
`item:subject`) that reads grouping values straight off the raw data instead of
through the fitted contrasts dict. This also resolves the §8 NOTE ("item-1 must
not deepen reliance on `_ranef_refs`") — the package no longer uses it at all.

Verified via scratch script (fixef, ranef intercept+slope incl. predict on
fitted data / new subset data / new data with an unseen level both with and
without `allow_new_levels`, ranef slope-only): all pass, unseen-level case now
correctly warns + zero-fills or errors per the flag instead of crashing.

## §8 REVIEW RESOLUTIONS (accepted)

- **loglik (was BLOCK).** `_pointwise_loglik` + the model's `loglik` return are
  DELETED. loglik is computed in Julia POST-HOC from stored ORIGINAL-scale params
  + raw data (reuse `linpred`→μ, then `logpdf` per family). Original scale (V-note:
  `loo_compare` unaffected — per-obs Jacobian constant cancels). `psis_loo`
  (comparison.jl) + turingregression.jl:217 stop calling `generated_quantities`;
  they call the new `pointwise_loglik(TR)` helper. Model return becomes params-only
  → single flat DimArray auto-stacks (R9), no NamedTuple.
- **V-new (predict raw):** `posterior_predict`/`predict` NEVER re-standardise.
  `TR.modeldata` X stays RAW; `md_std` is transient inside `fit!`. Guard the raw-X
  and new-data paths.
- **V-new (round-trip all shapes):** `unstandardise∘standardise == id` test covers
  5 families × 3 ranef shapes; inverse branches on `scale_y` for ranef too
  (count/binary drop the y_std factor).
- **NOTE:** item-1 must not deepen reliance on `_ranef_refs` (MixedModels internal);
  reduce if a public path exists.
