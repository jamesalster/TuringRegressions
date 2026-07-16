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

Committed already (`scaling_refactor`):
```julia
struct LinearModelData          # reused for fixef AND each ranef
    has_intercept::Bool
    has_fixed_effects::Bool
    X::AbstractMatrix
    X_names::Union{Nothing,Vector{String}}
end
struct RandomEffect
    variable::Symbol; levels::Vector; level_index::Vector{Int}
    linearmodeldata::LinearModelData
end
struct ModelData
    f::FormulaTerm; y::AbstractVector
    linearmodeldata::LinearModelData      # fixef
    Z::Vector{RandomEffect}               # empty ⇒ no ranef
    weights::Union{Nothing,Vector{Float64}}
end
```
`ModelInfo` is DELETED — its 4 bools now derive: intercept/fixef from
`linearmodeldata`, ranef from `!isempty(Z)`, weighted from `!isnothing(weights)`.

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

## §6 CURRENT WIP STATE (scaling_refactor) — broken spots to resolve

Structural rename done, half-wired. Before/alongside the refactor, fix:
- `formula_handlers.jl`: `has_intercept = has_intercept(formula)` shadows the fn
  (recursion/UndefVar) — rename local.
- `model.jl` `_random_effects`: `size(ranef..linearmodeldata.X,2)` double-dot typo;
  `size(ranef.X,2)` → `ranef.linearmodeldata.X`.
- `model.jl` `_linear_model`: refs `modelinfo.linearmodeldata` but param is
  `modeldata` — wrong var name throughout.
- `model.jl` `_pointwise_loglik`: branch inverted (§5.4).
- `model.jl` `construct_model`: signature stub `(y, X, ngrou)` + TODO → resolve via §5.3.
- `turing_glm`: still refs deleted `model_info`, `Z`; `cached_construct_model` sig
  needs `(family, md, ...)`.
- `predict.jl` `_resolve_z`: `TR.modeldata.z`/`TR.modelinfo` → `TR.modeldata.Z` /
  derive flags; ranef branch reads `re.linearmodeldata`.
- `fit!`/`psis_loo`: `TR.model(TR.modeldata)` call convention must match §5.3.

## §7 SEQUENCE (user-ordered)

Build + scratch-test at every step. NO `Pkg.test()` until the very end.

1. **Simplify formula_handlers** — lean on StatsModels/MixedModels existing
   functions instead of hand-rolled parsing; add ONE reusable `LinearModelData`
   extraction fn used by both fixef and each ranef. Fix §6 formula_handlers bugs.
2. **Fix fit!/model.jl grouping** — resurrect the slim helper that computes
   n_groups/group_idx/group_predictors once from the (std) ModelData and calls the
   unpacked model (§5.3). Fix §6 model.jl bugs. Branch compiles + scratch fit works.
3. **Scaling extraction** — `Transform` + `standardise`; move scaling OUT of
   model.jl into Julia; feed std data to model. Round-trip test. Priors as args (5.1).
4. **FlexiChains + unscaling** — swap chain type; `returned(...; stack=true)`;
   `unstandardise` builds the DimStack (deletes _generated_quantities + extraction
   loop). Then T4 prior-map, T5 likelihood merge, T3c dim renames.

Then, and only then, `Pkg.test()` full re-verify (V5/V13/V14/V15/V20).
```

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
