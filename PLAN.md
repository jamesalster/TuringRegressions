# PLAN — T3 remaining work (step 4)

Architecture (structs, functions, invariants) distilled into SPEC.md — see
Types (`Predictors`/`RandomEffect`/`ModelData`/`LinearTransform`/`Transform`),
V10/V20-23, T3/T4/T5 task rows. §7 steps 1-3 DONE. This file tracks ONLY
what's left: step 4.

## Step 4 — FlexiChains + unstandardise (design locked, not started)

Root cause today: `fit!` forces `chain_type=MCMCChains.Chains`
(turingregression.jl:230-239) to fight Turing's FlexiChains default, because
every `.samples` access site (`name_map`, `.value`, `summarize(...).nt`) is
written against MCMCChains. Back-transform math also runs *inside* the
sampled `@model` (`_generated_quantities`, model.jl:224-335), returning a
`NamedTuple` unpacked by a manual Dict+stack+vcat loop
(turingregression.jl:241-321).

R9 (confirmed via playground.jl T8 scratch): `DynamicPPL.returned(model,
chain; stack=true)` auto-stacks a model's **flat `DimArray`** return into
`(iter, chain, param)`. A `NamedTuple`-of-`DimArray` does NOT auto-stack.
So: model must return ONE flat labeled array; all back-transform logic moves
out of the model into ordinary post-hoc Julia.

**Loglik decision:** stop computing `:loglik` inside the model. Write
`pointwise_loglik(TR)` as a standalone function in **src/comparison.jl**
(its only caller) — ports today's scalar-per-draw `_pointwise_loglik` math
(model.jl:195-221) to array math, run post-hoc from the raw standardised
param `DimArray` + `TR.modeldata`/`TR.tf` (standardise on demand via
`apply_transform`). Cleaner than smuggling per-obs loglik back into the flat
array as fake extra param labels (would bloat `:param` by `nobs` entries).

**Dim rename:** rename dim *names* to FlexiChains' own (`:draw`→ whatever
Step A finds, etc). Keep our existing dim **order** (param/effect dims, then
draw-equivalent, then chain-equivalent) — do not adopt FlexiChains' own axis
ordering.

### Execution order

**A. Empirical verification first (no source edits).** Extend playground.jl
T8 scratch, run in temp env, to pin down the 3 items SPEC.md flags `?`:
1. Exact dim names from `DynamicPPL.returned(...; stack=true)`.
2. `summarystats(chain)` field names (replaces `summarize(chain).nt`,
   summary.jl:230).
3. Where per-chain sampler internals live under `VNChain` (replaces
   `TR.samples.name_map[:internals]`, turingregression.jl:316-318).

Per C10: scratch first, `Pkg.test()` only once green.

**B. Project.toml.** `FlexiChains` `[extras]`→`[deps]` (`[compat]` already
pins `"0.6"`, line 40).

**C. src/model.jl.**
- Strip back-transform `Expr` blocks out of `_generated_quantities`
  (~lines 232-324: `β_original`/`α_original`, per-group `β_z_*`/`σ_z_*`/
  `R_z_*`/`offset_z_*`) — keep only standardised-scale values already
  computed for the likelihood term (α, β, σ/ν/ϕ, raw per-group z's,
  Cholesky `L`).
- Remove the `_pointwise_loglik` call/embedding (lines 326-327) — moves to
  `pointwise_loglik(TR)` in comparison.jl.
- Change final return to one flat `DimArray` over `Dim{:param}` with a
  deterministic label scheme (`:α`, `:β_<name>`, `:σ`, `:z_<group>_α`,
  `:z_<group>_β_<name>`, `:L_<group>_<i>_<j>`, ...) — the contract
  `unstandardise` decodes.

**D. src/transform.jl.** New
`unstandardise(raw::DimArray, tf::Transform, modelinfo, z::Vector{RandomEffect})::DimStack`
— pure post-hoc back-transform (math ported from model.jl). Builds layers:
`:fixef`, per-group `:{group}`, `:{group}_sd`, `:{group}_corr` (correlated
ranef only), `:{group}_offset` (slope-only/no-intercept ranef only, consumed
by predict.jl) — preserve mean-zero ranef constraint (V23) and offset trick.

**E. src/turingregression.jl `fit!` (~219-322).**
- `chain_type=MCMCChains.Chains` → `chain_type=FlexiChains.VNChain`; replace
  the old forced-Chains comment with the new rationale.
- Replace `generated_quantities(model_with_data, TR.samples)` + manual loop
  (241-321) with `raw = DynamicPPL.returned(model_with_data, TR.samples;
  stack=true)` then `TR.parameters = unstandardise(raw, TR.tf,
  TR.modelinfo, TR.z)`.
- Drop `filter(!=(:loglik), ...)` handling entirely.
- Rebuild `:internals` layer per Step A's VNChain finding.

**F. src/comparison.jl.** `psis_loo` (12-16): stop recomputing
`generated_quantities(...)`+`.loglik` itself — call new `pointwise_loglik(TR)`.

**G. Rename dims, keep order** (per Step A findings + dim-rename decision
above), across:
- parametermethods.jl — `_process_draws`/`_aggregate_draws` hardcode
  `:draw`/`:chain` as dim-keyword literals (lines 6,7,9,11,24).
- predict.jl — explicit `Dim{:draw}`/`Dim{:chain}` construction (~line 110).
- summary.jl:230-231 — `summarize(TR.samples).nt` → `summarystats` equivalent.
- statsapi.jl — depends only on `:fixef` layer name via `draws(TR, :fixef)`
  (51,63,77,99); likely no change needed beyond what `draws` handles.

**H. test/runtests.jl.** `using MCMCChains` + direct `:chain`/`:draw`/
`:{group}_sd`/`:{group}_corr`/`:{group}` assertions (lines 178, 319, 338,
358, 378, 410, 420, 424, 425, 437, 450) — rename alongside source.

**I. Re-verify.** All 5 families × 3 ranef shapes (none/intercept-only/
slope-only/correlated), scratch script first (C10), covering V3-V9,
V15/V22 (predict never re-standardises), V19 (loglik excluded from param
layers), V20 (cache key still structural), V21 (round-trip), V23
(mean-zero ranef) — then `Pkg.test()` once green.

Files touched: Project.toml, src/model.jl, src/transform.jl,
src/turingregression.jl, src/comparison.jl, src/parametermethods.jl,
src/predict.jl, src/summary.jl, test/runtests.jl, playground.jl (scratch,
not committed).

Out of scope here: T4 (prior flip, `AffineDistribution`) and T5 (likelihood
merge, code-gen re-verify) — both blocked on step 4 landing, per SPEC.md's
task table.
