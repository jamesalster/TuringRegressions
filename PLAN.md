# PLAN — T3 remaining work (step 4)

Architecture (structs, functions, invariants) distilled into SPEC.md — see
Types (`Predictors`/`RandomEffect`/`ModelData`/`LinearTransform`/`Transform`),
V10/V20-23, T3/T4/T5 task rows. §7 steps 1-3 DONE. This file tracks ONLY
what's left: step 4.

## Step 4 — FlexiChains + unscaling (not started)

- Swap chain type: `chain_type=VNChain` (not `MCMCChains.Chains`). Promote
  FlexiChains `[extras]`→`[deps]` in Project.toml (R5).
- `@model` return stmt becomes ONE flat `DimArray` (not NamedTuple) — a
  NamedTuple-of-DimArray does NOT auto-stack (R9). Pull raw stacked params via
  `DynamicPPL.returned(model, chain; stack=true)` → `(iter,chain,param)`
  DimArray, replacing today's manual per-layer Dict+stack+vcat loop
  (`turingregression.jl:279-358`).
- Delete `_generated_quantities` (model.jl). Write `unstandardise(raw::DimArray,
  tf::Transform)::DimStack` — back-transform (fixef/ranef/Σ, §2b/2d math
  already in SPEC's V10) applied AFTER unstacking, in Julia, not inside the
  model. One layer per group (`:fixef`, `:{group}`, `:{group}_sd`,
  `:{group}_corr`, `:{group}_offset`, `:internals`) — preserve the
  ranef-offset centering trick (`:{group}_offset`, consumed by predict.jl)
  and the mean-zero ranef constraint (V23).
- `pointwise_loglik(TR)` becomes standalone helper (post-hoc, same math as
  today's `:loglik` computation) — `psis_loo`/`comparison.jl` call it instead
  of reading `:loglik` out of `generated_quantities`'s NamedTuple (V19).
- Rename dims/labels to FlexiChains' own conventions (`:iter` not `:draw`,
  its native ordering) rather than relabeling into today's scheme — re-verify
  V7 (draws shape), V9 (param labels), C3 (layer/dim naming) under the new
  names.
- `summarize`→`summarystats` (confirm field names match today's usage in
  `summary.jl:230`, R4). `.name_map`/`.value` indexing → `VarName`-keyed
  access (R2, R3).
- Re-verify V21 (round-trip) and V22 (predict never re-standardises) still
  hold once params flow through the new path.
- Full re-verify: all 5 families × 3 ranef shapes, scratch script first
  (no `Pkg.test()` per C10), then `Pkg.test()` once green.

After step 4 lands: T4 (flip priors to original scale, `AffineDistribution`)
and T5 (likelihood merge, code-gen re-verify) can proceed — both already
speced in SPEC.md's task table.
