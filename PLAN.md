# PLAN — T3 remaining work (step 5: downstream fixes)

Architecture (structs, functions, invariants) distilled into SPEC.md — see
Types (`Predictors`/`RandomEffect`/`ModelData`/`LinearTransform`/`Transform`),
V10/V20-23, T3/T4/T5 task rows. §7 steps 1-4 DONE, verified. This file tracks
ONLY what's left: step 5.

## Step 4 (DONE, verified) FlexiChains + reshape + unstandardise

`fit!` (src/turingregression.jl:218-244) now: samples with `chain_type=VNChain`,
pulls the flat sampled-VarName array via `DimArray(TR.samples)` (R10), splits it
into named standardised-scale layers via `reshape_params` (src/reshape.jl, new
file), then back-transforms to original scale via `unstandardise`
(src/transform.jl:158-164, ported from the old in-model `_generated_quantities`
math). `TR.parameters::DimStack` populated same shape/labels as before
(`:fixef`, per-group `:{group}`/`:{group}_sd`/`:{group}_corr`/`:{group}_offset`).
`model.jl`'s `@model` body no longer has a return statement at all (build_model_body,
src/model.jl:228-264) — likelihood is `Turing.@addlogprob!` only.

`draws`/`predict.jl`/`summary.jl`'s fixef+ranef tables/`statsapi.jl` all read off
`TR.parameters` (a plain DimStack with manually-labelled `:draw`/`:chain` dims,
set in reshape.jl) — unaffected by the VNChain switch, confirmed working as-is.
Dim-rename DONE: `:draw` → `:iter` throughout (SPEC.md §T3, FlexiChains' own
naming). Changed src/reshape.jl (Dim{:iter} construction sites),
src/transform.jl (`_unstandardise_*`), src/parametermethods.jl
`_process_draws`/`_aggregate_draws` (`DA[iter=...]`, `size(DA,:iter)`,
collapse target `(:iter,:chain)=>:iter`), src/predict.jl (`Dim{:iter}` in
`linpred`), src/metrics.jl, test/runtests.jl:425. `drop_warmup`/`n_draws`
kwarg names unchanged (those are param names, not the dim symbol). Verified
no remaining `:draw`/`draw=`/`Dim{:draw}` in src/, test/, ext/.

## Step 5 (not started) — fix what the model's return-value removal broke

Root cause: model.jl's `@model` used to return a `:loglik` key (old
`_generated_quantities`, V19) that `psis_loo` read via
`generated_quantities(...)`. That return statement is gone (step 4). Two call
sites still assume the old shape:

**A. src/comparison.jl:10-17 `psis_loo`.** Currently:
```julia
gq = generated_quantities(model_with_data, TR.samples)
nobs = length(gq[1, 1].loglik)
ll = [gq[i, j].loglik[n] for i in axes(gq, 1), j in axes(gq, 2), n in 1:nobs]
```
Broken — `model_with_data` has no return value now. Per PLAN's original Loglik
decision: write a standalone `pointwise_loglik(TR)` in comparison.jl (its only
caller), computed post-hoc, not inside the model. Needs:
  - standardised `ModelData` — `apply_transform(TR.tf, TR.modeldata)` (same as
    `_build_model_with_data`, turingregression.jl:207-216)
  - standardised-scale param draws — re-derive via
    `reshape_params(DimArray(TR.samples), TR.modeldata, T)` (fit!'s std_params
    isn't retained on TR, cheap to recompute — pure reshuffle, no MCMC)
  - port the per-family math from the now-dead `_pointwise_loglik`
    (model.jl:199-225) into array form (μ from the standardised linear model +
    fixef/ranef draws, then per-family logpdf, weighted variant if
    `is_weighted(TR.modeldata)`)
  - output shape `(nobs, draw, chain)` or whatever `PosteriorStats.loo` expects
    — check today's `ll` shape convention in the working (pre-step-4) version
    via git history if needed
  - delete `_pointwise_loglik` from model.jl once ported (dead code, no other
    caller)

**B. src/summary.jl:228-232 `model_warnings(TR)`.** Currently:
```julia
chain_info = summarize(TR.samples; sections=:parameters)
model_warnings(chain_info.nt)
```
Broken — MCMCChains-only API, `TR.samples` is a `VNChain`. Simplest fix: this
file already has `_diagnostics_table` (summary.jl:4-36) computing
rhat/ess/mcse/mcse directly off a `(label,draw,chain)` DimArray via
MCMCDiagnosticTools' `rhat`/`ess`/`mcse` — those already work on plain arrays,
not Chains objects (used today on `TR.parameters` slices in `Base.summary`).
Reuse that instead of touching `TR.samples` at all: build a diagnostics table
over `TR.parameters` (fixef + all ranef layers, no warmup-drop needed — this
fn is a soundness check not a report) and feed it to the existing
`model_warnings(chain_info::NamedTuple)` (summary.jl:197-221), which is
already chain-object-agnostic. Delete the FlexiChains-summarize approach
entirely, don't try to find its `summarystats`/`FlexiSummary` equivalent —
unnecessary now that `_diagnostics_table` covers it.

**C. Cleanup once A/B land.**
- `test/runtests.jl:4` `using MCMCChains` — check if still needed after B
  (likely not; grep found no other `.value`/`name_map`/`Chains` use in tests).
- SPEC.md V19's wording ("`_generated_quantities` ... always includes a
  `:loglik` key") is now stale — describes the pre-step-4 mechanism. Rewrite
  once `pointwise_loglik(TR)` lands, describing the new post-hoc call site
  instead of the deleted in-model one.
- SPEC.md R9/R10 dim-naming caveat, and old step-4 text under §T3, can be
  trimmed to "done" once this file's step 4 section above is folded back into
  SPEC.md (or just leave as history — not blocking).

**D. Re-verify.** `psis_loo`/`loo_compare` tests already exist
(test/runtests.jl:219-226,281) — `Pkg.test()` (or a targeted script per C10,
this is a big-suite change so full `Pkg.test()` once green is warranted) is
the actual verification once A/B are in. Also re-check V7/V9/V10 hold (spot
check already done per user for reshape/unstandardise, but psis_loo path is
untested since step 4 landed).

Files touched (step 5): src/comparison.jl, src/model.jl (delete dead fn),
src/summary.jl, test/runtests.jl (maybe), SPEC.md (V19 wording).

Out of scope here: T4 (prior flip, `AffineDistribution`) and T5 (likelihood
merge, code-gen re-verify) — both blocked on step 5 landing, per SPEC.md's
task table.
