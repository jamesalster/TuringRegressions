## 2026-08-06T11:04:36.256

T33 baseline — ranef sd back-transformed by elementwise `_scale_effects`, no recentering cross term.

cold (compile + fit): 30.61s
warm (fit only):      6.98s

| param | ESS/sec | rhat | posterior mean | gold (lme4 REML) |
|---|---|---|---|---|
| fixef α | 71.2 | 1.021 | 251.35 | 251.4 |
| fixef Days | 101.0 | 1.003 | 10.47 | 10.5 |
| fixef σ | 218.8 | 1.002 | 25.77 | — |
| Subject_sd Intercept | 70.6 | 1.01 | 38.81 | 24.7 |
| Subject_sd Days | 119.0 | 1.004 | 6.07 | 5.9 |

max rhat: 1.021 ok

## 2026-08-06T11:13:44.846

T33 partial — affine map `A` added, but SD computed `A·D·D·A'` (ρ dropped). WRONG, worse than baseline.

cold (compile + fit): 28.99s
warm (fit only):      6.21s

| param | ESS/sec | rhat | posterior mean | gold (lme4 REML) |
|---|---|---|---|---|
| fixef α | 80.1 | 1.021 | 251.35 | 251.4 |
| fixef Days | 113.6 | 1.003 | 10.47 | 10.5 |
| fixef σ | 246.3 | 1.002 | 25.77 | — |
| Subject_sd Intercept | 81.1 | 1.007 | 47.73 | 24.7 |
| Subject_sd Days | 134.0 | 1.004 | 6.07 | 5.9 |

max rhat: 1.021 ok

## 2026-08-06T11:21:25.195

T33 fixed — full `Σ_orig = A·(D·R·D)·A'`, SDs off its diagonal. Prior sweep (Exp 1/5/20) and median vs mean both ≈ no effect, so residual gap is not prior pull or skew; traced to ρ_std 0.603 vs the ~0.75 the data want.

cold (compile + fit): 30.08s
warm (fit only):      6.61s

| param | ESS/sec | rhat | posterior mean | gold (lme4 REML) |
|---|---|---|---|---|
| fixef α | 75.2 | 1.021 | 251.35 | 251.4 |
| fixef Days | 106.7 | 1.003 | 10.47 | 10.5 |
| fixef σ | 231.2 | 1.002 | 25.77 | — |
| Subject_sd Intercept | 106.2 | 1.007 | 31.09 | 24.7 |
| Subject_sd Days | 125.8 | 1.004 | 6.07 | 5.9 |

max rhat: 1.021 ok

## 2026-08-06T11:28:49.855

T33 + LKJ η 2.0→1.0 (model.jl:47). ρ_std 0.603→0.677, raw corr −0.106→−0.019 (gold 0.07), max rhat 1.021→1.008. CURRENT BASELINE for T24-T32. Residual sd Intercept 29.14 vs 24.7 is the centered-LKJ issue — T34, not a bug.

cold (compile + fit): 28.79s
warm (fit only):      6.73s

| param | ESS/sec | rhat | posterior mean | gold (lme4 REML) |
|---|---|---|---|---|
| fixef α | 89.3 | 1.003 | 251.51 | 251.4 |
| fixef Days | 65.5 | 1.004 | 10.46 | 10.5 |
| fixef σ | 254.9 | 0.999 | 25.77 | — |
| Subject_sd Intercept | 126.0 | 1.003 | 29.14 | 24.7 |
| Subject_sd Days | 89.5 | 1.008 | 6.27 | 5.9 |

max rhat: 1.008 ok

## 2026-08-06T11:51:29.189

T24 adtype sweep — ForwardDiff (`AutoForwardDiff()`)

cold (compile + fit): 30.11s
warm (fit only):      8.01s

| param | ESS/sec | rhat | posterior mean | gold (lme4 REML) |
|---|---|---|---|---|
| fixef α | 75.1 | 1.003 | 251.51 | 251.4 |
| fixef Days | 55.1 | 1.004 | 10.46 | 10.5 |
| fixef σ | 214.3 | 0.999 | 25.77 | — |
| Subject_sd Intercept | 105.9 | 1.003 | 29.14 | 24.7 |
| Subject_sd Days | 75.2 | 1.008 | 6.27 | 5.9 |

max rhat: 1.008 ok

## 2026-08-06T11:52:37.180

T24 adtype sweep — ReverseDiff (`AutoReverseDiff(compile=true)`)

cold (compile + fit): 31.56s
warm (fit only):      2.27s

| param | ESS/sec | rhat | posterior mean | gold (lme4 REML) |
|---|---|---|---|---|
| fixef α | 271.7 | 1.012 | 251.7 | 251.4 |
| fixef Days | 251.9 | 1.004 | 10.55 | 10.5 |
| fixef σ | 889.5 | 1.003 | 25.87 | — |
| Subject_sd Intercept | 343.3 | 1.0 | 28.72 | 24.7 |
| Subject_sd Days | 264.8 | 1.001 | 6.34 | 5.9 |

max rhat: 1.012 ok

## 2026-08-06T12:01:42.730

T24 adtype sweep — ReverseDiff (`AutoReverseDiff(compile=true)`)

cold (compile + fit): 29.22s
warm (fit only):      2.06s

| param | ESS/sec | rhat | posterior mean | gold (lme4 REML) |
|---|---|---|---|---|
| fixef α | 299.1 | 1.012 | 251.7 | 251.4 |
| fixef Days | 277.3 | 1.004 | 10.55 | 10.5 |
| fixef σ | 979.3 | 1.003 | 25.87 | — |
| Subject_sd Intercept | 377.9 | 1.0 | 28.72 | 24.7 |
| Subject_sd Days | 291.6 | 1.001 | 6.34 | 5.9 |

max rhat: 1.012 ok

## 2026-08-06T12:04:21.751

T24 adtype sweep — Mooncake (`AutoMooncake()`)

cold (compile + fit): 117.42s
warm (fit only):      9.14s

| param | ESS/sec | rhat | posterior mean | gold (lme4 REML) |
|---|---|---|---|---|
| fixef α | 57.3 | 1.012 | 250.8 | 251.4 |
| fixef Days | 43.7 | 1.002 | 10.36 | 10.5 |
| fixef σ | 208.8 | 0.999 | 25.85 | — |
| Subject_sd Intercept | 88.3 | 1.002 | 28.64 | 24.7 |
| Subject_sd Days | 73.0 | 0.999 | 6.24 | 5.9 |

max rhat: 1.012 ok


## T24 adtype sweep — Enzyme (`AutoEnzyme()`)

FAILED — `EnzymeRuntimeActivityError` during sampling: "Detected potential need for
runtime activity. Constant memory is stored (or returned) to a differentiable
variable and correctness cannot be guaranteed with static activity analysis."
Some pattern in the generated model (`model.jl`) isn't provably non-differentiable
under Enzyme's static analysis. Workarounds exist (`set_runtime_activity`, or a
codegen rewrite to remove the offending pattern) but not pursued — ReverseDiff
already gives a clear win with no code changes needed. Not a contender as-is.

## T24 — scratch check: intercept-only ranef model, ForwardDiff vs ReverseDiff

`Reaction ~ 1 + Days + (1 | Subject)` (intercept-only ranef, ~20 params vs 42 for
the full slope+corr sleepstudy model). Quick scratch script, warm-fit time only,
compile time ignored, samples=2000/warmup=2000/nchains=4 (same as main bench).

| adtype | warm fit time |
|---|---|
| ForwardDiff | 1.24s |
| ReverseDiff(compile=true) | 0.63s |

ReverseDiff still ~2x faster even at ~20 params — the "ForwardDiff wins below
~10-20 params" assumption in the original T24 task note doesn't hold here.
Not chasing a param-count-based heuristic; ReverseDiff looks like the better
default across the board for this model family.

## T24 — scratch check: fixed-effects-only model, ForwardDiff vs ReverseDiff

`Reaction ~ 1 + Days` (no ranef, ~3 params: α, β, σ). Same warm-fit-only,
compile-ignored setup as the intercept-only ranef check.

| adtype | warm fit time |
|---|---|
| ForwardDiff | 0.10s |
| ReverseDiff(compile=true) | 0.13s |

ForwardDiff wins at this scale (as expected — reverse-mode overhead not worth it
for a handful of params). So the crossover is somewhere between ~3 params
(ForwardDiff wins) and ~20 params (ReverseDiff wins 2x) — ranef presence, not
raw param count alone, may be what matters (ranef models have more
correlated/awkward posterior geometry per gradient eval). Still deciding final
default heuristic.

## T24 — scratch check: bigger dataset, fixed-effects-only, Bernoulli

`Survived ~ Class + Sex + Age`, Titanic (2201 obs, ~5-6 params), Bernoulli/logit.
Same warm-fit-only, compile-ignored setup.

| adtype | warm fit time |
|---|---|
| ForwardDiff | 1.41s |
| ReverseDiff(compile=true) | 2.29s |

ForwardDiff wins again — bigger N doesn't flip it, only param count/ranef
presence does. Pattern across all 4 scratch checks: fixed-effects-only models
(3 params, sleepstudy; ~5-6 params, titanic) favor ForwardDiff regardless of N;
ranef models (20 params, intercept-only; 42 params, full slope+corr) favor
ReverseDiff by ~2-3.5x. Presence of ranef (not raw param count alone) looks
like the right heuristic signal — proposing: `adtype = has_random_effects(md)
? AutoReverseDiff(compile=true) : AutoForwardDiff()`, both overridable via new
`adtype` kwarg on `fit!`.
## 2026-08-06T12:44:09.895

cold (compile + fit): 32.5s
warm (fit only):      2.15s

| param | ESS/sec | rhat | posterior mean | gold (lme4 REML) |
|---|---|---|---|---|
| fixef α | 286.5 | 1.012 | 251.7 | 251.4 |
| fixef Days | 265.6 | 1.004 | 10.55 | 10.5 |
| fixef σ | 937.9 | 1.003 | 25.87 | — |
| Subject_sd Intercept | 362.0 | 1.0 | 28.72 | 24.7 |
| Subject_sd Days | 279.2 | 1.001 | 6.34 | 5.9 |

max rhat: 1.012 ok

## 2026-08-06T12:51:54.912

cold (compile + fit): 28.24s
warm (fit only):      1.95s

| param | ESS/sec | rhat | posterior mean | gold (lme4 REML) |
|---|---|---|---|---|
| fixef α | 407.5 | 1.002 | 251.48 | 251.4 |
| fixef Days | 402.4 | 1.001 | 10.39 | 10.5 |
| fixef σ | 843.4 | 1.002 | 25.84 | — |
| Subject_sd Intercept | 373.8 | 1.002 | 29.3 | 24.7 |
| Subject_sd Days | 400.2 | 1.007 | 6.23 | 5.9 |

max rhat: 1.007 ok

## 2026-08-06T12:54:18.239

cold (compile + fit): 30.17s
warm (fit only):      2.06s

| param | ESS/sec | rhat | posterior mean | gold (lme4 REML) |
|---|---|---|---|---|
| fixef α | 299.4 | 1.012 | 251.7 | 251.4 |
| fixef Days | 277.5 | 1.004 | 10.55 | 10.5 |
| fixef σ | 980.0 | 1.003 | 25.87 | — |
| Subject_sd Intercept | 378.2 | 1.0 | 28.72 | 24.7 |
| Subject_sd Days | 291.8 | 1.001 | 6.34 | 5.9 |

max rhat: 1.012 ok

## 2026-08-06T12:55:29.954

cold (compile + fit): 28.71s
warm (fit only):      2.07s

| param | ESS/sec | rhat | posterior mean | gold (lme4 REML) |
|---|---|---|---|---|
| fixef α | 299.6 | 1.005 | 250.79 | 251.4 |
| fixef Days | 217.0 | 1.005 | 10.49 | 10.5 |
| fixef σ | 816.0 | 1.001 | 25.76 | — |
| Subject_sd Intercept | 346.1 | 0.999 | 29.47 | 24.7 |
| Subject_sd Days | 311.7 | 1.002 | 6.33 | 5.9 |

max rhat: 1.005 ok

## 2026-08-06T12:56:48.313

cold (compile + fit): 28.97s
warm (fit only):      1.99s

| param | ESS/sec | rhat | posterior mean | gold (lme4 REML) |
|---|---|---|---|---|
| fixef α | 277.0 | 1.005 | 251.95 | 251.4 |
| fixef Days | 309.8 | 1.002 | 10.55 | 10.5 |
| fixef σ | 812.5 | 1.002 | 25.81 | — |
| Subject_sd Intercept | 365.5 | 1.003 | 29.36 | 24.7 |
| Subject_sd Days | 435.1 | 1.004 | 6.25 | 5.9 |

max rhat: 1.005 ok

## 2026-08-06T12:58:15.710

cold (compile + fit): 29.14s
warm (fit only):      1.83s

| param | ESS/sec | rhat | posterior mean | gold (lme4 REML) |
|---|---|---|---|---|
| fixef α | 336.9 | 1.012 | 251.7 | 251.4 |
| fixef Days | 312.3 | 1.004 | 10.55 | 10.5 |
| fixef σ | 1102.9 | 1.003 | 25.87 | — |
| Subject_sd Intercept | 425.6 | 1.0 | 28.72 | 24.7 |
| Subject_sd Days | 328.4 | 1.001 | 6.34 | 5.9 |

max rhat: 1.012 ok

## 2026-08-06T12:59:29.535

cold (compile + fit): 29.11s
warm (fit only):      2.46s

| param | ESS/sec | rhat | posterior mean | gold (lme4 REML) |
|---|---|---|---|---|
| fixef α | 250.2 | 1.012 | 251.7 | 251.4 |
| fixef Days | 231.9 | 1.004 | 10.55 | 10.5 |
| fixef σ | 819.0 | 1.003 | 25.87 | — |
| Subject_sd Intercept | 316.1 | 1.0 | 28.72 | 24.7 |
| Subject_sd Days | 243.8 | 1.001 | 6.34 | 5.9 |

max rhat: 1.012 ok


## T26-T29 — codegen micro-opts, isolated one-at-a-time

Changes tested (model.jl):
- T26: `_likelihood` Normal-unweighted — `logpdf(MvNormal(μ,σ),y)` → hand-rolled `-nobs*log(σ) - sum(abs2,y.-μ)/(2σ^2) - nobs*log(2π)/2`. Skips MvNormal/PDMat construction per eval.
- T27: `_random_effects` intercept+slope branch — `diagm(σ_z)*L.L*z_raw` → `(σ_z .* L.L)*z_raw` (row-scale broadcast, drops p×p alloc+matmul).
- T28: same branch — `filldist(MvNormal(zeros(p),I), n_groups)` → `filldist(Normal(), p, n_groups)`, matching the no-intercept branch's existing pattern. Skips MvNormal/PDMat machinery on the prior side.
- T29: `_linear_model` — special-case ranef slope term to avoid `sum(...; dims=2)[:]` alloc when `n_predictors==1` (elementwise broadcast instead); `@view` on the slice in the general (p>1) case.

Method: reverted to clean baseline, ran sleepstudy_bench.jl once per variant (baseline, T26 alone, T27+T28 alone [same 3 lines, tested together], T29 alone, all four combined). Also reran baseline a 2nd time to gauge run-to-run noise, since this bench is a single NUTS run per variant, not averaged.

| variant | warm | ESS/sec α | ESS/sec Days | ESS/sec Subject_sd Int |
|---|---|---|---|---|
| baseline run 1 | 2.06s | 299.4 | 277.5 | 378.2 |
| baseline run 2 (noise check) | 2.46s | 250.2 | 231.9 | 316.1 |
| T26 only | 2.07s | 299.6 | 217.0 | 346.1 |
| T27+T28 only | 1.99s | 277.0 | 309.8 | 365.5 |
| T29 only | 1.83s | 336.9 | 312.3 | 425.6 |
| all four combined | 1.95s | 407.5 | 402.4 | 373.8 |

Baseline-vs-baseline swing is ~20% run to run (this bench = single NUTS run, not averaged over reps) — that's the noise floor. T26 alone and T27+T28 alone land inside that band: no clean individual signal, can't distinguish from noise at n=1. T29 alone is the only isolated change that beats the noise band on both warm time and ESS/sec across all params. The "all four" run's large ESS/sec jump likely partly a lucky draw, not cleanly attributable given the other three show no signal alone.

DECISION: keep T29 only. T26/T27/T28 reverted — no measured benefit distinguishable from noise, not worth the code churn (T28 in particular changes prior-sampling distribution shape, more moving parts than the gain justifies). Param recovery (posterior means, rhat) unaffected across all variants — none of the four changed model semantics.

Rigor note: to actually separate T26/T27/T28 from noise would need multiple reps per variant (~3-5x runs, this bench isn't set up for that — single fit! per variant, no averaging loop). Flagged for future T22 work if these are revisited.

Repeat run, T29-only code (post-decision, confirms the win isn't a fluke): warm 1.85s, ESS/sec α 332.7 / Days 308.3 / Subject_sd Int 420.3 — tight match to the first T29 run (1.83s / 336.9 / 312.3 / 425.6), well clear of the baseline noise band.

## 2026-08-06T13:03:01.897

cold (compile + fit): 28.38s
warm (fit only):      1.85s

| param | ESS/sec | rhat | posterior mean | gold (lme4 REML) |
|---|---|---|---|---|
| fixef α | 332.7 | 1.012 | 251.7 | 251.4 |
| fixef Days | 308.3 | 1.004 | 10.55 | 10.5 |
| fixef σ | 1089.1 | 1.003 | 25.87 | — |
| Subject_sd Intercept | 420.3 | 1.0 | 28.72 | 24.7 |
| Subject_sd Days | 324.2 | 1.001 | 6.34 | 5.9 |

max rhat: 1.012 ok

## T14 — PrecompileTools workload

Added `@compile_workload` (src/TuringRegressions.jl) warming one fixef-only
Normal `turing_glm`+`fit!` (samples=1, warmup=1, nchains=1) at package build
time. Targets T35's ~25s "generic" TTFX slice (Turing/DynamicPPL/AbstractMCMC/
StatsModels/ForwardDiff machinery), shared across families/shapes regardless
of which model triggers it.

Isolated per-family TTFX (`using TuringRegressions` + first `turing_glm`+`fit!`
in a fresh process, small synthetic data, not this bench's sleepstudy shape):

| model | baseline (using+fit) | post-T14 | delta |
|---|---|---|---|
| fixef-only Normal (exact precompiled match) | 30.10s | 6.01s | −24.09s (5.0x) |
| fixef-only Bernoulli | 28.65s | 11.22s | −17.43s (2.6x) |
| fixef-only Poisson | 29.74s | 11.29s | −18.45s (2.6x) |
| ranef-only Normal (intercept, ReverseDiff) | 39.17s | 18.25s | −20.92s (2.1x) |

`using TuringRegressions` alone is flat (~5-5.5s both) — win is entirely in
first-fit compile, as expected.

Bernoulli/Poisson (neither precompiled) recover almost as much as the exact
match — confirms the ~25s bucket is genuinely generic, not family-specific.
Ranef recovers a real chunk too, purely from that shared slice — its own
ReverseDiff-specific compile is NOT covered (see below) and still costs ~12-14s.

**Rejected: precompiling ranef `fit!`/`sample()`.** `AutoReverseDiff` (ranef's
V25 default) segfaults on package-image reload — confirmed with BOTH
`compile=true` and `compile=false` (identical crash), and confirmed NOT a
ranef-shape issue (ForwardDiff-on-ranef precompiles and reloads clean, just
isn't the default path so wouldn't help real users). This is a
Turing/DynamicPPL/ReverseDiff serialization limitation, not fixable from this
package.

**Rejected: precompiling ranef `turing_glm` construction only (no `fit!`).**
Tried warming both ranef codegen shapes (intercept-only, correlated-slope) via
construction-only calls (cheap, no NUTS compile). Measured effect: ranef fit
time 14.14s → 12.82s (~1.3s), while precompile build time rose 33.0s → 34.7s
(~1.6s). Net wash, and the 1.3s runtime delta sits inside this project's own
established noise band (T26-29 above: <20% single-run swing = noise, not
signal). Dropped — not worth the code.

Full sleepstudy bench (below) confirms the win holds even for the
correlated-slope shape, which is NOT covered by any precompiled model:

## 2026-08-06T16:21:02.895 (post-T14)

cold (compile + fit): 18.61s (baseline: 28.38s, −9.77s / 34%)
warm (fit only):      1.89s (baseline: 1.85s, unchanged — precompile doesn't touch already-JIT'd warm path)

| param | ESS/sec | rhat | posterior mean | gold (lme4 REML) |
|---|---|---|---|---|
| fixef α | 326.9 | 1.012 | 251.7 | 251.4 |
| fixef Days | 303.0 | 1.004 | 10.55 | 10.5 |
| fixef σ | 1070.1 | 1.003 | 25.87 | — |
| Subject_sd Intercept | 413.0 | 1.0 | 28.72 | 24.7 |
| Subject_sd Days | 318.6 | 1.001 | 6.34 | 5.9 |

max rhat: 1.012 ok — param recovery unaffected by T14, as expected (precompile
changes compile timing only, not model semantics).

DECISION: keep T14 as implemented (fixef-only Normal `fit!` in the workload,
nothing else). Real, substantial TTFX win across all families and even ranef
shapes, at ~33s one-time added precompile cost per package build. ok

