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
