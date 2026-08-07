using TuringRegressions
using Test
using RDatasets
using StatsModels
using StatsBase: mean, std, var, CoefTable
using Suppressor: @suppress
using Random
using GLM: GLM
using StatisticalMeasures
using CategoricalDistributions
using DataFrames
using CairoMakie
CairoMakie.activate!()
using PrettyTables

Random.seed!(1)

# Full-sampling-budget calls read these so the benchmark budget can be tuned
# via env vars without editing tests. BENCH_SAMPLES = real draws kept TOTAL
# across chains (new `samples` API); BENCH_WARMUP = adaptation draws per chain.
const BENCH_SAMPLES = parse(Int, get(ENV, "TR_BENCH_SAMPLES", "2000"))
const BENCH_WARMUP = parse(Int, get(ENV, "TR_BENCH_WARMUP", "2000"))
const BENCH_NCHAINS = parse(Int, get(ENV, "TR_BENCH_NCHAINS", "2"))

# Dev-loop subset — one Normal fit, one Bernoulli fit, the sleepstudy mixedmodels
# benchmark fit. Skips everything else for fast iteration.
# `TR_DEV_SUBSET=true julia --project=. -e 'using Pkg; Pkg.test()'`
const DEV_SUBSET = get(ENV, "TR_DEV_SUBSET", "false") == "true"

@info "Setting up tests"

mtcars = dataset("datasets", "mtcars")

titanic_df = dataset("datasets", "Titanic")
# Expand frequency cases into one row per observation
titanic = vcat([repeat(DataFrame(row[1:4]), row.Freq) for row in eachrow(titanic_df)]...)
titanic.Survived = titanic.Survived .== "Yes"

# Canonical mixed-model dataset (lme4): Reaction ~ Days + (Days|Subject), 18 subjects.
# Published lme4 REML estimates used below as loose sanity bounds:
# fixef ≈ (Intercept=251.4, Days=10.5); ranef sd ≈ (Intercept=24.7, Days=5.9); corr ≈ 0.07
sleepstudy = dataset("lme4", "sleepstudy")

# RE x non-Normal family: Incidence ~ Period + (1|Herd)
cbpp = dataset("lme4", "cbpp")

# small/quick fit for tests that only check API shape, not parameter recovery
quickfit!(TR) = @suppress fit!(TR; samples=600, warmup=1000, nchains=2, quiet=true)

# Benchmark table: posterior mean vs canonical (GLM MLE / lme4 REML), full-budget
# models only (N=BENCH_N, nchains=BENCH_NCHAINS). Printed at the end of the run — a running record
# of how tight our tolerances actually are, not just whether they pass.
const BENCHMARK_ROWS = NamedTuple[]

function record_benchmark!(model_name, param_name, ours, canonical)
    push!(
        BENCHMARK_ROWS,
        (
            model=model_name,
            param=param_name,
            ours=round(ours; digits=3),
            canonical=round(canonical; digits=3),
            abs_err=round(abs(ours - canonical); digits=3),
            rel_err_pct=round(100 * abs(ours - canonical) / max(abs(canonical), 1e-8); digits=1),
        ),
    )
end

# Fit-time table: wall-clock seconds per full-budget fit. First recorded fit
# eats TTFX compile, so read the first row as an upper bound, not a like-for-like.
const TIMING_ROWS = NamedTuple[]

function record_timing!(model_name, seconds)
    push!(TIMING_ROWS, (model=model_name, seconds=round(seconds; digits=1)))
end

try # keep going through sibling testsets on failure, still print benchmark table
@testset "TuringRegressions" begin

if !DEV_SUBSET
@testset "Predict" begin
    Random.seed!(123)
    mod = turing_glm(@formula(MPG ~ Cyl + Disp), mtcars, Normal)
    quickfit!(mod)

    @testset "call forms" begin
        @test_nowarn posterior_predict(mod) # fitted data
        @test_nowarn posterior_predict(mod, mod.modeldata.predictors.X) # matrix
        new_data = mtcars[1:5, :]
        @test_nowarn posterior_predict(mod, new_data) # new DataFrame
    end

    @testset "type variants and link relations" begin
        linp = posterior_predict(mod; type=:linpred)
        ep = posterior_predict(mod; type=:epred)
        post = posterior_predict(mod; type=:posterior)

        # Identity link (Normal): linpred == epred
        @test isapprox(Array(linp), Array(ep))
        # Posterior predictive adds observation noise -> higher variance than epred
        @test mean(var(Array(post); dims=2)) > mean(var(Array(ep); dims=2))
    end

    @testset "log-link relation (Poisson)" begin
        mod_count = turing_glm(@formula(HP ~ Cyl + Disp), mtcars, Poisson)
        quickfit!(mod_count)
        linp = posterior_predict(mod_count; type=:linpred)
        ep = posterior_predict(mod_count; type=:epred)
        @test isapprox(Array(linp), log.(Array(ep)))
    end

    @testset "new_data uses raw scale, no re-standardisation" begin
        new_data = mtcars[3:8, :]
        glm_mod = GLM.lm(@formula(MPG ~ Cyl + Disp), mtcars)
        glm_pred = GLM.predict(glm_mod, new_data)
        tr_pred = Array(posterior_predict(mean, mod, new_data; type=:epred))
        @test isapprox(glm_pred, tr_pred, atol=1.5)
    end
end
end # !DEV_SUBSET

if !DEV_SUBSET
@testset "StatsAPI interface" begin
    Random.seed!(123)
    mod = turing_glm(@formula(MPG ~ Cyl + Disp), mtcars, Normal)
    quickfit!(mod)

    @test coefnames(mod) == ["α", "Cyl", "Disp"]
    @test length(coef(mod)) == 3
    @test nobs(mod) == 32
    @test isfitted(mod) == true
    @test islinear(mod) == true
    @test weights(mod) == ones(32)
    @test responsename(mod) == "MPG"
    @test meanresponse(mod) == mean(mtcars.MPG)
    @test size(modelmatrix(mod)) == (32, 2)
    @test length(stderror(mod)) == 3
    @test size(vcov(mod)) == (3, 3)
    @test size(confint(mod)) == (3, 2)
    @test coeftable(mod) isa CoefTable
    @test response(mod) == mod.modeldata.y
    @test isapprox(residuals(mod), response(mod) .- fitted(mod))

    # predict (StatsAPI point estimate) vs posterior_predict (full posterior, primary API)
    pt = @test_logs (:warn, r"posterior_predict") predict(mod)
    @test pt isa Vector{Float64}
    @test length(pt) == 32
    @test posterior_predict(mod) isa DimArray

    # modelmatrix is misleading for random-effects models (no Z structure) — should error
    mod_re = turing_glm(@formula(Reaction ~ 1 + Days + (1 + Days | Subject)), sleepstudy, Normal)
    quickfit!(mod_re)
    @test_throws ArgumentError modelmatrix(mod_re)
    @test length(coef(mod_re)) == 2 # fixef only, still well-defined

    lp = @test_logs (:warn, r"posterior_predict") linearpredictor(mod)
    @test lp isa Vector{Float64}
    @test length(lp) == 32
    @test isapprox(lp, fitted(mod)) # identity link (Normal): linpred == epred

    @test isnothing(offset(mod))

    # StatsModels-inherited vif/gvif assume an explicit intercept column in modelmatrix,
    # which ours never has (α fit separately) — overridden to error instead of mislead
    @test_throws ArgumentError vif(mod)
    @test_throws ArgumentError gvif(mod)

    # No Bayesian analogue exists for these MLE-only diagnostics — clear error, not a number
    for fn in (score, informationmatrix, leverage, cooksdistance)
        @test_throws ArgumentError fn(mod)
    end
    @test_throws ArgumentError reconstruct(mod)
    @test_throws ArgumentError reconstruct!(mod)
    @test_throws ArgumentError predict!(mod)

    # No single well-defined value on a posterior (dof/aic/bic assume a fixed param count)
    for fn in (loglikelihood, dof, mss, rss, nulldeviance, nullloglikelihood, aic, aicc, bic, r2, adjr2)
        @test_throws ArgumentError fn(mod)
    end
end
end # !DEV_SUBSET

if !DEV_SUBSET
@testset "draws API" begin
    Random.seed!(123)
    mod = turing_glm(@formula(MPG ~ Cyl + Disp), mtcars, Normal)
    quickfit!(mod)

    @test draws(mod) isa DimStack
    @test draws(mod, :fixef) isa DimArray
    @test_throws ArgumentError draws(mod, :not_a_real_type)
    @test_throws AssertionError draws(mod, :fixef; drop_draws=0, n_draws=10_000)

    collapsed = draws(mod, :fixef; collapse=true)
    uncollapsed = draws(mod, :fixef; collapse=false)
    @test ndims(uncollapsed) == ndims(collapsed) + 1
    @test hasdim(uncollapsed, :chain)
end
end # !DEV_SUBSET

if !DEV_SUBSET
@testset "Metrics" begin
    Random.seed!(123)
    mod = turing_glm(@formula(MPG ~ Cyl + Disp), mtcars, Normal)
    quickfit!(mod)

    tab = @test_nowarn calculate_metrics(mod, [rsq, rmse])
    @test tab isa DimArray
    @test all(dims(tab, 1) .== ["RSquared", "RootMeanSquaredError"])

    default_tab = @test_nowarn default_metrics(mod)
    @test all(dims(default_tab, 1) .== ["RSquared", "RootMeanSquaredError", "MeanAbsoluteError"])

    mtcars_binom = copy(mtcars)
    mtcars_binom.binom = mtcars_binom.MPG .> 20
    mod_bin = turing_glm(@formula(binom ~ Cyl + Disp), mtcars_binom, Bernoulli)
    quickfit!(mod_bin)

    tab_bin = @test_nowarn calculate_metrics(mod_bin, [accuracy, kappa])
    @test all(dims(tab_bin, 1) .== ["Accuracy", "Kappa"])

    default_bin = @test_nowarn default_metrics(mod_bin)
    @test all(
        dims(default_bin, 1) .==
        ["Accuracy", "Kappa", "TruePositiveRate", "TrueNegativeRate", "AreaUnderCurve", "Pseudo r2"],
    )

    @test outcome_as_distribution(mod_bin) isa UnivariateFinite
    @test_throws ArgumentError outcome_as_distribution(mod)
end
end # !DEV_SUBSET

if !DEV_SUBSET
@testset "Model Comparison" begin
    Random.seed!(123)
    mod_full = turing_glm(@formula(MPG ~ Cyl + Disp), mtcars, Normal)
    quickfit!(mod_full)
    mod_small = turing_glm(@formula(MPG ~ Cyl), mtcars, Normal)
    quickfit!(mod_small)

    mod_unfit = turing_glm(@formula(MPG ~ Cyl), mtcars, Normal)
    @test_throws ArgumentError psis_loo(mod_unfit)

    loo_full = psis_loo(mod_full)
    @test isfinite(loo_full.estimates.elpd)
    @test loo_full.estimates.se_elpd > 0

    mc_vec = loo_compare([mod_full, mod_small])
    mc_vararg = loo_compare(mod_full, mod_small)
    @test mc_vec.rank == mc_vararg.rank
    @test mc_vec.elpd_diff == mc_vararg.elpd_diff
    @test length(mc_vec.rank) == 2
    @test minimum(mc_vec.elpd_diff) == 0.0 # best model has elpd_diff 0
end
end # !DEV_SUBSET

if !DEV_SUBSET
@testset "Show/summary output" begin
    Random.seed!(123)
    mod = turing_glm(@formula(MPG ~ Cyl + Disp), mtcars, Normal)
    quickfit!(mod)

    show_output = sprint(show, mod)
    @test contains(show_output, "TuringRegression Model")
    @test contains(show_output, "Normal")
    @test contains(show_output, "MPG")

    summary_output = sprint((io, x) -> summary(io, x; show_metrics=true), mod)
    @test contains(summary_output, "Fixed Effects")
    @test all(contains.(Ref(summary_output), ["mean", "std", "ess_bulk", "ess_tail", "rhat"]))
    @test contains(summary_output, "Prediction Metrics")

    summary_no_metrics = sprint(summary, mod)
    @test !contains(summary_no_metrics, "Prediction Metrics")

    Random.seed!(123)
    mod_re = turing_glm(
        @formula(Reaction ~ 1 + Days + (1 + Days | Subject)), sleepstudy, Normal
    )
    quickfit!(mod_re)
    re_output = sprint(summary, mod_re)
    @test contains(re_output, "Random Effects: Subject")
    @test contains(re_output, "(SD)")
    @test contains(re_output, "Correlation")
end
end # !DEV_SUBSET

if !DEV_SUBSET
@testset "Plots (Makie extension smoke test)" begin
    # Not substantive — just checks the Makie extension still loads and runs.
    Random.seed!(123)
    mod = turing_glm(@formula(MPG ~ Cyl + Disp), mtcars, Normal)
    quickfit!(mod)

    @test_nowarn lineribbon(1:10, randn(50, 10))
    @test_nowarn conditional_dependency(mod, :Cyl)
    @test_nowarn pp_check_hist(mod)
    @test_nowarn pp_check_dens(mod)
    @test_nowarn pp_check_dens_overlay(mod; n_draws=20)
end
end # !DEV_SUBSET

if !DEV_SUBSET
@testset "Warnings & errors" begin
    @test_throws ErrorException turing_glm(@formula(MPG ~ Cyl + Disp), mtcars, Gamma)

    Random.seed!(123)
    mod = turing_glm(@formula(MPG ~ Cyl + Disp), mtcars, Normal)
    @suppress fit!(mod; samples=50, warmup=50, nchains=1, quiet=true)
    @test_logs (:warn,) match_mode = :any model_warnings(mod)
end
end # !DEV_SUBSET

if !DEV_SUBSET
@testset "fit! budget API" begin
    Random.seed!(123)
    mod = turing_glm(@formula(MPG ~ Cyl + Disp), mtcars, Normal)

    # samples splits evenly over chains: 200 total / 2 chains = 100 kept per chain
    Random.seed!(123)
    @suppress fit!(mod; samples=200, warmup=10, nchains=2, quiet=true)
    @test size(mod.samples, 1) == 100

    # inexact division rounds UP (never below request): cld(101, 2) = 51 per chain
    Random.seed!(123)
    @suppress fit!(mod; samples=101, warmup=10, nchains=2, quiet=true)
    @test size(mod.samples, 1) == 51

    # warmup=0 disables discarding: kept draws unaffected
    Random.seed!(123)
    @suppress fit!(mod; samples=120, warmup=0, nchains=2, quiet=true)
    @test size(mod.samples, 1) == 60
end
end # !DEV_SUBSET

if !DEV_SUBSET
@testset "Weighted fit" begin
    Random.seed!(42)
    mod_unweighted = turing_glm(@formula(MPG ~ Cyl + Disp), mtcars, Normal)
    quickfit!(mod_unweighted)

    Random.seed!(42)
    mod_weighted = turing_glm(
        @formula(MPG ~ Cyl + Disp), mtcars, Normal; weights=ones(nrow(mtcars))
    )
    quickfit!(mod_weighted)

    # weights≡1 is mathematically identical to unweighted, but `_likelihood`'s weighted
    # per-obs `@addlogprob!` loop vs its unweighted vectorized `MvNormal` logpdf take a
    # different numeric path through NUTS, so posterior means land close but not
    # bit-identical. Compare in pooled-SD units instead of a raw atol so the check
    # scales with actual MCMC noise rather than each param's raw magnitude.
    u_mean = Array(draws(mean, mod_unweighted, :fixef))
    w_mean = Array(draws(mean, mod_weighted, :fixef))
    u_sd = Array(draws(std, mod_unweighted, :fixef))
    w_sd = Array(draws(std, mod_weighted, :fixef))
    z = abs.(u_mean .- w_mean) ./ sqrt.(u_sd .^ 2 .+ w_sd .^ 2)

    @test all(z .< 1.0)
end
end # !DEV_SUBSET

# --- Big fits (Fixed/Random effects) run last: heaviest, slowest testsets ---

@testset "Fixed effects vs GLM" begin
    @testset "Normal" begin
        Random.seed!(123)
        mod = turing_glm(@formula(MPG ~ Cyl + Disp), mtcars, Normal)
        record_timing!("Normal (mtcars)", @elapsed @suppress fit!(mod; samples=BENCH_SAMPLES, warmup=BENCH_WARMUP, nchains=BENCH_NCHAINS, quiet=true))

        glm_mod = GLM.lm(@formula(MPG ~ Cyl + Disp), mtcars)
        est = Array(draws(mean, mod, :fixef))
        names = string.(collect(dims(draws(mean, mod, :fixef), :fixef))[1:3])
        for (n, o, c) in zip(names, est[1:3], GLM.coef(glm_mod))
            record_benchmark!("Normal (mtcars)", n, o, c)
        end

        # calibrated against N=2000/nchains=4 posterior mean, see benchmark table
        @test isapprox(GLM.coef(glm_mod), est[1:3], atol=0.3)
        @test isapprox(
            GLM.predict(glm_mod), Array(posterior_predict(mean, mod; type=:epred)), atol=1.0
        )
    end

    if !DEV_SUBSET
    @testset "Poisson" begin
        Random.seed!(123)
        mod = turing_glm(@formula(HP ~ Cyl + Disp), mtcars, Poisson)
        record_timing!("Poisson (mtcars)", @elapsed @suppress fit!(mod; samples=BENCH_SAMPLES, warmup=BENCH_WARMUP, nchains=BENCH_NCHAINS, quiet=true))

        glm_mod = GLM.glm(@formula(HP ~ Cyl + Disp), mtcars, Poisson(), GLM.LogLink())
        est = Array(draws(mean, mod, :fixef))
        names = string.(collect(dims(draws(mean, mod, :fixef), :fixef))[1:3])
        for (n, o, c) in zip(names, est[1:3], GLM.coef(glm_mod))
            record_benchmark!("Poisson (mtcars)", n, o, c)
        end

        @test isapprox(GLM.coef(glm_mod), est[1:3], atol=0.05)
        @test isapprox(
            GLM.predict(glm_mod), Array(posterior_predict(mean, mod; type=:epred)), atol=10.0
        )
    end

    @testset "NegativeBinomial" begin
        Random.seed!(123)
        mod = turing_glm(@formula(HP ~ Cyl + Disp), mtcars, NegativeBinomial)
        record_timing!("NegativeBinomial (mtcars)", @elapsed @suppress fit!(mod; samples=BENCH_SAMPLES, warmup=BENCH_WARMUP, nchains=BENCH_NCHAINS, quiet=true))

        glm_mod = GLM.glm(
            @formula(HP ~ Cyl + Disp), mtcars, NegativeBinomial(), GLM.LogLink()
        )
        est = Array(draws(mean, mod, :fixef))
        names = string.(collect(dims(draws(mean, mod, :fixef), :fixef))[1:3])
        for (n, o, c) in zip(names, est[1:3], GLM.coef(glm_mod))
            record_benchmark!("NegativeBinomial (mtcars)", n, o, c)
        end

        @test isapprox(GLM.coef(glm_mod), est[1:3], atol=0.1)
        @test isapprox(
            GLM.predict(glm_mod), Array(posterior_predict(mean, mod; type=:epred)), atol=15.0
        )
    end
    end # !DEV_SUBSET

    @testset "Bernoulli" begin
        Random.seed!(123)
        mod = turing_glm(@formula(Survived ~ Class + Sex + Age), titanic, Bernoulli)
        record_timing!("Bernoulli (titanic)", @elapsed @suppress fit!(mod; samples=BENCH_SAMPLES, warmup=BENCH_WARMUP, nchains=BENCH_NCHAINS, quiet=true))

        glm_mod = GLM.glm(
            @formula(Survived ~ Class + Sex + Age), titanic, Binomial(), GLM.LogitLink()
        )
        est = Array(draws(mean, mod, :fixef))
        names = string.(collect(dims(draws(mean, mod, :fixef), :fixef))[1:6])
        for (n, o, c) in zip(names, est[1:6], GLM.coef(glm_mod))
            record_benchmark!("Bernoulli (titanic)", n, o, c)
        end

        @test isapprox(GLM.coef(glm_mod), est[1:6], atol=0.1)
        @test isapprox(
            GLM.predict(glm_mod), Array(posterior_predict(mean, mod; type=:epred)), atol=0.1
        )
    end
end

@testset "Random effects — sleepstudy" begin
    @testset "correlated intercept + slope (1+Days|Subject)" begin
        Random.seed!(123)
        mod = turing_glm(
            @formula(Reaction ~ 1 + Days + (1 + Days | Subject)), sleepstudy, Normal
        )
        record_timing!("sleepstudy RE (1+Days|Subject)", @elapsed @suppress fit!(mod; samples=BENCH_SAMPLES, warmup=BENCH_WARMUP, nchains=BENCH_NCHAINS, quiet=true))

        d = draws(mod)
        @test :fixef ∈ propertynames(d)
        @test :Subject ∈ propertynames(d)
        @test :Subject_sd ∈ propertynames(d)
        @test :Subject_corr ∈ propertynames(d)

        fixef = Array(draws(mean, mod, :fixef))
        record_benchmark!("sleepstudy RE (lme4 REML)", "Intercept", fixef[1], 251.4)
        record_benchmark!("sleepstudy RE (lme4 REML)", "Days", fixef[2], 10.5)
        @test isapprox(fixef[1], 251.4, atol=15) # Intercept
        @test isapprox(fixef[2], 10.5, atol=5)   # Days

        subj_sd = draws(mean, mod, :Subject_sd)
        record_benchmark!(
            "sleepstudy RE (lme4 REML)", "Subject_sd[Intercept]", subj_sd[effect=At(:Intercept)], 24.7
        )
        record_benchmark!(
            "sleepstudy RE (lme4 REML)", "Subject_sd[Days]", subj_sd[effect=At(:Days)], 5.9
        )
        @test isapprox(subj_sd[effect=At(:Intercept)], 24.7, atol=15)
        @test isapprox(subj_sd[effect=At(:Days)], 5.9, atol=5)

        subj = draws(mod, :Subject)
        @test size(subj, :group) == length(unique(sleepstudy.Subject))
        @test collect(dims(subj, :effect)) == [:Intercept, :Days]

        corr = draws(mod, :Subject_corr)
        @test size(corr) == (2, 2, size(corr, :iter))
    end

    if !DEV_SUBSET
    @testset "intercept-only (1|Subject)" begin
        Random.seed!(123)
        mod = turing_glm(@formula(Reaction ~ 1 + Days + (1 | Subject)), sleepstudy, Normal)
        quickfit!(mod)

        d = draws(mod)
        @test :Subject ∈ propertynames(d)
        @test :Subject_sd ∈ propertynames(d)
        @test :Subject_corr ∉ propertynames(d)
        @test collect(dims(draws(mod, :Subject), :effect)) == [:Intercept]
    end

    @testset "slope-only, no intercept (0+Days|Subject)" begin
        Random.seed!(123)
        mod = turing_glm(
            @formula(Reaction ~ 1 + Days + (0 + Days | Subject)), sleepstudy, Normal
        )
        quickfit!(mod)

        d = draws(mod)
        @test :Subject ∈ propertynames(d)
        @test :Subject_corr ∉ propertynames(d)
        @test collect(dims(draws(mod, :Subject), :effect)) == [:Days]

        # Guards B4: slope-only ranef must not silently drop the real predictor
        @test_nowarn posterior_predict(mod; type=:epred)
    end
    end # !DEV_SUBSET
end

if !DEV_SUBSET
@testset "Random effects — Poisson (cbpp)" begin
    Random.seed!(123)
    mod = turing_glm(@formula(Incidence ~ 1 + Period + (1 | Herd)), cbpp, Poisson)
    quickfit!(mod)

    d = draws(mod)
    @test :Herd ∈ propertynames(d)
    @test :Herd_sd ∈ propertynames(d)
    ep = posterior_predict(mean, mod; type=:epred)
    @test all(Array(ep) .> 0) # Poisson mean is strictly positive
end
end # !DEV_SUBSET

end # @testset "TuringRegressions"
finally
    println()
    println("="^78)
    println("BENCHMARK: posterior mean vs canonical (GLM MLE / lme4 REML)")
    println("Full-sampling-budget models only (samples=BENCH_SAMPLES total, warmup=BENCH_WARMUP total, nchains=BENCH_NCHAINS)")
    println("="^78)
    pretty_table(DataFrame(BENCHMARK_ROWS); column_labels=["model", "param", "ours", "canonical", "abs err", "rel err %"])

    println()
    println("="^78)
    println("FIT TIMINGS: wall-clock seconds per full-budget fit (samples=BENCH_SAMPLES total, warmup=BENCH_WARMUP total, nchains=BENCH_NCHAINS)")
    println("First row includes TTFX compile — treat as upper bound")
    println("="^78)
    pretty_table(DataFrame(TIMING_ROWS); column_labels=["model", "seconds"])
end # try
