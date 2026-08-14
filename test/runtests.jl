# How to run (from the package root):
#   julia --project=. -e 'using Pkg; Pkg.test()'                       # standard, ~5 min
#   TR_TEST_LEVEL=fast julia --project=. -e 'using Pkg; Pkg.test()'    # ~2 min
#   TR_TEST_LEVEL=benchmarks julia --project=. -e 'using Pkg; Pkg.test()'  # ~8 min
# Always via Pkg.test(), never `julia --project=. test/runtests.jl` — Pkg.test()
# resolves the test deps in an isolated env, which is what CI and users get.
# Levels are defined below.

using TuringRegressions
using Test
using RDatasets
using StatsModels
using StatsBase: mean, std, var, CoefTable
using Suppressor: @suppress
using Random
using GLM: GLM
using DataFrames
using LinearAlgebra: diag
using CairoMakie
CairoMakie.activate!()
using PrettyTables

const TR = TuringRegressions

Random.seed!(1)

# --- Test levels -------------------------------------------------------------
# Fits are the only expensive thing in this suite, so the levels are defined purely
# by which fits they pay for. Each level is a superset of the one before:
#   fast     — no-MCMC unit tests + two small fixed-effect fits      (~2 min)
#   standard — + other families, random effects, weights, LOO, plots (~5 min)
#   benchmarks — + full-budget fits benchmarked against GLM / lme4   (~8 min)
# Times are totals for that level, not increments.
# `TR_TEST_LEVEL=fast julia --project=. -e 'using Pkg; Pkg.test()'`
const LEVELS = (:fast, :standard, :benchmarks)
const LEVEL = Symbol(get(ENV, "TR_TEST_LEVEL", "standard"))
LEVEL ∈ LEVELS || error("TR_TEST_LEVEL must be one of $LEVELS, got :$LEVEL")
atleast(level::Symbol) = findfirst(==(LEVEL), LEVELS) ≥ findfirst(==(level), LEVELS)

# Full-level budget, overridable so the benchmark can be re-run tighter/looser without
# editing tests. BENCH_SAMPLES/BENCH_WARMUP are TOTALS across chains (the `fit!` API).
const BENCH_SAMPLES = parse(Int, get(ENV, "TR_BENCH_SAMPLES", "2000"))
const BENCH_WARMUP = parse(Int, get(ENV, "TR_BENCH_WARMUP", "2000"))
const BENCH_NCHAINS = parse(Int, get(ENV, "TR_BENCH_NCHAINS", "2"))

@info "Setting up tests" level = LEVEL

# --- Data --------------------------------------------------------------------

mtcars = dataset("datasets", "mtcars")
mtcars.Binom = mtcars.MPG .> 20  # binary outcome for Bernoulli tests

# Normal GLM benchmark set. mtcars is n=32 with Cyl/Disp collinear (r≈0.9), so the
# standardised-scale prior visibly shrinks its coefficients (measured: 0.11 on the
# intercept, 0.023 on Cyl) and it CANNOT hit a tight tolerance — n, not collinearity,
# is the main driver; n=31 trees misses by 0.14 too. iris is n=150 with r≈-0.43
# between the two predictors and recovers OLS to <0.02.
iris = dataset("datasets", "iris")

titanic_df = dataset("datasets", "Titanic")
# Expand frequency cases into one row per observation
titanic = vcat([repeat(DataFrame(row[1:4]), row.Freq) for row in eachrow(titanic_df)]...)
titanic.Survived = titanic.Survived .== "Yes"

# Canonical mixed-model dataset (lme4): Reaction ~ Days + (Days|Subject), 18 subjects.
# Published lme4 REML estimates used below as loose sanity bounds:
# fixef ≈ (Intercept=251.4, Days=10.5); ranef sd ≈ (Intercept=24.7, Days=5.9); corr ≈ 0.07
sleepstudy = dataset("lme4", "sleepstudy")
# Second grouping variable, for nested/interaction grouping terms
sleepstudy.Batch = repeat(["p", "q"], inner=90)

# RE x non-Normal family: Incidence ~ Period + (1|Herd)
cbpp = dataset("lme4", "cbpp")

# --- Fixtures ----------------------------------------------------------------

# Short budget: enough draws for API shape and loose parameter recovery, not for the
# tight tolerances of the :benchmarks level below.
quickfit!(mod) = @suppress fit!(mod; samples=600, warmup=1000, nchains=2, quiet=true)

# Every fixture is fitted ONCE and shared read-only across testsets — refitting the
# same model per testset was most of the old suite's runtime. Anything that mutates a
# model (fit!, set_model_code!) builds its own.
function fitmodel(formula, data, family; seed=123, kwargs...)
    Random.seed!(seed)
    mod = turing_glm(formula, data, family; kwargs...)
    quickfit!(mod)
    return mod
end

# Benchmark table: posterior mean vs canonical (GLM MLE / lme4 REML), :benchmarks only —
# a running record of how tight our tolerances actually are, not just whether they pass.
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

# Prediction table: worst-row epred error against the canonical fit, next to the tolerance
# actually asserted. Printed so pred_atol can be set from measurement rather than guessed —
# a tolerance 100× the observed error is not testing anything.
const PRED_ROWS = NamedTuple[]

function record_pred!(model_name, max_abs_err, tol)
    push!(PRED_ROWS, (
        model=model_name,
        max_abs_err=round(max_abs_err; digits=4),
        tol=tol,
        headroom=round(tol / max(max_abs_err, 1e-8); digits=1),
    ))
end

# Fit-time table: wall-clock seconds per full-budget fit. The first recorded fit eats
# TTFX compile, so read the first row as an upper bound, not a like-for-like.
const TIMING_ROWS = NamedTuple[]
record_timing!(model_name, seconds) = push!(TIMING_ROWS, (model=model_name, seconds=round(seconds; digits=1)))

try # keep going through sibling testsets on failure, still print the benchmark table
@testset "TuringRegressions" begin

# =============================================================================
# No MCMC — pure functions on formulas, priors and transforms. Always run.
# =============================================================================

@testset "Formula handling" begin
    @testset "fixed effects" begin
        md = TR.extract_model_data(@formula(MPG ~ Cyl + Disp), mtcars)
        @test TR.has_intercept(md)
        @test TR.has_fixed_effects(md)
        @test !TR.has_random_effects(md)
        @test size(md.predictors.X) == (32, 2)   # intercept column stripped, α fit separately
        @test md.predictors.X_names == ["Cyl", "Disp"]
        @test md.y == mtcars.MPG

        md0 = TR.extract_model_data(@formula(MPG ~ 0 + Cyl + Disp), mtcars)
        @test !TR.has_intercept(md0)
        @test size(md0.predictors.X) == (32, 2)
    end

    # V16: the ranef predictor slice depends on has_intercept(term.lhs) — an
    # off-by-one here silently drops or duplicates a random slope.
    @testset "random-effect term shapes" begin
        correlated = TR.extract_model_data(@formula(Reaction ~ 1 + Days + (1 + Days | Subject)), sleepstudy).Z[1]
        @test correlated.predictors.has_intercept
        @test correlated.predictors.X_names == ["Days"]
        @test length(correlated.levels) == 18
        @test sort(unique(correlated.level_index)) == 1:18

        intercept_only = TR.extract_model_data(@formula(Reaction ~ 1 + Days + (1 | Subject)), sleepstudy).Z[1]
        @test intercept_only.predictors.has_intercept
        @test !TR.has_fixed_effects(intercept_only)

        slope_only = TR.extract_model_data(@formula(Reaction ~ 1 + Days + (0 + Days | Subject)), sleepstudy).Z[1]
        @test !slope_only.predictors.has_intercept
        @test slope_only.predictors.X_names == ["Days"]
    end

    @testset "nested and interaction grouping" begin
        nested = TR.extract_model_data(@formula(Reaction ~ 1 + Days + (1 | Batch / Subject)), sleepstudy)
        # MixedModels expands a/b into two separate grouping terms, not one
        @test [z.variable for z in nested.Z] == [:Batch, :Batch__Subject]
        @test length(nested.Z[1].levels) == 2
        @test length(nested.Z[2].levels) == 18  # every Subject sits in exactly one Batch

        interaction = TR.extract_model_data(@formula(Reaction ~ 1 + Days + (1 | Batch & Subject)), sleepstudy)
        @test [z.variable for z in interaction.Z] == [:Batch__Subject]
    end

    @testset "layer keys disambiguate shared grouping variables" begin
        # A group used once keeps its bare name — the public layer names must not change.
        single = TR.extract_model_data(@formula(Reaction ~ 1 + Days + (1 + Days | Subject)), sleepstudy)
        @test TR.ranef_layer_keys(single.Z) == [:Subject]

        nested = TR.extract_model_data(@formula(Reaction ~ 1 + Days + (1 | Batch / Subject)), sleepstudy)
        @test TR.ranef_layer_keys(nested.Z) == [:Batch, :Batch__Subject]

        # `(1|g) + (0+x|g)` is the lme4/brms spelling for uncorrelated intercept + slope.
        # Keyed on :Subject alone the second term's layers overwrite the first's.
        shared = TR.extract_model_data(
            @formula(Reaction ~ 1 + Days + (1 | Subject) + (0 + Days | Subject)), sleepstudy
        )
        @test TR.ranef_layer_keys(shared.Z) == [:Subject_1, :Subject_2]
    end

    @testset "new-data grouping levels" begin
        subjects = unique(sleepstudy.Subject)
        train = filter(row -> row.Subject != subjects[end], sleepstudy)
        held_out = filter(row -> row.Subject == subjects[end], sleepstudy)
        md = TR.extract_model_data(@formula(Reaction ~ 1 + Days + (1 + Days | Subject)), train)

        @test_throws ErrorException TR.new_random_effects(md, held_out)

        z = @test_logs (:warn, r"unseen level") TR.new_random_effects(md, held_out; allow_new_levels=true)
        @test all(z[1].level_index .== 0)      # sentinel: population-mean (zero) ranef
        @test z[1].levels == md.Z[1].levels    # level order stays the fitted model's
    end
end

@testset "Standardise/unstandardise round trip (V21)" begin
    families = [Normal, TDist, NegativeBinomial, Bernoulli, Poisson]
    ranef_formulas = [
        @formula(Reaction ~ 1 + Days + (1 + Days | Subject)),  # correlated intercept + slope
        @formula(Reaction ~ 1 + Days + (1 | Subject)),          # intercept-only
        @formula(Reaction ~ 1 + Days + (0 + Days | Subject)),   # slope-only, no intercept
    ]
    for formula in ranef_formulas
        md = TR.extract_model_data(formula, sleepstudy)
        for family in families
            tf = TR.compute_transform(md, family)
            md_std = TR.apply_transform(tf, md)
            md_back = TR.unstandardise_data(md_std, tf)

            @test md_back.predictors.X ≈ md.predictors.X
            @test md_back.y ≈ md.y
            for (re_back, re) in zip(md_back.Z, md.Z)
                @test re_back.predictors.X ≈ re.predictors.X
            end
        end
    end
end

@testset "Priors" begin
    @test default_prior(Normal).auxiliary == Exponential(1)
    @test default_prior(TDist).auxiliary == Gamma(2, 0.1)
    @test default_prior(Bernoulli).intercept == Normal(0, 5)
    @test_throws ErrorException default_prior(Gamma)

    # each keyword overrides exactly one field
    custom = default_prior(Normal; fixed_effects=Normal(0, 10), lkj_eta=20.0)
    @test custom.fixed_effects == Normal(0, 10)
    @test custom.lkj_eta == 20.0
    @test custom.intercept == default_prior(Normal).intercept
    @test default_prior(Normal).lkj_eta == 1.0

    mod = turing_glm(@formula(MPG ~ Cyl + Disp), mtcars, Normal)
    @test default_prior(mod).auxiliary == default_prior(Normal).auxiliary
    prior_text = sprint(prior_summary, mod)
    @test contains(prior_text, "standardised scale")
    @test !contains(prior_text, "LKJ")  # no random effects, so no correlation prior

    re_mod = turing_glm(@formula(Reaction ~ 1 + Days + (1 + Days | Subject)), sleepstudy, Normal)
    @test contains(sprint(prior_summary, re_mod), "LKJ")
end

@testset "Model construction" begin
    @test_throws ErrorException turing_glm(@formula(MPG ~ Cyl + Disp), mtcars, Gamma)

    @test_throws ArgumentError turing_glm(@formula(MPG ~ Cyl), mtcars, Normal; weights=ones(5))
    @test_throws ArgumentError turing_glm(@formula(MPG ~ Cyl), mtcars, Normal; weights=fill(-1.0, 32))

    # no-intercept fixef coefficients get shrunk by the fixed-width prior — warn, don't fail
    @test_logs (:warn, r"No-intercept formula") turing_glm(@formula(MPG ~ 0 + Cyl + Disp), mtcars, Normal)

    @testset "array form" begin
        y, X = randn(20), randn(20, 2)
        auto = turing_glm(y, X, Normal)
        @test auto.modeldata.predictors.X_names == ["X1", "X2"]
        @test TR.has_intercept(auto)
        @test auto.modeldata.y == y
        @test turing_glm(y, X, Normal; names=[:age, :income]).modeldata.predictors.X_names == ["age", "income"]
        @test_throws ArgumentError turing_glm(y, X, Normal; names=[:only_one])
        @test_throws ArgumentError turing_glm(y, X, Normal; names=[:y, :x])
    end

    @testset "generated model code" begin
        mod = turing_glm(@formula(MPG ~ Cyl + Disp), mtcars, Normal)
        code = @suppress modelcode(mod)
        @test code isa Expr

        # round-tripping the unmodified code must produce a model that still fits
        set_model_code!(mod, code)
        @test isnothing(mod.samples)  # stale draws from the old model are dropped
        @suppress fit!(mod; samples=20, warmup=20, nchains=1, quiet=true)
        @test size(draws(mod, :fixef), :fixef) == 4  # α, Cyl, Disp, σ

        @test_throws ArgumentError set_model_code!(mod, :(1 + 1))
    end

    @testset "unfitted model" begin
        unfit = turing_glm(@formula(MPG ~ Cyl), mtcars, Normal)
        @test_throws ArgumentError draws(unfit)
        @test_throws ArgumentError draws(unfit, :fixef)
        @test_throws ArgumentError psis_loo(unfit)
        @test_throws ArgumentError model_summary(unfit)
        @test !isfitted(unfit)
        @test isnothing(model_warnings(unfit))
    end
end

# =============================================================================
# :fast and up — two small fixed-effect fits, shared by everything below.
# =============================================================================

normal_mod = fitmodel(@formula(MPG ~ Cyl + Disp), mtcars, Normal)
bernoulli_mod = fitmodel(@formula(Binom ~ Cyl + Disp), mtcars, Bernoulli)

@testset "draws API" begin
    @test draws(normal_mod) isa DimStack
    @test propertynames(draws(normal_mod)) == (:fixef,)
    @test collect(dims(draws(normal_mod, :fixef), :fixef)) == [:α, :Cyl, :Disp, :σ]
    @test_throws ArgumentError draws(normal_mod, :not_a_real_type)
    @test_throws ArgumentError draws(normal_mod, :fixef; n_draws=10_000)

    # V7: collapsed shape is (params, iter*chain); uncollapsed keeps :chain
    collapsed = draws(normal_mod, :fixef; collapse=true)
    uncollapsed = draws(normal_mod, :fixef; collapse=false)
    @test size(collapsed) == (4, 600)
    @test size(uncollapsed) == (4, 300, 2)
    @test hasdim(uncollapsed, :chain)
    @test size(draws(normal_mod, :fixef; drop_draws=100), :iter) == 400

    # reducer form drops the aggregated dims
    @test size(draws(mean, normal_mod, :fixef)) == (4,)
    @test Array(draws(mean, normal_mod, :fixef)) ≈ vec(mean(Array(collapsed); dims=2))

    # readme indexing idioms — orderless DimensionalData selectors on the :fixef dim
    @test Array(collapsed[fixef=At(:Cyl)]) == Array(collapsed)[2, :]
    @test collect(dims(uncollapsed[chain=1:2, fixef=Where(x -> occursin(r"yl", string(x)))], :fixef)) == [:Cyl]

    @test Array(outcome(normal_mod)) == mtcars.MPG
    @test collect(dims(get_fixef_predictors(normal_mod), :var)) == ["Cyl", "Disp"]
end

@testset "Predict" begin
    @testset "call forms" begin
        @test posterior_predict(normal_mod) isa DimArray
        @test size(posterior_predict(normal_mod)) == (32, 600)
        @test size(posterior_predict(normal_mod, mtcars[1:5, :])) == (5, 600)
        @test size(posterior_predict(mean, normal_mod)) == (32,)
        @test_throws ArgumentError posterior_predict(normal_mod, randn(32, 5))
        @test_throws ArgumentError posterior_predict(normal_mod; type=:not_a_type)
    end

    @testset "type variants and link relations" begin
        # V5, identity link (Normal): linpred == epred
        @test Array(posterior_predict(normal_mod; type=:linpred)) ≈
              Array(posterior_predict(normal_mod; type=:epred))
        # V5, logit link (Bernoulli)
        @test Array(posterior_predict(bernoulli_mod; type=:linpred)) ≈
              logit.(Array(posterior_predict(bernoulli_mod; type=:epred)))
    end

    @testset "posterior predictive noise" begin
        Random.seed!(123)
        post = Array(posterior_predict(normal_mod; type=:posterior, collapse=false))
        ep = Array(posterior_predict(normal_mod; type=:epred, collapse=false))
        σ = Array(draws(normal_mod, :fixef; collapse=false)[fixef=At([:σ])])

        # V6: extra variance across draws...
        @test mean(var(post; dims=2)) > mean(var(ep; dims=2))
        # ...and independent noise ACROSS ROWS within each draw — a single shared draw of
        # noise would still pass the check above. Residual row-variance should track σ².
        @test isapprox(mean(var(post .- ep; dims=1)), mean(σ .^ 2); rtol=0.3)
    end

    @testset "new data uses the raw scale" begin
        # V15/V22: stored β are original-scale, new X must go straight through unscaled.
        # Re-standardising against the new data's own mean/sd would break this badly.
        new_data = mtcars[3:8, :]
        glm_mod = GLM.lm(@formula(MPG ~ Cyl + Disp), mtcars)
        @test isapprox(
            GLM.predict(glm_mod, new_data),
            Array(posterior_predict(mean, normal_mod, new_data; type=:epred)),
            atol=1.5,
        )
    end
end

@testset "StatsAPI interface" begin
    @test coefnames(normal_mod) == ["α", "Cyl", "Disp"]
    @test length(coef(normal_mod)) == 3
    @test nobs(normal_mod) == 32
    @test isfitted(normal_mod)
    @test islinear(normal_mod)
    @test weights(normal_mod) == ones(32)
    @test responsename(normal_mod) == "MPG"
    @test meanresponse(normal_mod) == mean(mtcars.MPG)
    @test size(modelmatrix(normal_mod)) == (32, 2)
    @test length(stderror(normal_mod)) == 3
    @test size(vcov(normal_mod)) == (3, 3)
    @test size(confint(normal_mod)) == (3, 2)
    @test coeftable(normal_mod) isa CoefTable
    @test response(normal_mod) == normal_mod.modeldata.y
    @test residuals(normal_mod) ≈ response(normal_mod) .- fitted(normal_mod)
    @test isnothing(offset(normal_mod))

    # point estimates collapse the posterior — must say so, once
    point = @test_logs (:warn, r"posterior_predict") predict(normal_mod)
    @test point isa Vector{Float64}
    @test length(point) == 32
    lp = @test_logs (:warn, r"posterior_predict") linearpredictor(normal_mod)
    @test lp ≈ fitted(normal_mod)  # identity link (Normal)

    # StatsModels-inherited vif/gvif assume an explicit intercept column in modelmatrix,
    # which ours never has (α fit separately) — overridden to error instead of mislead
    @test_throws ArgumentError vif(normal_mod)
    @test_throws ArgumentError gvif(normal_mod)

    # No Bayesian analogue exists for these MLE-only diagnostics — clear error, not a number
    for fn in (score, informationmatrix, leverage, cooksdistance, reconstruct, reconstruct!, predict!)
        @test_throws ArgumentError fn(normal_mod)
    end
    # No single well-defined value on a posterior (dof/aic/bic assume a fixed param count)
    for fn in (loglikelihood, dof, mss, rss, nulldeviance, nullloglikelihood, aic, aicc, bic, r2, adjr2)
        @test_throws ArgumentError fn(normal_mod)
    end
end

@testset "Display" begin
    compact = sprint(show, normal_mod)
    @test contains(compact, "TuringRegression{Normal}")
    @test contains(compact, "fitted")
    @test !contains(compact, "Fixed Effects")

    @test contains(sprint(summary, normal_mod), "n=32")

    full = sprint(show, MIME("text/plain"), normal_mod)
    @test all(contains.(Ref(full), ["TuringRegression Model", "Normal", "MPG", "Prior (standardised scale)"]))

    # tables crop to the display width rather than wrapping, so a default 80-col `sprint`
    # loses the right-hand columns — render wide to see them all
    summary_text = sprint(
        (io, x) -> model_summary(io, x), normal_mod; context=(:displaysize => (24, 200))
    )
    @test contains(summary_text, "Family:")
    @test contains(summary_text, "Fixed Effects")
    @test all(contains.(Ref(summary_text), ["mean", "std", "ess_bulk", "ess_tail", "rhat"]))
    # ...and narrow really does crop, not wrap: one row stays one line
    narrow = sprint((io, x) -> model_summary(io, x), normal_mod; context=(:displaysize => (24, 60)))
    @test contains(narrow, "columns omitted")
    @test all(line -> length(line) <= 60, filter(contains('│'), split(narrow, '\n')))

    @test model_summary(devnull, normal_mod; return_table=true) isa NamedTuple

    # displaying a Vector{TuringRegression} must stay compact and never warn
    vec_output = sprint(show, [normal_mod, normal_mod])
    @test contains(vec_output, "TuringRegression{Normal}")
    @test !contains(vec_output, "rhat")
end

@testset "Pointwise log-likelihood and LOO" begin
    ll = TR.pointwise_loglik(normal_mod)
    @test size(ll) == (300, 2, 32)  # (iter, chain, obs)
    @test all(isfinite, ll)

    result = psis_loo(normal_mod)
    @test isfinite(result.estimates.elpd)
    @test result.estimates.se_elpd > 0
end

@testset "fit! budget" begin
    # V24: samples/warmup are TOTALS across chains; warmup is extra and never returned.
    mod = turing_glm(@formula(MPG ~ Cyl + Disp), mtcars, Normal)

    @suppress fit!(mod; samples=200, warmup=10, nchains=2, quiet=true)
    @test size(mod.samples, 1) == 100

    # inexact division rounds UP, so the realised total is never below the request
    @suppress fit!(mod; samples=101, warmup=10, nchains=2, quiet=true)
    @test size(mod.samples, 1) == 51

    @suppress fit!(mod; samples=120, warmup=0, nchains=2, quiet=true)
    @test size(mod.samples, 1) == 60

    # kwargs fit! computes itself must not be silently overridden
    @test_throws ArgumentError fit!(mod; samples=20, nchains=1, N=10)
    @test_throws ArgumentError fit!(mod; samples=20, nchains=1, discard_initial=10)

    # a deliberately under-sampled fit must trip the diagnostics
    @suppress fit!(mod; samples=50, warmup=50, nchains=1, quiet=true)
    @test_logs (:warn,) match_mode = :any model_warnings(mod)
end

# =============================================================================
# :standard and up — remaining families, random effects, weights, plots.
# =============================================================================

if atleast(:standard)

tdist_mod = fitmodel(@formula(MPG ~ Cyl + Disp), mtcars, TDist)
poisson_mod = fitmodel(@formula(HP ~ Cyl + Disp), mtcars, Poisson)
negbin_mod = fitmodel(@formula(HP ~ Cyl + Disp), mtcars, NegativeBinomial)

@testset "All families" begin
    all_mods = (normal_mod, tdist_mod, poisson_mod, negbin_mod, bernoulli_mod)

    @test collect(dims(draws(tdist_mod, :fixef), :fixef)) == [:α, :Cyl, :Disp, :σ, :ν]
    @test collect(dims(draws(poisson_mod, :fixef), :fixef)) == [:α, :Cyl, :Disp]
    @test collect(dims(draws(negbin_mod, :fixef), :fixef)) == [:α, :Cyl, :Disp, :ϕ]

    # V5, log link
    for mod in (poisson_mod, negbin_mod)
        @test Array(posterior_predict(mod; type=:linpred)) ≈ log.(Array(posterior_predict(mod; type=:epred)))
    end

    for mod in all_mods
        # V6 for every family, including the count/binary ones where the noise is in the
        # sampling distribution rather than an explicit σ
        ep = Array(posterior_predict(mod; type=:epred, collapse=false))
        post = Array(posterior_predict(mod; type=:posterior, collapse=false))
        @test size(post) == size(ep) == (32, 300, 2)
        @test mean(var(post; dims=2)) > mean(var(ep; dims=2))
    end

    @test all(Array(posterior_predict(poisson_mod; type=:epred)) .> 0)
    @test all(0 .≤ Array(posterior_predict(bernoulli_mod; type=:epred)) .≤ 1)
    @test all(∈((0, 1)), Array(posterior_predict(bernoulli_mod; type=:posterior)))
end

@testset "Random effects — sleepstudy" begin
    re_corr = fitmodel(@formula(Reaction ~ 1 + Days + (1 + Days | Subject)), sleepstudy, Normal)
    re_intercept = fitmodel(@formula(Reaction ~ 1 + Days + (1 | Subject)), sleepstudy, Normal)
    re_slope = fitmodel(@formula(Reaction ~ 1 + Days + (0 + Days | Subject)), sleepstudy, Normal)

    @testset "correlated intercept + slope (1+Days|Subject)" begin
        d = draws(re_corr)
        @test propertynames(d) == (:fixef, :Subject, :Subject_sd, :Subject_corr)

        subject = draws(re_corr, :Subject)
        @test collect(dims(subject, :effect__Subject)) == [:Intercept, :Days]
        @test size(subject, :group__Subject) == 18
        # readme idiom: one subject's slope offset. Levels are Strings, not the
        # integers the raw sleepstudy column looks like.
        @test eltype(dims(subject, :group__Subject)) <: AbstractString
        @test draws(mean, re_corr, :Subject)[effect__Subject=At(:Days), group__Subject=At("308")] isa Real

        corr = draws(re_corr, :Subject_corr)
        @test size(corr) == (2, 2, size(corr, :iter))
        corr_point = Array(draws(mean, re_corr, :Subject_corr))
        @test diag(corr_point) ≈ [1.0, 1.0]
        @test corr_point[1, 2] ≈ corr_point[2, 1]

        # loose sanity vs lme4 REML — the tight version is in the :benchmarks level
        fixef = Array(draws(mean, re_corr, :fixef))
        @test isapprox(fixef[1], 251.4, atol=20)
        @test isapprox(fixef[2], 10.5, atol=6)
        subject_sd = draws(mean, re_corr, :Subject_sd)
        @test isapprox(subject_sd[effect__Subject=At(:Intercept)], 24.7, atol=20)
        @test isapprox(subject_sd[effect__Subject=At(:Days)], 5.9, atol=6)

        # no grouping information in a bare matrix, so this must error rather than
        # silently predict without the random effects. `copy` matters: _resolve_z only lets
        # a matrix through when it is `===` the fitted one, so passing the fitted matrix
        # itself takes the fast path and does NOT error.
        @test_throws ErrorException posterior_predict(re_corr, copy(re_corr.modeldata.predictors.X))
        @test_throws ArgumentError modelmatrix(re_corr)
        @test length(coef(re_corr)) == 2  # fixef only, still well-defined
    end

    @testset "intercept-only (1|Subject)" begin
        @test propertynames(draws(re_intercept)) == (:fixef, :Subject, :Subject_sd)
        @test collect(dims(draws(re_intercept, :Subject), :effect__Subject)) == [:Intercept]
    end

    @testset "slope-only (0+Days|Subject)" begin
        d = draws(re_slope)
        @test propertynames(d) == (:fixef, :Subject, :Subject_sd)
        @test collect(dims(draws(re_slope, :Subject), :effect__Subject)) == [:Days]

        # epred vs lme4 `lmer(Reaction ~ 1 + Days + (0 + Days | Subject), sleepstudy)`
        # fitted values (REML), spot-checked on subject 1 (rows 1:10, full Days=0:9 range).
        # A centred ranef predictor would fit an implicit per-group intercept instead of a
        # through-origin slope, diverging from lme4 everywhere Days != 0.
        lme4_fitted_subject1 = [
            251.405, 271.492, 291.578, 311.665, 331.752, 351.839, 371.925, 392.012, 412.099, 432.185,
        ]
        epred_mean = Array(posterior_predict(mean, re_slope; type=:epred))
        @test isapprox(epred_mean[1:10], lme4_fitted_subject1, atol=15, norm=x -> maximum(abs, x))
    end

    # `(1|g) + (0+x|g)`: two terms, one grouping variable. Keying layers on the bare
    # group name dropped the first term entirely — its layers were overwritten on merge,
    # and `_add_random_effects!` then found no :Intercept in the survivor and silently
    # left the random intercept out of every prediction.
    @testset "two terms on one grouping variable" begin
        uncorr = fitmodel(
            @formula(Reaction ~ 1 + Days + (1 | Subject) + (0 + Days | Subject)), sleepstudy, Normal
        )
        @test propertynames(draws(uncorr)) ==
              (:fixef, :Subject_1, :Subject_1_sd, :Subject_2, :Subject_2_sd)
        @test collect(dims(draws(uncorr, :Subject_1), :effect__Subject_1)) == [:Intercept]
        @test collect(dims(draws(uncorr, :Subject_2), :effect__Subject_2)) == [:Days]

        # Both terms must reach the prediction. Identity link + posterior mean is linear,
        # so the reconstruction is exact, not approximate.
        fixef = Array(draws(mean, uncorr, :fixef))
        u_int = vec(Array(draws(mean, uncorr, :Subject_1)))   # one effect per term, so
        u_slope = vec(Array(draws(mean, uncorr, :Subject_2))) # dropdims leaves 18 levels
        g = uncorr.modeldata.Z[1].level_index
        expected = fixef[1] .+ fixef[2] .* sleepstudy.Days .+ u_int[g] .+ u_slope[g] .* sleepstudy.Days
        @test Array(posterior_predict(mean, uncorr; type=:epred)) ≈ expected

        # dropping the intercept term would collapse the per-subject offsets to zero
        @test maximum(abs, u_int) > 1.0

        text = sprint(model_summary, uncorr)
        @test contains(text, "Random Effects: Subject_1 (SD)")
        @test contains(text, "Random Effects: Subject_2 (SD)")
    end

    @testset "predict on new grouping levels" begin
        # seen levels: predictions must use that subject's own offsets
        seen = sleepstudy[1:10, :]
        @test Array(posterior_predict(mean, re_corr, seen; type=:epred)) ≈
              Array(posterior_predict(mean, re_corr; type=:epred))[1:10]

        # relabelling to a different (but still seen) level must pick up that level's offsets
        relabelled = copy(sleepstudy[1:10, :])
        relabelled.Subject = fill(sleepstudy.Subject[end], 10)
        @test Array(posterior_predict(mean, re_corr, relabelled; type=:epred)) !=
              Array(posterior_predict(mean, re_corr, seen; type=:epred))

        # a genuinely unseen level errors by default, and falls back to the population
        # mean (zero ranef) when explicitly allowed
        subjects = unique(sleepstudy.Subject)
        train = filter(row -> row.Subject != subjects[end], sleepstudy)
        held_out = filter(row -> row.Subject == subjects[end], sleepstudy)
        partial = fitmodel(@formula(Reaction ~ 1 + Days + (1 | Subject)), train, Normal)

        @test_throws ErrorException posterior_predict(partial, held_out)
        allowed = @test_logs (:warn, r"unseen level") posterior_predict(
            mean, partial, held_out; type=:epred, allow_new_levels=true
        )
        # zero ranef ⇒ the population-level fit, i.e. the fixed effects alone
        fixef = Array(draws(mean, partial, :fixef))
        @test Array(allowed) ≈ fixef[1] .+ held_out.Days .* fixef[2]
    end

    @testset "lkj_eta prior override" begin
        wide = fitmodel(
            @formula(Reaction ~ 1 + Days + (1 + Days | Subject)), sleepstudy, Normal;
            priors=default_prior(Normal; lkj_eta=20.0),
        )
        @test wide.prior.lkj_eta == 20.0
        # high eta shrinks the correlation toward 0
        @test abs(mean(Array(draws(wide, :Subject_corr))[1, 2, :])) < 0.5
    end

    @testset "summary output" begin
        text = sprint(model_summary, re_corr)
        @test contains(text, "Random Effects: Subject (SD)")
        @test contains(text, "Random Effects: Subject (Correlation)")
        @test contains(text, "Random Effects: Subject (Intercept)")
        # no correlation matrix when there is only one effect per group
        @test !contains(sprint(model_summary, re_intercept), "Correlation")
    end
end

# T59: two ranef terms of different level counts (Batch=2, Batch:Subject=18) used to
# collide in TR.parameters — both terms shared the literal dim name `:group`, which
# DimStack requires to have one consistent length across all its layers.
@testset "Random effects — nested/interaction grouping (Batch/Subject)" begin
    nested = fitmodel(@formula(Reaction ~ 1 + Days + (1 | Batch / Subject)), sleepstudy, Normal)
    interaction = fitmodel(@formula(Reaction ~ 1 + Days + (1 | Batch & Subject)), sleepstudy, Normal)

    @test propertynames(nested.parameters) == (:fixef, :Batch, :Batch_sd, :Batch__Subject, :Batch__Subject_sd)
    @test propertynames(interaction.parameters) == (:fixef, :Batch__Subject, :Batch__Subject_sd)

    batch_draws = draws(nested, :Batch)
    @test collect(dims(batch_draws, :group__Batch)) == ["p", "q"]

    nested_draws = draws(nested, :Batch__Subject)
    @test size(nested_draws, :group__Batch__Subject) == 18

    interaction_draws = draws(interaction, :Batch__Subject)
    @test size(interaction_draws, :group__Batch__Subject) == 18

    @test contains(sprint(model_summary, nested), "Random Effects: Batch (Intercept)")
    @test contains(sprint(model_summary, nested), "Random Effects: Batch__Subject (Intercept)")
end

@testset "Random effects — Poisson (cbpp)" begin
    mod = fitmodel(@formula(Incidence ~ 1 + Period + (1 | Herd)), cbpp, Poisson)
    @test :Herd ∈ propertynames(draws(mod))
    @test :Herd_sd ∈ propertynames(draws(mod))
    @test all(Array(posterior_predict(mean, mod; type=:epred)) .> 0)
end

@testset "Weighted fits" begin
    # weights ≡ 1 is mathematically identical to unweighted, but `_likelihood`'s weighted
    # per-obs `@addlogprob!` loop vs its unweighted vectorised logpdf take a different
    # numeric path through NUTS, so posterior means land close but not bit-identical.
    # Compare in pooled-SD units so the check scales with actual MCMC noise rather than
    # each parameter's raw magnitude.
    function agrees_with(unweighted, weighted, layer)
        z = abs.(Array(draws(mean, unweighted, layer)) .- Array(draws(mean, weighted, layer))) ./
            sqrt.(Array(draws(std, unweighted, layer)) .^ 2 .+ Array(draws(std, weighted, layer)) .^ 2)
        return all(z .< 1.0)
    end

    weighted_fixef = fitmodel(
        @formula(MPG ~ Cyl + Disp), mtcars, Normal; seed=42, weights=ones(nrow(mtcars))
    )
    unweighted_fixef = fitmodel(@formula(MPG ~ Cyl + Disp), mtcars, Normal; seed=42)
    @test agrees_with(unweighted_fixef, weighted_fixef, :fixef)

    # V14 with random effects: the weighted likelihood branch and the ranef branch are
    # generated independently, so their combination needs its own check
    formula = @formula(Reaction ~ 1 + Days + (1 | Subject))
    weighted_re = fitmodel(formula, sleepstudy, Normal; seed=42, weights=ones(nrow(sleepstudy)))
    unweighted_re = fitmodel(formula, sleepstudy, Normal; seed=42)
    @test agrees_with(unweighted_re, weighted_re, :fixef)
    @test agrees_with(unweighted_re, weighted_re, :Subject_sd)

    @test weights(weighted_re) == ones(nrow(sleepstudy))

    # Weights ≡ 1 only exercises the loop, not the weighting itself: a `w` dropped from
    # the `@addlogprob!` term would still pass above. Zero-weighting rows must reproduce
    # the fit on the subset — the only non-uniform case with an exact reference fit.
    keep = mtcars.Cyl .!= 8
    zero_weighted = fitmodel(
        @formula(MPG ~ Disp), mtcars, Normal; seed=42, weights=Float64.(keep)
    )
    subset_only = fitmodel(@formula(MPG ~ Disp), mtcars[keep, :], Normal; seed=42)
    @test agrees_with(subset_only, zero_weighted, :fixef)
    # ...and the two datasets must give genuinely different slopes, or the check above is
    # vacuous. Checked by OLS rather than a fourth fit.
    ols(df) = GLM.coef(GLM.lm(GLM.@formula(MPG ~ Disp), df))
    @test abs(ols(mtcars)[2] - ols(mtcars[keep, :])[2]) > 0.02
end

@testset "Model comparison" begin
    small_mod = fitmodel(@formula(MPG ~ Cyl), mtcars, Normal)

    by_vector = loo_compare([normal_mod, small_mod])
    by_vararg = loo_compare(normal_mod, small_mod)
    @test by_vector.rank == by_vararg.rank
    @test by_vector.elpd_diff == by_vararg.elpd_diff
    @test length(by_vector.rank) == 2
    @test minimum(by_vector.elpd_diff) == 0.0  # best model has elpd_diff 0
end

@testset "Plots (Makie extension smoke test)" begin
    # Not substantive — just checks the Makie extension still loads and runs.
    @test_nowarn lineribbon(1:10, randn(50, 10))
    @test_nowarn pp_check_hist(normal_mod)
    @test_nowarn pp_check_dens(normal_mod)
    @test_nowarn pp_check_dens_overlay(normal_mod; n_draws=20)

    # The readme shows plain Makie plots over `draws` output, relying on the
    # DimensionalData integration rather than any recipe of ours — still worth a
    # smoke test, since a dim rename would break every plotting example in the docs.
    coefs = draws(normal_mod, :fixef)
    pos, vals, axis = categorical_layout(coefs)

    # Each category must get its own parameter's draws and nothing else —
    # `categorical_layout` exists because DimensionalData 0.30 gets this wrong.
    @test length(pos) == length(vals) == length(coefs)
    @test axis.xticks == (1:size(coefs, 1), string.(lookup(coefs, :fixef)))
    for i in 1:size(coefs, 1)
        @test vals[pos .== i] == parent(coefs)[i, :]
    end

    # Categorical dim second: same layout, transposed input.
    @test categorical_layout(permutedims(coefs), 2) == (pos, vals, axis)
    @test_throws ArgumentError categorical_layout(coefs, :nope)

    @test_nowarn violin(pos, vals; axis, scale=:width, show_median=true, side=:left)
    @test_nowarn rainclouds(pos, vals; axis)
    @test_nowarn boxplot(pos, vals; axis)
    @test_nowarn scatter(coefs[fixef=At([:α, :σ])])
    @test_nowarn scatter(parent(coefs[fixef=At([:α, :σ, :Cyl])]))
    @test_nowarn Makie.series(draws(normal_mod, :fixef; collapse=false)[fixef=At(:α)]'; linewidth=0.3)

    # Readme's lineribbon example: linear predictor over a grid of one predictor.
    grid = [fill(mean(mtcars.Cyl), 50) collect(range(extrema(mtcars.Disp)...; length=50))]
    lp = posterior_predict(normal_mod, grid; type=:linpred)
    @test size(lp, 1) == 50
    @test_nowarn lineribbon(grid[:, 2], parent(lp)')
end

end # atleast(:standard)

# =============================================================================
# :benchmarks only — full sampling budget, tight tolerances against GLM / lme4.
# =============================================================================

if atleast(:benchmarks)

# Fits at the full budget and records timing; returns the fitted model.
function benchfit(name, formula, data, family)
    Random.seed!(123)
    mod = turing_glm(formula, data, family)
    seconds = @elapsed @suppress fit!(
        mod; samples=BENCH_SAMPLES, warmup=BENCH_WARMUP, nchains=BENCH_NCHAINS, quiet=true
    )
    record_timing!(name, seconds)
    return mod
end

# Posterior mean vs GLM MLE, coefficient by coefficient, recorded for the benchmark table.
function check_against_glm(name, mod, glm_mod; coef_atol, pred_atol)
    est = Array(draws(mean, mod, :fixef))
    n = length(GLM.coef(glm_mod))
    names = string.(collect(dims(draws(mean, mod, :fixef), :fixef))[1:n])
    for (param, ours, canonical) in zip(names, est[1:n], GLM.coef(glm_mod))
        record_benchmark!(name, param, ours, canonical)
    end
    # Elementwise max, NOT isapprox: isapprox on vectors compares the L2 norm, so the
    # same per-row accuracy needs a bigger tolerance on a bigger dataset (√n). Worst
    # single coefficient / row is the quantity we actually care about.
    @test maximum(abs, GLM.coef(glm_mod) .- est[1:n]) < coef_atol

    ours_pred = Array(posterior_predict(mean, mod; type=:epred))
    record_pred!(name, maximum(abs, GLM.predict(glm_mod) .- ours_pred), pred_atol)
    @test maximum(abs, GLM.predict(glm_mod) .- ours_pred) < pred_atol
end

@testset "Fixed effects vs GLM" begin
    # V3/V4: posterior mean ≈ MLE, epred ≈ GLM.predict, per family
    check_against_glm(
        "Normal (iris)",
        benchfit("Normal (iris)", @formula(SepalLength ~ SepalWidth + PetalLength), iris, Normal),
        GLM.lm(@formula(SepalLength ~ SepalWidth + PetalLength), iris);
        coef_atol=0.02, pred_atol=0.005,  # measured: coef 0.003, epred 0.0012
    )
    # No-intercept does NOT recover the MLE and is not meant to: with no intercept the
    # slopes carry the whole mean of y, so they sit far from 0 and the standardised-scale
    # Normal(0,2) fixef prior shrinks them (verified: widening to Normal(0,10) lands on the
    # OLS estimate). turing_glm warns about exactly this (see "Model construction"). Kept as
    # a sanity check that it is shrunk, not broken — tolerances are deliberately loose.
    check_against_glm(
        "Normal no-intercept (mtcars)",
        benchfit("Normal no-intercept (mtcars)", @formula(MPG ~ 0 + Cyl + Disp), mtcars, Normal),
        GLM.lm(@formula(MPG ~ 0 + Cyl + Disp), mtcars);
        coef_atol=1.0, pred_atol=2.0,  # measured: coef 0.334, epred 1.236
    )
    check_against_glm(
        "Poisson (mtcars)",
        benchfit("Poisson (mtcars)", @formula(HP ~ Cyl + Disp), mtcars, Poisson),
        GLM.glm(@formula(HP ~ Cyl + Disp), mtcars, Poisson(), GLM.LogLink());
        coef_atol=0.02, pred_atol=0.5,  # measured: coef 0.002, epred 0.081
    )
    check_against_glm(
        "NegativeBinomial (mtcars)",
        benchfit("NegativeBinomial (mtcars)", @formula(HP ~ Cyl + Disp), mtcars, NegativeBinomial),
        GLM.glm(@formula(HP ~ Cyl + Disp), mtcars, NegativeBinomial(), GLM.LogLink());
        coef_atol=0.02, pred_atol=5.0,  # measured: coef 0.005, epred 2.014
    )
    check_against_glm(
        "Bernoulli (titanic)",
        benchfit("Bernoulli (titanic)", @formula(Survived ~ Class + Sex + Age), titanic, Bernoulli),
        GLM.glm(@formula(Survived ~ Class + Sex + Age), titanic, Binomial(), GLM.LogitLink());
        coef_atol=0.02, pred_atol=0.02,  # measured: coef 0.006, epred 0.0037
    )
end

@testset "Random effects vs lme4 — sleepstudy" begin
    mod = benchfit(
        "sleepstudy RE (1+Days|Subject)",
        @formula(Reaction ~ 1 + Days + (1 + Days | Subject)), sleepstudy, Normal,
    )

    fixef = Array(draws(mean, mod, :fixef))
    record_benchmark!("sleepstudy RE (lme4 REML)", "Intercept", fixef[1], 251.4)
    record_benchmark!("sleepstudy RE (lme4 REML)", "Days", fixef[2], 10.5)
    # fixef recover lme4 REML closely — measured error across seeds is ≤0.33 and ≤0.08,
    # so atol=1 is a real regression check, not a rubber stamp
    @test isapprox(fixef[1], 251.4, atol=1)
    @test isapprox(fixef[2], 10.5, atol=1)

    subject_sd = draws(mean, mod, :Subject_sd)
    record_benchmark!("sleepstudy RE (lme4 REML)", "Subject_sd[Intercept]", subject_sd[effect=At(:Intercept)], 24.7)
    record_benchmark!("sleepstudy RE (lme4 REML)", "Subject_sd[Days]", subject_sd[effect=At(:Days)], 5.9)
    # KNOWN ISSUE — intercept SD comes out ~29 against lme4's 24.7, consistently across
    # seeds (measured: 28.8-29.2), so it is bias not noise. V26: the LKJ prior is flat on
    # the STANDARDISED-scale correlation, which shrinks the correlation and pushes the
    # intercept SD up to compensate. Bounded rather than point-checked: it must stay under
    # 30, so the known inflation cannot silently get worse.
    @test 22 < subject_sd[effect=At(:Intercept)] < 30
    @test isapprox(subject_sd[effect=At(:Days)], 5.9, atol=1)
end

end # atleast(:benchmarks)

end # @testset "TuringRegressions"
finally
    if atleast(:benchmarks)
        println()
        println("="^78)
        println("BENCHMARK: posterior mean vs canonical (GLM MLE / lme4 REML)")
        println("samples=$BENCH_SAMPLES total, warmup=$BENCH_WARMUP total, nchains=$BENCH_NCHAINS")
        println("="^78)
        # crop off: the sleepstudy rows and the rel-err column are the first things the
        # default terminal-fitting drops, and they are the whole point of the table
        pretty_table(DataFrame(BENCHMARK_ROWS); column_labels=["model", "param", "ours", "canonical", "abs err", "rel err %"],
            fit_table_in_display_horizontally=false, fit_table_in_display_vertically=false)

        println()
        println("="^78)
        println("PREDICTIONS: worst-row epred error vs canonical, against asserted tolerance")
        println("="^78)
        pretty_table(DataFrame(PRED_ROWS); column_labels=["model", "max abs err", "tol", "headroom"],
            fit_table_in_display_horizontally=false, fit_table_in_display_vertically=false)

        println()
        println("="^78)
        println("FIT TIMINGS: wall-clock seconds per full-budget fit")
        println("First row includes TTFX compile — treat as upper bound")
        println("="^78)
        pretty_table(DataFrame(TIMING_ROWS); column_labels=["model", "seconds"])
    end
end # try
