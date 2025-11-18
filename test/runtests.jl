
using TuringRegressions
using Test
using RDatasets
using MCMCChains
using ParetoSmooth
using StatsModels
using StatsBase: mean, std
using Suppressor: @suppress
using Random
using GLM: GLM
using StatisticalMeasures
using CategoricalDistributions

@info "Setting up tests"
mtcars = dataset("datasets", "mtcars")

titanic_df = dataset("datasets", "Titanic")
# Expand the frequency table to individual cases
titanic_expanded = vcat([repeat(DataFrame(row[1:4]), row.Freq) for row in eachrow(titanic_df)]...)
titanic_expanded.Survived = titanic_expanded.Survived .== "Yes"

@warn "No tests yet implemented for plots"

@testset "Vs. GLM" begin
    mod1 = turing_glm(@formula(MPG ~ Cyl + Disp), mtcars, Normal);
    Random.seed!(123)
    fit!(mod1, N=15000)
    mod_glm = GLM.lm(@formula(MPG ~ Cyl + Disp), mtcars);
    @test isapprox(
        GLM.coef(mod_glm), coef(mod1, median; drop_warmup=2000), atol=0.025
    )
    @test isapprox(
        GLM.predict(mod_glm), predict(mod1, median; drop_warmup=2000, type=:epred), atol=0.1
    )

    mod2 = turing_glm(@formula(HP ~ Cyl + Disp), mtcars, Poisson);
    Random.seed!(123)
    fit!(mod2, N=15000);
    mod2_glm = GLM.glm(@formula(HP ~ Cyl + Disp), mtcars, Poisson(), GLM.LogLink());
    @test isapprox(
        GLM.coef(mod2_glm), parameters(mod2, median; drop_warmup=2000), atol=0.025
    )
    @test isapprox(
        GLM.predict(mod2_glm),
        predict(mod2, median; drop_warmup=2000, type=:epred),
        atol=1,
    )

    #mod3 = turing_glm(@formula(HP ~ Cyl + Disp), mtcars, NegativeBinomial; priors=prior);
    #Random.seed!(123)
    #fit!(mod3, N=15000);
    #mod3_glm = GLM.glm(@formula(HP ~ Cyl + Disp), mtcars, NegativeBinomial(), GLM.LogLink());
    #@test isapprox(
    #    GLM.coef(mod3_glm), parameters(mod3, median; drop_warmup=2000)[1:3], atol=0.025
    #)
    #@test isapprox(
    #    GLM.predict(mod3_glm), predict(mod3, median; drop_warmup=2000, type=:epred), atol=2
    #)

    Random.seed!(123)
    mod4 = turing_glm(@formula(Survived ~ Class + Sex + Age), titanic_expanded, Bernoulli);
    fit!(mod4, N=15000);
    mod4_glm = GLM.glm(@formula(Survived ~ Class + Sex + Age), titanic_expanded, Binomial(), GLM.LogitLink());
    GLM.coef(mod4_glm)
    @test isapprox(
        GLM.coef(mod4_glm), parameters(mod4, median; drop_warmup=2000), atol=0.05
    )
    @test isapprox(
        GLM.predict(mod4_glm),
        predict(mod4, median; drop_warmup=2000, type=:epred),
        atol=0.05,
    )
end

#@testset "Model Creation" begin
#    mod1 = @test_nowarn turing_glm(@formula(MPG ~ Cyl + Disp), mtcars, Normal);
#
#    predmat = Matrix(mtcars[:, [:Cyl, :Disp]])
#    y = mtcars.MPG
#
#    @test mod1.formula isa StatsModels.FormulaTerm
#    @test mod1.prior isa RegressionPrior
#    @test mod1.link == identity
#    @test mod1.y isa Vector
#    @test mod1.X isa Matrix
#    @test isnothing(mod1.z)
#    @test mod1.X_names == (:Cyl, :Disp)
#    @test mod1.Z_names == ()
#    @test isnothing(mod1.samples)
#
#    # CUstrom prior
#    prior = Regression(Normal(0, 2), Normal(0, 1), Exponential(1));
#    mod1b = turing_glm(@formula(MPG ~ Cyl + Disp), mtcars, Normal; priors=prior);
#    @test mod1b.prior == prior
#
#    # Test X y method
#    mod2 = turing_glm(y, predmat, Normal; names=[:Cyl, :Disp]);
#    @test mod2.formula isa StatsModels.FormulaTerm
#    @test mod2.y == mod1.y
#    @test mod2.X == mod1.X
#    @test mod2.X_names == mod1.X_names
#
#    mod2b = turing_glm(y, predmat, Normal)
#    @test mod2b.X_names == (:X1, :X2)
#    @test mod2b.X == mod1.X
#
#    # Model creation
#    mod3 = @test_throws ArgumentError turing_glm(
#        @formula(MPG ~ Cyl + (1|Disp)), mtcars, Normal
#    );
#
#    # Link
#    mod3 = turing_glm(@formula(MPG ~ Cyl + Disp), mtcars, TDist);
#    @test mod3.link == identity
#    mod4 = turing_glm(@formula(MPG ~ Cyl + Disp), mtcars, Bernoulli);
#    @test mod4.link == logit
#    mod5 = turing_glm(@formula(MPG ~ Cyl + Disp), mtcars, Poisson);
#    @test mod5.link == log
#    mod6 = turing_glm(@formula(MPG ~ Cyl + Disp), mtcars, NegativeBinomial);
#    @test mod6.link == log
#    # Test error (from TuringGLM) on wrong family
#    mod7 = @test_throws ArgumentError turing_glm(
#        @formula(MPG ~ Cyl + Disp), mtcars, Categorical
#    )
#
#    # Standardized
#    mod8 = @test_warn "standardi" turing_glm(
#        @formula(MPG ~ Cyl + Disp), mtcars, Normal; standardize=false
#    );
#    @test mod8.X == predmat
#    @test mod8.y == y
#    @test !mod8.standardized
#
#    # Standardization for count outcome
#    mod9 = turing_glm(@formula(HP ~ Cyl + Disp), mtcars, Poisson);
#    @test isapprox(mean(mod9.X; dims=1), zeros(1, size(mod9.X, 2)); atol=1e-12)
#    @test isapprox(std(mod9.X; dims=1), ones(1, size(mod9.X, 2)); atol=1e-12)
#    @test mod9.y == vec(mtcars.HP)
#    @test !mod8.standardized
#end
#
#@testset "Model Fit" begin
#    mod1 = turing_glm(@formula(MPG ~ Cyl + Disp), mtcars, Normal);
#
#    # Fit
#    mod1 = @test_nowarn fit!(mod1);
#    @test mod1.samples isa MCMCChains.Chains
#    @test mod1.unstd_params isa MCMCChains.Chains
#
#    # Default size
#    @test size(mod1.samples) == (2000, 16, 4)
#
#    # Kwargs to fit
#    @test_nowarn fit!(mod1; parallel=MCMCSerial(), N=4000, nchains=1);
#    @test size(mod1.samples) == (4000, 16, 1)
#
#    # Fit warnings
#    mod1 = fit!(mod1, N=10);
#    @test_warn "rhat" show(mod1);
#    @test_warn "ess" show(mod1);
#    @test_warn "MCSE" show(mod1);
#end
#
## Init global model for following sections
@info "Fitting models for tests"

model = turing_glm(@formula(MPG ~ Cyl + Disp), mtcars, TDist);
model_empty = deepcopy(model);
model = @suppress fit!(model);
mod_count = turing_glm(@formula(HP ~ Cyl + Disp), mtcars, Poisson);
fit!(mod_count);

@testset "Parameter Methods" begin
    default_samples = 2000
    default_chains = 4
    default_dropwarmup = 200
    expected_out = (default_samples - default_dropwarmup) * default_chains

    param_names = [ :α, :Cyl, :Disp, :σ, :ν]
    @test parameter_names(model) == param_names

    # Test the main get_parameters method
    pars = [:α, :σ, :ν]
    ps = get_parameters(model, pars)
    @test ps isa DimArray
    @test size(ps) == (length(pars), expected_out)
    @test Array(dims(ps, 1)) == string.(pars)
    expected_idx = vec([
        (i, j) for i in (default_dropwarmup + 1):default_samples, j in 1:default_chains
    ])
    @test Array(dims(ps, 2)) == expected_idx

    # Test kwargs
    @test size(get_parameters(model, pars; drop_warmup=0)) ==
        (length(pars), default_samples * default_chains)
    @test size(get_parameters(model, pars; drop_warmup=800)) ==
        (length(pars), (default_samples - 800) * default_chains)
    @test size(get_parameters(model, pars; n_draws=50)) == (length(pars), 50 * default_chains)
    @test size(get_parameters(model, pars; n_draws=50)) == (length(pars), 50 * default_chains)
    @test size(get_parameters(model, pars; n_draws=50)) == (length(pars), 50 * default_chains)
    @test size(get_parameters(model, pars; collapse=false)) ==
        ( length(pars), default_samples - default_dropwarmup, default_chains)
    @test_throws ErrorException get_parameters(model, pars; drop_warmup=2000, n_draws=5000)

    # Test derivative methods with funciton and kwargs
    pars = parameters(model)
    @test pars == get_parameters(model, [:α, Symbol("β[1]"), Symbol("β[2]"), :σ, :ν])
    @test isapprox(parameters(model, mean), mean(parameters(model), dims=2))
    @test ndims(parameters(model, median; dropdims=false)) == 2
    @test ndims(parameters(model, median; collapse=false, dropdims=false)) == 3

    # Test other methods more simply
    @test fixef(model) == get_parameters(model, [:α, Symbol("β[1]"), Symbol("β[2]")])
    @test isapprox(fixef(model, mean), mean(fixef(model), dims=2))
    @test ndims(fixef(model, median; dropdims=false)) == 2

    @test coef(model) == fixef(model, median)

    ints = internals(model)
    @test Array(dims(ints, 1)) == Symbol.([
        "lp",
        "n_steps",
        "is_accept",
        "acceptance_rate",
        "log_density",
        "hamiltonian_energy",
        "hamiltonian_energy_error",
        "max_hamiltonian_energy_error",
        "tree_depth",
        "numerical_error",
        "step_size",
        "nom_step_size",
    ])
    @test isapprox(internals(model, mean), mean(ints, dims=2))
    @test ndims(internals(model, median; dropdims=false)) == 2

    out = outcome(model)
    @test out isa DimArray
end

@testset "Prediction" begin
    # Basic prediction, with a count model
    predmat = Matrix(mtcars[5:9, [:Cyl, :Disp]])
    pred_data = (predmat .- mean(predmat; dims=1)) ./ std(predmat; dims=1)

    # Methods pass
    lp = @test_nowarn predict(mod_count, pred_data; type=:linpred)
    ep = @test_nowarn predict(mod_count, pred_data; type=:epred)
    pp = @test_nowarn predict(mod_count, pred_data; type=:posterior)

    # Relations: link
    @test isapprox(lp, log.(ep))
    @test var(pp) > var(ep) #higher variance
    # Relations: without link
    @test predict(model, pred_data; type=:linpred) == predict(model, pred_data; type=:epred)
    @test var(predict(model, pred_data; type=:posterior)) >
        var(predict(model, pred_data; type=:epred))
end


@testset "Display Methods" begin
    show_output = sprint(show, model)
    @test contains(show_output, "TuringRegression Model")
    @test contains(show_output, "TDist (link: identity)")
    @test contains(show_output, "MPG ~ Cyl + Disp")
    @test contains(show_output, "Prior")
    @test contains(show_output, "Fixed Effects")
    @test contains(show_output, "Intercept")
    @test contains(show_output, "Normal(μ=0.0, σ=2.0)")
    @test contains(show_output, "Normal(μ=0.0, σ=5.0)")
    @test contains(show_output, "Auxiliary")
    @test contains(show_output, "TDist")
    @test contains(show_output, "32") #Observations
    @test contains(show_output, "8000 samples")
    @test contains(show_output, "4 chains")
    # Empty
    @test contains(sprint(show, mod_empty), "empty")

    # Pretty version
    pretty_output = sprint(pretty, model)
    @test contains(pretty_output, show_output)
    @test contains(pretty_output, "Fixed Effects")
    @test all(
        contains.(
            Ref(pretty_output),
            ["mean", "std", "q2.5", "q97.5", "mcse", "ess_bulk", "ess_tail"],
        ),
    )
    @test contains(pretty_output, "Prediction Metrics")
    @test contains(pretty_output, "RSquared")
    @test all(contains.(Ref(pretty_output), ["Cyl", "Disp"]))
    coef = string.(round.(parent(fixef(model, mean; drop_warmup=0)); digits=2))
    @test all(contains.(Ref(pretty_output), coef))
end

@testset "Model Comparison" begin
    @test psis_loo(mod) isa ParetoSmooth.PsisLoo
    comp1 = loo_compare(model, model_count)
    @test comp1 isa ParetoSmooth.ModelComparison
    comp2 = loo_compare([model, model_count])
    @test sprint(show, comp1) == sprint(show, comp2)
    comp1b = loo_compare(model, model_count; model_names=("Mod1", "Mod2"))
    @test contains(sprint(show, comp1b), "Mod1")
end

@testset "Metrics" begin
    tab = @test_nowarn calculate_metrics(model, [rsq, rmse])
    @test tab isa DimArray
    @test all(dims(tab, 1) .== ["RSquared", "RootMeanSquaredError"])
    tab1b = @test_nowarn default_metrics(model)
    @test all(dims(tab1b, 1) .== ["RSquared", "RootMeanSquaredError", "MeanAbsoluteError"])

    # Categorical
    mtcars.binom = mtcars.MPG .> 20;
    mod4 = turing_glm(@formula(binom ~ Cyl + Disp), mtcars, Bernoulli);
    fit!(mod4);
    tab2 = @test_nowarn calculate_metrics(mod4, [accuracy, kappa])
    @test all(dims(tab2, 1) .== ["Accuracy", "Kappa"])
    tab2b = @test_nowarn default_metrics(mod4; collapse=false)
    @test all(
        dims(tab2b, 1) .==
        ["Accuracy", "Kappa", "TruePositiveRate", "TrueNegativeRate", "AreaUnderCurve", "Pseudo r2"],
    )

    out = @test_nowarn outcome_as_distribution(mod4)
    @test out isa UnivariateFinite
end
