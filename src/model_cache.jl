#### Cache for gensym'd/eval'd model functions (T13)

# construct_model gensyms+evals a fresh model type per call (see model.jl), needed
# because DynamicPPL dispatches on typeof(f) and shared names corrupt AD/sampler
# caches across structurally different models (R11). But identical specs fit
# repeatedly (e.g. same formula/family/priors/ranef shape) shouldn't pay that
# ~25s recompile every time (T13). Key on everything baked into the generated
# Expr as a literal (R12, R13): family, ModelData's derived structural bools,
# per-ranef shape, and the 4 prior distributions themselves (not runtime args to the model).
const MODEL_CACHE = Dict{Any,Tuple{Function,Expr}}()
const MODEL_CACHE_LOCK = ReentrantLock()

function _ranef_cache_key(ranef::RandomEffect)
    n_predictors = has_fixed_effects(ranef.predictors) ? size(ranef.predictors.X, 2) : 0
    return (ranef.variable, ranef.predictors.has_intercept, has_fixed_effects(ranef.predictors), n_predictors)
end

function _prior_cache_key(prior::RegressionPrior)
    dists = (prior.intercept, prior.fixed_effects, prior.random_effects, prior.auxiliary)
    return Tuple((typeof(d), Distributions.params(d)) for d in dists)
end

function _model_cache_key(family::Type{<:Distribution}, modeldata::ModelData, prior::RegressionPrior)
    structural_key = (has_intercept(modeldata), has_fixed_effects(modeldata), has_random_effects(modeldata), is_weighted(modeldata))
    ranef_key = Tuple(_ranef_cache_key.(modeldata.Z))
    return (family, structural_key, ranef_key, _prior_cache_key(prior))
end

# Memoized wrapper around construct_model: identical specs reuse the same
# eval'd model function (and its already-JIT'd AD/sampler methods); only
# genuinely new shapes pay gensym+eval.
function cached_construct_model(family::Type{<:Distribution}, modeldata::ModelData,
    prior::RegressionPrior, show_code::Bool=false)
    key = _model_cache_key(family, modeldata, prior)
    lock(MODEL_CACHE_LOCK) do
        get!(MODEL_CACHE, key) do
            construct_model(family, modeldata, prior, show_code)
        end
    end
end
