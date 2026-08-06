#### Cache for gensym'd/eval'd model functions

# DynamicPPL dispatches on typeof(f), so shared model names corrupt AD/sampler caches
# across structurally different models — each shape needs its own gensym. But refitting
# the same shape shouldn't re-pay the ~25s compile, so key on everything baked into the
# Expr as a literal: family, structural bools, per-ranef shape. Priors are runtime args,
# not literals — excluded from the key so different priors on one shape reuse one model.
const MODEL_CACHE = Dict{Any,Tuple{Function,Expr}}()
const MODEL_CACHE_LOCK = ReentrantLock()

function _ranef_cache_key(ranef::RandomEffect)
    # Generated symbols are positional (σ_z_<i>, model.jl), not the group's variable
    # name — two ranef terms with the same shape but different grouping-variable names
    # compile to identical code, so `variable` isn't part of the key.
    n_predictors = has_fixed_effects(ranef.predictors) ? size(ranef.predictors.X, 2) : 0
    return (ranef.predictors.has_intercept, has_fixed_effects(ranef.predictors), n_predictors)
end

function _model_cache_key(family::Type{<:Distribution}, modeldata::ModelData)
    structural_key = (has_intercept(modeldata), has_fixed_effects(modeldata), has_random_effects(modeldata), is_weighted(modeldata))
    ranef_key = Tuple(_ranef_cache_key.(modeldata.Z))
    return (family, structural_key, ranef_key)
end

# Memoized wrapper around construct_model: identical specs reuse the same
# eval'd model function (and its already-JIT'd AD/sampler methods); only
# genuinely new shapes pay gensym+eval.
function cached_construct_model(family::Type{<:Distribution}, modeldata::ModelData)
    key = _model_cache_key(family, modeldata)
    lock(MODEL_CACHE_LOCK) do
        get!(MODEL_CACHE, key) do
            construct_model(family, modeldata)
        end
    end
end
