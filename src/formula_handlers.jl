
#Data structure to hold model information

struct Predictors
    has_intercept::Bool
    X::AbstractMatrix
    X_names::Union{Nothing,Vector{String}} #TODO would love to cut this
end
struct RandomEffect
    variable::Symbol           # grouping variable (e.g., :subject)
    levels::Vector            # unique levels
    level_index::Vector{Int} # data coded as integer index into levels
    predictors::Predictors
end

struct ModelData
    f::FormulaTerm
    y::AbstractVector
    predictors::Predictors
    Z::Vector{RandomEffect}
    weights::Union{Nothing,Vector{Float64}}
end

# Derived flags — replace the old (deleted) ModelInfo struct. Multiple-dispatch
# accessors so callers don't need to know whether they hold a ModelData or a TR.
has_intercept(md::ModelData) = md.predictors.has_intercept
has_fixed_effects(p::Predictors) = size(p.X, 2) > 0
has_fixed_effects(re::RandomEffect) = has_fixed_effects(re.predictors)
has_fixed_effects(md::ModelData) = has_fixed_effects(md.predictors)
has_random_effects(md::ModelData) = !isempty(md.Z)
is_weighted(md::ModelData) = !isnothing(md.weights)

#### Functions to extract information from the formula

# Single reusable extractor: after `apply_schema(...; MixedModel)`, the fixed-effect
# part of the RHS is always exactly one MatrixTerm — whether or not any random-effects
# terms are present — and each RandomEffectsTerm's `.lhs` is the same kind of MatrixTerm.
# So this one function builds a Predictors for both fixef and every ranef term.
function extract_predictors(term::MatrixTerm, d::NamedTuple)
    term_has_intercept = StatsModels.hasintercept(term)
    cols = modelcols(term, d)
    X = term_has_intercept ? cols[:, 2:end] : cols
    X_names = term_has_intercept ? coefnames(term)[2:end] : coefnames(term)
    return Predictors(term_has_intercept, X, X_names)
end

# Get model data out, y, X and Z. Thanks to claude for a bit of help
function extract_model_data(formula, data, weights=nothing)
    # Apply schema - validates and types everything
    f = apply_schema(formula, schema(formula, data), MixedModel)
    d = columntable(data)

    # Extract y
    y = modelcols(f.lhs, d)

    # Separate fixed and random terms. The fixed part is always a single MatrixTerm
    # (possibly intercept-only / zero predictors), whether or not ranef terms exist.
    all_terms = f.rhs isa Tuple ? collect(f.rhs) : [f.rhs]
    is_re(t) = t isa RandomEffectsTerm
    fixed_term = only(filter(!is_re, all_terms))
    re_terms = filter(is_re, all_terms)

    predictors = extract_predictors(fixed_term, d)
    Z = [extract_random_effect(t, d) for t in re_terms]

    vars = [z.variable for z in Z]
    if length(unique(vars)) < length(vars)
        dupes = [v for v in unique(vars) if count(==(v), vars) > 1]
        @warn "Multiple random-effects terms share grouping variable(s) $dupes — their DimStack layers collide, later term overwrites earlier"
    end

    return ModelData(f, y, predictors, Z, weights)
end

# Grouping-variable values for a random-effects term's RHS, read straight off the raw
# data — handles a single grouping variable (`(1|g)`) and an interaction of several
# (`(1|item:subject)`). Deliberately NOT MixedModels._ranef_refs: that function looks
# values up in the term's fitted contrasts dict and KeyErrors on any level unseen at
# fit time — which breaks posterior_predict on new data with a new grouping level
# before `_remap_levels`'s allow_new_levels handling ever gets a chance to run.
_ranef_group_values(rhs::CategoricalTerm, d::NamedTuple) = string.(d[rhs.sym])
function _ranef_group_values(rhs::InteractionTerm, d::NamedTuple)
    columns = [string.(d[t.sym]) for t in rhs.terms]
    return [join(row, ":") for row in zip(columns...)]
end

# Get random effect datastructure from formula
function extract_random_effect(term::RandomEffectsTerm, d::NamedTuple)
    # Get variable name - handle simple and interaction terms
    variable = if term.rhs isa CategoricalTerm
        term.rhs.sym
    else  # InteractionTerm like (item:subject)
        Symbol(join([t.sym for t in term.rhs.terms], ":"))
    end

    group_values = _ranef_group_values(term.rhs, d)
    levels = sort(unique(group_values))
    level_index = [findfirst(==(v), levels) for v in group_values]

    predictors = extract_predictors(term.lhs, d)

    return RandomEffect(variable, levels, level_index, predictors)
end

"""
    new_random_effects(reference::ModelData, new_data; allow_new_levels=false)

Rebuild ranef structure for `new_data` via `extract_model_data` (same path used at fit
time), then remap each grouping level onto `reference`'s (the fitted model's) original
level order/index.

By default errors clearly if `new_data` contains a grouping level not seen during
fitting. With `allow_new_levels=true`, unseen levels get a `@warn` and are marked (level
index `0`) so prediction uses the population-mean (zero) random effect for those rows,
instead of erroring.
"""
function new_random_effects(reference::ModelData, new_data; allow_new_levels::Bool=false)
    has_random_effects(reference) || return nothing
    md_new = extract_model_data(reference.f, new_data)
    return [
        _remap_levels(re_new, re_orig; allow_new_levels) for
        (re_new, re_orig) in zip(md_new.Z, reference.Z)
    ]
end

# Sentinel level index 0 (never a valid 1-based level) marks an unseen level.
function _remap_levels(re_new::RandomEffect, re_orig::RandomEffect; allow_new_levels::Bool=false)
    unseen = Any[]
    level_index = map(re_new.level_index) do i
        key = re_new.levels[i]
        idx = findfirst(==(key), re_orig.levels)
        !isnothing(idx) && return idx
        if !allow_new_levels
            error(
                "predict: unseen level '$key' for grouping variable :$(re_orig.variable) — " *
                "all grouping levels must have been present when the model was fitted. " *
                "Pass allow_new_levels=true to use population-mean random effects for new levels.",
            )
        end
        push!(unseen, key)
        return 0
    end
    if !isempty(unseen)
        @warn "predict: unseen level(s) for grouping variable :$(re_orig.variable); using population-mean (zero) random effect for these rows." levels =
            unique(unseen)
    end
    return RandomEffect(re_orig.variable, re_orig.levels, level_index, re_new.predictors)
end
