
#Data structure to hold the random effects part of a model
struct RandomEffect
    variable::Symbol           # grouping variable (e.g., :subject)
    levels::Vector            # unique levels
    level_index::Vector{Int} # data coded as integer index into levels
    predictors::Matrix{Float64} #predictor matrix
    predictor_names::Vector{String} #names of predictors
    has_intercept::Bool       # random intercept?
    has_fixed_effects::Bool # random fixed effects?
end

#### Functions to extract information from the formula 

function has_intercept(formula) # allow implicit intercepts
    rhs = formula.rhs isa Term ? [formula.rhs] : formula.rhs
    for term in rhs
        term isa ConstantTerm || continue
        term.n == 0 && return false
        term.n == 1 && return true
        error("Intercept must be 0 or 1, got $(term.n)")
    end
    true  # implicit intercept when no ConstantTerm found
end

function get_fixef_names(formula, data)
    coefs = coefnames(ModelFrame(formula, data))
    filter!(x -> !occursin(" | ", x), coefs) # Drop random effects TODO do this properly
    if has_intercept(formula)
        return coefs[2:end]
    else
        return coefs
    end
end

# Get model data out, y, X and Z. Thanks to claude for a bit of help
function extract_model_data(formula, data)
    # Apply schema - validates and types everything
    f = apply_schema(formula, schema(formula, data), MixedModel)
    d = columntable(data)

    # Extract y
    y = modelcols(f.lhs, d)

    # Separate fixed and random terms
    all_terms = f.rhs isa Tuple ? collect(f.rhs) : [f.rhs]
    is_re(t) = t isa RandomEffectsTerm
    re_terms = filter(is_re, all_terms)

    if isempty(re_terms)
        X = modelcols(f.rhs, d)
        X = has_intercept(formula) ? X[:, 2:end] : X
        Z = nothing
    else
        X = MixedModels.modelmatrix(MixedModel(formula, data))
        X = X[:, 2:end]
        Z = [extract_random_effect(t, d) for t in re_terms]
    end

    return (y=y, X=X, Z=Z, formula=f)
end

# Random Effect Handling Functions
function _ranef_predictors(term_lhs, d, has_intercept::Bool)
    cols = modelcols(term_lhs, d)
    return has_intercept ? cols[:, 2:end] : cols
end
function _ranef_predictor_names(term_lhs, has_intercept::Bool)
    names = [string(t) for t in term_lhs.terms]
    return has_intercept ? names[2:end] : names
end

# Get random effect datastructure from formula
function extract_random_effect(term::RandomEffectsTerm, d::NamedTuple)
    # Use their _ranef_refs function to get grouping info
    refs, levels = _ranef_refs(term.rhs, d)

    # Convert refs to Vector{Int}
    level_index = Vector{Int}(refs)

    # Get variable name - handle simple and interaction terms
    variable = if term.rhs isa CategoricalTerm
        term.rhs.sym
    else  # InteractionTerm like (item:subject)
        Symbol(join([t.sym for t in term.rhs.terms], ":"))
    end

    # Check for intercept in column names
    has_intercept = StatsModels.hasintercept(term.lhs)

    # Get predictor matrix from LHS using modelcols
    predictors = _ranef_predictors(term.lhs, d, has_intercept)
    predictor_names = _ranef_predictor_names(term.lhs, has_intercept)

    has_fixed_effects = size(predictors, 2) > 0

    return RandomEffect(variable, levels, level_index, predictors, predictor_names, has_intercept, has_fixed_effects)
end

