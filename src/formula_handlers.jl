
#Data structure to hold the random effects part of a model
struct RandomEffect
    variable::Symbol           # grouping variable (e.g., :subject)
    levels::Vector            # unique levels
    level_index::Vector{Int} # data coded as integer index into levels
    predictors::Matrix{Float64} #predictor matrix
    has_intercept::Bool       # random intercept?
    has_fixed_effects::Bool
end

#### Functions to extract information from the formula 

#function data_response(formula::FormulaTerm, data::D) where {D}
#    return response(formula, data)
#end

# From TuringGLM
#function data_fixed_effects(formula::FormulaTerm, data::D) where {D}
#    if has_ranef(formula)
#        X = MixedModels.modelmatrix(MixedModel(formula, data))
#        X = X[:, 2:end]
#    else
#        X = StatsModels.modelmatrix(formula, data)
#        if hasintercept(formula)
#            X = X[:, 2:end]
#        end
#    end
#    return X
#end

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

# From TuringGLM
#function has_ranef(formula)
#    if formula.rhs isa StatsModels.Term
#        return false
#    else
#        return any(t -> t isa FunctionTerm{typeof(|)}, formula.rhs)
#    end
#end

function get_fixef_names(formula, data)
    coefs = coefnames(ModelFrame(formula, data))
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
    d = Tables.columntable(data)
    
    # Extract y
    y = modelcols(f.lhs, d)
    
    # Separate fixed and random terms
    all_terms = f.rhs isa Tuple ? collect(f.rhs) : [f.rhs]
    is_re(t) = t isa RandomEffectsTerm 
    
    # Build X from fixed effects only
    X = MixedModels.modelmatrix(MixedModel(formula, data))
    X = X[:, 2:end]
    
    # Manually extract each random effect
    re_terms = filter(is_re, all_terms)
    if isempty(re_terms)
        Z = nothing
    else
        Z = [extract_random_effect(t, d) for t in re_terms]
    end
    
    return (y=y, X=X, Z=Z)
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
    
    # Get predictor matrix from LHS using modelcols
    predictors = modelcols(term.lhs, d)[:, 2:end]
    println(typeof(predictors))

    has_fixed_effects = size(predictors, 2) > 0
    
    # Check for intercept in column names
    has_intercept = StatsModels.hasintercept(term.lhs)
    
    return RandomEffect(variable, levels, level_index, predictors, has_intercept, has_fixed_effects)
end