
#Data structure to hold model information

struct Predictors
    has_intercept::Bool
    has_fixed_effects::Bool
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
has_fixed_effects(md::ModelData) = md.predictors.has_fixed_effects
has_random_effects(md::ModelData) = !isempty(md.Z)
is_weighted(md::ModelData) = !isnothing(md.weights)

#### Functions to extract information from the formula

## TODO drop for StatsModels.has_intercept? rework somehow
function has_intercept(formula) # allow implicit intercepts
    rhs = formula.rhs
    rhs = if rhs isa MatrixTerm
        rhs.terms
    elseif rhs isa Term
        [rhs]
    else
        rhs
    end
    for term in rhs
        term isa ConstantTerm || continue
        term.n == 0 && return false
        term.n == 1 && return true
        error("Intercept must be 0 or 1, got $(term.n)")
    end
    true  # implicit intercept when no ConstantTerm found
end

#TODO replace with known statsAPI stuff if we can?
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
function extract_model_data(formula, data, weights=nothing)
    # Apply schema - validates and types everything
    formula_has_intercept = has_intercept(formula)
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
        Z = RandomEffect[]
    else
        X = MixedModels.modelmatrix(MixedModel(formula, data))
        Z = [extract_random_effect(t, d) for t in re_terms]
        vars = [z.variable for z in Z]
        if length(unique(vars)) < length(vars)
            dupes = [v for v in unique(vars) if count(==(v), vars) > 1]
            @warn "Multiple random-effects terms share grouping variable(s) $dupes — their DimStack layers collide, later term overwrites earlier"
        end
    end

    #TODO check this intercept handling is it right?
    X = formula_has_intercept ? X[:, 2:end] : X
    X_names = get_fixef_names(formula, data)

    predictors = Predictors(formula_has_intercept, size(X, 2) > 0, X, X_names)
    return ModelData(f, y, predictors, Z, weights)
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
    term_has_intercept = StatsModels.hasintercept(term.lhs)

    # Get predictor matrix from LHS using modelcols
    X = let
        cols = modelcols(term.lhs, d)
        term_has_intercept ? cols[:, 2:end] : cols
    end
    X_names = let
        predictor_terms = filter(t -> !(t isa ConstantTerm || t isa InterceptTerm), term.lhs.terms)
        [string(t) for t in predictor_terms]
    end

    has_fixed_effects = size(X, 2) > 0

    predictors = Predictors(term_has_intercept, has_fixed_effects, X, X_names)

    return RandomEffect(variable, levels, level_index, predictors)
end
