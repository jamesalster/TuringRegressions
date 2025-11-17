
#### Functions to extract information from the formula 

function data_response(formula::FormulaTerm, data::D) where {D}
    return response(formula, data)
end

# From TuringGLM
function data_fixed_effects(formula::FormulaTerm, data::D) where {D}
    if has_ranef(formula)
        X = MixedModels.modelmatrix(MixedModel(formula, data))
        X = X[:, 2:end]
    else
        X = StatsModels.modelmatrix(formula, data)
        if hasintercept(formula)
            X = X[:, 2:end]
        end
    end
    return X
end

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
function has_ranef(formula)
    if formula.rhs isa StatsModels.Term
        return false
    else
        return any(t -> t isa FunctionTerm{typeof(|)}, formula.rhs)
    end
end

function get_fixef_names(formula, data)
    coefs = coefnames(ModelFrame(formula, data))
    if has_intercept(formula)
        return coefs[2:end]
    else
        return coefs
    end
end


