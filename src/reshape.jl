
#### Splitting the flat sampled-VarName DimArray into named layers
# `reshape_params` is pure structure: flat raw array -> DimStack of standardised-scale
# layers. `unstandardise` (below) is pure rescale: standardised DimStack -> original-scale
# DimStack. Kept separate so a bug in one is easy to isolate from the other.

_select_param(raw::DimArray, root::Symbol) = raw[param=findall(==(root), getsym.(dims(raw, :param)))]

# Unflatten the trailing (raw) idx dim of `sub` (dims :iter, :chain, :idx) into `shape`,
# relying on FlexiChains preserving the sampled container's column-major element order.
function _unflatten(sub::DimArray, shape::Tuple{Vararg{Int}})
    arr = permutedims(Array(sub), (3, 1, 2)) # (idx, draw, chain)
    return reshape(arr, shape..., size(arr, 2), size(arr, 3))
end

_scalar_row(raw::DimArray, root::Symbol) = _unflatten(_select_param(raw, root), (1,)) # (1,draw,chain)
_vector(raw::DimArray, root::Symbol, n::Int) = _unflatten(_select_param(raw, root), (n,)) # (n,draw,chain)

# Reassemble the LKJCholesky lower-triangular factor L (n,n,draw,chain) from its
# n*(n+1)/2 sampled entries — FlexiChains only samples the structural lower triangle,
# in column-major order (col 1 rows 1:n, col 2 rows 2:n, ...).
function _cholesky_L(raw::DimArray, root::Symbol, n::Int)
    sub = Array(_select_param(raw, root)) # (draw,chain,nidx)
    ndraw, nchain, _ = size(sub)
    L = zeros(eltype(sub), n, n, ndraw, nchain)
    k = 1
    for j in 1:n, i in j:n
        L[i, j, :, :] .= sub[:, :, k]
        k += 1
    end
    return L
end

# Ranef effect layout, read straight off the RandomEffect. `correlated` ⇔ intercept
# and slopes are modelled jointly (an LKJ factor is sampled, a _corr layer emitted).
function _effect_names(re::RandomEffect)
    names = Symbol[]
    re.predictors.has_intercept && push!(names, :Intercept)
    has_fixed_effects(re.predictors) && append!(names, Symbol.(re.predictors.X_names))
    return names
end
_is_correlated(re::RandomEffect) = re.predictors.has_intercept & has_fixed_effects(re.predictors)

# Shared rescaling atoms. Coefficients live in arrays whose leading axis is `:effect`,
# so both work for fixef (effect,draw,chain) and ranef (effect,group,draw,chain).
#   _scale_effects — multiply each effect row by its own factor
#   _center        — contract the effect axis against per-effect X means (Σ mean·coef)
_scale_effects(coefs, v) = coefs .* reshape(v, length(v), ntuple(_ -> 1, ndims(coefs) - 1)...)
_center(mean_x, coefs) = dropdims(sum(_scale_effects(coefs, mean_x); dims=1); dims=1)

function _fixef_layer(raw::DimArray, md::ModelData, family::Type{<:Distribution})
    names = Symbol[]
    parts = Array{Float64,3}[]

    if has_intercept(md)
        push!(names, :α)
        push!(parts, _scalar_row(raw, :α))
    end
    if has_fixed_effects(md)
        append!(names, Symbol.(md.predictors.X_names))
        push!(parts, _vector(raw, :β, size(md.predictors.X, 2)))
    end
    for root in family_spec(family).aux_roots
        push!(names, root)
        push!(parts, _scalar_row(raw, root))
    end

    fixef = vcat(parts...) # (nfixef, draw, chain)
    ndraw, nchain = size(fixef, 2), size(fixef, 3)
    return DimArray(fixef, (Dim{:fixef}(names), Dim{:iter}(1:ndraw), Dim{:chain}(1:nchain)))
end

# One ranef term's layers, keyed on the sampled positional index `i` (model.jl's
# σ_z_<i>/L_z_<i>/r_z_<i>), returned relabelled under the real group variable name.
function _ranef_layers(raw::DimArray, i::Int, ranef::RandomEffect)
    group = ranef.variable
    n_groups = length(ranef.levels)
    names = _effect_names(ranef)
    n = length(names)
    σz_root, r_root, L_root = Symbol("σ_z_", i), Symbol("r_z_", i), Symbol("L_z_", i)

    σz = _vector(raw, σz_root, n)                              # (effect, draw, chain)
    r = _unflatten(_select_param(raw, r_root), (n, n_groups)) # (effect, group, draw, chain)
    ndraw, nchain = size(σz, 2), size(σz, 3)
    draw_chain = (Dim{:iter}(1:ndraw), Dim{:chain}(1:nchain))

    if _is_correlated(ranef)
        # Effects covary: reconstruct via the LKJ factor, M = diag(σz)·L·r per draw.
        L = _cholesky_L(raw, L_root, n)
        M = similar(r, n, n_groups, ndraw, nchain)
        R = similar(L, n, n, ndraw, nchain)
        for c in 1:nchain, d in 1:ndraw
            M[:, :, d, c] = Diagonal(σz[:, d, c]) * L[:, :, d, c] * r[:, :, d, c]
            R[:, :, d, c] = L[:, :, d, c] * L[:, :, d, c]'
        end
        return (;
            Symbol(group) => DimArray(M, (Dim{:effect}(names), Dim{:group}(ranef.levels), draw_chain...)),
            Symbol(group, "_sd") => DimArray(σz, (Dim{:effect}(names), draw_chain...)),
            Symbol(group, "_corr") => DimArray(R, (Dim{:effect}(names), Dim{:effect2}(names), draw_chain...)),
        )
    else
        # Effects independent (intercept-only or slope-only): just scale unit draws by σz.
        M = reshape(σz, n, 1, ndraw, nchain) .* r
        return (;
            Symbol(group) => DimArray(M, (Dim{:effect}(names), Dim{:group}(ranef.levels), draw_chain...)),
            Symbol(group, "_sd") => DimArray(σz, (Dim{:effect}(names), draw_chain...)),
        )
    end
end

"""
    reshape_params(raw::DimArray, md::ModelData, family) -> DimStack

Split the flat `VarName`-keyed sampler output (`DimArray(TR.samples)`) into named,
standardised-scale layers: `:fixef` (α, β, aux, in that order), and per ranef term a
`:{group}` matrix (`:effect` × `:group`) plus its `:{group}_sd` / `:{group}_corr`
companions. Pure structural reshuffle — no rescaling, see `unstandardise`.
"""
function reshape_params(raw::DimArray, md::ModelData, family::Type{<:Distribution})
    layers = (; fixef=_fixef_layer(raw, md, family))
    for (i, ranef) in enumerate(md.Z)
        layers = merge(layers, _ranef_layers(raw, i, ranef))
    end
    return DimStack(layers)
end
