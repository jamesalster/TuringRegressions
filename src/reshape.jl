
#### Splitting the flat sampled-VarName DimArray into named layers
# `reshape.jl` only handles structure: flat raw array -> DimStack of standardised-scale layers
# `unstandardise.jl` handles the rescaling.

# The sampler emits one column per SCALAR: `β[1]`, `β[2]`... are separate `:param` entries.
# `getsym` strips a VarName back to its root (`β[2]` -> `:β`) so all of one variable's
# columns can be grabbed together, in axis order — that order is what `_unflatten` decodes.
_select_param(raw::DimArray, root::Symbol) = raw[param=findall(==(root), getsym.(dims(raw, :param)))]

# DimStack requires a dim NAME to have one length across all layers, so two ranef terms
# with different level counts (e.g. nested `a/b` grouping) can't both use a literal
# `:group` dim. Give each term's group axis a unique name instead; `draws(TR, type)`
# (parametermethods.jl) renames it back to `:group` on the single-layer array it returns.
_group_dim_name(group) = Symbol(:group_, group)

# Unflatten the trailing (raw) idx dim of `sub` (dims :iter, :chain, :idx) into `shape`,
# relying on FlexiChains preserving the sampled container's column-major element order.
# That ordering is an unguarded upstream contract: if it ever changed, a matrix param
# would come back transposed with no error raised, so check at least the count matches.
function _unflatten(sub::DimArray, shape::Tuple{Vararg{Int}})
    arr = permutedims(Array(sub), (3, 1, 2)) # (idx, draw, chain)
    n_sampled, n_expected = size(arr, 1), prod(shape)
    n_sampled == n_expected || throw(DimensionMismatch(
        "reshape_params: sampler returned $n_sampled scalars, expected $n_expected for " *
        "shape $shape. The FlexiChains parameter layout has changed.",
    ))
    return reshape(arr, shape..., size(arr, 2), size(arr, 3))
end

_scalar_row(raw::DimArray, root::Symbol) = _unflatten(_select_param(raw, root), (1,)) # (1,draw,chain)
_vector(raw::DimArray, root::Symbol, n::Int) = _unflatten(_select_param(raw, root), (n,)) # (n,draw,chain)

# Reassemble the LKJCholesky lower-triangular factor L (n,n,draw,chain) from its
# n*(n+1)/2 sampled entries — FlexiChains only samples the structural lower triangle,
# in column-major order (col 1 rows 1:n, col 2 rows 2:n, ...).
# Walked by hand rather than via `_unflatten`: a triangle is not a rectangular reshape.
# NB no permutedims here, so `sub` keeps the raw (draw,chain,idx) order, unlike everywhere else.
function _cholesky_L(raw::DimArray, root::Symbol, n::Int)
    sub = Array(_select_param(raw, root)) # (draw,chain,nidx)
    ndraw, nchain, nidx = size(sub)
    n_expected = n * (n + 1) ÷ 2
    nidx == n_expected || throw(DimensionMismatch(
        "reshape_params: LKJCholesky factor $root has $nidx sampled entries, expected " *
        "$n_expected for a $n×$n lower triangle. The FlexiChains parameter layout has changed.",
    ))
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
#
# Intercept-then-slopes order must match the row order the model builds its ranef
# matrix in (`_random_effects`, model.jl) — these are only labels, so a mismatch
# mislabels effects silently rather than erroring. Change one, change the other.
function _effect_names(re::RandomEffect)
    names = Symbol[]
    re.predictors.has_intercept && push!(names, :Intercept) # intercept first, if present
    has_fixed_effects(re.predictors) && append!(names, Symbol.(re.predictors.X_names)) # then each varying slope
    return names
end
# Both intercept AND slopes ⇒ they were given a joint prior, so an LKJ factor exists to read.
_is_correlated(re::RandomEffect) = re.predictors.has_intercept & has_fixed_effects(re.predictors)

# α, then β..., then aux — `unstandardise` indexes this layer positionally, so the order is necessary.
function _fixef_layer(raw::DimArray, md::ModelData, family::Type{<:Distribution})
    names = Symbol[]
    parts = Array{Float64,3}[]

    if has_intercept(md)
        push!(names, :α)                        # single intercept, always first
        push!(parts, _scalar_row(raw, :α))      # kept as a 1-row block so vcat works below
    end
    if has_fixed_effects(md)
        append!(names, Symbol.(md.predictors.X_names))              # one name per predictor column
        push!(parts, _vector(raw, :β, size(md.predictors.X, 2)))    # β was sampled as one vector
    end
    for root in family_spec(family).aux_roots
        push!(names, root)                      # σ / ν / ϕ, whichever this family samples
        push!(parts, _scalar_row(raw, root))    # each is a single scalar per draw
    end

    # Each part is already (nfixef, draw, chain)
    fixef = vcat(parts...) # stack the blocks onto one leading parameter axis
    ndraw, nchain = size(fixef, 2), size(fixef, 3)
    return DimArray(fixef, (Dim{:fixef}(names), Dim{:iter}(1:ndraw), Dim{:chain}(1:nchain)))
end

# One ranef term's layers, keyed on the sampled positional index `i` (model.jl's
# σ_z_<i>/L_z_<i>/r_z_<i>), returned relabelled under the real group variable name.
#
# The per-group effects are rebuilt here: the Turing model code assigns its ranef
# matrix with `=` rather than `~`, so only the σ/L/z pieces are ever sampled and stored.
function _ranef_layers(raw::DimArray, i::Int, ranef::RandomEffect)
    group = ranef.variable
    n_groups = length(ranef.levels) # e.g. number of distinct Subjects
    names = _effect_names(ranef)
    n = length(names)               # effects per group: 1 (intercept only), or 1+slopes
    σz_root, r_root, L_root = Symbol("σ_z_", i), Symbol("r_z_", i), Symbol("L_z_", i)

    σz = _vector(raw, σz_root, n)                              # (effect, draw, chain)
    r = _unflatten(_select_param(raw, r_root), (n, n_groups)) # (effect, group, draw, chain)
    ndraw, nchain = size(σz, 2), size(σz, 3)
    draw_chain = (Dim{:iter}(1:ndraw), Dim{:chain}(1:nchain))

    if _is_correlated(ranef)
        # Effects covary: reconstruct via the LKJ factor, M = diag(σz)·L·r per draw.
        # Same expression as the model's, untransposed — it wants rows=groups for
        # indexing, the output layer wants (effect, group). R = L·L' is the implied
        # correlation matrix
        L = _cholesky_L(raw, L_root, n)
        M = similar(r, n, n_groups, ndraw, nchain) # per-group effects, rebuilt below
        R = similar(L, n, n, ndraw, nchain)        # correlation matrix, one per draw
        for c in 1:nchain, d in 1:ndraw
            # Scale unit draws by the SDs, then let L tilt them into the correlated shape.
            M[:, :, d, c] = Diagonal(σz[:, d, c]) * L[:, :, d, c] * r[:, :, d, c]
            R[:, :, d, c] = L[:, :, d, c] * L[:, :, d, c]' # L·L' undoes the Cholesky split
        end
        return (;
            Symbol(group) => DimArray(M, (Dim{:effect}(names), Dim{_group_dim_name(group)}(ranef.levels), draw_chain...)),
            Symbol(group, "_sd") => DimArray(σz, (Dim{:effect}(names), draw_chain...)),
            Symbol(group, "_corr") => DimArray(R, (Dim{:effect}(names), Dim{:effect2}(names), draw_chain...)),
        )
    else
        # Effects independent (intercept-only or slope-only): just scale unit draws by σz.
        # The reshape inserts a length-1 group axis so σz broadcasts across every group.
        M = reshape(σz, n, 1, ndraw, nchain) .* r
        return (;
            Symbol(group) => DimArray(M, (Dim{:effect}(names), Dim{_group_dim_name(group)}(ranef.levels), draw_chain...)),
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