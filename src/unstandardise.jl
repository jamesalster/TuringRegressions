
#### Back-transform of drawn params: standardised-scale DimStack -> original-scale
# DimStack. Structural reshuffling is handled in reshape.jl.

"""
    coef_map(has_intercept::Bool, xt::ZScoreTransform, y_scale::Float64) -> AbstractMatrix

Matrix `A` taking a standardised-scale coefficient vector `[α?; β...]` to the original
scale, so `A * coef_std` gives `[α; β...]` describing raw y against raw X. Column order
matches `xt`'s own predictor columns. Shared by fixef and every ranef term.

Substituting `x_std = (x - mean_x)/sd_x` and `y_std = (y - mean_y)/sd_y` into
`y_std = α_s + Σ β_s·x_std` and collecting terms gives

    β_j = β_s,j · y_scale/sd_x_j                        # each slope scales independently
    α   = y_scale·α_s - Σ_j mean_x_j·(y_scale/sd_x_j)·β_s,j  + mean_y

which is exactly this matrix's rows: the diagonal holds the slope factors, and the
first row additionally carries the `-mean_x·slope_scale` cross terms by which the
intercept absorbs the X-centring the slopes were fitted against.

The trailing `+ mean_y` is deliberately NOT in `A`: random effects share this map and
are mean-zero deviations, so only fixef adds it (see `_unstandardise_fixef`).
"""
function coef_map(has_intercept::Bool, xt::ZScoreTransform, y_scale::Float64)
    slope_scale = y_scale ./ xt.scale  # empty when no slopes
    # No intercept ⇒ `compute_transform` never centred X (standardise.jl), so `xt.mean`
    # is empty and there is no shift for anyone to absorb — pure diagonal rescale.
    has_intercept || return Diagonal(slope_scale)
    isempty(slope_scale) && return fill(y_scale, 1, 1) # intercept alone: just rescale it

    n = 1 + length(slope_scale)
    A = zeros(n, n)
    A[1, 1] = y_scale                            # α itself stretches with y
    A[1, 2:end] .= .-xt.mean .* slope_scale      # ...minus what each slope shifted it by
    for j in 2:n
        A[j, j] = slope_scale[j-1]               # every slope scales on its own, no mixing
    end
    return A
end

# Back-transform the :fixef layer. Names/order read straight off `std` — the
# same layout `_fixef_layer` (reshape.jl) produced — so there is one source of truth
# for fixef ordering, not two.
function _unstandardise_fixef(std::DimArray, tf::Transform, family::Type{<:Distribution})
    names = collect(dims(std, :fixef))
    aux_roots = family_spec(family).aux_roots
    # Two unenforced contracts with `_fixef_layer` (reshape.jl), both silent if broken:
    # aux params sit LAST (so the count splits coefs from aux), and the intercept is
    # labelled `:α` (structural truth lives on ModelData, which isn't passed here).
    n_coef = length(names) - length(aux_roots) # everything before the aux block is α/β
    has_int = n_coef > 0 && names[1] == :α

    raw = Array(std) # (fixef, draw, chain)
    ndraw, nchain = size(raw, 2), size(raw, 3)

    A = coef_map(has_int, tf.fixef, tf.y_scale) # same map for every draw, so build it once
    coef_raw = similar(raw, n_coef, ndraw, nchain)
    for c in 1:nchain, d in 1:ndraw
        coef_raw[:, d, c] = A * raw[1:n_coef, d, c] # one matrix-vector product per draw
    end
    # The one piece of the back-transform `A` deliberately omits, so ranef can share it:
    # α is the only mean-carrying param, ranef deviations are mean-zero.
    has_int && (coef_raw[1, :, :] .+= tf.y_mean)

    # σ is a spread on y, so rescales with y; ν/ϕ are shape params, scale-free.
    aux_raw = raw[n_coef+1:end, :, :] # range-indexing an Array already copies
    for (k, root) in enumerate(aux_roots)
        root === :σ && (aux_raw[k, :, :] .*= tf.y_scale) # ν/ϕ fall through untouched
    end

    return DimArray(vcat(coef_raw, aux_raw), (Dim{:fixef}(names), Dim{:iter}(1:ndraw), Dim{:chain}(1:nchain)))
end

# Back-transform one ranef term's layers. Returns a NamedTuple of layers to merge into
# the output DimStack, keyed the same as `_ranef_layers`. Point estimates and their
# spread (sd/corr) go through the SAME linear map `A`, so a group's reported SD always
# describes the same quantity (at raw X=0) as its reported point estimate. Ranef terms
# are mean-zero by construction — no intercept offset here, unlike fixef.
function _unstandardise_ranef(std_layers::DimStack, tf::Transform, ranef::RandomEffect, xt::ZScoreTransform, key::Symbol)
    has_int = ranef.predictors.has_intercept

    effects_std = Array(std_layers[key])                 # (effect,group,draw,chain)
    sds_std = Array(std_layers[Symbol(key, "_sd")])      # (effect,draw,chain)
    names = collect(dims(std_layers[key], _effect_dim_name(key)))
    levels = collect(dims(std_layers[key], _group_dim_name(key)))
    ndraw, nchain = size(effects_std, 3), size(effects_std, 4)
    draw_chain = (Dim{:iter}(1:ndraw), Dim{:chain}(1:nchain))

    # `tf.y_scale` is already 1.0 for families that don't scale y — no branch needed here.
    A = coef_map(has_int, xt, tf.y_scale)
    # One covariance transform per draw: Σ_orig = A·(D·R·D)·A'. SDs are its diagonal, so
    # they carry the same cross terms as the corr matrix — no diagonal-only shortcut here,
    # which would drop ρ and misreport the Intercept SD for correlated terms.
    corr_std = _is_correlated(ranef) ? Array(std_layers[Symbol(key, "_corr")]) : nothing

    effects_raw = similar(effects_std)
    sds_raw = similar(sds_std)   # recomputed from Σ below, not scaled directly
    corr_raw = isnothing(corr_std) ? nothing : similar(corr_std)
    # A is provably diagonal in the uncorrelated cases (intercept-only ⇒ 1×1, slopes-only
    # ⇒ Diagonal), so the full products are redundant there — kept anyway for one path.
    for c in 1:nchain, d in 1:ndraw
        D = Diagonal(sds_std[:, d, c])           # SDs on the diagonal, so D·R·D is a covariance
        # Transforming a covariance needs A on both sides — that's what mixes the slopes'
        # spread into the intercept's, exactly as A's first row mixes their point estimates.
        Σ_raw = isnothing(corr_std) ? A * D * D * A' : A * D * corr_std[:, :, d, c] * D * A'
        sd = sqrt.(diag(Σ_raw))                  # variances live on the diagonal; SD = √variance
        effects_raw[:, :, d, c] = A * effects_std[:, :, d, c] # no y_mean: mean-zero, unlike α
        sds_raw[:, d, c] = sd
        isnothing(corr_raw) || (corr_raw[:, :, d, c] = Σ_raw ./ (sd * sd')) # renormalise Σ back to a corr matrix
    end

    base = (;
        key => DimArray(effects_raw, (Dim{_effect_dim_name(key)}(names), Dim{_group_dim_name(key)}(levels), draw_chain...)),
        Symbol(key, "_sd") => DimArray(sds_raw, (Dim{_effect_dim_name(key)}(names), draw_chain...)),
    )

    isnothing(corr_raw) || return merge(base, (;
        Symbol(key, "_corr") => DimArray(
            corr_raw, (Dim{_effect_dim_name(key)}(names), Dim{_effect2_dim_name(key)}(names), draw_chain...)
        ),
    ))
    return base
end

"""
    unstandardise(std_params::DimStack, tf::Transform, md::ModelData, family) -> DimStack

Back-transform `reshape_params`'s standardised-scale layers to the original data scale.
Pure rescale, no structural reshuffling — see `reshape_params`.
"""
function unstandardise(std_params::DimStack, tf::Transform, md::ModelData, family::Type{<:Distribution})
    # `family` is only needed for the aux params, which are fixef-only.
    layers = (; fixef=_unstandardise_fixef(std_params.fixef, tf, family))
    # `tf.ranef` and the layer keys are both index-aligned with `md.Z` by construction
    # (compute_transform / ranef_layer_keys).
    for (ranef, xt, key) in zip(md.Z, tf.ranef, ranef_layer_keys(md.Z))
        layers = merge(layers, _unstandardise_ranef(std_params, tf, ranef, xt, key))
    end
    return DimStack(layers)
end
