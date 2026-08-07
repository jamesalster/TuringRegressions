
#### Back-transform of drawn params: standardised-scale DimStack -> original-scale
# DimStack. Pure rescale, no structural reshuffling — see reshape.jl.

"""
    coef_map(has_intercept::Bool, xt::ZScoreTransform, y_scale::Float64) -> AbstractMatrix

Affine map from a standardised-scale coef vector `[α?; β...]` (order matching `xt`'s
own predictor columns) to original scale. Each slope scales by `y_scale/sd_x`; when an
intercept is present it also absorbs the X-centring those slopes carry
(`-mean_x*y_scale/sd_x` cross term), so `A * coef_std` reproduces `α + Σβx` on raw X.
Shared by fixef and every ranef term — same shape, same algebra.
"""
function coef_map(has_intercept::Bool, xt::ZScoreTransform, y_scale::Float64)
    slope_scale = y_scale ./ xt.scale  # empty when no slopes
    has_intercept || return Diagonal(slope_scale)
    isempty(slope_scale) && return fill(y_scale, 1, 1)

    n = 1 + length(slope_scale)
    A = zeros(n, n)
    A[1, 1] = y_scale
    A[1, 2:end] .= .-xt.mean .* slope_scale
    for j in 2:n
        A[j, j] = slope_scale[j-1]
    end
    return A
end

# Back-transform the :fixef layer. Names/order read straight off `std` — the
# same layout `_fixef_layer` (reshape.jl) produced — so there is one source of truth
# for fixef ordering, not two.
function _unstandardise_fixef(std::DimArray, tf::Transform, family::Type{<:Distribution})
    names = collect(dims(std, :fixef))
    aux_roots = family_spec(family).aux_roots
    n_coef = length(names) - length(aux_roots)
    has_int = n_coef > 0 && names[1] == :α

    raw = Array(std) # (fixef, draw, chain)
    ndraw, nchain = size(raw, 2), size(raw, 3)

    A = coef_map(has_int, tf.fixef, tf.y_scale)
    coef_raw = similar(raw, n_coef, ndraw, nchain)
    for c in 1:nchain, d in 1:ndraw
        coef_raw[:, d, c] = A * raw[1:n_coef, d, c]
    end
    has_int && (coef_raw[1, :, :] .+= tf.y_mean) # population mean carried by α only

    # σ is a spread on y, so rescales with y; ν/ϕ are shape params, scale-free.
    aux_raw = copy(raw[n_coef+1:end, :, :])
    for (k, root) in enumerate(aux_roots)
        root === :σ && (aux_raw[k, :, :] .*= tf.y_scale)
    end

    return DimArray(vcat(coef_raw, aux_raw), (Dim{:fixef}(names), Dim{:iter}(1:ndraw), Dim{:chain}(1:nchain)))
end

# Back-transform one ranef term's layers. Returns a NamedTuple of layers to merge into
# the output DimStack, keyed the same as `_ranef_layers`. Point estimates and their
# spread (sd/corr) go through the SAME linear map `A`, so a group's reported SD always
# describes the same quantity (at raw X=0) as its reported point estimate. Ranef terms
# are mean-zero by construction — no intercept offset here, unlike fixef.
function _unstandardise_ranef(std_layers::DimStack, tf::Transform, ranef::RandomEffect, xt::ZScoreTransform, family::Type{<:Distribution})
    group = ranef.variable
    y_scale = family_spec(family).scales_y ? tf.y_scale : 1.0
    has_int = ranef.predictors.has_intercept

    effects_std = Array(std_layers[Symbol(group)])         # (effect,group,draw,chain)
    sds_std = Array(std_layers[Symbol(group, "_sd")])      # (effect,draw,chain)
    names = collect(dims(std_layers[Symbol(group)], :effect))
    levels = collect(dims(std_layers[Symbol(group)], :group))
    ndraw, nchain = size(effects_std, 3), size(effects_std, 4)
    draw_chain = (Dim{:iter}(1:ndraw), Dim{:chain}(1:nchain))

    A = coef_map(has_int, xt, y_scale)
    # One covariance transform per draw: Σ_orig = A·(D·R·D)·A'. SDs are its diagonal, so
    # they carry the same cross terms as the corr matrix — no diagonal-only shortcut here,
    # which would drop ρ and misreport the Intercept SD for correlated terms.
    corr_std = _is_correlated(ranef) ? Array(std_layers[Symbol(group, "_corr")]) : nothing

    effects_raw = similar(effects_std)
    sds_raw = similar(sds_std)
    corr_raw = isnothing(corr_std) ? nothing : similar(corr_std)
    for c in 1:nchain, d in 1:ndraw
        D = Diagonal(sds_std[:, d, c])
        Σ_raw = isnothing(corr_std) ? A * D * D * A' : A * D * corr_std[:, :, d, c] * D * A'
        sd = sqrt.(diag(Σ_raw))
        effects_raw[:, :, d, c] = A * effects_std[:, :, d, c]
        sds_raw[:, d, c] = sd
        isnothing(corr_raw) || (corr_raw[:, :, d, c] = Σ_raw ./ (sd * sd'))
    end

    base = (;
        Symbol(group) => DimArray(effects_raw, (Dim{:effect}(names), Dim{:group}(levels), draw_chain...)),
        Symbol(group, "_sd") => DimArray(sds_raw, (Dim{:effect}(names), draw_chain...)),
    )

    isnothing(corr_raw) ||
        return merge(base, (; Symbol(group, "_corr") => DimArray(corr_raw, (Dim{:effect}(names), Dim{:effect2}(names), draw_chain...))))
    return base
end

"""
    unstandardise(std_params::DimStack, tf::Transform, md::ModelData, family) -> DimStack

Back-transform `reshape_params`'s standardised-scale layers to the original data scale.
Pure rescale, no structural reshuffling — see `reshape_params`.
"""
function unstandardise(std_params::DimStack, tf::Transform, md::ModelData, family::Type{<:Distribution})
    layers = (; fixef=_unstandardise_fixef(std_params.fixef, tf, family))
    for (ranef, xt) in zip(md.Z, tf.ranef)
        layers = merge(layers, _unstandardise_ranef(std_params, tf, ranef, xt, family))
    end
    return DimStack(layers)
end
