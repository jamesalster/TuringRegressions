
#### Affine standardisation, computed once outside the model
# StatsBase.ZScoreTransform (fit/transform/reconstruct) does the per-column mean/std
# bookkeeping for us, and guards zero-variance columns (scale→1.0, not division by
# zero) for free — no need to hand-roll it.

struct Transform
    fixef::ZScoreTransform
    y::ZScoreTransform              # center=scale=false when family doesn't scale y
    ranef::Vector{ZScoreTransform}  # aligned with ModelData.Z
end

# Single source of truth for family-specific behaviour, consumed by every pass.
#   scales_y  — is y standardised? (Gaussian-like families only)
#   aux_roots — extra sampled scalars (dispersion / shape), in output order
struct FamilySpec
    scales_y::Bool
    aux_roots::Tuple{Vararg{Symbol}}
end
family_spec(::Type{Normal})            = FamilySpec(true, (:σ,))
family_spec(::Type{TDist})             = FamilySpec(true, (:σ, :ν))
family_spec(::Type{NegativeBinomial})  = FamilySpec(false, (:ϕ,))
family_spec(::Type{<:Distribution})    = FamilySpec(false, ())  # Bernoulli, Poisson

"""
    compute_transform(md::ModelData, family) -> Transform

Compute standardisation constants from raw `ModelData`. Pure — does not touch `md`.
"""
function compute_transform(md::ModelData, family::Type{<:Distribution})
    fixef = fit(ZScoreTransform, md.predictors.X, dims=1)
    ranef = [fit(ZScoreTransform, re.predictors.X, dims=1) for re in md.Z]
    scale_y = family_spec(family).scales_y
    y = scale_y ? fit(ZScoreTransform, md.y) : fit(ZScoreTransform, md.y; center=false, scale=false)
    return Transform(fixef, y, ranef)
end

# fn is StatsBase.transform or StatsBase.reconstruct — same traversal either direction.
function apply_transform(fn::Function, tf::Transform, md::ModelData)
    predictors = Predictors(md.predictors.has_intercept, fn(tf.fixef, md.predictors.X), md.predictors.X_names)
    Z = [
        RandomEffect(re.variable, re.levels, re.level_index,
            Predictors(re.predictors.has_intercept, fn(lt, re.predictors.X), re.predictors.X_names))
        for (re, lt) in zip(md.Z, tf.ranef)
    ]
    return ModelData(md.f, fn(tf.y, md.y), predictors, Z, md.weights)
end

"""
    apply_transform(tf::Transform, md::ModelData) -> ModelData

Apply a `Transform`'s constants to raw `ModelData`, returning a scaled `ModelData`.
"""
apply_transform(tf::Transform, md::ModelData) = apply_transform(transform, tf, md)

"""
    unstandardise_data(md_std::ModelData, tf::Transform) -> ModelData

Inverse of `apply_transform`, at the data level (not posterior draws). Used by the
round-trip test guarding the standardisation constants.
"""
unstandardise_data(md_std::ModelData, tf::Transform) = apply_transform(StatsBase.reconstruct, tf, md_std)

"""
    standardise(md::ModelData, family) -> (md_std::ModelData, tf::Transform)

`compute_transform` + `apply_transform` in one call.
"""
function standardise(md::ModelData, family::Type{<:Distribution})
    tf = compute_transform(md, family)
    return apply_transform(tf, md), tf
end


function _unstandardise_fixef(std::DimArray, tf::Transform, md::ModelData, family::Type{<:Distribution})
    spec = family_spec(family)
    y_scale, y_mean = spec.scales_y ? (tf.y.scale[1], tf.y.mean[1]) : (1.0, 0.0)
    ndraw, nchain = size(std, 2), size(std, 3)
    y = y_scale  # y's scale if standardised, else identity

    # Slopes: divide out each predictor's sd, times y's scale.
    β_orig = if has_fixed_effects(md)
        β = Array(std[fixef=At(Symbol.(md.predictors.X_names))]) # (npred,draw,chain)
        _scale_effects(β, y ./ tf.fixef.scale)
    else
        zeros(0, ndraw, nchain)
    end

    names = Symbol[]
    parts = Array{Float64,3}[]
    if has_intercept(md)
        α = Array(std[fixef=At([:α])])[1, :, :]         # (draw,chain)
        spec.scales_y && (α = y_mean .+ y_scale .* α)   # undo y centring/scaling
        has_fixed_effects(md) && (α = α .- _center(tf.fixef.mean, β_orig))  # undo X centring
        push!(names, :α)
        push!(parts, reshape(α, 1, ndraw, nchain))
    end
    if has_fixed_effects(md)
        append!(names, Symbol.(md.predictors.X_names))
        push!(parts, β_orig)
    end
    # σ is a spread on y, so rescales with y; ν/ϕ are shape params, scale-free.
    for root in spec.aux_roots
        v = Array(std[fixef=At([root])])
        push!(names, root)
        push!(parts, root === :σ ? y_scale .* v : v)
    end

    return DimArray(vcat(parts...), (Dim{:fixef}(names), Dim{:iter}(1:ndraw), Dim{:chain}(1:nchain)))
end

# Linear map from std-scale ranef-effect vector to original-scale. Row 1 (Intercept,
# only when both intercept and slopes present) also recenters from std-scale (mean-X)
# back to raw X=0 — `-mean_x_j*(y/sd_x_j)` per slope column — since lme4's gold standard
# fits raw X directly and carries no such shift. Slope rows are plain elementwise
# `y/sd_x` scaling, no cross terms.
function _ranef_transform_matrix(names, mean_x, sd_x, y, has_int::Bool)
    n = length(names)
    if has_int && n > 1
        A = zeros(n, n)
        A[1, 1] = y
        for j in 2:n
            A[1, j] = -mean_x[j-1] * y / sd_x[j-1]
            A[j, j] = y / sd_x[j-1]
        end
        return A
    end
    return Diagonal(has_int ? fill(y, n) : y ./ sd_x)
end

# Back-transform one ranef term's layers. Returns a NamedTuple of layers to merge into
# the output DimStack, keyed the same as `_ranef_layers`. Point estimates (`M`) and their
# spread (`sd`/`corr`) go through the SAME linear map `A`, so a group's reported SD
# always describes the same quantity (at raw X=0) as its reported point estimate.
function _unstandardise_ranef(std_layers::DimStack, tf::Transform, i::Int, ranef::RandomEffect, family::Type{<:Distribution})
    group = ranef.variable
    y = family_spec(family).scales_y ? tf.y.scale[1] : 1.0  # y's scale if standardised
    mean_x, sd_x = tf.ranef[i].mean, tf.ranef[i].scale       # per slope effect, X order
    has_int = ranef.predictors.has_intercept
    has_slopes = !isempty(mean_x)

    M = Array(std_layers[Symbol(group)])          # (effect,group,draw,chain)
    σz = Array(std_layers[Symbol(group, "_sd")])  # (effect,draw,chain)
    names = collect(dims(std_layers[Symbol(group)], :effect))
    levels = collect(dims(std_layers[Symbol(group)], :group))
    n = length(names)
    ndraw, nchain = size(M, 3), size(M, 4)
    draw_chain = (Dim{:iter}(1:ndraw), Dim{:chain}(1:nchain))

    A = _ranef_transform_matrix(names, mean_x, sd_x, y, has_int)
    # One covariance transform per draw: Σ_orig = A·(D·R·D)·A'. SDs are its diagonal, so
    # they carry the same cross terms as the corr matrix — no diagonal-only shortcut here,
    # which would drop ρ and misreport the Intercept SD for correlated terms.
    R_std = _is_correlated(ranef) ? Array(std_layers[Symbol(group, "_corr")]) : nothing
    M_orig = similar(M)
    sd_orig = similar(σz)
    R_orig = isnothing(R_std) ? nothing : similar(R_std)
    for c in 1:nchain, d in 1:ndraw
        D = Diagonal(σz[:, d, c])
        Σ_orig = isnothing(R_std) ? A * D * D * A' : A * D * R_std[:, :, d, c] * D * A'
        sd = sqrt.(diag(Σ_orig))
        M_orig[:, :, d, c] = A * M[:, :, d, c]
        sd_orig[:, d, c] = sd
        isnothing(R_orig) || (R_orig[:, :, d, c] = Σ_orig ./ (sd * sd'))
    end

    base = (;
        Symbol(group) => DimArray(M_orig, (Dim{:effect}(names), Dim{:group}(levels), draw_chain...)),
        Symbol(group, "_sd") => DimArray(sd_orig, (Dim{:effect}(names), draw_chain...)),
    )

    isnothing(R_orig) ||
        return merge(base, (; Symbol(group, "_corr") => DimArray(R_orig, (Dim{:effect}(names), Dim{:effect2}(names), draw_chain...))))
    has_slopes && !has_int &&
        return merge(base, (; Symbol(group, "_offset") => DimArray(-_center(mean_x, M_orig), (Dim{:group}(levels), draw_chain...))))
    return base
end

"""
    unstandardise(std_params::DimStack, tf::Transform, md::ModelData, family) -> DimStack

Back-transform `reshape_params`'s standardised-scale layers to the original data scale.
Pure rescale, no structural reshuffling — see `reshape_params`.
"""
function unstandardise(std_params::DimStack, tf::Transform, md::ModelData, family::Type{<:Distribution})
    layers = (; fixef=_unstandardise_fixef(std_params.fixef, tf, md, family))
    for (i, ranef) in enumerate(md.Z)
        layers = merge(layers, _unstandardise_ranef(std_params, tf, i, ranef, family))
    end
    return DimStack(layers)
end
