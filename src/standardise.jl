
#### Affine standardisation, computed once outside the model
# StatsBase.ZScoreTransform (fit/transform/reconstruct) does the per-column mean/std
# bookkeeping for us, and guards zero-variance columns (scale→1.0, not division by
# zero) for free — no need to hand-roll it.

struct Transform
    fixef::ZScoreTransform
    y::ZScoreTransform              # center=scale=false when family doesn't scale y
    ranef::Vector{ZScoreTransform}  # aligned with ModelData.Z
    y_mean::Float64                 # 0.0 when family doesn't scale y
    y_scale::Float64                # 1.0 when family doesn't scale y
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
    # No-intercept ranef terms skip centring: an intercept would absorb the mean_x*slope
    # cross term on back-transform, but there's nowhere to put it without one.
    ranef = [fit(ZScoreTransform, re.predictors.X, dims=1; center=re.predictors.has_intercept) for re in md.Z]
    scale_y = family_spec(family).scales_y
    y = scale_y ? fit(ZScoreTransform, md.y) : fit(ZScoreTransform, md.y; center=false, scale=false)
    y_mean, y_scale = scale_y ? (y.mean[1], y.scale[1]) : (0.0, 1.0)
    return Transform(fixef, y, ranef, y_mean, y_scale)
end

# fn is StatsBase.transform or StatsBase.reconstruct — same traversal either direction.
function apply_transform(fn::Function, tf::Transform, md::ModelData)
    predictors = Predictors(md.predictors.has_intercept, fn(tf.fixef, md.predictors.X), md.predictors.X_names)
    Z = [
        RandomEffect(re.variable, re.levels, re.level_index,
            Predictors(re.predictors.has_intercept, fn(xt, re.predictors.X), re.predictors.X_names))
        for (re, xt) in zip(md.Z, tf.ranef)
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
