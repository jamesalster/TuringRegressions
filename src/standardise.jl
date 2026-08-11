
#### Affine standardisation, computed once outside the model
# Why: without it we couldn't reproduce brms/GLM/lme4 numbers, and it samples faster.
# Constants of the data, not the parameters, so they're computed here rather than
# redone every leapfrog step. Draws come back std-scale, mapped back in unstandardise.jl.
# ZScoreTransform also guards zero-variance columns (scale→1.0) for free.

struct Transform
    fixef::ZScoreTransform
    y::ZScoreTransform              # center=scale=false when family doesn't scale y
    ranef::Vector{ZScoreTransform}  # aligned with ModelData.Z
    # Unpacked: ZScoreTransform leaves .mean/.scale EMPTY when center/scale=false, so
    # store identity values instead of branching at every use site.
    y_mean::Float64                 # 0.0 when family doesn't scale y
    y_scale::Float64                # 1.0 when family doesn't scale y
end

# Single source of truth for family-specific behaviour, consumed by every pass.
#   scales_y  — is y standardised? Identity-link families only; rescaling counts or
#               0/1 outcomes would break their support.
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
    has_int = has_intercept(md)
    # Only the intercept can absorb the mean_x*slope shift on back-transform, so no
    # intercept, no centring. Scaling needs no such home and is always applied.
    fixef = fit(ZScoreTransform, md.predictors.X, dims=1; center=has_int)
    ranef = [fit(ZScoreTransform, re.predictors.X, dims=1; center=re.predictors.has_intercept) for re in md.Z]
    scale_y = family_spec(family).scales_y
    center_y = scale_y && has_int  # same argument, applied to y's mean
    y = fit(ZScoreTransform, md.y; center=center_y, scale=scale_y)
    y_mean = center_y ? y.mean[1] : 0.0
    y_scale = scale_y ? y.scale[1] : 1.0
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
