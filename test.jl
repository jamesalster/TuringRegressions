
using Turing
using RDatasets

@model function model1(y, X)
    N, K = size(X)
    alpha ~ Normal(0, 5)
    beta ~ filldist(Normal(0, 2), K)
    sigma ~ Exponential(1)
    mu = alpha .+ X * beta
    y ~ MvNormal(mu, sigma^2 * I)
end

@model function model2(y, X)
    N, K = size(X)
    pred_prior = vcat(Normal(0,5), filldist(Normal(0, 2), K))
    sigma ~ Exponential(1)
    mu = X * pred_prior
    y ~ MvNormal(mu, sigma^2 * I)
end

df= dataset("datasets", "iris")

mod1 = model1(df[:,1], Matrix(df[:,2:4]))
fit = sample(mod1, NUTS(), MCMCThreads(), 2000, 4)
describe(fit)

preds2 = hcat(fill(1, nrow(df)), Matrix(df[:,2:4]))

mod2 = model2(df[:,1], preds2)
fit2 = sample(mod1, NUTS(), MCMCThreads(), 2000, 4)
describe(fit)