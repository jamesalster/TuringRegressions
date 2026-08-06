# T30: isolated cost of LKJCholesky's bijector transform (constrain + logabsdetjac),
# the per-gradient-eval overhead DynamicPPL pays for the `L_ranef ~ LKJCholesky(p,1.0)`
# line (model.jl:47), vs a rough per-gradient-eval budget backed out from the T23
# sleepstudy bench (p=2, correlated-slope shape). Diagnostic only, no model.jl edit.
using Bijectors, Distributions
using Random

Random.seed!(1)

p = 2  # sleepstudy: Intercept + Days
dist = LKJCholesky(p, 1.0)
x = rand(dist)
b = Bijectors.bijector(dist)
y = b(x)

N = 200_000
# warm up
b(x); Bijectors.logabsdetjac(b, x)

t_transform = @elapsed for _ in 1:N
    b(x)
end
t_jac = @elapsed for _ in 1:N
    Bijectors.logabsdetjac(b, x)
end
t_logpdf = @elapsed for _ in 1:N
    logpdf(dist, x)
end

ns_transform = t_transform / N * 1e9
ns_jac = t_jac / N * 1e9
ns_logpdf = t_logpdf / N * 1e9
ns_total = ns_transform + ns_jac + ns_logpdf

println("LKJCholesky(p=$p, η=1.0) per-call cost (forward path, no gradient):")
println("  bijector transform : $(round(ns_transform; digits=1)) ns")
println("  logabsdetjac       : $(round(ns_jac; digits=1)) ns")
println("  logpdf              : $(round(ns_logpdf; digits=1)) ns")
println("  total               : $(round(ns_total; digits=1)) ns")
println()

# Back-of-envelope per-gradient-eval budget from the T23 sleepstudy bench:
# warm fit 1.85s (post-T29 baseline, BENCHLOG), 2000 kept + 2000 warmup draws total
# across 4 chains run in parallel (MCMCThreads) => wall time ~= per-chain time.
# 4000/4 = 1000 draws/chain; NUTS typically averages ~10-20 leapfrog (=gradient) evals
# per draw for a well-adapted posterior of this size/geometry.
warm_time = 1.85
draws_per_chain = 1000
for leapfrog_per_draw in (10, 20)
    grad_evals = draws_per_chain * leapfrog_per_draw
    ns_per_eval = warm_time / grad_evals * 1e9
    pct = ns_total / ns_per_eval * 100
    println("assuming $leapfrog_per_draw leapfrog/draw: ~$(round(ns_per_eval; digits=0)) ns/grad-eval budget, LKJ forward-path = $(round(pct; digits=2))% of it")
end
