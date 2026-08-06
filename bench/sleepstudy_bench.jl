# T23: fixed benchmark fit — Reaction ~ 1 + Days + (1 + Days | Subject), Normal,
# ~42 params, 180 obs. Run fresh (`julia --project=bench bench/sleepstudy_bench.jl`)
# for a cold (compile-included) number; the script's own second fit! call in the
# same process gives the warm number (model function reused via V20 cache).
# Appends each run's numbers to bench/BENCHLOG.md — a running record, not overwritten.
# Bench-only deps (RDatasets, MCMCDiagnosticTools, ReverseDiff, Mooncake) live in
# bench/Project.toml, not the package's own Project.toml.
using TuringRegressions
using RDatasets
using Random
using Dates
using MCMCDiagnosticTools: ess, rhat
using Statistics: mean

Random.seed!(1)

sleepstudy = dataset("lme4", "sleepstudy")
f = @formula(Reaction ~ 1 + Days + (1 + Days | Subject))

# Published lme4 REML gold-standard estimates (see test/runtests.jl).
const GOLD = Dict(
    "fixef α" => 251.4,
    "fixef Days" => 10.5,
    "Subject_sd Intercept" => 24.7,
    "Subject_sd Days" => 5.9,
)

# `turing_glm` eval's the model function at runtime (V20); calling it and `fit!`
# must each be a separate top-level statement, not both inside one function body,
# or the eval'd method isn't visible yet (world age).
mod_cold = turing_glm(f, sleepstudy, Normal)
cold_time = @elapsed fit!(mod_cold; samples=2000, warmup=2000, nchains=4)

mod = turing_glm(f, sleepstudy, Normal)
warm_time = @elapsed fit!(mod; samples=2000, warmup=2000, nchains=4)

fixef = draws(mod, :fixef; collapse=false)
fixef_labels = collect(dims(fixef, :fixef))

sd = draws(mod, :Subject_sd; collapse=false)
sd_labels = collect(dims(sd, :effect))

lines = String[]
push!(lines, "## $(now())")
push!(lines, "")
push!(lines, "cold (compile + fit): $(round(cold_time; digits=2))s")
push!(lines, "warm (fit only):      $(round(warm_time; digits=2))s")
push!(lines, "")
push!(lines, "| param | ESS/sec | rhat | posterior mean | gold (lme4 REML) |")
push!(lines, "|---|---|---|---|---|")

for l in fixef_labels
    data = fixef[fixef=At(l)]
    key = "fixef $l"
    gold = get(GOLD, key, nothing)
    push!(lines, "| $key | $(round(ess(data) / warm_time; digits=1)) | $(round(rhat(data); digits=3)) | $(round(mean(data); digits=2)) | $(gold === nothing ? "—" : gold) |")
end
for l in sd_labels
    data = sd[effect=At(l)]
    key = "Subject_sd $l"
    gold = get(GOLD, key, nothing)
    push!(lines, "| $key | $(round(ess(data) / warm_time; digits=1)) | $(round(rhat(data); digits=3)) | $(round(mean(data); digits=2)) | $(gold === nothing ? "—" : gold) |")
end

max_rhat = max(maximum(rhat(fixef[fixef=At(l)]) for l in fixef_labels),
               maximum(rhat(sd[effect=At(l)]) for l in sd_labels))
push!(lines, "")
push!(lines, "max rhat: $(round(max_rhat; digits=3)) $(max_rhat > 1.05 ? "⚠ REGRESSION (>1.05)" : "ok")")
push!(lines, "")

report = join(lines, "\n")
println(report)

open(joinpath(@__DIR__, "BENCHLOG.md"), "a") do io
    println(io, report)
end
