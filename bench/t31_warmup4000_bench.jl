# T31: warmup budget sweep. Default (C1) is warmup=samples => 2000 total => 500/chain.
# Stan defaults 1000/chain. Test warmup=4000 (1000/chain) against the T29/T14 baseline
# (warmup=2000, 500/chain) on the same T23 sleepstudy harness. samples held at 2000.
using TuringRegressions
using RDatasets
using Random
using Dates
using MCMCDiagnosticTools: ess, rhat
using Statistics: mean

Random.seed!(1)

sleepstudy = dataset("lme4", "sleepstudy")
f = @formula(Reaction ~ 1 + Days + (1 + Days | Subject))

const GOLD = Dict(
    "fixef α" => 251.4,
    "fixef Days" => 10.5,
    "Subject_sd Intercept" => 24.7,
    "Subject_sd Days" => 5.9,
)

# warm up model compile first (untimed), then time the warmup=4000 fit alone
mod_warm = turing_glm(f, sleepstudy, Normal)
fit!(mod_warm; samples=2000, warmup=2000, nchains=4)

mod = turing_glm(f, sleepstudy, Normal)
warm_time = @elapsed fit!(mod; samples=2000, warmup=4000, nchains=4)

fixef = draws(mod, :fixef; collapse=false)
fixef_labels = collect(dims(fixef, :fixef))

sd = draws(mod, :Subject_sd; collapse=false)
sd_labels = collect(dims(sd, :effect))

lines = String[]
push!(lines, "## $(now()) — T31 warmup=4000 (1000/chain) vs baseline warmup=2000 (500/chain)")
push!(lines, "")
push!(lines, "warm (fit only, warmup=4000): $(round(warm_time; digits=2))s")
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
