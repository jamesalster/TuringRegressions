# T24: AD backend sweep, same fixed benchmark as sleepstudy_bench.jl (T23) but with
# `sampler=NUTS(;adtype=...)` swapped in. Run each backend as its own fresh process
# (`TR_ADTYPE=ForwardDiff julia --project=bench bench/adtype_sweep.jl`) so "cold" includes
# that backend's own first-use compile, not just the model function's. Appends to the
# same bench/BENCHLOG.md as sleepstudy_bench.jl. Bench-only deps (RDatasets,
# MCMCDiagnosticTools, ReverseDiff, Mooncake) live in bench/Project.toml.
using TuringRegressions
using Turing
import ReverseDiff
import Mooncake
import Enzyme
using RDatasets
using Random
using Dates
using MCMCDiagnosticTools: ess, rhat
using Statistics: mean

Random.seed!(1)

adtype_name = get(ENV, "TR_ADTYPE", "ForwardDiff")
adtype = if adtype_name == "ForwardDiff"
    AutoForwardDiff()
elseif adtype_name == "ReverseDiff"
    AutoReverseDiff(; compile=true)
elseif adtype_name == "Mooncake"
    AutoMooncake()
elseif adtype_name == "Enzyme"
    AutoEnzyme()
else
    error("Unknown TR_ADTYPE=$adtype_name")
end
sampler = NUTS(; adtype=adtype)

sleepstudy = dataset("lme4", "sleepstudy")
f = @formula(Reaction ~ 1 + Days + (1 + Days | Subject))

const GOLD = Dict(
    "fixef α" => 251.4,
    "fixef Days" => 10.5,
    "Subject_sd Intercept" => 24.7,
    "Subject_sd Days" => 5.9,
)

mod_cold = turing_glm(f, sleepstudy, Normal)
cold_time = @elapsed fit!(mod_cold; sampler=sampler, samples=2000, warmup=2000, nchains=4)

mod = turing_glm(f, sleepstudy, Normal)
warm_time = @elapsed fit!(mod; sampler=sampler, samples=2000, warmup=2000, nchains=4)

fixef = draws(mod, :fixef; collapse=false)
fixef_labels = collect(dims(fixef, :fixef))

sd = draws(mod, :Subject_sd; collapse=false)
sd_labels = collect(dims(sd, :effect))

lines = String[]
push!(lines, "## $(now())")
push!(lines, "")
push!(lines, "T24 adtype sweep — $(adtype_name) (`$(adtype)`)")
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
