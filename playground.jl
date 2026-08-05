
using TuringRegressions
using RDatasets
#using CairoMakie

mtcars = dataset("datasets", "mtcars")

#mtcars.gp = string.(mtcars.Cyl);

model = turing_glm(@formula(MPG ~ 1 + HP + Cyl),
                    mtcars, Normal)
modelcode(model)

fit!(model; samples = 500)



