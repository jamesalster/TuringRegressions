
module TuringRegressionsMakieExt

using TuringRegressions
using Makie
using Statistics: mean, median, quantile
import TuringRegressions: lineribbon, lineribbon!, conditional_dependency, pp_check_dens, pp_check_dens_overlay, pp_check_hist
import TuringRegressions: TuringRegression, predictors, predict

"""
    lineribbon(x, y; widths=[0.66, 0.95], colorscale="Greys", kwargs...)

Create a ribbon plot showing median line with confidence interval ribbons.

# Arguments
- `x`: vector of x-positions 
- `y`: matrix where each column corresponds to x positions
- `widths`: percentile widths for intervals (default: [66, 95])

# Example
```julia
using GLMakie
# Ribbon plot
x = 1:0.1:5
y = randn(1000, length(x))  # 1000 samples for each x position
lineribbon(x, y)
```
"""
Makie.@recipe(LineRibbon, x, y) do scene
    Makie.Theme(;
        widths=[0.66, 0.95], linewidth=2.0, colorscale=:grays, linecolor=:black, alpha=1
    )
end

function Makie.plot!(plot::LineRibbon)
    # Extract values from observables
    x = plot[1][]
    y = plot[2][]
    widths = plot.widths[]
    linewidth = plot.linewidth[]
    linecolor = plot.linecolor[]
    alpha = plot.alpha[]

    if size(y, 2) != length(x)
        throw(DimensionMismatch("size(y, 2) must be equal to the length of x"))
    end

    # Calculate medians
    medians = [median(y[:, i]) for i in 1:size(y, 2)]

    # Sort widths and create colors
    sorted_widths = sort(widths; rev=true)
    cmap = reverse(Makie.to_colormap(Makie.cgrad(colorscale, 125; categorical=true))) #cut the whitest bits
    colors = cmap[round.(Int, sorted_widths*100)]

    # Calculate and plot ribbons
    for (i, width) in enumerate(sorted_widths)
        lower_p = (1 - width) / 2
        upper_p = 1 - lower_p

        lower = [quantile(y[:, j], lower_p) for j in 1:size(y, 2)]
        upper = [quantile(y[:, j], upper_p) for j in 1:size(y, 2)]

        Makie.band!(plot, x, lower, upper; color=colors[i], alpha=alpha)
    end

    # Plot median line
    Makie.lines!(plot, x, medians; color=linecolor, linewidth=linewidth)

    return plot
end


"""
    conditional_dependency(TR::TuringRegression, variable::Symbol; type=:posterior, kwargs...)

Plot how predictions change when varying one variable while holding others at their means.  
    Kwargs are passed to Makie.Figure(). 
"""
function conditional_dependency(
    TR::TuringRegression, variable::Symbol; type=:posterior, kwargs...
)
    if TR.modelinfo.has_random_effects
        @warn "conditional_dependency only varies fixed effects; random effects are held at their fitted values, not marginalised or re-predicted."
    end
    N = 200
    pp = predictors(TR, :fixef)
    id = findfirst(==(variable), TR.X_names)
    means = mean(pp; dims=1)
    predict_range = range.(extrema(pp[var = At(variable)])..., N)

    # make prediction_grid
    predgrid = zeros(N, length(means))
    for i in eachindex(means)
        if i == id
            predgrid[:, i] = collect(predict_range)
        else
            predgrid[:, i] = fill(means[i], N)
        end
    end

    preds = predict(TR, predgrid; type=type)

    # plot
    fig = Makie.Figure(kwargs...)
    ax = Makie.Axis(
        fig[1, 1];
        title="Conditional Dependency Plot",
        subtitle="Other variables held at their mean",
        ylabel="Outcome",
        xlabel=string(variable),
    )
    lineribbon!(ax, predgrid[:, id], preds')
    Makie.scatter!(ax, hcat(pp[:, id], outcome(TR)))
    fig
end

"""
    pp_check_hist(TR::TuringRegression; bins=20, type=:posterior, kwargs...)

Compare predicted vs observed values using histograms. Kwargs are passed to Makie.Figure().
"""
function pp_check_hist(TR::TuringRegression; bins=20, type=:posterior, kwargs...)
    preds = predict(median, TR; type=type)
    fig = Makie.Figure(kwargs...)
    ax = Makie.Axis(fig[1, 1]; title="Posterior Predictive Check", xlabel="Outcome")
    Makie.hist!(ax, preds; label="Predictions", bins=bins)
    Makie.hist!(ax, outcome(TR); label="Data", bins=bins)
    Makie.axislegend(ax; position=:rt)
    fig
end

"""
    pp_check_dens(TR::TuringRegression; bandwidth = Makie.automatic, type=:posterior, kwargs...)

Compare predicted vs observed values using density curves. Kwargs are passed to Makie.Figure().
"""
function pp_check_dens(
    TR::TuringRegression; bandwidth=Makie.automatic, type=:posterior, kwargs...
)
    preds = predict(median, TR; type=type)
    fig = Makie.Figure(kwargs...)
    ax = Makie.Axis(fig[1, 1]; title="Posterior Predictive Check", xlabel="Outcome")
    Makie.density!(ax, preds; bandwidth=bandwidth, label="Predictions")
    Makie.density!(ax, outcome(TR); bandwidth=bandwidth, label="Data")
    Makie.axislegend(ax; position=:rt)
    fig
end

"""
    pp_check_dens_overlay(TR::TuringRegression; n_draws=100, type=:posterior, kwargs...)

Overlay multiple prediction density curves against observed data density. Kwargs are passed to Makie.Figure().
"""
function pp_check_dens_overlay(TR::TuringRegression; n_draws=100, type=:posterior, kwargs...)
    preds = predict(TR; n_draws=n_draws, type=type)
    fig = Makie.Figure(kwargs...)
    ax = Makie.Axis(fig[1, 1]; title="Posterior Predictive Check", xlabel="Outcome")
    for i in 1:n_draws
        Makie.density!(
            ax,
            preds[:, i];
            color="#FFFFFF00",
            alpha=0.2,
            strokewidth=1,
            strokecolor=:dodgerblue,
            strokearound=true,
        )
    end
    Makie.density!(
        ax,
        outcome(TR);
        color="#FFFFFF00",
        strokewidth=2,
        strokecolor=:dodgerblue4,
        strokearound=true,
    )
    fig
end


end