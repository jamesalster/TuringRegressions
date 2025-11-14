
"""
    lineribbon(x, y; widths=[0.66, 0.95], colormap=:greys, kwargs...)

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
        widths=[0.66, 0.95], linewidth=2.0, colorscale="Grays", linecolor=:black, alpha=1
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
    cmap = reverse(colormap("Grays", 125)) #cut the whitest bits
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
