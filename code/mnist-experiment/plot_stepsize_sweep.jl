"""Render the MNIST step-size sensitivity figure at journal-ready dimensions."""
function render_stepsize_sweep_figure(
    αs,
    accept_rates,
    αs_valid,
    Δnovelty,
    Δdiversity,
    Δenergy;
    output_dir,
)
    mkpath(output_dir)

    blue = colorant"#0072B2"
    orange = colorant"#D55E00"
    green = colorant"#009E73"
    charcoal = colorant"#303030"
    guide_gray = colorant"#666666"
    energy_display = [isfinite(x) && x < 0.001 ? NaN : x for x in Δenergy]

    common = (
        fontfamily = "Helvetica",
        background_color = :white,
        foreground_color = charcoal,
        framestyle = :box,
        grid = true,
        gridalpha = 0.16,
        gridlinewidth = 0.6,
        minorgrid = false,
        tickfontsize = 11,
        guidefontsize = 13,
        titlefontsize = 13,
        legendfontsize = 10,
        linewidth = 2.8,
        markersize = 5.5,
        markerstrokewidth = 0.8,
        dpi = 300,
    )

    p1 = plot(
        αs,
        100 .* accept_rates;
        xlabel = "Integration step size",
        ylabel = "MALA acceptance (%)",
        xscale = :log10,
        xlims = (8e-4, 0.6),
        xticks = ([1e-3, 1e-2, 1e-1], ["0.001", "0.01", "0.1"]),
        ylims = (-3, 103),
        yticks = 0:25:100,
        label = nothing,
        color = blue,
        marker = :circle,
        markercolor = :white,
        markerstrokecolor = blue,
        title = "(a)",
        titlelocation = :left,
        left_margin = 4Plots.mm,
        bottom_margin = 4Plots.mm,
        top_margin = 4Plots.mm,
        common...,
    )
    hline!(p1, [95]; color = guide_gray, linestyle = :dash, linewidth = 1.5,
        label = nothing)
    vline!(p1, [0.01]; color = charcoal, linestyle = :dot, linewidth = 1.4,
        label = nothing)
    annotate!(p1, 0.00105, 91.5, text("95%", 9, guide_gray, :left))
    annotate!(p1, 0.0115, 12, text("main protocol", 9, charcoal, :left))

    p2 = plot(
        αs_valid,
        Δnovelty;
        xlabel = "Integration step size",
        ylabel = "Absolute ULA-MALA difference",
        xscale = :log10,
        xlims = (8e-4, 0.13),
        xticks = ([1e-3, 1e-2, 1e-1], ["0.001", "0.01", "0.1"]),
        ylims = (-0.0003, 0.0122),
        yticks = 0:0.003:0.012,
        label = "Novelty",
        color = blue,
        marker = :circle,
        markercolor = :white,
        markerstrokecolor = blue,
        title = "(b)",
        titlelocation = :left,
        legend = :topleft,
        legend_columns = 1,
        background_color_legend = :white,
        foreground_color_legend = :transparent,
        left_margin = 5Plots.mm,
        right_margin = 2Plots.mm,
        bottom_margin = 4Plots.mm,
        top_margin = 4Plots.mm,
        common...,
    )
    plot!(p2, αs_valid, Δdiversity; label = "Diversity", color = orange,
        marker = :square, markercolor = :white, markerstrokecolor = orange)
    plot!(p2, αs_valid, energy_display; label = "Energy", color = green,
        marker = :diamond, markercolor = :white, markerstrokecolor = green)
    vline!(p2, [0.01]; color = charcoal, linestyle = :dot, linewidth = 1.4,
        label = nothing)

    combined = plot(
        p1,
        p2;
        layout = grid(1, 2, widths = [0.48, 0.52]),
        size = (650, 275),
        margin = 1Plots.mm,
    )

    pdf_path = joinpath(output_dir, "Fig_stepsize_sweep_ula_vs_mala.pdf")
    png_path = joinpath(output_dir, "Fig_stepsize_sweep_ula_vs_mala.png")
    savefig(combined, pdf_path)
    savefig(combined, png_path)
    return (pdf = pdf_path, png = png_path)
end
