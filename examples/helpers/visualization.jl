# SPDX-License-Identifier: AGPL-3.0
# Copyright © 2025 Roy Chih Chung Wang <roy.c.c.wang@proton.me>

function generatecolours(N::Int)
    # oversample since the HSV starts from red, and ends at red.
    M = round(Int, 1.3 * N)

    colours = range(Colors.HSV(0, 1, 1), stop = Colors.HSV(-360, 1, 1), length = M)
    colours = convert(Vector{Colors.RGB}, colours)
    colours = collect([colours[n].r; colours[n].g; colours[n].b] for n in 1:N)

    return colours
end

function visualize2Dpartition(
        X_parts::Vector{Vector{Vector{T}}},
        y_set,
        t_set,
        fig_num::Int,
        title_string::String;
        display_range = 1:length(X_parts),
        new_fig_flag = true
    ) where {T <: Real}

    colours_for_pts = generatecolours(length(X_parts))

    if new_fig_flag
        PLT.figure(fig_num)
        fig_num += 1
    end

    ax = PLT.axes()

    # partitions of points.
    for i in display_range

        # points.
        X = X_parts[i]

        x1 = collect(X[n][1] for n in eachindex(X))
        x2 = collect(X[n][2] for n in eachindex(X))

        PLT.scatter(x1, x2, label = "$(i)", color = colours_for_pts[i])

        # annotate centroid.
        z1, z2 = Statistics.mean(X)
        # PLT.annotate("$(i)",
        # xy=[z1;z2],
        # xycoords="data",
        # fontsize=20.0,
        # #bbox = ["boxstyle"=>"rarrow,pad=0.3", "fc"=>"cyan", "ec"=>"b", "lw"=>2])
        # bbox = (z1,z2, z1+0.5, z2+0.5) )

        #PLT.text(z1, z2, "$(i)", backgroundcolor = "white")
        bbox = Dict("boxstyle" => "round, pad=0.3", "fc" => "cyan", "ec" => "b", "lw" => 2)
        PLT.text(z1, z2, "$(i)", bbox = bbox)
    end

    ### boundaries of partitions.
    @assert length(t_set) == length(y_set)
    @assert !isempty(X_parts[1][1])

    #colours_for_boundaries = generatecolours(length(t_set))

    for i in eachindex(t_set, y_set)

        # PLT.plot(t_set[i], y_set[i], color = colours_for_boundaries[i])
        PLT.plot(t_set[i], y_set[i], color = "black")
    end

    PLT.axis("scaled")
    PLT.title(title_string)
    PLT.legend()

    return fig_num, ax
end

function visualizesingleregions(
        X_set, y_set, t_set, fig_num;
        display_region_range = 1:length(X_set)
    )

    for i in display_region_range
        fig_num, ax = visualize2Dpartition(
            X_set, y_set, t_set, fig_num,
            "X_set, levels = $(levels)"; display_range = i:i
        )

        # ## I am here. debug the visualization of the complement. then, mixGP, use X_set as training set.
        # # display the rest of the points.
        # Xc_set = collect( X_set[n] for n = 1:length(X_set) if n != i )
        # Xc_flatten = combinevectors(combinevectors(Xc_set[]))

        # Xc_display = similar(X_set)
        # resize!(Xc_display, 1)
        # Xc_display[1] = Xc_flatten

        # fig_num, ax = visualize2Dpartition(Xc_display, y_set, t_set, fig_num,
        # "Xc_set, levels = $(levels)"; display_range = i:i, new_fig_flag = false)

    end

    return fig_num
end

function visualizemeshgridpcolorx1horizontal(
        x_ranges::Vector{LinRange{T, L}},
        Y::Matrix{T},
        marker_locations::Vector,
        marker_symbol::String,
        fig_num::Int,
        title_string::String;
        x1_title_string::String = "Dimension 1",
        x2_title_string::String = "Dimension 2",
        cmap = "Greens_r"
    ) where {T <: Real, L}

    #
    @assert length(x_ranges) == 2
    x_coords = collect(collect(x_ranges[d]) for d in 1:2)

    PLT.figure(fig_num)
    fig_num += 1
    PLT.pcolormesh(x_coords[1], x_coords[2], Y, cmap = cmap, shading = "auto")
    PLT.xlabel(x1_title_string)
    PLT.ylabel(x2_title_string)
    PLT.title(title_string)

    for i in 1:length(marker_locations)
        #pt = reverse(marker_locations[i])
        pt = marker_locations[i]
        PLT.annotate(marker_symbol, xy = pt, xycoords = "data")
    end

    PLT.plt.colorbar()
    PLT.axis("scaled")

    return fig_num
end


##### debug.
## debug.
function parsedebugvarsset(N_regions::Int, debug_vars_set, dummy_val::T) where {T}

    Nq = length(debug_vars_set)

    # allocate.
    Ws_tilde = Vector{Vector{T}}(undef, Nq)
    Us = Vector{Vector{T}}(undef, Nq)
    Rs = Vector{Vector{Int}}(undef, Nq)
    Ps = Vector{Int}(undef, Nq)

    for j in 1:Nq

        ### initialize.
        Ws_tilde[j] = Vector{T}(undef, N_regions)
        Us[j] = Vector{T}(undef, N_regions)

        # debug: initialize to known value.
        fill!(Ws_tilde[j], 0.0)
        fill!(Us[j], Inf)

        ### parse.
        w_tilde = debug_vars_set[j].w_tilde_set[1]

        u = debug_vars_set[j].u_set[1]
        # v = debug_vars_set[j].v_set[1]

        region_inds = debug_vars_set[j].region_inds_set[1]
        p_region_ind = debug_vars_set[j].p_region_ind_set[1]

        @assert length(u) == length(w_tilde) == length(region_inds) + 1

        for i in 1:length(region_inds)

            r = region_inds[i]
            Ws_tilde[j][r] = w_tilde[i]
            Us[j][r] = u[i]
        end
        Ws_tilde[j][p_region_ind] = one(T)
        Us[j][p_region_ind] = u[end]

        Ps[j] = p_region_ind
        Rs[j] = region_inds

        #hps_keep_flags_set = debug_vars_set[j].hps_keep_flags_set_set[1]
        #zs_set = debug_vars_set[j].zs_set_set[1]
        #ts_set = debug_vars_set[j].ts_set_set[1]
    end

    return Ws_tilde, Us, Ps, Rs
end
