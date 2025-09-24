# fire_simulator/visualize.py

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.animation import FuncAnimation
from matplotlib.patches import FancyArrowPatch
from .fire_grid import FireGrid
from .fire_model import update_fire

def format_time(seconds: float) -> str:
    """Convert a number of seconds into HhMmSs string."""
    total = int(round(seconds))
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    parts = []
    if h:
        parts.append(f"{h}h")
    if m:
        parts.append(f"{m}min")
    # always show seconds if there’s no higher unit, or if s>0
    if s or not parts:
        parts.append(f"{s}sec")
    return "".join(parts)

def save_static_map(grid: FireGrid,
                    class_color_map: dict,
                    class_name_map: dict,
                    unknown_color=(0.6, 0.8, 1.0),
                    unknown_label="Other",
                    output_path="map_and_flammability.png"):
    """
    Saves a side-by-side figure of:
      • top-left:  Land-cover (colored by class_color_map)
      • top-right: per-cell flammability heatmap
      • bottom-left: legend for land-cover
      • bottom-right: horizontal colorbar for flammability
    """
    rows, cols = grid.rows, grid.cols

    # Build RGB landcover image
    rgb_lc = np.zeros((rows, cols, 3), dtype=float)
    for code, col in class_color_map.items():
        rgb_lc[grid.landcover == code] = col
    mask_unknown = ~np.isin(grid.landcover, list(class_color_map.keys()))
    rgb_lc[mask_unknown] = unknown_color

    # Create figure with 2×2 GridSpec
    fig = plt.figure(figsize=(10, 5), dpi=150)
    gs = fig.add_gridspec(
        nrows=2, ncols=2,
        height_ratios=[1, 0.1],
        width_ratios=[1, 1],
        hspace=0.0,
        wspace=0.1
    )

    # Top-left: landcover
    ax_lc = fig.add_subplot(gs[0, 0])
    ax_lc.imshow(rgb_lc, origin="upper")
    ax_lc.set_title("Land-Cover Map")
    ax_lc.axis("off")

    # Top-right: flammability
    ax_fm = fig.add_subplot(gs[0, 1])
    im = ax_fm.imshow(grid.flammability, cmap="YlOrBr", vmin=0, vmax=1, origin="upper")
    ax_fm.set_title("Flammability Map [0–1]")
    ax_fm.axis("off")

    # Bottom-left: legend for landcover
    ax_leg = fig.add_subplot(gs[1, 0])
    ax_leg.axis("off")
    patches = [
        Patch(color=col, label=class_name_map.get(code, str(code)))
        for code, col in class_color_map.items()
    ]
    patches.append(Patch(color=unknown_color, label=unknown_label))
    ax_leg.legend(
        handles=patches,
        loc="center",
        ncol=3,
        fontsize="small",
        frameon=False
    )

    # Bottom-right: horizontal colorbar
    ax_cbar = fig.add_subplot(gs[1, 1])
    cbar = fig.colorbar(
        im,
        cax=ax_cbar,
        orientation="horizontal",
        label="Flammability"
    )
    ax_cbar.xaxis.set_ticks_position('bottom')

    # Save & close
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"✔ Saved static map to {output_path}")

def visualize_fire(
    sim,
    class_color_map: dict,
    delay_sec: float,
    steps_per_frame: int,
    *,
    record: bool = False,
    record_path: str = "fire_simulation.mp4",
    record_fps: int = 10
):
    """
    Interactive fire simulation with blitting, wind arrow + scale bar + optional video recording.
    Window is exactly the figure canvas (no extra margins) and centered on screen.
    Arrow sizing/positioning happens only inside update_frame.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch
    from matplotlib.animation import FuncAnimation, FFMpegWriter
    from .fire_grid import FireGrid
    from .fire_model import update_fire

    # unpack
    grid       = sim.grid
    rows, cols = grid.rows, grid.cols
    cell_size  = grid.cell_size  # metres per pixel

    # static background
    rgb_bg = np.zeros((rows, cols, 3), float)
    for code, col in class_color_map.items():
        rgb_bg[grid.landcover == code] = col
    mask_unknown = ~np.isin(grid.landcover, list(class_color_map.keys()))
    rgb_bg[mask_unknown] = (0.6, 0.8, 1.0)

    # overlay for burning/burned
    overlay = np.zeros((rows, cols, 4), float)

    # figure + axes (no margins)
    fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.set_position([0, 0, 1, 1])
    ax.axis('off')
    ax.imshow(rgb_bg, origin='upper')
    im_overlay = ax.imshow(overlay, origin='upper')

    # force draw so canvas size is known
    fig.canvas.draw()

    # resize & center window
    mgr = plt.get_current_fig_manager()
    ww, wh = fig.canvas.get_width_height()
    try:
        # Qt5
        mgr.window.resize(ww, wh)
        screen = mgr.window.screen().geometry()
        sw, sh = screen.width(), screen.height()
        mgr.window.move((sw - ww)//2, (sh - wh)//2)
    except Exception:
        try:
            # TkAgg
            mgr.window.wm_geometry(
                f"{ww}x{wh}+{(mgr.window.winfo_screenwidth()-ww)//2}"
                f"+{(mgr.window.winfo_screenheight()-wh)//2}"
            )
        except Exception:
            pass

    # Wind arrow and text (will be updated each frame)
    max_arrow_cell_size = 0.15 * min(rows, cols)
    print(f"Max arrow size: {max_arrow_cell_size:.2f} cells")
    center_x_arrow = max_arrow_cell_size
    print(f"Arrow center X: {center_x_arrow:.2f} cells")
    center_y_arrow = max_arrow_cell_size
    print(f"Arrow center Y: {center_y_arrow:.2f} cells")
    arrow = FancyArrowPatch(
        (center_x_arrow, center_y_arrow), (center_x_arrow, center_y_arrow),         # tail & head at center
        transform=ax.transData,
        color='black',
        arrowstyle='-|>',
        mutation_scale=20,
        linewidth=2
    )
    ax.add_patch(arrow)

    wind_text = ax.text(
        center_x_arrow, center_y_arrow - max_arrow_cell_size/1.5,
        "",
        transform=ax.transData,
        color='white', ha='center', va='center', fontsize='small',
        bbox=dict(facecolor='black', alpha=0.5, pad=2)
    )

    # 7) step/time text (axes coords)
    time_text = ax.text(
        center_x_arrow + max_arrow_cell_size, center_y_arrow - max_arrow_cell_size/1.5,
        "",
        transform=ax.transData,
        color='white', ha='center', va='center', fontsize='small',
        bbox=dict(facecolor='black', alpha=0.5, pad=2)
    )

    # 8) scale bar (axes coords), static length fraction
    total_m   = cols * cell_size
    candidates = np.array([1, 10, 100, 1_000, 10_000, 20_000, 50_000])
    max_bar    = total_m * 0.2
    choices    = candidates[candidates <= max_bar]
    scale_len  = choices[-1] if len(choices) else candidates[0]
    if scale_len >= 1_000:
        # display in km
        scale_val = scale_len // 1_000
        label = f"{scale_val} km"
    else:
        # display in m
        label = f"{scale_len} m"
    bar_len_cells = scale_len / cell_size
    bar_center_col = center_x_arrow
    bar_center_row = rows - 100
    bar_col0 = bar_center_col - bar_len_cells/2
    bar_col1 = bar_center_col + bar_len_cells/2
    bar_row  = bar_center_row
    ax.plot([bar_col0, bar_col1], [bar_row, bar_row], transform=ax.transData, color='black', linewidth=2)

    ax.text(
        center_x_arrow, rows - 100 -50,
        label,
        transform=ax.transData,
        color='black', ha='center', va='center', fontsize='small'
    )

    # 9) click to ignite
    def on_click(event):
        if event.inaxes is not ax or event.xdata is None:
            return
        j, i = int(event.xdata + 0.5), int(event.ydata + 0.5)
        if 0 <= i < rows and 0 <= j < cols:
            sim.grid.ignite(i, j)
    fig.canvas.mpl_connect('button_press_event', on_click)

    # 10) optional recorder
    if record:
        writer = FFMpegWriter(fps=record_fps)
        writer.setup(fig, record_path, dpi=150)

    # 11) frame update
    def update_frame(frame):
        nonlocal arrow, wind_text, time_text

        # advance simulation
        for _ in range(steps_per_frame):
            sim.step()

        # update fire overlay
        overlay[...,3] = 0.0
        mask_b = (grid.grid_state == FireGrid.BURNING)
        overlay[mask_b,:3] = (0.545,0.0,0.0); overlay[mask_b,3] = 0.9
        mask_d = (grid.grid_state == FireGrid.BURNED)
        overlay[mask_d,:3] = (0.212,0.212,0.212); overlay[mask_d,3] = 0.9
        im_overlay.set_data(overlay)

        # — d) ARROW SIZING & POSITION
        arrow_size_frac     = sim.wind_speed / (sim.wind_speed_mu + 2*sim.wind_speed_sigma)
        arrow_cell_size     = arrow_size_frac * max_arrow_cell_size
        tail = (center_x_arrow, center_y_arrow)
        head = (center_x_arrow + arrow_cell_size * np.cos(sim.wind_direction),  center_y_arrow + arrow_cell_size * np.sin(sim.wind_direction))
        arrow.set_positions(tail, head)

        wind_kmh = sim.wind_speed * 3.6
        wind_text.set_text(f"{wind_kmh:.1f} km/h")

        elapsed_time = sim.step_count * sim.dt
        time_text.set_text(f"t={format_time(elapsed_time)}")

        # — f) record frame if desired
        if record:
            writer.grab_frame()

        return [im_overlay, arrow, wind_text, time_text]

    ani = FuncAnimation(fig, update_frame, blit=True, interval=delay_sec * 1000, cache_frame_data=False)

    # 12) quit‐key handler
    def on_key(event):
        if event.key in ('q','Q','escape'):
            ani.event_source.stop()
            plt.close(fig)
            if record:
                writer.finish()
                print(f"✔ Saved video to {record_path}")
    fig.canvas.mpl_connect('key_press_event', on_key)

    # 13) show
    plt.show()